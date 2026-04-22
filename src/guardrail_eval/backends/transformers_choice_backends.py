from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..types import ChoiceSample, ChoiceVerdict
from .transformers_common import TransformersMultimodalBackend
from .transformers_gemma3_backend import _encode_image_base64

DEFAULT_CHOICE_SYSTEM_PROMPT = (
    "You are a careful multimodal reasoning assistant. "
    "Choose the single best answer for the multiple-choice question."
)


class TransformersChoiceBackend(TransformersMultimodalBackend):
    error_name = "choice_model"

    def __init__(
        self,
        model_ref: str,
        *,
        system_prompt: str = DEFAULT_CHOICE_SYSTEM_PROMPT,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.system_prompt = system_prompt.strip()
        backend_kwargs = backend_kwargs or {}
        self.max_choice_rows = int(backend_kwargs.get("max_choice_rows", 0))
        super().__init__(model_ref=model_ref, backend_kwargs=backend_kwargs)

    def _prompt_messages(self, sample: ChoiceSample) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _choice_messages(self, sample: ChoiceSample, choice_target: str) -> list[dict[str, Any]]:
        messages = list(self._prompt_messages(sample))
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": choice_target}],
            }
        )
        return messages

    def _score_choice_messages(
        self,
        prompt_messages_batch: list[list[dict[str, Any]]],
        choice_messages: list[list[dict[str, Any]]],
        prompt_lengths: list[int] | None = None,
    ) -> list[float | None]:
        import torch.nn.functional as F

        processor_kwargs = self._processor_kwargs([])
        template_kwargs = self._chat_template_kwargs(None)
        apply_kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            **template_kwargs,
        }
        if processor_kwargs:
            apply_kwargs["processor_kwargs"] = processor_kwargs

        if prompt_lengths is None:
            prompt_inputs = self.processor.apply_chat_template(
                prompt_messages_batch,
                add_generation_prompt=True,
                **apply_kwargs,
            )
            prompt_attention = prompt_inputs["attention_mask"]
            prompt_lengths = [int(mask.sum().item()) for mask in prompt_attention]

        full_inputs = self.processor.apply_chat_template(
            choice_messages,
            add_generation_prompt=False,
            **apply_kwargs,
        ).to(self.device)

        inference_mode = getattr(self._torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        with context:
            outputs = self.model(**full_inputs, use_cache=False)

        logits = outputs.logits
        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs.get("attention_mask")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size(0), shift_labels.size(1))

        losses: list[float | None] = []
        for row_idx, prompt_len in enumerate(prompt_lengths):
            if attention_mask is None:
                candidate_mask = self._torch.zeros_like(shift_labels[row_idx], dtype=self._torch.bool)
                candidate_mask[prompt_len - 1 :] = True
            else:
                nonpad_positions = attention_mask[row_idx].nonzero(as_tuple=False).flatten()
                candidate_token_positions = nonpad_positions[prompt_len:]
                candidate_mask = self._torch.zeros_like(shift_labels[row_idx], dtype=self._torch.bool)
                if len(candidate_token_positions) > 0:
                    candidate_mask[candidate_token_positions - 1] = True

            candidate_count = int(candidate_mask.sum().item())
            if candidate_count == 0:
                losses.append(None)
                continue

            row_loss = per_token_loss[row_idx][candidate_mask].mean().item()
            losses.append(float(row_loss))
        return losses

    def score_choice_samples(self, samples: list[ChoiceSample]) -> list[ChoiceVerdict]:
        if not samples:
            return []

        t0 = time.perf_counter()
        prompt_messages_batch = [self._prompt_messages(sample) for sample in samples]

        processor_kwargs = self._processor_kwargs([])
        template_kwargs = self._chat_template_kwargs(None)
        apply_kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            **template_kwargs,
        }
        if processor_kwargs:
            apply_kwargs["processor_kwargs"] = processor_kwargs

        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages_batch,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        prompt_attention = prompt_inputs["attention_mask"]
        sample_prompt_lengths = [int(mask.sum().item()) for mask in prompt_attention]

        row_messages: list[list[dict[str, Any]]] = []
        row_prompt_lengths: list[int] = []
        row_refs: list[tuple[int, str]] = []
        per_sample_losses: list[dict[str, float | None]] = [
            {label: None for label in sample.choice_labels}
            for sample in samples
        ]

        for sample_idx, sample in enumerate(samples):
            prompt_messages = prompt_messages_batch[sample_idx]
            prompt_len = sample_prompt_lengths[sample_idx]
            for choice_label, choice_target in zip(sample.choice_labels, sample.choice_targets):
                row_messages.append(
                    prompt_messages
                    + [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": choice_target}],
                        }
                    ]
                )
                row_prompt_lengths.append(prompt_len)
                row_refs.append((sample_idx, choice_label))

        if not row_messages:
            return []

        max_rows = self.max_choice_rows if self.max_choice_rows > 0 else len(row_messages)
        max_rows = max(max_rows, 1)
        for start in range(0, len(row_messages), max_rows):
            end = min(start + max_rows, len(row_messages))
            chunk_losses = self._score_choice_messages(
                prompt_messages_batch=[],
                choice_messages=row_messages[start:end],
                prompt_lengths=row_prompt_lengths[start:end],
            )
            for row_offset, loss in enumerate(chunk_losses):
                sample_idx, choice_label = row_refs[start + row_offset]
                per_sample_losses[sample_idx][choice_label] = loss

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        batch_avg_latency_ms = elapsed_ms / len(samples)

        verdicts: list[ChoiceVerdict] = []
        for sample_idx, sample in enumerate(samples):
            choice_losses = per_sample_losses[sample_idx]
            valid_losses = [
                (label, loss) for label, loss in choice_losses.items() if loss is not None
            ]
            if not valid_losses:
                verdicts.append(
                    ChoiceVerdict(
                        pred_choice="error",
                        choice_losses=choice_losses,
                        raw="",
                        error_reason="no_valid_choice_losses",
                        batch_avg_latency_ms=batch_avg_latency_ms,
                    )
                )
                continue

            pred_choice = min(valid_losses, key=lambda item: item[1])[0]
            verdicts.append(
                ChoiceVerdict(
                    pred_choice=pred_choice,
                    choice_losses=choice_losses,
                    raw=pred_choice,
                    error_reason=None,
                    batch_avg_latency_ms=batch_avg_latency_ms,
                )
            )
        return verdicts


class TransformersGemma3ChoiceBackend(TransformersChoiceBackend):
    error_name = "gemma3_choice"

    def _model_class(self):
        from transformers import Gemma3ForConditionalGeneration

        return Gemma3ForConditionalGeneration

    def _prompt_messages(self, sample: ChoiceSample) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

        content: list[dict[str, Any]] = []
        for image_path in sample.image_paths:
            content.append({"type": "image", "image": _encode_image_base64(str(Path(image_path)))})
        content.append({"type": "text", "text": sample.prompt})
        messages.append({"role": "user", "content": content})
        return messages


class TransformersQwen25VLChoiceBackend(TransformersChoiceBackend):
    error_name = "qwen2_5_vl_choice"

    def _model_class(self):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration

    def _prompt_messages(self, sample: ChoiceSample) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

        content: list[dict[str, Any]] = []
        for image_path in sample.image_paths:
            content.append({"type": "image", "image": str(Path(image_path))})
        content.append({"type": "text", "text": sample.prompt})
        messages.append({"role": "user", "content": content})
        return messages


class TransformersQwen25OmniChoiceBackend(TransformersChoiceBackend):
    error_name = "qwen2_5_omni_choice"

    def _load_model(self, model_ref: str, backend_kwargs: dict[str, Any]):
        from .transformers_common import resolve_torch_dtype

        from transformers import Qwen2_5OmniForConditionalGeneration

        load_kwargs: dict[str, Any] = {
            "device_map": backend_kwargs.get("device_map", self.device),
            "torch_dtype": resolve_torch_dtype(str(backend_kwargs.get("dtype", "bfloat16"))),
            "enable_audio_output": bool(backend_kwargs.get("enable_audio_output", False)),
        }
        for key in ("attn_implementation", "trust_remote_code", "low_cpu_mem_usage"):
            if key in backend_kwargs:
                load_kwargs[key] = backend_kwargs[key]
        return Qwen2_5OmniForConditionalGeneration.from_pretrained(model_ref, **load_kwargs)

    def _model_class(self):
        from transformers import Qwen2_5OmniForConditionalGeneration

        return Qwen2_5OmniForConditionalGeneration

    def _chat_template_kwargs(self, chat_template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        kwargs = dict(chat_template_kwargs or {})
        kwargs.setdefault("load_audio_from_video", False)
        return kwargs

    def _processor_kwargs(self, samples: list[Any]) -> dict[str, Any]:
        del samples
        return {"padding": True, "use_audio_in_video": False}

    def _prompt_messages(self, sample: ChoiceSample) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

        content: list[dict[str, Any]] = []
        for image_path in sample.image_paths:
            content.append({"type": "image", "path": str(Path(image_path))})
        content.append({"type": "text", "text": sample.prompt})
        messages.append({"role": "user", "content": content})
        return messages
