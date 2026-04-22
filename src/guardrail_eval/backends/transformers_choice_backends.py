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
        prompt_messages: list[dict[str, Any]],
        choice_messages: list[list[dict[str, Any]]],
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

        prompt_inputs = self.processor.apply_chat_template(
            [prompt_messages],
            add_generation_prompt=True,
            **apply_kwargs,
        ).to(self.device)
        full_inputs = self.processor.apply_chat_template(
            choice_messages,
            add_generation_prompt=False,
            **apply_kwargs,
        ).to(self.device)

        prompt_attn = prompt_inputs["attention_mask"][0]
        prompt_len = int(prompt_attn.sum().item())

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
        for row_idx in range(shift_labels.size(0)):
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

        verdicts: list[ChoiceVerdict] = []
        for sample in samples:
            t0 = time.perf_counter()
            prompt_messages = self._prompt_messages(sample)
            choice_messages = [
                self._choice_messages(sample, choice_target)
                for choice_target in sample.choice_targets
            ]
            losses = self._score_choice_messages(prompt_messages, choice_messages)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            choice_losses = {
                label: loss for label, loss in zip(sample.choice_labels, losses)
            }
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
                        batch_avg_latency_ms=elapsed_ms,
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
                    batch_avg_latency_ms=elapsed_ms,
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

    def _model_class(self):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        return Qwen2_5OmniThinkerForConditionalGeneration

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
