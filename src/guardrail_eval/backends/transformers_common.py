from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from ..types import Sample


def resolve_torch_dtype(dtype_name: str):
    import torch

    value = getattr(torch, dtype_name, None)
    if value is None:
        raise ValueError(f"Unsupported torch dtype for transformers backend: {dtype_name}")
    return value


class TransformersMultimodalBackend:
    """Shared inference loop for multimodal transformers backends."""

    default_max_tokens = 20
    error_name = "model"

    def __init__(self, model_ref: str, backend_kwargs: dict[str, Any] | None = None) -> None:
        import torch
        from transformers import AutoProcessor

        backend_kwargs = backend_kwargs or {}
        self.backend_kwargs = dict(backend_kwargs)
        self.model_ref = model_ref
        self.device = str(backend_kwargs.get("device", "cuda"))
        self.use_cache = bool(backend_kwargs.get("use_cache", True))
        self.processor = AutoProcessor.from_pretrained(model_ref)
        self.model = self._load_model(model_ref, backend_kwargs)
        self._torch = torch
        self._post_load(backend_kwargs)

    def _load_model(self, model_ref: str, backend_kwargs: dict[str, Any]):
        model_cls = self._model_class()
        load_kwargs: dict[str, Any] = {
            "device_map": backend_kwargs.get("device_map", self.device),
            "torch_dtype": resolve_torch_dtype(str(backend_kwargs.get("dtype", "bfloat16"))),
        }
        for key in ("attn_implementation", "trust_remote_code", "low_cpu_mem_usage"):
            if key in backend_kwargs:
                load_kwargs[key] = backend_kwargs[key]
        return model_cls.from_pretrained(
            model_ref,
            **load_kwargs,
        )

    def _model_class(self):
        raise NotImplementedError

    def _post_load(self, backend_kwargs: dict[str, Any]) -> None:
        del backend_kwargs

    def _build_messages(self, sample: Sample) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _chat_template_kwargs(self, chat_template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        return chat_template_kwargs or {}

    def _processor_kwargs(self, samples: list[Sample]) -> dict[str, Any]:
        del samples
        kwargs = {"padding": True}
        backend_kwargs = getattr(self, "backend_kwargs", {})
        extra = backend_kwargs.get("processor_kwargs") if isinstance(backend_kwargs, dict) else None
        if isinstance(extra, dict):
            kwargs.update(extra)
        return kwargs

    def _generation_kwargs(self, sampling: dict[str, Any]) -> dict[str, Any]:
        max_new_tokens = int(sampling.get("max_tokens", self.default_max_tokens))
        temperature = float(sampling.get("temperature", 0.0))
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": self.use_cache,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(sampling.get("top_p", 1.0))
        return kwargs

    def _messages_batch(self, samples: list[Sample]) -> list[list[dict[str, Any]]]:
        return [self._build_messages(sample) for sample in samples]

    def _sample_id_preview(self, samples: list[Sample]) -> str:
        sample_ids = ", ".join(sample.id for sample in samples[:3])
        if len(samples) > 3:
            sample_ids += ", ..."
        return sample_ids

    def _apply_messages_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        samples: list[Sample] | None = None,
        add_generation_prompt: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ):
        template_kwargs = self._chat_template_kwargs(chat_template_kwargs)
        apply_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": "pt",
            "return_dict": True,
            **template_kwargs,
        }
        if processor_kwargs:
            apply_kwargs["processor_kwargs"] = processor_kwargs

        try:
            return self.processor.apply_chat_template(messages_batch, **apply_kwargs).to(self.device)
        except Exception as exc:
            sample_ids = self._sample_id_preview(samples or [])
            batch_desc = f"batch of {len(messages_batch)} prompts"
            if samples:
                batch_desc += f" ({sample_ids})"
            raise RuntimeError(
                f"Failed to preprocess {self.error_name} {batch_desc}"
            ) from exc

    def prepare_inputs(
        self,
        samples: list[Sample],
        *,
        add_generation_prompt: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        processor_kwargs = self._processor_kwargs(samples)
        return self._apply_messages_batch(
            self._messages_batch(samples),
            samples=samples,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
            processor_kwargs=processor_kwargs,
        )

    def forward_messages_hidden_states(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        samples: list[Sample] | None = None,
        add_generation_prompt: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        if not messages_batch:
            raise ValueError("forward_messages_hidden_states() requires at least one prompt")

        inputs = self._apply_messages_batch(
            messages_batch,
            samples=samples,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
            processor_kwargs=processor_kwargs,
        )
        inference_mode = getattr(self._torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        forward_kwargs: dict[str, Any] = {
            **inputs,
            "output_hidden_states": True,
            "return_dict": True,
            "use_cache": self.use_cache,
        }
        if model_kwargs:
            forward_kwargs.update(model_kwargs)

        try:
            with context:
                outputs = self.model(**forward_kwargs)
        except Exception as exc:
            sample_ids = self._sample_id_preview(samples or [])
            batch_desc = f"batch of {len(messages_batch)} prompts"
            if samples:
                batch_desc += f" ({sample_ids})"
            raise RuntimeError(
                f"Failed to collect {self.error_name} hidden states for {batch_desc}"
            ) from exc

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(f"{self.error_name} model forward pass did not return hidden_states")
        return inputs, hidden_states

    def forward_samples_hidden_states(
        self,
        samples: list[Sample],
        *,
        add_generation_prompt: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        if not samples:
            raise ValueError("forward_samples_hidden_states() requires at least one sample")
        return self.forward_messages_hidden_states(
            self._messages_batch(samples),
            samples=samples,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
            processor_kwargs=self._processor_kwargs(samples),
            model_kwargs=model_kwargs,
        )

    def chat_samples(
        self,
        samples: list[Sample],
        *,
        sampling: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if not samples:
            return []

        generation_kwargs = self._generation_kwargs(sampling)
        inputs = self.prepare_inputs(
            samples,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )

        inference_mode = getattr(self._torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()

        t0 = time.perf_counter()
        try:
            with context:
                generated = self.model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            sample_ids = self._sample_id_preview(samples)
            raise RuntimeError(
                f"Failed to generate {self.error_name} output for batch of {len(samples)} samples ({sample_ids})"
            ) from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        batch_avg_ms = elapsed_ms / len(samples)

        prompt_tokens = inputs["input_ids"].shape[-1]
        trimmed = generated[:, prompt_tokens:]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        return [(text, batch_avg_ms) for text in texts]

    def close(self) -> None:
        self.model = None  # type: ignore[assignment]
        self.processor = None  # type: ignore[assignment]
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
