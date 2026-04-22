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
        return {"padding": True}

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
        template_kwargs = self._chat_template_kwargs(chat_template_kwargs)
        processor_kwargs = self._processor_kwargs(samples)
        messages_batch = [self._build_messages(sample) for sample in samples]

        try:
            apply_kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
                **template_kwargs,
            }
            if processor_kwargs:
                apply_kwargs["processor_kwargs"] = processor_kwargs
            inputs = self.processor.apply_chat_template(messages_batch, **apply_kwargs).to(self.device)
        except Exception as exc:
            sample_ids = ", ".join(sample.id for sample in samples[:3])
            if len(samples) > 3:
                sample_ids += ", ..."
            raise RuntimeError(
                f"Failed to preprocess {self.error_name} batch of {len(samples)} samples ({sample_ids})"
            ) from exc

        inference_mode = getattr(self._torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()

        t0 = time.perf_counter()
        try:
            with context:
                generated = self.model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            sample_ids = ", ".join(sample.id for sample in samples[:3])
            if len(samples) > 3:
                sample_ids += ", ..."
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
