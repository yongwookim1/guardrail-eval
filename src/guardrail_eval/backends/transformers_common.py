from __future__ import annotations

import time
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
        self.processor = AutoProcessor.from_pretrained(model_ref)
        self.model = self._load_model(model_ref, backend_kwargs)
        self._torch = torch
        self._post_load(backend_kwargs)

    def _load_model(self, model_ref: str, backend_kwargs: dict[str, Any]):
        model_cls = self._model_class()
        return model_cls.from_pretrained(
            model_ref,
            device_map=backend_kwargs.get("device_map", self.device),
            torch_dtype=resolve_torch_dtype(str(backend_kwargs.get("dtype", "bfloat16"))),
        )

    def _model_class(self):
        raise NotImplementedError

    def _post_load(self, backend_kwargs: dict[str, Any]) -> None:
        del backend_kwargs

    def _build_messages(self, sample: Sample) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _chat_template_kwargs(self, chat_template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        return chat_template_kwargs or {}

    @classmethod
    def _generation_kwargs(cls, sampling: dict[str, Any]) -> dict[str, Any]:
        max_new_tokens = int(sampling.get("max_tokens", cls.default_max_tokens))
        temperature = float(sampling.get("temperature", 0.0))
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": False,
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
        outputs: list[tuple[str, float]] = []
        generation_kwargs = self._generation_kwargs(sampling)
        template_kwargs = self._chat_template_kwargs(chat_template_kwargs)

        for sample in samples:
            messages = self._build_messages(sample)
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    **template_kwargs,
                ).to(self.device)
            except Exception as exc:
                raise RuntimeError(f"Failed to preprocess {self.error_name} sample {sample.id}") from exc

            t0 = time.perf_counter()
            try:
                with self._torch.inference_mode():
                    generated = self.model.generate(**inputs, **generation_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate {self.error_name} output for sample {sample.id}"
                ) from exc
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            trimmed = generated[:, inputs["input_ids"].shape[-1]:]
            text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            outputs.append((text, elapsed_ms))

        return outputs

    def close(self) -> None:
        self.model = None  # type: ignore[assignment]
        self.processor = None  # type: ignore[assignment]
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
