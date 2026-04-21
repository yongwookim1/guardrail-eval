from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from ..types import Sample


def _resolve_torch_dtype(dtype_name: str):
    import torch

    value = getattr(torch, dtype_name, None)
    if value is None:
        raise ValueError(f"Unsupported torch dtype for transformers backend: {dtype_name}")
    return value


def _normalize_llama4_config(config: Any, *, attention_chunk_size: int = 8192) -> None:
    text_config = getattr(config, "text_config", None)
    targets = [target for target in (config, text_config) if target is not None]
    for target in targets:
        if getattr(target, "attention_chunk_size", None) is None:
            setattr(target, "attention_chunk_size", attention_chunk_size)


class TransformersLlama4Backend:
    """Transformers-based backend for Llama Guard 4.

    This follows the model card's recommended load path instead of the native
    vLLM model implementation, which has proven brittle for this repo's setup.
    """

    def __init__(self, model_ref: str, backend_kwargs: dict[str, Any] | None = None) -> None:
        import torch
        from transformers import AutoProcessor, Llama4ForConditionalGeneration

        backend_kwargs = backend_kwargs or {}
        self.model_ref = model_ref
        self.device = str(backend_kwargs.get("device", "cuda"))
        self.processor = AutoProcessor.from_pretrained(model_ref)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_ref,
            device_map=backend_kwargs.get("device_map", self.device),
            torch_dtype=_resolve_torch_dtype(str(backend_kwargs.get("dtype", "bfloat16"))),
        )
        _normalize_llama4_config(
            self.model.config,
            attention_chunk_size=int(backend_kwargs.get("attention_chunk_size", 8192)),
        )
        self._torch = torch

    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if sample.image_path:
            content.append({"type": "image", "path": sample.image_path})
        if sample.text:
            content.append({"type": "text", "text": sample.text})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _generation_kwargs(sampling: dict[str, Any]) -> dict[str, Any]:
        max_new_tokens = int(sampling.get("max_tokens", 20))
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
        del chat_template_kwargs  # unused in the transformers path

        outputs: list[tuple[str, float]] = []
        generation_kwargs = self._generation_kwargs(sampling)

        for sample in samples:
            messages = self._build_messages(sample)
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(self.device)
            except Exception as exc:
                raise RuntimeError(f"Failed to preprocess llama_guard_4 sample {sample.id}") from exc

            t0 = time.perf_counter()
            try:
                with self._torch.inference_mode():
                    generated = self.model.generate(**inputs, **generation_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate llama_guard_4 output for sample {sample.id}"
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
