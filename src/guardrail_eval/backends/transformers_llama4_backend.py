from __future__ import annotations

import time
from typing import Any

from ..types import Sample


def _resolve_torch_dtype(dtype_name: str):
    import torch

    value = getattr(torch, dtype_name, None)
    if value is None:
        raise ValueError(f"Unsupported torch dtype for transformers backend: {dtype_name}")
    return value


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
        self._torch = torch

    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if sample.image_path:
            content.append({"type": "image", "path": sample.image_path})
        if sample.text:
            content.append({"type": "text", "text": sample.text})
        return [{"role": "user", "content": content}]

    def chat_samples(
        self,
        samples: list[Sample],
        *,
        sampling: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        del chat_template_kwargs  # unused in the transformers path

        outputs: list[tuple[str, float]] = []
        max_new_tokens = int(sampling.get("max_tokens", 20))
        temperature = float(sampling.get("temperature", 0.0))
        top_p = float(sampling.get("top_p", 1.0))
        do_sample = temperature > 0.0

        for sample in samples:
            messages = self._build_messages(sample)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)

            t0 = time.perf_counter()
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )
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
