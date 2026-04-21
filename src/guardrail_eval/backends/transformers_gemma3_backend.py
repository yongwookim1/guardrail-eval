from __future__ import annotations

import base64
import functools
import io
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ..types import Sample
from .transformers_llama4_backend import _resolve_torch_dtype


@functools.lru_cache(maxsize=8192)
def _encode_image_base64(path: str) -> str:
    image = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", optimize=True, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TransformersGemma3Backend:
    """Transformers-based backend for Gemma-3 style multimodal chat models."""

    def __init__(self, model_ref: str, backend_kwargs: dict[str, Any] | None = None) -> None:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        backend_kwargs = backend_kwargs or {}
        self.model_ref = model_ref
        self.device = str(backend_kwargs.get("device", "cuda"))
        self.processor = AutoProcessor.from_pretrained(model_ref)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_ref,
            device_map=backend_kwargs.get("device_map", self.device),
            torch_dtype=_resolve_torch_dtype(str(backend_kwargs.get("dtype", "bfloat16"))),
        )
        self._torch = torch

    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if sample.image_path:
            image_path = str(Path(sample.image_path))
            content.append({"type": "image", "image": _encode_image_base64(image_path)})
        if sample.text:
            content.append({"type": "text", "text": sample.text})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _generation_kwargs(sampling: dict[str, Any]) -> dict[str, Any]:
        max_new_tokens = int(sampling.get("max_tokens", 128))
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
        chat_template_kwargs = chat_template_kwargs or {}
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
                    **chat_template_kwargs,
                ).to(self.device)
            except Exception as exc:
                raise RuntimeError(f"Failed to preprocess nemotron_cs sample {sample.id}") from exc

            t0 = time.perf_counter()
            try:
                with self._torch.inference_mode():
                    generated = self.model.generate(**inputs, **generation_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate nemotron_cs output for sample {sample.id}"
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
