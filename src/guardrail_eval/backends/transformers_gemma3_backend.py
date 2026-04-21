from __future__ import annotations

import base64
import functools
import io
from pathlib import Path

from PIL import Image

from ..io import IMAGE_CACHE_MAXSIZE
from ..types import Sample
from .transformers_common import TransformersMultimodalBackend


@functools.lru_cache(maxsize=IMAGE_CACHE_MAXSIZE)
def _encode_image_base64(path: str) -> str:
    image = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", optimize=True, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TransformersGemma3Backend(TransformersMultimodalBackend):
    """Transformers-based backend for Gemma-3 style multimodal chat models."""

    default_max_tokens = 128
    error_name = "nemotron_cs"

    def _model_class(self):
        from transformers import Gemma3ForConditionalGeneration

        return Gemma3ForConditionalGeneration

    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if sample.image_path:
            image_path = str(Path(sample.image_path))
            content.append({"type": "image", "image": _encode_image_base64(image_path)})
        if sample.text:
            content.append({"type": "text", "text": sample.text})
        return [{"role": "user", "content": content}]
