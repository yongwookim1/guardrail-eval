from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import Sample
from .transformers_common import TransformersMultimodalBackend


class TransformersQwen25VLBackend(TransformersMultimodalBackend):
    """Transformers-based backend for Qwen2.5-VL style multimodal chat models."""

    error_name = "qwen2_5_vl"

    def _model_class(self):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration

    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if sample.image_path:
            content.append({"type": "image", "image": str(Path(sample.image_path))})
        if sample.text:
            content.append({"type": "text", "text": sample.text})
        return [{"role": "user", "content": content}]
