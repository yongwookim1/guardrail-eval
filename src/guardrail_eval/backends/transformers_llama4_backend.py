from __future__ import annotations

from typing import Any

from ..types import Sample
from .transformers_common import TransformersMultimodalBackend


def _coerce_float_field(mapping: Any, key: str) -> None:
    if not isinstance(mapping, dict) or key not in mapping:
        return
    value = mapping[key]
    if isinstance(value, int):
        mapping[key] = float(value)


def _normalize_llama4_rope(target: Any) -> None:
    rope_parameters = getattr(target, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        _coerce_float_field(rope_parameters, "factor")
        _coerce_float_field(rope_parameters, "high_freq_factor")
        _coerce_float_field(rope_parameters, "low_freq_factor")

    rope_scaling = getattr(target, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        _coerce_float_field(rope_scaling, "factor")
        _coerce_float_field(rope_scaling, "high_freq_factor")
        _coerce_float_field(rope_scaling, "low_freq_factor")


def _normalize_llama4_config(config: Any, *, attention_chunk_size: int = 8192) -> None:
    text_config = getattr(config, "text_config", None)
    targets = [target for target in (config, text_config) if target is not None]
    for target in targets:
        if getattr(target, "attention_chunk_size", None) is None:
            setattr(target, "attention_chunk_size", attention_chunk_size)
        _normalize_llama4_rope(target)


class TransformersLlama4Backend(TransformersMultimodalBackend):
    """Transformers-based backend for Llama Guard 4.

    This follows the model card's recommended load path instead of the native
    vLLM model implementation, which has proven brittle for this repo's setup.
    """

    default_max_tokens = 20
    error_name = "llama_guard_4"

    def _model_class(self):
        from transformers import Llama4ForConditionalGeneration

        return Llama4ForConditionalGeneration

    def _post_load(self, backend_kwargs: dict[str, Any]) -> None:
        _normalize_llama4_config(
            self.model.config,
            attention_chunk_size=int(backend_kwargs.get("attention_chunk_size", 8192)),
        )

    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if sample.image_path:
            content.append({"type": "image", "path": sample.image_path})
        if sample.text:
            content.append({"type": "text", "text": sample.text})
        return [{"role": "user", "content": content}]

    def _chat_template_kwargs(self, chat_template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        del chat_template_kwargs
        return {}
