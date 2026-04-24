from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .backends.transformers_gemma3_backend import (
    TransformersGemma3Backend,
    _encode_image_base64,
)
from .backends.transformers_qwen25_vl_backend import TransformersQwen25VLBackend
from .types import Sample


@dataclass
class MIRTokenSpans:
    active_positions: Any
    vision_positions: Any
    text_positions: Any

    @property
    def prompt_token_count(self) -> int:
        return int(self.active_positions.numel())

    @property
    def vision_token_count(self) -> int:
        return int(self.vision_positions.numel())

    @property
    def text_token_count(self) -> int:
        return int(self.text_positions.numel())


@dataclass
class MIRSampleLayers:
    vision_layers: list[Any]
    text_layers: list[Any]
    debug: dict[str, Any]


def _active_positions_for_row(attention_mask, input_ids, row: int):
    if attention_mask is None:
        return torch.arange(
            int(input_ids.shape[1]),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    return attention_mask[row].nonzero(as_tuple=False).flatten()


def _spans_from_image_mask(input_ids, attention_mask, image_mask, row: int) -> MIRTokenSpans:
    active_positions = _active_positions_for_row(attention_mask, input_ids, row)
    row_image_mask = image_mask[row].bool()
    vision_positions = active_positions[row_image_mask[active_positions]]
    if vision_positions.numel() == 0:
        raise ValueError("No vision tokens were found in the prompt")

    last_vision_pos = int(vision_positions[-1].item())
    text_positions = active_positions[active_positions > last_vision_pos]
    if text_positions.numel() == 0:
        raise ValueError("No text tokens were found after the vision-token span")

    return MIRTokenSpans(
        active_positions=active_positions,
        vision_positions=vision_positions,
        text_positions=text_positions,
    )


class _TransformersMIRBackendMixin:
    def collect_mir_sample(self, *, sample_id: str, image_path: str, text: str) -> MIRSampleLayers:
        sample = Sample(
            id=sample_id,
            text=text,
            image_path=image_path,
            expected_label="safe",
        )
        inputs, hidden_states = self.forward_samples_hidden_states(
            [sample],
            add_generation_prompt=False,
        )
        spans = self._extract_mir_token_spans(inputs, row=0)
        vision_layers = [
            layer_hidden[0].index_select(0, spans.vision_positions).detach().cpu()
            for layer_hidden in hidden_states
        ]
        text_layers = [
            layer_hidden[0].index_select(0, spans.text_positions).detach().cpu()
            for layer_hidden in hidden_states
        ]

        input_ids = inputs["input_ids"][0]
        debug = {
            "sample_id": sample_id,
            "image_path": image_path,
            "prompt_token_count": spans.prompt_token_count,
            "vision_token_count": spans.vision_token_count,
            "text_token_count": spans.text_token_count,
            "first_vision_position": int(spans.vision_positions[0].item()),
            "last_vision_position": int(spans.vision_positions[-1].item()),
            "first_text_position": int(spans.text_positions[0].item()),
            "last_text_position": int(spans.text_positions[-1].item()),
            "input_ids_length": int(input_ids.shape[0]),
            "input_keys": sorted(inputs.keys()),
        }
        return MIRSampleLayers(
            vision_layers=vision_layers,
            text_layers=text_layers,
            debug=debug,
        )

    def _extract_mir_token_spans(self, inputs, row: int) -> MIRTokenSpans:
        raise NotImplementedError


class TransformersGemma3MIRBackend(_TransformersMIRBackendMixin, TransformersGemma3Backend):
    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        if sample.image_path:
            user_content.append(
                {"type": "image", "image": _encode_image_base64(str(Path(sample.image_path)))}
            )

        assistant_text = (sample.text or "").strip()
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

    def _extract_mir_token_spans(self, inputs, row: int) -> MIRTokenSpans:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        image_token_id = int(self.model.config.image_token_index)
        image_mask = input_ids == image_token_id
        return _spans_from_image_mask(input_ids, attention_mask, image_mask, row)


class TransformersQwen25VLMIRBackend(_TransformersMIRBackendMixin, TransformersQwen25VLBackend):
    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        if sample.image_path:
            user_content.append({"type": "image", "image": str(Path(sample.image_path))})

        assistant_text = (sample.text or "").strip()
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

    def _extract_mir_token_spans(self, inputs, row: int) -> MIRTokenSpans:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            image_mask = mm_token_type_ids == 1
        else:
            image_token_id = int(self.model.config.image_token_id)
            image_mask = input_ids == image_token_id
        return _spans_from_image_mask(input_ids, attention_mask, image_mask, row)
