from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .backends.transformers_gemma3_backend import TransformersGemma3Backend
from .backends.transformers_qwen25_vl_backend import TransformersQwen25VLBackend
from .types import Sample


@dataclass
class ERankTokenSpans:
    active_positions: Any
    vision_positions: Any

    @property
    def prompt_token_count(self) -> int:
        return int(self.active_positions.numel())

    @property
    def vision_token_count(self) -> int:
        return int(self.vision_positions.numel())


@dataclass
class ERankSampleLayers:
    image_tokens: Any
    debug: dict[str, Any]


def _active_positions_for_row(attention_mask, input_ids, row: int):
    if attention_mask is None:
        return torch.arange(
            int(input_ids.shape[1]),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    return attention_mask[row].nonzero(as_tuple=False).flatten()


def _vision_spans_from_image_mask(input_ids, attention_mask, image_mask, row: int) -> ERankTokenSpans:
    active_positions = _active_positions_for_row(attention_mask, input_ids, row)
    row_image_mask = image_mask[row].bool()
    vision_positions = active_positions[row_image_mask[active_positions]]
    if vision_positions.numel() == 0:
        raise ValueError("No vision tokens were found in the prompt")
    return ERankTokenSpans(
        active_positions=active_positions,
        vision_positions=vision_positions,
    )


class _TransformersERankBackendMixin:
    def collect_erank_sample(self, sample: Sample) -> ERankSampleLayers:
        inputs, hidden_states = self.forward_samples_hidden_states(
            [sample],
            add_generation_prompt=True,
        )
        spans = self._extract_erank_token_spans(inputs, row=0)
        last_hidden = hidden_states[-1][0].index_select(0, spans.vision_positions).detach().cpu()
        debug = {
            "sample_id": sample.id,
            "question_id": sample.meta.get("question_id"),
            "prompt_token_count": spans.prompt_token_count,
            "vision_token_count": spans.vision_token_count,
            "first_vision_position": int(spans.vision_positions[0].item()),
            "last_vision_position": int(spans.vision_positions[-1].item()),
            "image_path": sample.image_path,
            "input_keys": sorted(inputs.keys()),
            "hidden_size": int(last_hidden.shape[1]),
        }
        return ERankSampleLayers(image_tokens=last_hidden, debug=debug)

    def _extract_erank_token_spans(self, inputs, row: int) -> ERankTokenSpans:
        raise NotImplementedError


class TransformersGemma3ERankBackend(_TransformersERankBackendMixin, TransformersGemma3Backend):
    def _extract_erank_token_spans(self, inputs, row: int) -> ERankTokenSpans:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        image_token_id = int(self.model.config.image_token_index)
        image_mask = input_ids == image_token_id
        return _vision_spans_from_image_mask(input_ids, attention_mask, image_mask, row)


class TransformersQwen25VLERankBackend(_TransformersERankBackendMixin, TransformersQwen25VLBackend):
    def _extract_erank_token_spans(self, inputs, row: int) -> ERankTokenSpans:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            image_mask = mm_token_type_ids == 1
        else:
            image_token_id = int(self.model.config.image_token_id)
            image_mask = input_ids == image_token_id
        return _vision_spans_from_image_mask(input_ids, attention_mask, image_mask, row)
