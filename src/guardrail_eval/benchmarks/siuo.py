"""SIUO — Safe Inputs but Unsafe Output (multimodal safety benchmark).

Dataset: MMInstruction/SIUO on the Hugging Face Hub. Every sample is expected
to be flagged `unsafe` by a guardrail model.
"""
from __future__ import annotations

from typing import Iterator

from ..types import Sample
from ._hf_common import iter_hf_samples
from .base import Benchmark
from .registry import register_benchmark


@register_benchmark("siuo")
class SIUO(Benchmark):
    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        yield from iter_hf_samples(
            hf_id=self.config["hf_id"],
            split=self.config.get("split", "test"),
            columns=self.columns,
            expected_label=self.expected_label,
            name=self.name,
            limit=limit,
        )
