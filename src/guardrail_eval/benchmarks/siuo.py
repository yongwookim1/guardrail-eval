"""SIUO — Safe Inputs but Unsafe Output, generation task.

HF repo: https://huggingface.co/datasets/sinwang/SIUO (imagefolder + JSONs).
Layout used here:
    siuo_gen.json   — list of {question_id, image, question, category,
                      safety_warning, reference_answer}
    images/<name>.png   — referenced by record["image"]

All 168 samples are expected to be flagged `unsafe`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import load_json
from ..types import Sample
from .base import resolve_dataset_path
from ._hf_common import LocalFileBenchmark
from .registry import register_benchmark


@register_benchmark("siuo")
class SIUO(LocalFileBenchmark):
    def _prepare(self) -> tuple[list[dict[str, Any]], Path]:
        prompts_file = self.config.get("prompts_file", "siuo_gen.json")
        images_dir = self.config.get("images_dir", "images")
        local = resolve_dataset_path(self.config)
        prompts_path = local / prompts_file
        image_root = local / images_dir
        if not prompts_path.exists():
            raise FileNotFoundError(f"SIUO prompts file not found: {prompts_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"SIUO images directory not found: {image_root}")
        records = load_json(local / prompts_file)
        if not isinstance(records, list):
            raise ValueError(f"{prompts_file} is not a JSON list at the top level")
        return records, image_root

    def _record_to_sample(self, idx: int, record: dict[str, Any], image_root: Path) -> Sample:
        qid = record.get("question_id", idx)
        image_name = record["image"]
        return Sample(
            id=f"siuo_{qid}",
            text=record["question"],
            image_path=str((image_root / image_name).resolve()),
            expected_label=self.expected_label,  # type: ignore[arg-type]
            category=record.get("category"),
            meta={
                "image_file": image_name,
                "safety_warning": record.get("safety_warning"),
                "reference_answer": record.get("reference_answer"),
            },
        )
