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

from ..types import Sample
from ._hf_common import HFFileBenchmark, load_json, open_image
from .registry import register_benchmark


@register_benchmark("siuo")
class SIUO(HFFileBenchmark):
    def _prepare(self) -> tuple[list[dict[str, Any]], Path]:
        from huggingface_hub import snapshot_download

        prompts_file = self.config.get("prompts_file", "siuo_gen.json")
        images_dir = self.config.get("images_dir", "images")

        # Whole repo is ~210 MB; cheaper than ad-hoc per-file downloads.
        local = Path(snapshot_download(
            repo_id=self.config["hf_id"],
            repo_type="dataset",
            allow_patterns=[prompts_file, f"{images_dir}/*"],
        ))

        records = load_json(local / prompts_file)
        if not isinstance(records, list):
            raise ValueError(f"{prompts_file} is not a JSON list at the top level")
        return records, local / images_dir

    def _record_to_sample(self, idx: int, record: dict[str, Any], image_root: Path) -> Sample:
        qid = record.get("question_id", idx)
        image_name = record["image"]
        return Sample(
            id=f"siuo_{qid}",
            text=record["question"],
            image=open_image(image_root / image_name),
            expected_label=self.expected_label,  # type: ignore[arg-type]
            category=record.get("category"),
            meta={
                "image_file": image_name,
                "safety_warning": record.get("safety_warning"),
                "reference_answer": record.get("reference_answer"),
            },
        )
