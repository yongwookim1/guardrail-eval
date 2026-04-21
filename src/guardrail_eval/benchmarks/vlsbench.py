"""VLSBench — visual leakless multimodal safety benchmark.

HF repo: https://huggingface.co/datasets/Foreshhh/vlsbench
Layout:
    data.json       — list of {instruction_id, instruction, image_path,
                      category, sub_category, source, image_description,
                      safety_reason}
    imgs.tar        — bundle of images; each record's `image_path` resolves
                      inside the extracted directory (typically "imgs/<n>.png")

All 2,241 samples are expected to be flagged `unsafe`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import load_json
from ..types import Sample
from .base import resolve_dataset_path
from ._hf_common import LocalFileBenchmark, extract_tar_once
from .registry import register_benchmark


@register_benchmark("vlsbench")
class VLSBench(LocalFileBenchmark):
    def _prepare(self) -> tuple[list[dict[str, Any]], Path]:
        local = resolve_dataset_path(self.config)
        metadata_file = self.config.get("metadata_file", "data.json")
        images_archive = self.config.get("images_archive", "imgs.tar")
        metadata_path = local / metadata_file
        tar_path = local / images_archive
        if not metadata_path.exists():
            raise FileNotFoundError(f"VLSBench metadata file not found: {metadata_path}")
        if not tar_path.exists():
            raise FileNotFoundError(f"VLSBench images archive not found: {tar_path}")

        # The tar extracts to <cache>/imgs/. image_paths in data.json are
        # relative to the tar root ("imgs/0.png"), so the image root is the
        # dataset dir itself.
        extract_root = local
        extract_tar_once(tar_path, into=extract_root, sentinel_subdir="imgs")

        records = load_json(metadata_path)
        if not isinstance(records, list):
            raise ValueError(f"{metadata_file} is not a JSON list at the top level")
        return records, extract_root

    def _record_to_sample(self, idx: int, record: dict[str, Any], image_root: Path) -> Sample:
        iid = record.get("instruction_id", idx)
        rel_path = record["image_path"]
        return Sample(
            id=f"vlsbench_{iid}",
            text=record["instruction"],
            image_path=str((image_root / rel_path).resolve()),
            expected_label=self.expected_label,  # type: ignore[arg-type]
            category=record.get("category"),
            meta={
                "image_path": rel_path,
                "sub_category": record.get("sub_category"),
                "source": record.get("source"),
                "image_description": record.get("image_description"),
                "safety_reason": record.get("safety_reason"),
            },
        )
