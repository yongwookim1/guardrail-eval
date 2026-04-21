"""HoliSafe-Bench multimodal safety benchmark.

Dataset card: https://huggingface.co/datasets/etri-vilab/holisafe-bench

Local layout expected by this repo:
    holisafe_bench.json   — list of records with fields including:
                            {id, image, query, category, subcategory,
                             type, image_safe, image_safety_label}
    images/<...>          — image tree referenced by record["image"]

Unlike SIUO and VLSBench, HoliSafe is a mixed-label benchmark. The final
expected label is derived from the last character of the dataset's `type`
field, whose notation is [Image][Query][Final input safeness].
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import load_json
from ..types import Sample
from ._hf_common import LocalFileBenchmark
from .base import resolve_dataset_path
from .registry import register_benchmark


def _expected_label_from_type(type_code: str) -> str:
    code = str(type_code).strip().upper()
    if len(code) < 3 or code[-1] not in {"S", "U"}:
        raise ValueError(f"Unsupported HoliSafe type code: {type_code!r}")
    return "safe" if code[-1] == "S" else "unsafe"


@register_benchmark("holisafe")
class HoliSafe(LocalFileBenchmark):
    def _prepare(self) -> tuple[list[dict[str, Any]], Path]:
        metadata_file = self.config.get("metadata_file", "holisafe_bench.json")
        images_dir = self.config.get("images_dir", "images")
        local = resolve_dataset_path(self.config)
        metadata_path = local / metadata_file
        image_root = local / images_dir
        if not metadata_path.exists():
            raise FileNotFoundError(f"HoliSafe metadata file not found: {metadata_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"HoliSafe images directory not found: {image_root}")

        records = load_json(metadata_path)
        if not isinstance(records, list):
            raise ValueError(f"{metadata_file} is not a JSON list at the top level")
        return records, image_root

    def _record_to_sample(self, idx: int, record: dict[str, Any], image_root: Path) -> Sample:
        sample_id = record.get("id", idx)
        rel_path = str(record["image"])
        type_code = str(record["type"])
        expected_label = _expected_label_from_type(type_code)
        return Sample(
            id=f"holisafe_{sample_id}",
            text=record.get("query"),
            image_path=str((image_root / rel_path).resolve()),
            expected_label=expected_label,  # type: ignore[arg-type]
            category=record.get("category"),
            meta={
                "image_path": rel_path,
                "subcategory": record.get("subcategory"),
                "type": type_code,
                "image_safe": record.get("image_safe"),
                "image_safety_label": record.get("image_safety_label"),
            },
        )
