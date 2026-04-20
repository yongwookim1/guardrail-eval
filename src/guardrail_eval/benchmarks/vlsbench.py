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

from ..types import Sample
from ._hf_common import HFFileBenchmark, extract_tar_once, load_json, open_image
from .registry import register_benchmark


@register_benchmark("vlsbench")
class VLSBench(HFFileBenchmark):
    def _prepare(self) -> tuple[list[dict[str, Any]], Path]:
        from huggingface_hub import hf_hub_download

        repo_id = self.config["hf_id"]
        metadata_file = self.config.get("metadata_file", "data.json")
        images_archive = self.config.get("images_archive", "imgs.tar")

        # Pull only the two files we need (skip the 2.5 GB parquet shards).
        metadata_path = Path(hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename=metadata_file,
        ))
        tar_path = Path(hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename=images_archive,
        ))

        # The tar extracts to <cache>/imgs/. image_paths in data.json are
        # relative to the tar root ("imgs/0.png"), so the image root is the
        # cache dir itself.
        extract_root = tar_path.parent
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
            image=open_image(image_root / rel_path),
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
