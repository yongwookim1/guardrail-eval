"""Shared helpers for benchmarks that live as raw files in a HF dataset repo.

Both SIUO and VLSBench ship as `{metadata.json, image_dir_or_tar}` rather than
as uniform parquet with an `Image()` feature. That means the generic
`datasets.load_dataset` path is awkward — we'd still need out-of-band image
resolution. Instead, each loader:

  1. Fetches only the files it needs with `huggingface_hub.hf_hub_download`
     (or `snapshot_download` with `allow_patterns`).
  2. Reads the JSON metadata.
  3. Resolves images from the local cache directory on iteration.
"""
from __future__ import annotations

import json
import tarfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Iterator

from PIL import Image as PILImage

from ..types import Sample
from .base import Benchmark


class HFFileBenchmark(Benchmark):
    """Base for benchmarks whose HF repo stores JSON metadata + image files."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._records: list[dict[str, Any]] | None = None
        self._image_root: Path | None = None

    # ---- subclass hooks ----------------------------------------------------

    @abstractmethod
    def _prepare(self) -> tuple[list[dict[str, Any]], Path]:
        """Download required files and return (records, image_root)."""

    @abstractmethod
    def _record_to_sample(self, idx: int, record: dict[str, Any], image_root: Path) -> Sample:
        """Map a raw record + image root to a Sample (loads the PIL image)."""

    # ---- Benchmark protocol ------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._records is None:
            self._records, self._image_root = self._prepare()

    def num_samples(self, limit: int | None = None) -> int:
        self._ensure_loaded()
        assert self._records is not None
        n = len(self._records)
        return min(n, limit) if limit is not None else n

    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        self._ensure_loaded()
        assert self._records is not None and self._image_root is not None
        for idx, record in enumerate(self._records):
            if limit is not None and idx >= limit:
                break
            yield self._record_to_sample(idx, record, self._image_root)


# ---- utilities -------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def open_image(path: Path) -> PILImage.Image:
    """Open and fully load a PIL image (so the file handle can be released)."""
    img = PILImage.open(path)
    img.load()
    return img


def extract_tar_once(tar_path: Path, into: Path, sentinel_subdir: str) -> Path:
    """Extract `tar_path` into `into` if `into/sentinel_subdir` doesn't exist.

    Returns the sentinel path. Idempotent — safe to call every run.
    """
    target = into / sentinel_subdir
    if target.exists():
        return target
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(into)
    return target
