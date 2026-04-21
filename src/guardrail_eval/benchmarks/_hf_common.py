"""Shared helpers for benchmarks that live as raw local files.

These benchmarks store JSON metadata plus image files or image archives rather
than a uniform parquet `Image()` feature. Loaders therefore resolve a local
dataset directory, read JSON metadata, and hand image paths through to the
evaluation pipeline without decoding them up front.
"""
from __future__ import annotations

import tarfile
from abc import abstractmethod
from pathlib import Path
from typing import Iterator

from ..types import Sample
from .base import Benchmark


class LocalFileBenchmark(Benchmark):
    """Base for benchmarks backed by local JSON metadata + image files."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._records: list[dict[str, Any]] | None = None
        self._image_root: Path | None = None

    # ---- subclass hooks ----------------------------------------------------

    @abstractmethod
    def _prepare(self) -> tuple[list[dict[str, Any]], Path]:
        """Resolve required files and return (records, image_root)."""

    @abstractmethod
    def _record_to_sample(self, idx: int, record: dict[str, Any], image_root: Path) -> Sample:
        """Map a raw record + image root to a Sample."""

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
