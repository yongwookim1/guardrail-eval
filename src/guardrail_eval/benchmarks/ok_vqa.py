from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from ..types import Sample
from .base import Benchmark, resolve_dataset_path
from .registry import register_benchmark


def _materialize_image(value: Any, path: Path) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.exists():
            return str(candidate.resolve())
        return None
    if isinstance(value, dict):
        raw_path = value.get("path")
        if raw_path:
            candidate = Path(str(raw_path))
            if candidate.exists():
                return str(candidate.resolve())
        raw_bytes = value.get("bytes")
        if raw_bytes:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="PNG")
            return str(path.resolve())
        return None
    if isinstance(value, Image.Image):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            value.convert("RGB").save(path, format="PNG")
        return str(path.resolve())
    return None


def _find_parquet_files(dataset_root: Path, split: str) -> list[str]:
    candidates = [
        sorted(str(path) for path in (dataset_root / "data").glob(f"{split}-*.parquet")),
        sorted(str(path) for path in dataset_root.glob(f"{split}-*.parquet")),
        sorted(str(path) for path in dataset_root.rglob(f"{split}-*.parquet")),
        sorted(str(path) for path in dataset_root.rglob("*.parquet")),
    ]
    for files in candidates:
        if files:
            return files
    raise FileNotFoundError(f"No OK-VQA parquet files found under {dataset_root} for split={split!r}")


@register_benchmark("okvqa_erank")
class OKVQAERankBenchmark(Benchmark):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.dataset_root = resolve_dataset_path(config)
        self.split = str(config.get("split", "val2014")).strip()
        self.prompt_prefix = str(config.get("prompt_prefix", "")).strip()
        self.prompt_suffix = str(config.get("prompt_suffix", "")).strip()
        self._dataset = None
        self._cache_dir = self.dataset_root / ".cache" / "okvqa_images"
        self._hf_cache_dir = self.dataset_root / ".cache" / "hf_datasets"

    def _ensure_loaded(self) -> None:
        if self._dataset is not None:
            return
        from datasets import load_dataset

        parquet_files = _find_parquet_files(self.dataset_root, self.split)
        self._dataset = load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train",
            cache_dir=str(self._hf_cache_dir),
        )

    def num_samples(self, limit: int | None = None) -> int:
        self._ensure_loaded()
        assert self._dataset is not None
        total = len(self._dataset)
        return min(total, limit) if limit is not None else total

    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        self._ensure_loaded()
        assert self._dataset is not None
        for idx, row in enumerate(self._dataset):
            if limit is not None and idx >= limit:
                break

            question_id = str(row.get("question_id", idx))
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"OK-VQA sample {question_id!r} is missing a question")

            image_path = _materialize_image(
                row.get("image"),
                self._cache_dir / f"{question_id}.png",
            )
            if not image_path:
                raise ValueError(f"OK-VQA sample {question_id!r} is missing a materializable image")

            prompt_parts = [part for part in (self.prompt_prefix, question, self.prompt_suffix) if part]
            prompt = "\n\n".join(prompt_parts)

            yield Sample(
                id=f"okvqa_{question_id}",
                text=prompt,
                image_path=image_path,
                expected_label="safe",
                category=str(row.get("question_type") or "_uncategorized"),
                meta={
                    "question_id": question_id,
                    "answers": list(row.get("answers", [])),
                    "question_type": row.get("question_type"),
                    "answer_type": row.get("answer_type"),
                    "split": self.split,
                },
            )
