"""MMMU-Pro standard (10 options) local-parquet loader."""
from __future__ import annotations

import ast
import io
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from ..types import CHOICE_LETTERS, ChoiceSample
from .base import MultipleChoiceBenchmark, resolve_dataset_path
from .registry import register_benchmark

DEFAULT_PROMPT_PREFIX = (
    "Answer with the option letter from the given choices directly."
)


def _parse_options(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [line.strip() for line in text.splitlines() if line.strip()]
    raise TypeError(f"Unsupported MMMU-Pro options payload: {type(raw)!r}")


def _normalize_answer(raw_answer: Any, options: list[str]) -> str:
    answer = str(raw_answer).strip()
    upper_answer = answer.upper()
    if upper_answer in CHOICE_LETTERS[: len(options)]:
        return upper_answer
    for idx, option in enumerate(options):
        if answer == option:
            return CHOICE_LETTERS[idx]
    raise ValueError(f"Unable to normalize MMMU-Pro answer {raw_answer!r}")


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


def _find_parquet_files(dataset_root: Path, subset_dir: str, split: str) -> list[str]:
    subset_root = dataset_root / subset_dir
    candidates = [
        sorted(str(path) for path in subset_root.glob(f"{split}-*.parquet")),
        sorted(str(path) for path in (subset_root / split).glob("*.parquet")),
        sorted(str(path) for path in subset_root.glob("*.parquet")) if split == "test" else [],
        sorted(str(path) for path in dataset_root.glob("*.parquet")),
    ]
    for files in candidates:
        if files:
            return files
    raise FileNotFoundError(
        f"No MMMU-Pro parquet files found under {dataset_root} (subset={subset_dir!r}, split={split!r})"
    )


@register_benchmark("mmmu_pro")
class MMMUProBenchmark(MultipleChoiceBenchmark):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.dataset_root = resolve_dataset_path(config)
        self.subset_dir = str(config.get("subset_dir", "standard (10 options)"))
        self.split = str(config.get("split", "test"))
        self.prompt_prefix = str(config.get("prompt_prefix", DEFAULT_PROMPT_PREFIX)).strip()
        self._dataset = None
        self._cache_dir = self.dataset_root / ".cache" / "mmmu_pro_images"

    def _ensure_loaded(self) -> None:
        if self._dataset is not None:
            return
        from datasets import load_dataset

        parquet_files = _find_parquet_files(self.dataset_root, self.subset_dir, self.split)
        self._dataset = load_dataset("parquet", data_files=parquet_files, split="train")

    def num_samples(self, limit: int | None = None) -> int:
        self._ensure_loaded()
        assert self._dataset is not None
        n = len(self._dataset)
        return min(n, limit) if limit is not None else n

    def iter_choice_samples(self, limit: int | None = None) -> Iterator[ChoiceSample]:
        self._ensure_loaded()
        assert self._dataset is not None
        for idx, row in enumerate(self._dataset):
            if limit is not None and idx >= limit:
                break

            options = _parse_options(row["options"])
            if not options:
                raise ValueError(f"MMMU-Pro sample {row.get('id', idx)!r} has no options")
            choice_labels = list(CHOICE_LETTERS[: len(options)])
            correct_choice = _normalize_answer(row["answer"], options)

            image_paths: list[str] = []
            for image_idx in range(1, 8):
                image_key = f"image_{image_idx}"
                if image_key not in row:
                    continue
                materialized = _materialize_image(
                    row.get(image_key),
                    self._cache_dir / f"{row.get('id', idx)}_{image_key}.png",
                )
                if materialized:
                    image_paths.append(materialized)

            question = str(row.get("question", "")).strip()
            options_text = "\n".join(
                f"{label}. {option}" for label, option in zip(choice_labels, options)
            )
            prompt = (
                f"{self.prompt_prefix}\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer:"
            )

            yield ChoiceSample(
                id=str(row.get("id", idx)),
                prompt=prompt,
                choice_labels=choice_labels,
                choice_targets=choice_labels,
                correct_choice=correct_choice,
                image_paths=image_paths,
                category=str(row.get("subject") or "_uncategorized"),
                meta={
                    "subset_dir": self.subset_dir,
                    "split": self.split,
                    "subject": row.get("subject"),
                    "img_type": row.get("img_type"),
                    "topic_difficulty": row.get("topic_difficulty"),
                    "options": options,
                },
            )
