"""MMBench local parquet loader with circular option shifts."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from ..types import CHOICE_LETTERS, ChoiceSample
from .base import MultipleChoiceBenchmark, resolve_dataset_path
from .registry import register_benchmark

DEFAULT_PROMPT_SUFFIX = "Please select the correct answer from the options above."
_SPLIT_ALIASES = {
    "validation": {"validation", "val", "dev"},
    "val": {"validation", "val", "dev"},
    "dev": {"validation", "val", "dev"},
    "test": {"test"},
    "train": {"train"},
}


def _normalize_split(raw_split: Any) -> str:
    text = str(raw_split or "").strip().lower()
    if text in {"validation", "val", "dev"}:
        return "dev"
    return text


def _split_matches(raw_split: Any, wanted_split: str) -> bool:
    normalized = _normalize_split(raw_split)
    aliases = _SPLIT_ALIASES.get(wanted_split, {wanted_split})
    return normalized in {_normalize_split(alias) for alias in aliases}


def _collect_options(record: dict[str, Any]) -> list[str]:
    options: list[str] = []
    for label in ("A", "B", "C", "D"):
        value = record.get(label)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            options.append(text)
    if len(options) < 2:
        raise ValueError(f"MMBench sample {record.get('index')!r} has fewer than 2 options")
    return options


def _normalize_answer(record: dict[str, Any], options: list[str]) -> str:
    raw_answer = record.get("answer", record.get("label"))
    if raw_answer is None:
        raise ValueError(f"MMBench sample {record.get('index')!r} is missing an answer")

    if isinstance(raw_answer, int):
        if 0 <= raw_answer < len(options):
            return CHOICE_LETTERS[raw_answer]
        raise ValueError(f"MMBench sample {record.get('index')!r} has invalid numeric answer {raw_answer!r}")

    answer = str(raw_answer).strip()
    upper_answer = answer.upper()
    if upper_answer in CHOICE_LETTERS[: len(options)]:
        return upper_answer
    if answer.isdigit():
        idx = int(answer)
        if 0 <= idx < len(options):
            return CHOICE_LETTERS[idx]
    for idx, option in enumerate(options):
        if answer == option:
            return CHOICE_LETTERS[idx]
    raise ValueError(f"Unable to normalize MMBench answer {raw_answer!r} for sample {record.get('index')!r}")


def _decode_base64_image(data: str, path: Path) -> str:
    payload = data.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
    return str(path.resolve())


def _materialize_image(value: Any, path: Path) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            candidate = Path(text)
            if candidate.exists():
                return str(candidate.resolve())
        except OSError:
            pass
        return _decode_base64_image(text, path)
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


def _build_prompt(question: str, hint: str | None, labeled_options: list[tuple[str, str]], prompt_suffix: str) -> str:
    parts: list[str] = []
    if hint:
        hint_text = hint.strip()
        if hint_text:
            parts.append(f"Hint: {hint_text}")
    parts.append(f"Question: {question.strip()}")
    options_text = "\n".join(f"{label}. {option}" for label, option in labeled_options)
    parts.append(f"Options:\n{options_text}")
    if prompt_suffix:
        parts.append(prompt_suffix)
    return "\n\n".join(parts).strip()


def _find_parquet_files(dataset_root: Path) -> list[str]:
    files = sorted(str(path) for path in dataset_root.rglob("*.parquet"))
    if files:
        return files
    raise FileNotFoundError(f"No parquet files found under {dataset_root}")


@register_benchmark("mmbench")
class MMBenchBenchmark(MultipleChoiceBenchmark):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.dataset_root = resolve_dataset_path(config)
        self.split = str(config.get("split", "validation")).strip().lower()
        self.prompt_suffix = str(config.get("prompt_suffix", DEFAULT_PROMPT_SUFFIX)).strip()
        self.circular_eval = bool(config.get("circular_eval", True))
        self._cache_dir = self.dataset_root / ".cache" / "mmbench_images"
        self._hf_cache_dir = self.dataset_root / ".cache" / "hf_datasets"
        self._records: list[dict[str, Any]] | None = None

    def _ensure_loaded(self) -> None:
        if self._records is not None:
            return
        from datasets import load_dataset

        parquet_files = _find_parquet_files(self.dataset_root)
        dataset = load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train",
            cache_dir=str(self._hf_cache_dir),
        )
        records: list[dict[str, Any]] = []
        for row in dataset:
            record = dict(row)
            raw_split = record.get("split")
            if raw_split is not None and not _split_matches(raw_split, self.split):
                continue
            records.append(record)
        self._records = records

    def _passes_for_record(self, record: dict[str, Any]) -> int:
        options = _collect_options(record)
        return len(options) if self.circular_eval else 1

    def num_samples(self, limit: int | None = None) -> int:
        self._ensure_loaded()
        assert self._records is not None
        records = self._records[:limit] if limit is not None else self._records
        return sum(self._passes_for_record(record) for record in records)

    def iter_choice_samples(self, limit: int | None = None) -> Iterator[ChoiceSample]:
        self._ensure_loaded()
        assert self._records is not None
        for idx, record in enumerate(self._records):
            if limit is not None and idx >= limit:
                break

            sample_index = record.get("index", idx)
            options = _collect_options(record)
            correct_choice = _normalize_answer(record, options)
            correct_idx = CHOICE_LETTERS.index(correct_choice)
            image_path = _materialize_image(
                record.get("image"),
                self._cache_dir / f"{sample_index}.png",
            )
            question = str(record.get("question", "")).strip()
            hint = record.get("hint")
            num_passes = len(options) if self.circular_eval else 1

            for rotation in range(num_passes):
                rotated_options = options[rotation:] + options[:rotation]
                rotated_labels = list(CHOICE_LETTERS[: len(rotated_options)])
                rotated_correct_choice = rotated_labels[(correct_idx - rotation) % len(rotated_options)]
                labeled_options = list(zip(rotated_labels, rotated_options))
                prompt = _build_prompt(
                    question,
                    None if hint is None else str(hint),
                    labeled_options,
                    self.prompt_suffix,
                )

                yield ChoiceSample(
                    id=f"mmbench_{sample_index}_pass_{rotation + 1}",
                    prompt=prompt,
                    choice_labels=rotated_labels,
                    choice_targets=rotated_labels,
                    correct_choice=rotated_correct_choice,
                    image_paths=[image_path] if image_path else [],
                    category=str(record.get("category") or "_uncategorized"),
                    meta={
                        "base_sample_id": sample_index,
                        "pass_index": rotation + 1,
                        "num_passes": num_passes,
                        "rotation": rotation,
                        "split": record.get("split", self.split),
                        "source": record.get("source"),
                        "l2_category": record.get("l2-category"),
                        "comment": record.get("comment"),
                        "base_correct_choice": correct_choice,
                        "base_options": options,
                    },
                )
