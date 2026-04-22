from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from ..types import ChoiceSample, Sample

REPO_ROOT = Path(__file__).resolve().parents[3]


class Benchmark(ABC):
    name: str

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.name = config["name"]
        self.task_type = str(config.get("task_type", "classification"))
        self.expected_label = config.get("expected_label", "unsafe")

    @abstractmethod
    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        """Yield Sample objects. Must honor `limit` (None means all)."""

    def num_samples(self, limit: int | None = None) -> int | None:
        """Return the total count if known (used for tqdm).

        Subclasses that can cheaply compute the count should override. Return
        None when unknown — tqdm will then show item count without a total.
        """
        return None

    def __iter__(self) -> Iterator[Sample]:
        return self.iter_samples()


class MultipleChoiceBenchmark(Benchmark):
    def __init__(self, config: dict[str, Any]) -> None:
        config = {**config, "task_type": "multiple_choice"}
        super().__init__(config)

    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        raise TypeError(f"{self.name} is a multiple-choice benchmark; use iter_choice_samples() instead")

    @abstractmethod
    def iter_choice_samples(self, limit: int | None = None) -> Iterator[ChoiceSample]:
        """Yield ChoiceSample objects. Must honor `limit` (None means all)."""


def resolve_dataset_path(config: dict[str, Any]) -> Path:
    """Resolve a benchmark's local dataset directory from config."""
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        raise KeyError("Benchmark config must define 'dataset_path' for local/offline loading")

    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configured dataset_path does not exist: {path}")
    return path.resolve()
