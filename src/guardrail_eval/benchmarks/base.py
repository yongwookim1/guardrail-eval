from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from ..types import Sample


class Benchmark(ABC):
    name: str

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.name = config["name"]
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
