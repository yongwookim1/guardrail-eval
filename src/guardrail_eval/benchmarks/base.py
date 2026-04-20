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
        self.columns = config.get("columns", {})

    @abstractmethod
    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        """Yield Sample objects. Must honor `limit` (None means all)."""

    def __iter__(self) -> Iterator[Sample]:
        return self.iter_samples()
