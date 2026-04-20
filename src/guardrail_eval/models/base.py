from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import Sample, Verdict


class GuardrailModel(ABC):
    """Abstract interface every guardrail model implements."""

    name: str

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.name = config["name"]

    @abstractmethod
    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        """Classify a batch of samples. Implementations should batch under the hood."""

    def classify(self, sample: Sample) -> Verdict:
        return self.classify_batch([sample])[0]

    def close(self) -> None:
        """Release GPU memory / shutdown inference engine. Override if needed."""
