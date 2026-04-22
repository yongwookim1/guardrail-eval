from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..types import ChoiceSample, ChoiceVerdict, Sample, Verdict

REPO_ROOT = Path(__file__).resolve().parents[3]


class GuardrailModel(ABC):
    """Abstract interface every guardrail model implements."""

    name: str

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.name = config["name"]
        task_types = config.get("task_types", ["classification"])
        self.task_types = {str(task_type) for task_type in task_types}

    @abstractmethod
    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        """Classify a batch of samples. Implementations should batch under the hood."""

    def classify(self, sample: Sample) -> Verdict:
        return self.classify_batch([sample])[0]

    def score_choice_batch(self, samples: list[ChoiceSample]) -> list[ChoiceVerdict]:
        raise NotImplementedError(f"{self.name} does not support multiple-choice scoring")

    def supports_task(self, task_type: str) -> bool:
        return str(task_type) in self.task_types

    def close(self) -> None:
        """Release GPU memory / shutdown inference engine. Override if needed."""


def resolve_model_source(config: dict[str, Any]) -> str:
    """Return the local vLLM model path from config."""
    model_path = config.get("model_path")
    if not model_path:
        raise KeyError("Model config must define 'model_path'")
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configured model_path does not exist: {path}")
    return str(path.resolve())
