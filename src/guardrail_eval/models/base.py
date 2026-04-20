from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..types import Sample, Verdict

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def resolve_model_source(config: dict[str, Any]) -> str:
    """Return the vLLM model reference from config.

    Prefer a local `model_path` when provided so offline deployments can keep
    weights under the repo's `models/` directory. Fall back to legacy `hf_id`
    for remote/HF-cache based usage.
    """
    model_path = config.get("model_path")
    if model_path:
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Configured model_path does not exist: {path}")
        return str(path.resolve())

    hf_id = config.get("hf_id")
    if hf_id:
        return str(hf_id)

    raise KeyError("Model config must define either 'model_path' or 'hf_id'")
