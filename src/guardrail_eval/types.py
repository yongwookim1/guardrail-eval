from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from PIL.Image import Image

Label = Literal["safe", "unsafe", "error"]


@dataclass
class Sample:
    id: str
    text: str | None
    image: Image | None
    expected_label: Label
    image_path: str | None = None
    image_data_uri: str | None = None
    category: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    label: Label
    categories: list[str]
    raw: str
    error_reason: str | None = None
    # Batch-average latency: the batch's wall-time divided by batch size. Not a
    # true per-sample latency (vLLM processes batch items concurrently) — use
    # it as a throughput proxy, not a p99 measurement.
    batch_avg_latency_ms: float = 0.0
