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
    category: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    label: Label
    categories: list[str]
    raw: str
    latency_ms: float = 0.0
