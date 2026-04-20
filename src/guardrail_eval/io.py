from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL.Image import Image


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def pil_to_data_uri(image: Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a data: URI for OpenAI-style multimodal messages."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_many(self, records: Iterable[dict[str, Any]]) -> None:
        for r in records:
            self.write(r)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
