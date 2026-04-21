from __future__ import annotations

import base64
import functools
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


def _cache_size_from_env(env_var: str, default: int) -> int:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, 0)


IMAGE_CACHE_MAXSIZE = _cache_size_from_env("GUARDRAIL_EVAL_IMAGE_CACHE_MAXSIZE", 1024)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=IMAGE_CACHE_MAXSIZE)
def file_to_data_uri(path: str) -> str:
    """Encode an existing image file as a data: URI.

    This avoids the decode -> re-encode cycle for datasets that already store
    images on disk in PNG/JPEG/WebP form.
    """
    file_path = Path(path)
    data = file_path.read_bytes()
    mime, _ = mimetypes.guess_type(file_path.name)
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


class JsonlWriter:
    def __init__(self, path: str | Path, *, mode: str = "w") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, mode, encoding="utf-8")

    def write(self, record: dict[str, Any], *, flush: bool = False) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if flush:
            self._fh.flush()

    def write_many(self, records: Iterable[dict[str, Any]], *, flush: bool = False) -> None:
        for r in records:
            self.write(r)
        if flush:
            self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

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


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
        return

    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in legacy results file: {path}")
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError(f"Legacy results file contains a non-object record: {path}")
        yield record


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))
