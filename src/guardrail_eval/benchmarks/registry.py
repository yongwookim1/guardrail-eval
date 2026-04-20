from __future__ import annotations

import importlib
from typing import Any, Callable

from ..io import load_yaml
from .base import Benchmark

_REGISTRY: dict[str, type[Benchmark]] = {}


def register_benchmark(name: str) -> Callable[[type[Benchmark]], type[Benchmark]]:
    def _wrap(cls: type[Benchmark]) -> type[Benchmark]:
        _REGISTRY[name] = cls
        return cls
    return _wrap


def _resolve_class(class_path: str) -> type[Benchmark]:
    module_name, _, cls_name = class_path.rpartition(".")
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    if not issubclass(cls, Benchmark):
        raise TypeError(f"{class_path} is not a Benchmark subclass")
    return cls


def load_benchmark(config_path: str) -> Benchmark:
    cfg: dict[str, Any] = load_yaml(config_path)
    cls = _REGISTRY.get(cfg["name"]) or _resolve_class(cfg["class"])
    return cls(cfg)


def list_benchmarks() -> list[str]:
    return sorted(_REGISTRY.keys())
