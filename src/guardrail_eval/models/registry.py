from __future__ import annotations

import importlib
from typing import Any, Callable

from ..io import load_yaml
from .base import GuardrailModel

_REGISTRY: dict[str, type[GuardrailModel]] = {}


def register_model(name: str) -> Callable[[type[GuardrailModel]], type[GuardrailModel]]:
    def _wrap(cls: type[GuardrailModel]) -> type[GuardrailModel]:
        _REGISTRY[name] = cls
        return cls
    return _wrap


def _resolve_class(class_path: str) -> type[GuardrailModel]:
    module_name, _, cls_name = class_path.rpartition(".")
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    if not issubclass(cls, GuardrailModel):
        raise TypeError(f"{class_path} is not a GuardrailModel subclass")
    return cls


def load_model(config_path: str) -> GuardrailModel:
    cfg: dict[str, Any] = load_yaml(config_path)
    cls = _REGISTRY.get(cfg["name"]) or _resolve_class(cfg["class"])
    return cls(cfg)


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())
