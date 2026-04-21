from __future__ import annotations

import types

import pytest

from guardrail_eval.models.llama_guard import _build_backend


def test_build_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        _build_backend({"backend": "unknown", "backend_kwargs": {}}, "/tmp/model")


def test_build_backend_vllm(monkeypatch):
    class FakeBackend:
        def __init__(self, model_ref, backend_kwargs):
            self.model_ref = model_ref
            self.backend_kwargs = backend_kwargs

    monkeypatch.setattr("guardrail_eval.models.llama_guard.VLLMBackend", FakeBackend)
    backend_name, backend = _build_backend({"backend": "vllm", "backend_kwargs": {"x": 1}}, "/tmp/model")
    assert backend_name == "vllm"
    assert backend.model_ref == "/tmp/model"
    assert backend.backend_kwargs == {"x": 1}


def test_build_backend_transformers(monkeypatch):
    class FakeBackend:
        def __init__(self, model_ref, backend_kwargs):
            self.model_ref = model_ref
            self.backend_kwargs = backend_kwargs

    fake_module = types.SimpleNamespace(TransformersLlama4Backend=FakeBackend)
    monkeypatch.setitem(__import__("sys").modules, "guardrail_eval.backends.transformers_llama4_backend", fake_module)
    backend_name, backend = _build_backend({"backend": "transformers", "backend_kwargs": {"y": 2}}, "/tmp/model")
    assert backend_name == "transformers"
    assert backend.model_ref == "/tmp/model"
    assert backend.backend_kwargs == {"y": 2}
