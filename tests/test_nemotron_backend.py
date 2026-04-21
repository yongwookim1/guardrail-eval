from __future__ import annotations

import types

import pytest

from guardrail_eval.models.nemotron import _build_backend
from guardrail_eval.backends.transformers_gemma3_backend import TransformersGemma3Backend


def test_build_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        _build_backend({"backend": "unknown", "backend_kwargs": {}}, "/tmp/model")


def test_build_backend_vllm(monkeypatch):
    class FakeBackend:
        def __init__(self, model_ref, backend_kwargs):
            self.model_ref = model_ref
            self.backend_kwargs = backend_kwargs

    monkeypatch.setattr("guardrail_eval.models.nemotron.VLLMBackend", FakeBackend)
    backend_name, backend = _build_backend({"backend": "vllm", "backend_kwargs": {"x": 1}}, "/tmp/model")
    assert backend_name == "vllm"
    assert backend.model_ref == "/tmp/model"
    assert backend.backend_kwargs == {"x": 1}


def test_build_backend_transformers(monkeypatch):
    class FakeBackend:
        def __init__(self, model_ref, backend_kwargs):
            self.model_ref = model_ref
            self.backend_kwargs = backend_kwargs

    fake_module = types.SimpleNamespace(TransformersGemma3Backend=FakeBackend)
    monkeypatch.setitem(__import__("sys").modules, "guardrail_eval.backends.transformers_gemma3_backend", fake_module)
    backend_name, backend = _build_backend({"backend": "transformers", "backend_kwargs": {"y": 2}}, "/tmp/model")
    assert backend_name == "transformers"
    assert backend.model_ref == "/tmp/model"
    assert backend.backend_kwargs == {"y": 2}


def test_gemma3_generation_kwargs_disable_cache_for_greedy_decode():
    kwargs = TransformersGemma3Backend._generation_kwargs({"max_tokens": 33, "temperature": 0.0})
    assert kwargs == {"max_new_tokens": 33, "do_sample": False, "use_cache": False}


def test_gemma3_generation_kwargs_include_sampling_fields():
    kwargs = TransformersGemma3Backend._generation_kwargs({"max_tokens": 33, "temperature": 0.6, "top_p": 0.85})
    assert kwargs["max_new_tokens"] == 33
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.6
    assert kwargs["top_p"] == 0.85
    assert kwargs["use_cache"] is False
