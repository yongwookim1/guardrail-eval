from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from guardrail_eval.backends.transformers_common import TransformersMultimodalBackend
from guardrail_eval.types import Sample


class _FakeBatch(dict):
    def to(self, device: str):
        self["device"] = device
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.apply_calls: list[tuple[object, dict[str, object]]] = []
        self.decode_calls: list[object] = []

    def apply_chat_template(self, messages_batch, **kwargs):
        self.apply_calls.append((messages_batch, kwargs))
        return _FakeBatch({"input_ids": np.array([[1, 2, 3], [4, 5, 0]])})

    def batch_decode(self, trimmed, skip_special_tokens: bool = True):
        self.decode_calls.append((trimmed, skip_special_tokens))
        rows = trimmed.tolist()
        return [f"decoded:{row[0]}" for row in rows]


class _FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return np.array([[1, 2, 3, 11], [4, 5, 0, 22]])


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class _DummyBackend(TransformersMultimodalBackend):
    error_name = "dummy"

    def _model_class(self):
        raise NotImplementedError

    @staticmethod
    def _build_messages(sample: Sample) -> list[dict[str, object]]:
        return [{"role": "user", "content": [{"type": "text", "text": sample.text or ""}]}]


def test_chat_samples_batches_transformers_requests() -> None:
    backend = _DummyBackend.__new__(_DummyBackend)
    backend.processor = _FakeProcessor()
    backend.model = _FakeModel()
    backend._torch = _FakeTorch()
    backend.device = "cuda"
    backend.use_cache = True

    samples = [
        Sample(id="a", text="first", image_path=None, expected_label="safe"),
        Sample(id="b", text="second", image_path=None, expected_label="unsafe"),
    ]

    outputs = backend.chat_samples(samples, sampling={"max_tokens": 7, "temperature": 0.0})

    assert len(outputs) == 2
    assert [text for text, _ in outputs] == ["decoded:11", "decoded:22"]
    assert len(backend.processor.apply_calls) == 1
    messages_batch, kwargs = backend.processor.apply_calls[0]
    assert len(messages_batch) == 2
    assert kwargs["padding"] is True
    assert len(backend.model.generate_calls) == 1
    assert backend.model.generate_calls[0]["use_cache"] is True
    assert backend.model.generate_calls[0]["max_new_tokens"] == 7
