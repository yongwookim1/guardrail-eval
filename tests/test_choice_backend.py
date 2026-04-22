from __future__ import annotations

import torch

from guardrail_eval.backends.transformers_choice_backends import TransformersChoiceBackend
from guardrail_eval.types import ChoiceSample


class _FakePromptBatch(dict):
    pass


class _FakeProcessor:
    def apply_chat_template(self, messages_batch, **kwargs):
        del kwargs
        attention_rows = []
        for idx, _ in enumerate(messages_batch):
            attention_rows.append([1] * (idx + 3))
        max_len = max(len(row) for row in attention_rows)
        padded = [row + [0] * (max_len - len(row)) for row in attention_rows]
        return _FakePromptBatch({"attention_mask": torch.tensor(padded, dtype=torch.long)})


class _DummyChoiceBackend(TransformersChoiceBackend):
    def __init__(self) -> None:
        self.system_prompt = ""
        self.max_choice_rows = 0
        self.calls: list[tuple[int, list[int]]] = []
        self.processor = _FakeProcessor()

    def _model_class(self):
        raise NotImplementedError

    def _prompt_messages(self, sample: ChoiceSample) -> list[dict[str, object]]:
        return [{"role": "user", "content": [{"type": "text", "text": sample.prompt}]}]

    def _score_choice_messages(
        self,
        prompt_messages_batch: list[list[dict[str, object]]],
        choice_messages: list[list[dict[str, object]]],
        prompt_lengths: list[int] | None = None,
    ) -> list[float | None]:
        del prompt_messages_batch
        assert prompt_lengths is not None
        self.calls.append((len(choice_messages), prompt_lengths))
        losses: list[float | None] = []
        for choice_message in choice_messages:
            label = str(choice_message[-1]["content"][0]["text"])
            losses.append(0.1 if label == "A" else 0.9)
        return losses


def _sample(sample_id: str, correct_choice: str) -> ChoiceSample:
    return ChoiceSample(
        id=sample_id,
        prompt=f"Question {sample_id}",
        choice_labels=["A", "B"],
        choice_targets=["A", "B"],
        correct_choice=correct_choice,
    )


def test_score_choice_samples_batches_all_choices_once() -> None:
    backend = _DummyChoiceBackend()

    verdicts = backend.score_choice_samples([_sample("s1", "A"), _sample("s2", "A")])

    assert len(backend.calls) == 1
    assert backend.calls[0][0] == 4
    assert [verdict.pred_choice for verdict in verdicts] == ["A", "A"]


def test_score_choice_samples_honors_max_choice_rows() -> None:
    backend = _DummyChoiceBackend()
    backend.max_choice_rows = 2

    verdicts = backend.score_choice_samples([_sample("s1", "A"), _sample("s2", "A")])

    assert len(backend.calls) == 2
    assert [call[0] for call in backend.calls] == [2, 2]
    assert [verdict.pred_choice for verdict in verdicts] == ["A", "A"]
