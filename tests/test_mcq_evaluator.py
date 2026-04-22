from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from guardrail_eval.benchmarks.base import MCQBenchmark
from guardrail_eval.evaluator import run_mcq
from guardrail_eval.models.base import GuardrailModel
from guardrail_eval.types import MCQSample, MCQVerdict, Sample, Verdict


class _DummyMCQBenchmark(MCQBenchmark):
    def __init__(self) -> None:
        super().__init__({"name": "dummy_mcq"})
        self._samples = [
            MCQSample(
                id="sample_1",
                prompt="Question\n\nAnswer:",
                choice_labels=["A", "B"],
                choice_targets=["A", "B"],
                correct_choice="A",
                category="math",
            ),
            MCQSample(
                id="sample_2",
                prompt="Question\n\nAnswer:",
                choice_labels=["A", "B"],
                choice_targets=["A", "B"],
                correct_choice="B",
                category="science",
            ),
        ]

    def iter_mcq_samples(self, limit: int | None = None) -> Iterator[MCQSample]:
        for idx, sample in enumerate(self._samples):
            if limit is not None and idx >= limit:
                break
            yield sample

    def num_samples(self, limit: int | None = None) -> int | None:
        return min(len(self._samples), limit) if limit is not None else len(self._samples)


class _DummyMCQModel(GuardrailModel):
    def __init__(self) -> None:
        super().__init__({"name": "dummy_mcq_model", "task_types": ["mcq"]})

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        raise NotImplementedError

    def score_mcq_batch(self, samples: list[MCQSample]) -> list[MCQVerdict]:
        verdicts: list[MCQVerdict] = []
        for sample in samples:
            pred_choice = "A" if sample.id == "sample_1" else "B"
            verdicts.append(
                MCQVerdict(
                    pred_choice=pred_choice,
                    choice_losses={"A": 0.1, "B": 0.2},
                    raw=pred_choice,
                )
            )
        return verdicts


def test_run_mcq_writes_summary_and_results(tmp_path: Path) -> None:
    summary = run_mcq(
        _DummyMCQModel(),
        _DummyMCQBenchmark(),
        output_dir=tmp_path / "results",
        batch_size=2,
        flush_every_batches=2,
    )

    assert summary["n"] == 2
    assert summary["correct"] == 2
    assert summary["by_subject"]["math"]["accuracy"] == 1.0

    result_dir = tmp_path / "results" / "dummy_mcq_model" / "dummy_mcq"
    records = [
        json.loads(line)
        for line in (result_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["pred_choice"] for record in records] == ["A", "B"]
