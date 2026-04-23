from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from guardrail_eval.benchmarks.base import Benchmark, MultipleChoiceBenchmark
from guardrail_eval.evaluator import run, run_choice
from guardrail_eval.models.base import GuardrailModel
from guardrail_eval.types import ChoiceSample, ChoiceVerdict, Sample, Verdict


class _DummyBenchmark(Benchmark):
    def __init__(self) -> None:
        super().__init__({"name": "dummy_benchmark"})
        self._samples = [
            Sample(id="sample_1", text="one", image_path=None, expected_label="safe", category="alpha"),
            Sample(id="sample_2", text="two", image_path=None, expected_label="unsafe", category="beta"),
        ]

    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        for idx, sample in enumerate(self._samples):
            if limit is not None and idx >= limit:
                break
            yield sample

    def num_samples(self, limit: int | None = None) -> int | None:
        return min(len(self._samples), limit) if limit is not None else len(self._samples)


class _DummyModel(GuardrailModel):
    def __init__(self) -> None:
        super().__init__({"name": "dummy_model"})

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for sample in samples:
            label = "safe" if sample.id == "sample_1" else "unsafe"
            verdicts.append(Verdict(label=label, categories=[], raw=label))
        return verdicts


class _DummyChoiceBenchmark(MultipleChoiceBenchmark):
    def __init__(self) -> None:
        super().__init__({"name": "dummy_choice_benchmark"})
        self._samples = [
            ChoiceSample(
                id="q1_pass_1",
                prompt="Question 1 pass 1",
                choice_labels=["A", "B"],
                choice_targets=["A", "B"],
                correct_choice="A",
                category="alpha",
                meta={
                    "base_sample_id": "q1",
                    "pass_index": 1,
                    "num_passes": 2,
                    "rotation": 0,
                    "base_options": ["cat", "dog"],
                },
            ),
            ChoiceSample(
                id="q1_pass_2",
                prompt="Question 1 pass 2",
                choice_labels=["A", "B"],
                choice_targets=["A", "B"],
                correct_choice="B",
                category="alpha",
                meta={
                    "base_sample_id": "q1",
                    "pass_index": 2,
                    "num_passes": 2,
                    "rotation": 1,
                    "base_options": ["cat", "dog"],
                },
            ),
            ChoiceSample(
                id="q2_pass_1",
                prompt="Question 2 pass 1",
                choice_labels=["A", "B"],
                choice_targets=["A", "B"],
                correct_choice="B",
                category="beta",
                meta={
                    "base_sample_id": "q2",
                    "pass_index": 1,
                    "num_passes": 2,
                    "rotation": 0,
                    "base_options": ["cat", "dog"],
                },
            ),
            ChoiceSample(
                id="q2_pass_2",
                prompt="Question 2 pass 2",
                choice_labels=["A", "B"],
                choice_targets=["A", "B"],
                correct_choice="A",
                category="beta",
                meta={
                    "base_sample_id": "q2",
                    "pass_index": 2,
                    "num_passes": 2,
                    "rotation": 1,
                    "base_options": ["cat", "dog"],
                },
            ),
        ]

    def iter_choice_samples(self, limit: int | None = None) -> Iterator[ChoiceSample]:
        for idx, sample in enumerate(self._samples):
            if limit is not None and idx >= limit:
                break
            yield sample

    def num_samples(self, limit: int | None = None) -> int:
        return min(len(self._samples), limit) if limit is not None else len(self._samples)


class _DummyChoiceModel(GuardrailModel):
    def __init__(self) -> None:
        super().__init__({"name": "dummy_choice_model", "task_types": ["multiple_choice"]})

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        raise NotImplementedError

    def score_choice_batch(self, samples: list[ChoiceSample]) -> list[ChoiceVerdict]:
        predictions = {
            "q1_pass_1": "A",
            "q1_pass_2": "B",
            "q2_pass_1": "A",
            "q2_pass_2": "A",
        }
        verdicts: list[ChoiceVerdict] = []
        for sample in samples:
            pred = predictions[sample.id]
            verdicts.append(
                ChoiceVerdict(
                    pred_choice=pred,
                    choice_losses={label: 0.0 for label in sample.choice_labels},
                    raw=pred,
                )
            )
        return verdicts


def test_run_resume_streams_legacy_results_into_jsonl(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    result_dir = output_dir / "dummy_model" / "dummy_benchmark"
    result_dir.mkdir(parents=True)
    legacy_path = result_dir / "results.json"
    legacy_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample_1",
                    "expected": "safe",
                    "expected_category": "alpha",
                    "pred_label": "safe",
                    "pred_categories": [],
                    "raw_output": "safe",
                    "error_reason": None,
                    "batch_avg_latency_ms": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = run(
        _DummyModel(),
        _DummyBenchmark(),
        output_dir=output_dir,
        resume=True,
        batch_size=2,
        flush_every_batches=2,
    )

    raw_path = result_dir / "results.jsonl"
    lines = raw_path.read_text(encoding="utf-8").splitlines()

    assert summary["n"] == 2
    assert summary["correct"] == 2
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [record["sample_id"] for record in records] == ["sample_1", "sample_2"]


def test_run_choice_writes_question_level_and_permutation_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"

    summary = run_choice(
        _DummyChoiceModel(),
        _DummyChoiceBenchmark(),
        output_dir=output_dir,
        batch_size=4,
        flush_every_batches=2,
    )

    assert summary["accuracy_scope"] == "pass"
    assert summary["n"] == 4
    assert summary["correct"] == 3
    assert summary["accuracy"] == 0.75
    assert summary["question_level"]["questions_total"] == 2
    assert summary["question_level"]["questions_complete"] == 2
    assert summary["question_level"]["questions_correct"] == 1
    assert summary["question_level"]["question_accuracy"] == 0.5
    assert summary["permutation_bias"]["questions_with_multiple_passes"] == 2
    assert summary["permutation_bias"]["questions_with_complete_passes"] == 2
    assert summary["permutation_bias"]["questions_inconsistent"] == 1
    assert summary["permutation_bias"]["inconsistency_rate"] == 0.5

    result_dir = output_dir / "dummy_choice_model" / "dummy_choice_benchmark"
    written_summary = json.loads((result_dir / "results_summary.json").read_text(encoding="utf-8"))
    assert written_summary["question_level"]["question_accuracy"] == 0.5
    assert written_summary["permutation_bias"]["questions_inconsistent"] == 1
