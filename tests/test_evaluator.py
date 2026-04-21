from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from guardrail_eval.benchmarks.base import Benchmark
from guardrail_eval.evaluator import run
from guardrail_eval.models.base import GuardrailModel
from guardrail_eval.types import Sample, Verdict


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
