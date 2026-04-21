from __future__ import annotations

import json
from pathlib import Path

from guardrail_eval.benchmarks.base import Benchmark
from guardrail_eval.evaluator import run
from guardrail_eval.models.base import GuardrailModel
from guardrail_eval.types import Sample, Verdict


class FakeBenchmark(Benchmark):
    def __init__(self, samples: list[Sample]) -> None:
        super().__init__({"name": "fake_benchmark"})
        self._samples = samples

    def iter_samples(self, limit: int | None = None):
        items = self._samples[:limit] if limit is not None else self._samples
        yield from items

    def num_samples(self, limit: int | None = None) -> int | None:
        n = len(self._samples)
        return min(n, limit) if limit is not None else n


class FakeModel(GuardrailModel):
    def __init__(self) -> None:
        super().__init__({"name": "fake_model"})

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for sample in samples:
            label = "unsafe" if "bad" in (sample.text or "") else "safe"
            verdicts.append(Verdict(label=label, categories=[], raw=label))
        return verdicts


def _sample(idx: int, text: str) -> Sample:
    return Sample(
        id=f"s{idx}",
        text=text,
        image_path=None,
        expected_label="unsafe",
    )


def test_run_writes_outputs_and_can_resume(tmp_path: Path):
    benchmark = FakeBenchmark([_sample(1, "bad"), _sample(2, "bad"), _sample(3, "bad")])
    model = FakeModel()

    out_dir = tmp_path / "results"
    summary = run(model, benchmark, out_dir, limit=2, batch_size=2)
    assert summary["n"] == 2

    raw_path = out_dir / "fake_model" / "fake_benchmark" / "results.json"
    lines = raw_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    resumed = run(model, benchmark, out_dir, limit=3, batch_size=2, resume=True)
    assert resumed["n"] == 3
    assert resumed["resumed_records"] == 2

    all_lines = raw_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(all_lines) == 3

    summary_path = out_dir / "fake_model" / "fake_benchmark" / "results_summary.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["n"] == 3
