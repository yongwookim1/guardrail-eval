from __future__ import annotations

from pathlib import Path

import pytest
import torch

from guardrail_eval.benchmarks.base import Benchmark
from guardrail_eval.erank import effective_rank, run_erank_evaluation
from guardrail_eval.erank_backends import ERankSampleLayers
from guardrail_eval.erank_cli import _public_model_names
from guardrail_eval.types import Sample


class _FakeBenchmark(Benchmark):
    def __init__(self, config: dict[str, object], samples: list[Sample]) -> None:
        super().__init__(config)
        self._samples = samples

    def num_samples(self, limit: int | None = None) -> int:
        return len(self._samples[:limit] if limit is not None else self._samples)

    def iter_samples(self, limit: int | None = None):
        samples = self._samples[:limit] if limit is not None else self._samples
        yield from samples


class _FakeERankBackend:
    def __init__(self) -> None:
        self.device = "cpu"
        self.closed = False

    def collect_erank_sample(self, sample: Sample) -> ERankSampleLayers:
        rows_by_id = {
            "q1": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            ),
            "q2": torch.tensor(
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
        }
        rows = rows_by_id[sample.id]
        return ERankSampleLayers(
            image_tokens=rows,
            debug={
                "sample_id": sample.id,
                "question_id": sample.meta.get("question_id"),
                "prompt_token_count": 20,
                "vision_token_count": int(rows.shape[0]),
                "first_vision_position": 4,
                "last_vision_position": 4 + int(rows.shape[0]) - 1,
                "image_path": sample.image_path,
                "input_keys": ["attention_mask", "input_ids"],
                "hidden_size": int(rows.shape[1]),
            },
        )

    def close(self) -> None:
        self.closed = True


def test_effective_rank_identity_matrix_matches_rank() -> None:
    matrix = torch.eye(3)
    score = effective_rank(matrix, metric_device="cpu")
    assert score == pytest.approx(3.0, rel=1e-5)


def test_public_model_names_only_returns_supported_choice_stems() -> None:
    names = _public_model_names()
    assert "gemma_3_4b_it" in names
    assert "nemotron_cs" in names
    assert "qwen2_5_vl_7b_instruct" in names
    assert "safeqwen2_5_vl_7b" in names
    assert "guardreasoner_vl_7b" in names


def test_run_erank_evaluation_with_fake_backend(monkeypatch, tmp_path: Path) -> None:
    samples = [
        Sample(
            id="q1",
            text="Question 1",
            image_path=str(tmp_path / "q1.png"),
            expected_label="safe",
            category="demo",
            meta={"question_id": "base_q1"},
        ),
        Sample(
            id="q2",
            text="Question 2",
            image_path=str(tmp_path / "q2.png"),
            expected_label="safe",
            category="demo",
            meta={"question_id": "base_q2"},
        ),
    ]
    benchmark = _FakeBenchmark(
        {
            "name": "okvqa_erank",
            "task_type": "classification",
        },
        samples,
    )
    fake_backend = _FakeERankBackend()
    monkeypatch.setattr(
        "guardrail_eval.erank.build_erank_backend",
        lambda model_config: ("qwen2_5_vl", fake_backend),
    )

    artifacts = run_erank_evaluation(
        model_config={"name": "qwen2_5_vl_7b_instruct"},
        benchmark=benchmark,
        limit=None,
        top_k=2,
        metric_device="cpu",
        debug_limit=1,
    )

    assert fake_backend.closed is True
    assert artifacts.summary["model_name"] == "qwen2_5_vl_7b_instruct"
    assert artifacts.summary["benchmark"] == "okvqa_erank"
    assert artifacts.summary["n_samples"] == 2
    assert artifacts.summary["shared_image_tokens"] == 2
    assert len(artifacts.per_position) == 2
    assert len(artifacts.top_positions) == 2
    assert len(artifacts.samples) == 2
    assert len(artifacts.debug_records) == 1


def test_run_erank_evaluation_rejects_multiple_choice_benchmark(tmp_path: Path) -> None:
    sample = Sample(
        id="q1",
        text="Question 1",
        image_path=str(tmp_path / "q1.png"),
        expected_label="safe",
        category="demo",
        meta={"question_id": "base_q1"},
    )
    benchmark = _FakeBenchmark(
        {
            "name": "mmbench",
            "task_type": "multiple_choice",
        },
        [sample],
    )

    with pytest.raises(ValueError, match="open-ended multimodal benchmark"):
        run_erank_evaluation(
            model_config={"name": "qwen2_5_vl_7b_instruct"},
            benchmark=benchmark,
            limit=None,
            top_k=10,
            metric_device="cpu",
        )
