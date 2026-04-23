from __future__ import annotations

from pathlib import Path

from guardrail_eval.benchmarks.mmmu_pro import MMMUProBenchmark


def test_mmmu_pro_expands_circular_passes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mmmu_pro"
    dataset_dir.mkdir()

    benchmark = MMMUProBenchmark(
        {
            "name": "mmmu_pro",
            "dataset_path": str(dataset_dir),
            "circular_eval": True,
        }
    )
    benchmark._dataset = [
        {
            "id": "sample-1",
            "question": "Pick the blue option.",
            "options": ["red", "blue", "green"],
            "answer": "B",
            "subject": "science",
        }
    ]

    samples = list(benchmark.iter_choice_samples())

    assert benchmark.num_samples() == 3
    assert [sample.id for sample in samples] == [
        "sample-1_pass_1",
        "sample-1_pass_2",
        "sample-1_pass_3",
    ]
    assert [sample.correct_choice for sample in samples] == ["B", "A", "C"]
    assert "A. blue" in samples[1].prompt
    assert "C. red" in samples[1].prompt
    assert samples[2].meta["base_sample_id"] == "sample-1"
    assert samples[2].meta["rotation"] == 2
    assert samples[2].meta["num_passes"] == 3


def test_mmmu_pro_limit_counts_base_questions_before_expansion(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mmmu_pro"
    dataset_dir.mkdir()

    benchmark = MMMUProBenchmark(
        {
            "name": "mmmu_pro",
            "dataset_path": str(dataset_dir),
            "circular_eval": True,
        }
    )
    benchmark._dataset = [
        {
            "id": "sample-1",
            "question": "Pick one.",
            "options": ["A1", "A2"],
            "answer": "A",
            "subject": "science",
        },
        {
            "id": "sample-2",
            "question": "Pick two.",
            "options": ["B1", "B2", "B3", "B4"],
            "answer": "D",
            "subject": "math",
        },
    ]

    samples = list(benchmark.iter_choice_samples(limit=1))

    assert benchmark.num_samples(limit=1) == 2
    assert len(samples) == 2
    assert all(sample.meta["base_sample_id"] == "sample-1" for sample in samples)
