from __future__ import annotations

import base64
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from guardrail_eval.benchmarks.mmbench import MMBenchBenchmark


def _tiny_image_b64() -> str:
    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_mmbench_parquet(path: Path) -> None:
    rows = [
        {
            "index": 7,
            "question": "Which option is correct?",
            "hint": "Read the image carefully.",
            "A": "alpha",
            "B": "beta",
            "C": None,
            "D": None,
            "answer": "B",
            "category": "reasoning",
            "image": _tiny_image_b64(),
            "source": "unit-test",
            "l2-category": "attribute_reasoning",
            "comment": None,
            "split": "dev",
        },
        {
            "index": 8,
            "question": "Pick the fourth choice.",
            "hint": None,
            "A": "one",
            "B": "two",
            "C": "three",
            "D": "four",
            "answer": "D",
            "category": "logic",
            "image": _tiny_image_b64(),
            "source": "unit-test",
            "l2-category": "logic_reasoning",
            "comment": "test",
            "split": "test",
        },
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_mmbench_expands_circular_passes_per_question(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mmbench"
    dataset_dir.mkdir()
    _write_mmbench_parquet(dataset_dir / "part-00000.parquet")

    benchmark = MMBenchBenchmark(
        {
            "name": "mmbench",
            "dataset_path": str(dataset_dir),
            "split": "validation",
            "circular_eval": True,
        }
    )

    samples = list(benchmark.iter_choice_samples())

    assert benchmark.num_samples() == 2
    assert [sample.id for sample in samples] == ["mmbench_7_pass_1", "mmbench_7_pass_2"]
    assert [sample.correct_choice for sample in samples] == ["B", "A"]
    assert samples[0].choice_labels == ["A", "B"]
    assert samples[1].choice_labels == ["A", "B"]
    assert samples[0].meta["pass_index"] == 1
    assert samples[1].meta["pass_index"] == 2
    assert samples[0].meta["num_passes"] == 2
    assert samples[1].meta["base_sample_id"] == 7
    assert "Hint: Read the image carefully." in samples[0].prompt
    assert "A. beta" in samples[1].prompt
    assert "B. alpha" in samples[1].prompt
    assert len(samples[0].image_paths) == 1
    assert Path(samples[0].image_paths[0]).exists()


def test_mmbench_limit_counts_base_questions_not_expanded_passes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mmbench"
    dataset_dir.mkdir()
    _write_mmbench_parquet(dataset_dir / "part-00000.parquet")

    benchmark = MMBenchBenchmark(
        {
            "name": "mmbench",
            "dataset_path": str(dataset_dir),
            "split": "test",
            "circular_eval": True,
        }
    )

    samples = list(benchmark.iter_choice_samples(limit=1))

    assert benchmark.num_samples(limit=1) == 4
    assert len(samples) == 4
    assert [sample.correct_choice for sample in samples] == ["D", "C", "B", "A"]
