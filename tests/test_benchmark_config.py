from pathlib import Path

import pytest

from guardrail_eval.benchmarks.base import REPO_ROOT, resolve_dataset_path


def test_resolve_dataset_path_uses_repo_relative_directory():
    dataset_dir = REPO_ROOT / "datasets" / "test-dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved = resolve_dataset_path({"name": "test", "dataset_path": "datasets/test-dataset"})
        assert resolved == dataset_dir.resolve()
    finally:
        dataset_dir.rmdir()


def test_resolve_dataset_path_missing_directory_raises():
    missing = REPO_ROOT / "datasets" / "does-not-exist"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        resolve_dataset_path({"name": "test", "dataset_path": "datasets/does-not-exist"})


def test_resolve_dataset_path_missing_key_raises():
    with pytest.raises(KeyError, match="dataset_path"):
        resolve_dataset_path({"name": "test"})
