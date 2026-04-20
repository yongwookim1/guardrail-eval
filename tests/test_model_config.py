from pathlib import Path

import pytest

from guardrail_eval.models.base import REPO_ROOT, resolve_model_source


def test_resolve_model_source_prefers_repo_relative_model_path():
    model_dir = REPO_ROOT / "models" / "test-model"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved = resolve_model_source({"name": "test", "model_path": "models/test-model"})
        assert resolved == str(model_dir.resolve())
    finally:
        model_dir.rmdir()


def test_resolve_model_source_falls_back_to_hf_id():
    assert resolve_model_source({"name": "test", "hf_id": "org/repo"}) == "org/repo"


def test_resolve_model_source_missing_local_path_raises():
    missing = REPO_ROOT / "models" / "does-not-exist"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        resolve_model_source({"name": "test", "model_path": "models/does-not-exist"})
