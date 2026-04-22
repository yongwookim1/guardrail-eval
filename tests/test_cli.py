from __future__ import annotations

from pathlib import Path

from guardrail_eval.cli import CONFIGS_DIR, _public_model_names, _resolve_model_config_for_task


def test_public_model_names_hide_mcq_suffix() -> None:
    public_names = _public_model_names()

    assert "nemotron_cs" in public_names
    assert "qwen2_5_vl_3b_instruct" in public_names
    assert "nemotron_cs_mcq" not in public_names
    assert "qwen2_5_vl_3b_instruct_mcq" not in public_names


def test_resolve_model_config_for_task_prefers_mcq_variant() -> None:
    mcq_path = _resolve_model_config_for_task("nemotron_cs", "mcq")
    cls_path = _resolve_model_config_for_task("nemotron_cs", "classification")

    assert mcq_path == CONFIGS_DIR / "models" / "nemotron_cs_mcq.yaml"
    assert cls_path == CONFIGS_DIR / "models" / "nemotron_cs.yaml"


def test_resolve_model_config_for_task_maps_mcq_only_models() -> None:
    mcq_path = _resolve_model_config_for_task("qwen2_5_vl_3b_instruct", "mcq")

    assert mcq_path == CONFIGS_DIR / "models" / "qwen2_5_vl_3b_instruct_mcq.yaml"
