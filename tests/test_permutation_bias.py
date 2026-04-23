from __future__ import annotations

from guardrail_eval.analysis.permutation_bias import summarize_permutation_bias


def _record(
    base_sample_id: str,
    pass_index: int,
    rotation: int,
    pred_choice: str,
    *,
    base_options: list[str] | None = None,
) -> dict[str, object]:
    return {
        "sample_id": f"{base_sample_id}_pass_{pass_index}",
        "pred_choice": pred_choice,
        "error_reason": None,
        "meta": {
            "base_sample_id": base_sample_id,
            "pass_index": pass_index,
            "num_passes": 3,
            "rotation": rotation,
            "base_options": base_options or ["cat", "dog", "horse"],
        },
    }


def test_summarize_permutation_bias_flags_inconsistent_semantic_predictions() -> None:
    records = [
        _record("q1", 1, 0, "A"),
        _record("q1", 2, 1, "C"),
        _record("q1", 3, 2, "B"),
        _record("q2", 1, 0, "A"),
        _record("q2", 2, 1, "A"),
        _record("q2", 3, 2, "A"),
    ]

    summary = summarize_permutation_bias(records, max_examples=5)

    assert summary["questions_with_multiple_passes"] == 2
    assert summary["questions_with_complete_passes"] == 2
    assert summary["questions_incomplete"] == 0
    assert summary["questions_inconsistent"] == 1
    assert summary["inconsistency_rate"] == 0.5
    assert summary["questions_invalid"] == 0
    assert summary["examples"][0]["base_sample_id"] == "q2"


def test_summarize_permutation_bias_marks_invalid_prediction_rows() -> None:
    records = [
        _record("q1", 1, 0, "A"),
        _record("q1", 2, 1, "error"),
        _record("q1", 3, 2, "B"),
    ]

    summary = summarize_permutation_bias(records, max_examples=5)

    assert summary["questions_with_multiple_passes"] == 1
    assert summary["questions_with_complete_passes"] == 1
    assert summary["questions_incomplete"] == 0
    assert summary["questions_invalid"] == 1
    assert summary["invalid_rate"] == 1.0
    assert summary["questions_inconsistent"] == 0
    assert summary["examples"][0]["type"] == "invalid"


def test_summarize_permutation_bias_excludes_incomplete_groups_from_rates() -> None:
    records = [
        _record("q1", 1, 0, "A"),
        _record("q1", 2, 1, "C"),
        _record("q1", 3, 2, "B"),
        _record("q2", 1, 0, "A"),
        _record("q2", 2, 1, "C"),
    ]

    summary = summarize_permutation_bias(records, max_examples=5)

    assert summary["questions_with_multiple_passes"] == 2
    assert summary["questions_with_complete_passes"] == 1
    assert summary["questions_incomplete"] == 1
    assert summary["questions_inconsistent"] == 0
    assert summary["inconsistency_rate"] == 0.0
    assert summary["examples"][0]["type"] == "incomplete"


def test_summarize_permutation_bias_uses_semantic_index_not_just_text() -> None:
    duplicated = ["same", "same", "other"]
    records = [
        _record("q1", 1, 0, "A", base_options=duplicated),
        _record("q1", 2, 1, "A", base_options=duplicated),
        _record("q1", 3, 2, "C", base_options=duplicated),
    ]

    summary = summarize_permutation_bias(records, max_examples=5)

    assert summary["questions_with_complete_passes"] == 1
    assert summary["questions_inconsistent"] == 1
    assert summary["inconsistency_rate"] == 1.0
