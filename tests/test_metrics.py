from __future__ import annotations

from guardrail_eval.metrics import summarize


def _record(
    expected: str,
    pred: str,
    category: str,
    error_reason: str | None = None,
    expected_type: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": f"{expected}-{pred}-{category}",
        "expected": expected,
        "expected_category": category,
        "expected_type": expected_type,
        "pred_label": pred,
        "pred_categories": [],
        "raw_output": "",
        "error_reason": error_reason,
        "batch_avg_latency_ms": 0.0,
    }


def test_summarize_mixed_label_metrics() -> None:
    summary = summarize(
        [
            _record("safe", "safe", "alpha", expected_type="SSS"),
            _record("safe", "unsafe", "alpha", expected_type="SSS"),
            _record("unsafe", "unsafe", "beta", expected_type="SUU"),
            _record("unsafe", "error", "beta", error_reason="backend_error:RuntimeError", expected_type="SUU"),
        ]
    )

    assert summary["n"] == 4
    assert summary["correct"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["balanced_accuracy"] == 0.5
    assert summary["safe_total"] == 2
    assert summary["unsafe_total"] == 2
    assert summary["predicted_counts"] == {"safe": 1, "unsafe": 2, "error": 1}
    assert summary["confusion_matrix"] == {
        "safe": {"safe": 1, "unsafe": 1, "error": 0},
        "unsafe": {"safe": 0, "unsafe": 1, "error": 1},
    }
    assert summary["safe_precision"] == 1.0
    assert summary["safe_recall"] == 0.5
    assert summary["unsafe_precision"] == 0.5
    assert summary["unsafe_recall"] == 0.5
    assert summary["safe_f1"] == 2 / 3
    assert summary["unsafe_f1"] == 0.5
    assert summary["errors"] == 1
    assert summary["safe_misses"] == 1
    assert summary["unsafe_misses"] == 1
    assert summary["by_expected_category"]["alpha"]["accuracy"] == 0.5
    assert summary["by_expected_category"]["alpha"]["safe_recall"] == 0.5
    assert summary["by_expected_category"]["beta"]["unsafe_recall"] == 0.5
    assert summary["by_expected_type"]["SSS"]["safe_recall"] == 0.5
    assert summary["by_expected_type"]["SUU"]["unsafe_recall"] == 0.5
    assert summary["error_reasons"] == {"backend_error:RuntimeError": 1}


def test_summarize_unsafe_only_keeps_legacy_fields() -> None:
    summary = summarize(
        [
            _record("unsafe", "unsafe", "cat_a", expected_type="UUU"),
            _record("unsafe", "safe", "cat_a", expected_type="UUU"),
            _record("unsafe", "error", "cat_b", error_reason="missing_label", expected_type="SSU"),
        ]
    )

    assert summary["true_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["unsafe_errors"] == 1
    assert summary["unsafe_total"] == 3
    assert summary["safe_total"] == 0
    assert summary["unsafe_recall"] == 1 / 3
    assert summary["balanced_accuracy"] == 1 / 3
    assert summary["by_expected_category"]["cat_a"]["recall"] == 0.5
    assert summary["by_expected_category"]["cat_b"]["recall"] == 0.0
    assert summary["by_expected_type"]["UUU"]["unsafe_recall"] == 0.5
    assert summary["by_expected_type"]["SSU"]["unsafe_recall"] == 0.0
