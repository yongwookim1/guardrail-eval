from __future__ import annotations

from guardrail_eval.metrics_mcq import MCQMetricsAccumulator


def test_mcq_metrics_summary_tracks_subject_accuracy_and_errors() -> None:
    acc = MCQMetricsAccumulator()
    acc.update(
        {
            "sample_id": "a",
            "subject": "math",
            "correct": True,
            "error_reason": None,
        }
    )
    acc.update(
        {
            "sample_id": "b",
            "subject": "math",
            "correct": False,
            "error_reason": "backend_error:RuntimeError",
        }
    )
    acc.update(
        {
            "sample_id": "c",
            "subject": "physics",
            "correct": False,
            "error_reason": None,
        }
    )

    summary = acc.summary()

    assert summary["n"] == 3
    assert summary["correct"] == 1
    assert summary["errors"] == 1
    assert summary["by_subject"]["math"]["n"] == 2
    assert summary["by_subject"]["math"]["correct"] == 1
    assert summary["by_subject"]["math"]["errors"] == 1
    assert summary["by_subject"]["physics"]["accuracy"] == 0.0
