from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .types import MCQSample, MCQVerdict


def _safe_div(num: int, den: int) -> float | None:
    return (num / den) if den else None


class MCQMetricsAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.correct = 0
        self.errors = 0
        self.by_subject: dict[str, Counter[str]] = defaultdict(Counter)
        self.error_reasons: Counter[str] = Counter()

    def update(self, record: dict[str, Any]) -> None:
        self.n += 1
        subject = str(record.get("subject") or "_uncategorized")
        self.by_subject[subject]["n"] += 1

        if record.get("correct"):
            self.correct += 1
            self.by_subject[subject]["correct"] += 1

        error_reason = record.get("error_reason")
        if error_reason:
            self.errors += 1
            self.by_subject[subject]["errors"] += 1
            self.error_reasons[str(error_reason)] += 1

    def update_many(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.update(record)

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0}

        by_subject = {
            subject: {
                "n": stats["n"],
                "correct": stats["correct"],
                "accuracy": _safe_div(stats["correct"], stats["n"]),
                "errors": stats["errors"],
            }
            for subject, stats in sorted(self.by_subject.items())
        }
        return {
            "n": self.n,
            "correct": self.correct,
            "accuracy": _safe_div(self.correct, self.n),
            "errors": self.errors,
            "by_subject": by_subject,
            "error_reasons": dict(self.error_reasons),
        }


def mcq_verdict_to_record(sample: MCQSample, verdict: MCQVerdict) -> dict[str, Any]:
    return {
        "sample_id": sample.id,
        "gold_choice": sample.correct_choice,
        "pred_choice": verdict.pred_choice,
        "correct": verdict.pred_choice == sample.correct_choice,
        "choice_labels": sample.choice_labels,
        "choice_losses": verdict.choice_losses,
        "raw_output": verdict.raw,
        "error_reason": verdict.error_reason,
        "batch_avg_latency_ms": verdict.batch_avg_latency_ms,
        "subject": sample.category,
        "meta": sample.meta,
    }
