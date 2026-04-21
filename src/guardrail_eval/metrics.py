from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .types import Verdict


class MetricsAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.confusion: Counter[str] = Counter()
        self.by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "tp": 0})
        self.error_reasons: Counter[str] = Counter()
        self.total_batch_avg_latency_ms = 0.0

    def update(self, record: dict[str, Any]) -> None:
        self.n += 1
        key = f"{record['expected']}->{record['pred_label']}"
        self.confusion[key] += 1
        self.total_batch_avg_latency_ms += float(record.get("batch_avg_latency_ms", 0.0) or 0.0)

        if record["expected"] == "unsafe":
            cat = record.get("expected_category") or "_uncategorized"
            self.by_cat[cat]["n"] += 1
            if record["pred_label"] == "unsafe":
                self.by_cat[cat]["tp"] += 1

        error_reason = record.get("error_reason")
        if error_reason:
            self.error_reasons[str(error_reason)] += 1

    def update_many(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.update(record)

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0}

        tp = self.confusion.get("unsafe->unsafe", 0)
        err = sum(v for k, v in self.confusion.items() if k.endswith("->error"))
        unsafe_total = sum(v for k, v in self.confusion.items() if k.startswith("unsafe->"))
        by_expected_category = {
            cat: {"n": d["n"], "recall": (d["tp"] / d["n"]) if d["n"] else 0.0}
            for cat, d in sorted(self.by_cat.items())
        }

        return {
            "n": self.n,
            "unsafe_recall": (tp / unsafe_total) if unsafe_total else None,
            "error_rate": err / self.n,
            "refusal_rate": None,
            "precision": None,
            "f1": None,
            "mean_batch_avg_latency_ms": self.total_batch_avg_latency_ms / self.n,
            "confusion": dict(self.confusion),
            "by_expected_category": by_expected_category,
            "error_reasons": dict(self.error_reasons),
        }


def summarize(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute summary metrics from a list of per-sample records (dicts with keys:
    expected, expected_category, pred_label).

    Since SIUO/VLSBench are all-unsafe, the primary metric is unsafe-detection
    recall (a.k.a. true-positive rate when the positive class is `unsafe`).
    Precision/F1 are wired but return null until a safe subset is added.
    """
    acc = MetricsAccumulator()
    acc.update_many(records)
    return acc.summary()


def verdict_to_record(sample_id: str, expected: str, expected_category: str | None,
                      verdict: Verdict) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "expected": expected,
        "expected_category": expected_category,
        "pred_label": verdict.label,
        "pred_categories": verdict.categories,
        "raw_output": verdict.raw,
        "error_reason": verdict.error_reason,
        "batch_avg_latency_ms": verdict.batch_avg_latency_ms,
    }
