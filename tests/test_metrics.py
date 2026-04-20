from guardrail_eval.metrics import summarize


def _rec(sid, expected, pred, cat=None):
    return {
        "sample_id": sid,
        "expected": expected,
        "expected_category": cat,
        "pred_label": pred,
        "pred_categories": [],
        "raw_output": "",
        "batch_avg_latency_ms": 0.0,
    }


def test_all_unsafe_correct_gives_recall_1():
    records = [_rec(f"s{i}", "unsafe", "unsafe") for i in range(10)]
    s = summarize(records)
    assert s["n"] == 10
    assert s["unsafe_recall"] == 1.0
    assert s["error_rate"] == 0.0


def test_partial_recall():
    records = [_rec("a", "unsafe", "unsafe"),
               _rec("b", "unsafe", "safe"),
               _rec("c", "unsafe", "unsafe"),
               _rec("d", "unsafe", "error")]
    s = summarize(records)
    # unsafe_total = 4 (all start with unsafe->), tp = 2
    assert s["unsafe_recall"] == 0.5
    assert s["error_rate"] == 0.25


def test_category_breakdown():
    records = [
        _rec("a", "unsafe", "unsafe", cat="violence"),
        _rec("b", "unsafe", "safe", cat="violence"),
        _rec("c", "unsafe", "unsafe", cat="hate"),
    ]
    s = summarize(records)
    assert s["by_expected_category"]["violence"]["n"] == 2
    assert s["by_expected_category"]["violence"]["recall"] == 0.5
    assert s["by_expected_category"]["hate"]["recall"] == 1.0


def test_empty():
    assert summarize([]) == {"n": 0}
