from __future__ import annotations

from guardrail_eval.models.gemma_3_it import parse_gemma_binary_output


def test_parse_gemma_binary_output_accepts_leading_label() -> None:
    label, categories, error_reason = parse_gemma_binary_output("unsafe\nBecause the prompt requests harm.")
    assert label == "unsafe"
    assert categories == []
    assert error_reason is None


def test_parse_gemma_binary_output_rejects_missing_label() -> None:
    label, categories, error_reason = parse_gemma_binary_output("This input should be blocked.")
    assert label == "error"
    assert categories == []
    assert error_reason == "missing_label"
