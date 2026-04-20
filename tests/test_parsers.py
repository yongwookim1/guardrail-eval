"""Parser unit tests. These do NOT load vLLM or any HF weights."""
from __future__ import annotations

from guardrail_eval.models.nemotron import parse_nemotron_output
from guardrail_eval.models.llama_guard import parse_llama_guard_output


# ---------- Nemotron ----------

def test_nemotron_unsafe_with_categories():
    raw = "User Safety: unsafe\nResponse Safety: unsafe\nSafety Categories: Criminal Planning/Confessions, Violence"
    label, cats = parse_nemotron_output(raw)
    assert label == "unsafe"
    assert cats == ["Criminal Planning/Confessions", "Violence"]


def test_nemotron_safe_no_categories_line():
    raw = "User Safety: safe"
    label, cats = parse_nemotron_output(raw)
    assert label == "safe"
    assert cats == []


def test_nemotron_case_insensitive():
    raw = "user safety: UNSAFE\nsafety categories: Hate/Identity Hate"
    label, cats = parse_nemotron_output(raw)
    assert label == "unsafe"
    assert cats == ["Hate/Identity Hate"]


def test_nemotron_unparseable_returns_error():
    label, cats = parse_nemotron_output("I'm sorry, I can't help with that.")
    assert label == "error"
    assert cats == []


def test_nemotron_empty_returns_error():
    assert parse_nemotron_output("") == ("error", [])


# ---------- Llama-Guard-4 ----------

def test_llama_guard_safe():
    label, cats = parse_llama_guard_output("\n\nsafe")
    assert label == "safe"
    assert cats == []


def test_llama_guard_unsafe_single_category():
    label, cats = parse_llama_guard_output("\n\nunsafe\nS9")
    assert label == "unsafe"
    assert cats == ["S9"]


def test_llama_guard_unsafe_multiple_categories_comma():
    label, cats = parse_llama_guard_output("unsafe\nS1,S9")
    assert label == "unsafe"
    assert cats == ["S1", "S9"]


def test_llama_guard_unsafe_multiple_categories_newline():
    label, cats = parse_llama_guard_output("unsafe\nS1\nS9")
    assert label == "unsafe"
    assert cats == ["S1", "S9"]


def test_llama_guard_noise_before_label_is_error():
    label, _ = parse_llama_guard_output("I cannot comply.\nunsafe")
    assert label == "error"


def test_llama_guard_empty_returns_error():
    assert parse_llama_guard_output("") == ("error", [])
