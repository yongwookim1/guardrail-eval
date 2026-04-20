"""meta-llama/Llama-Guard-4-12B (Llama-4 early-fusion, multimodal).

Model card: https://huggingface.co/meta-llama/Llama-Guard-4-12B

Output format is terse. The first non-empty line is "safe" or "unsafe"; any
subsequent non-empty lines hold comma-separated taxonomy codes (S1..S14).
Examples:
    "\n\nsafe"
    "\n\nunsafe\nS9"
    "\n\nunsafe\nS1,S9"

The HF card does not explicitly advertise vLLM support, but Llama-Guard-4
shares the Llama-4 early-fusion architecture, which vLLM ≥0.11 runs. If the
engine fails to load, switch `backend: vllm` to a transformers-based backend.
"""
from __future__ import annotations

import re
from typing import Any

from ..backends.vllm_backend import VLLMBackend, build_user_messages
from ..types import Sample, Verdict
from .base import GuardrailModel
from .registry import register_model


_CATEGORY_RE = re.compile(r"\bS\d{1,2}\b")


def parse_llama_guard_output(raw: str) -> tuple[str, list[str]]:
    """Return (label, categories). Label is 'safe', 'unsafe', or 'error' if unparseable."""
    if not raw:
        return "error", []
    tokens = [t.strip() for t in raw.replace(",", "\n").splitlines() if t.strip()]
    if not tokens:
        return "error", []
    first = tokens[0].lower()
    if first not in ("safe", "unsafe"):
        return "error", []
    cats = _CATEGORY_RE.findall(" ".join(tokens[1:]))
    return first, cats


@register_model("llama_guard_4")
class LlamaGuard4(GuardrailModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.sampling: dict[str, Any] = config.get("sampling", {"max_tokens": 20, "temperature": 0.0})
        self.backend = VLLMBackend(
            hf_id=config["hf_id"],
            backend_kwargs=config.get("backend_kwargs", {}),
        )

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        if not samples:
            return []
        conversations = build_user_messages(samples)
        outputs = self.backend.chat(conversations, sampling=self.sampling)
        verdicts: list[Verdict] = []
        for raw, latency in outputs:
            label, cats = parse_llama_guard_output(raw)
            verdicts.append(Verdict(label=label, categories=cats, raw=raw, latency_ms=latency))
        return verdicts

    def close(self) -> None:
        self.backend.close()
