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

from ..backends.vllm_backend import VLLMBackend, chat_samples as vllm_chat_samples
from ..types import Sample, Verdict
from .base import GuardrailModel, resolve_model_source
from .registry import register_model


_CATEGORY_RE = re.compile(r"\bS\d{1,2}\b")


def parse_llama_guard_output(raw: str) -> tuple[str, list[str], str | None]:
    """Return (label, categories, error_reason)."""
    if not raw:
        return "error", [], "empty_output"
    tokens = [t.strip() for t in raw.replace(",", "\n").splitlines() if t.strip()]
    if not tokens:
        return "error", [], "empty_output"

    label_idx = next((i for i, token in enumerate(tokens[:3]) if token.lower() in ("safe", "unsafe")), None)
    if label_idx is None:
        return "error", [], "missing_label"

    first = tokens[label_idx].lower()
    cats = _CATEGORY_RE.findall(" ".join(tokens[label_idx + 1:]))
    return first, cats, None


def _build_backend(config: dict[str, Any], model_ref: str):
    backend_name = str(config.get("backend", "vllm"))
    backend_kwargs = config.get("backend_kwargs", {})
    if backend_name == "vllm":
        return backend_name, VLLMBackend(model_ref=model_ref, backend_kwargs=backend_kwargs)
    if backend_name == "transformers":
        from ..backends.transformers_llama4_backend import TransformersLlama4Backend

        return backend_name, TransformersLlama4Backend(model_ref=model_ref, backend_kwargs=backend_kwargs)
    raise ValueError(f"Unsupported backend for llama_guard_4: {backend_name}")


@register_model("llama_guard_4")
class LlamaGuard4(GuardrailModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.sampling: dict[str, Any] = config.get("sampling", {"max_tokens": 20, "temperature": 0.0})
        self.backend_name, self.backend = _build_backend(config, resolve_model_source(config))

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        if not samples:
            return []
        if self.backend_name == "vllm":
            outputs = vllm_chat_samples(self.backend, samples, sampling=self.sampling)
            verdicts: list[Verdict] = []
            for raw, batch_avg_latency in outputs:
                label, cats, error_reason = parse_llama_guard_output(raw)
                verdicts.append(Verdict(
                    label=label, categories=cats, raw=raw,
                    error_reason=error_reason,
                    batch_avg_latency_ms=batch_avg_latency,
                ))
            return verdicts

        verdicts: list[Verdict] = []
        for sample in samples:
            try:
                outputs = self.backend.chat_samples([sample], sampling=self.sampling)
            except Exception as exc:
                verdicts.append(Verdict(
                    label="error",
                    categories=[],
                    raw="",
                    error_reason=f"backend_error:{type(exc).__name__}",
                    batch_avg_latency_ms=0.0,
                ))
                continue
            raw, batch_avg_latency = outputs[0]
            label, cats, error_reason = parse_llama_guard_output(raw)
            verdicts.append(Verdict(
                label=label, categories=cats, raw=raw,
                error_reason=error_reason,
                batch_avg_latency_ms=batch_avg_latency,
            ))
        return verdicts

    def close(self) -> None:
        self.backend.close()
