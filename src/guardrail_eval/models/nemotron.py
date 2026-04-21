"""nvidia/Nemotron-3-Content-Safety (Gemma-3-4B-it base, multimodal).

Model card: https://huggingface.co/nvidia/Nemotron-3-Content-Safety

Output format (line-delimited):
    User Safety: safe|unsafe
    Response Safety: safe|unsafe       # only when an assistant turn was provided
    Safety Categories: cat1, cat2, ... # only when request_categories="/categories"
"""
from __future__ import annotations

import re
from typing import Any

from ..backends.vllm_backend import VLLMBackend, chat_samples as vllm_chat_samples
from ..types import Sample, Verdict
from .base import GuardrailModel, resolve_model_source
from .registry import register_model


_LABEL_RE = re.compile(r"^\s*User Safety\s*:\s*(safe|unsafe)", re.IGNORECASE | re.MULTILINE)
_CATEGORIES_RE = re.compile(r"^\s*Safety Categories\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_BARE_LABEL_RE = re.compile(r"^\s*(safe|unsafe)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_nemotron_output(raw: str) -> tuple[str, list[str], str | None]:
    """Return (label, categories, error_reason)."""
    label_match = _LABEL_RE.search(raw or "")
    if label_match:
        label = label_match.group(1).lower()
        error_reason = None
    else:
        bare_match = _BARE_LABEL_RE.search(raw or "")
        if bare_match:
            label = bare_match.group(1).lower()
            error_reason = None
        else:
            return "error", [], "missing_user_safety_label"

    cats: list[str] = []
    cat_match = _CATEGORIES_RE.search(raw or "")
    if cat_match:
        cats = [c.strip() for c in cat_match.group(1).split(",") if c.strip()]
    return label, cats, error_reason


def _build_backend(config: dict[str, Any], model_ref: str):
    backend_name = str(config.get("backend", "transformers"))
    backend_kwargs = config.get("backend_kwargs", {})
    if backend_name == "vllm":
        return backend_name, VLLMBackend(model_ref=model_ref, backend_kwargs=backend_kwargs)
    if backend_name == "transformers":
        from ..backends.transformers_gemma3_backend import TransformersGemma3Backend

        return backend_name, TransformersGemma3Backend(model_ref=model_ref, backend_kwargs=backend_kwargs)
    raise ValueError(f"Unsupported backend for nemotron_cs: {backend_name}")


@register_model("nemotron_cs")
class NemotronContentSafety(GuardrailModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.request_categories: str = config.get("request_categories", "/categories")
        self.sampling: dict[str, Any] = config.get("sampling", {})
        self.backend_name, self.backend = _build_backend(config, resolve_model_source(config))

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        if not samples:
            return []
        if self.backend_name == "vllm":
            outputs = vllm_chat_samples(
                self.backend,
                samples,
                sampling=self.sampling,
                chat_template_kwargs={"request_categories": self.request_categories},
            )
            verdicts: list[Verdict] = []
            for raw, batch_avg_latency in outputs:
                label, cats, error_reason = parse_nemotron_output(raw)
                verdicts.append(Verdict(
                    label=label, categories=cats, raw=raw,
                    error_reason=error_reason,
                    batch_avg_latency_ms=batch_avg_latency,
                ))
            return verdicts

        try:
            outputs = self.backend.chat_samples(
                samples,
                sampling=self.sampling,
                chat_template_kwargs={"request_categories": self.request_categories},
            )
        except Exception:
            outputs = []
            verdicts = []
            for sample in samples:
                try:
                    outputs = self.backend.chat_samples(
                        [sample],
                        sampling=self.sampling,
                        chat_template_kwargs={"request_categories": self.request_categories},
                    )
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
                label, cats, error_reason = parse_nemotron_output(raw)
                verdicts.append(Verdict(
                    label=label, categories=cats, raw=raw,
                    error_reason=error_reason,
                    batch_avg_latency_ms=batch_avg_latency,
                ))
            return verdicts

        verdicts: list[Verdict] = []
        for raw, batch_avg_latency in outputs:
            label, cats, error_reason = parse_nemotron_output(raw)
            verdicts.append(Verdict(
                label=label, categories=cats, raw=raw,
                error_reason=error_reason,
                batch_avg_latency_ms=batch_avg_latency,
            ))
        return verdicts

    def close(self) -> None:
        self.backend.close()
