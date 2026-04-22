from __future__ import annotations

from typing import Any

from ..backends.transformers_mcq_backends import (
    DEFAULT_MCQ_SYSTEM_PROMPT,
    TransformersGemma3MCQBackend,
    TransformersQwen25OmniMCQBackend,
    TransformersQwen25VLMCQBackend,
)
from ..types import MCQSample, MCQVerdict, Sample, Verdict
from .base import GuardrailModel, resolve_model_source
from .registry import register_model


class _BaseTransformersMCQModel(GuardrailModel):
    backend_cls = None

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        backend_kwargs = config.get("backend_kwargs", {})
        system_prompt = str(config.get("system_prompt", DEFAULT_MCQ_SYSTEM_PROMPT))
        assert self.backend_cls is not None
        self.backend = self.backend_cls(
            resolve_model_source(config),
            system_prompt=system_prompt,
            backend_kwargs=backend_kwargs,
        )

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        raise NotImplementedError(f"{self.name} only supports MCQ benchmarks")

    def score_mcq_batch(self, samples: list[MCQSample]) -> list[MCQVerdict]:
        if not samples:
            return []
        try:
            return self.backend.score_mcq_samples(samples)
        except Exception:
            verdicts: list[MCQVerdict] = []
            for sample in samples:
                try:
                    verdicts.extend(self.backend.score_mcq_samples([sample]))
                except Exception as exc:
                    verdicts.append(
                        MCQVerdict(
                            pred_choice="error",
                            choice_losses={
                                label: None for label in sample.choice_labels
                            },
                            raw="",
                            error_reason=f"backend_error:{type(exc).__name__}",
                            batch_avg_latency_ms=0.0,
                        )
                    )
            return verdicts

    def close(self) -> None:
        self.backend.close()


@register_model("gemma_3_4b_it_mcq")
class Gemma3MCQModel(_BaseTransformersMCQModel):
    backend_cls = TransformersGemma3MCQBackend


@register_model("nemotron_cs_mcq")
class NemotronContentSafetyMCQModel(_BaseTransformersMCQModel):
    backend_cls = TransformersGemma3MCQBackend


@register_model("qwen2_5_vl_3b_instruct_mcq")
class Qwen25VL3BInstructMCQModel(_BaseTransformersMCQModel):
    backend_cls = TransformersQwen25VLMCQBackend


@register_model("guardreasoner_vl_3b_mcq")
class GuardReasonerVL3BMCQModel(_BaseTransformersMCQModel):
    backend_cls = TransformersQwen25VLMCQBackend


@register_model("qwen2_5_omni_3b_mcq")
class Qwen25Omni3BMCQModel(_BaseTransformersMCQModel):
    backend_cls = TransformersQwen25OmniMCQBackend


@register_model("omniguard_3b_mcq")
class OmniGuard3BMCQModel(_BaseTransformersMCQModel):
    backend_cls = TransformersQwen25OmniMCQBackend
