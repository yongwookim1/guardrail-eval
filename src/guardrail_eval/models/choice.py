from __future__ import annotations

from typing import Any

from ..backends.transformers_choice_backends import (
    DEFAULT_CHOICE_SYSTEM_PROMPT,
    TransformersGemma3ChoiceBackend,
    TransformersQwen25OmniChoiceBackend,
    TransformersQwen25VLChoiceBackend,
)
from ..types import ChoiceSample, ChoiceVerdict, Sample, Verdict
from .base import GuardrailModel, resolve_model_source
from .registry import register_model


class _BaseTransformersChoiceModel(GuardrailModel):
    backend_cls = None

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        backend_kwargs = config.get("backend_kwargs", {})
        system_prompt = str(config.get("system_prompt", DEFAULT_CHOICE_SYSTEM_PROMPT))
        assert self.backend_cls is not None
        self.backend = self.backend_cls(
            resolve_model_source(config),
            system_prompt=system_prompt,
            backend_kwargs=backend_kwargs,
        )

    def classify_batch(self, samples: list[Sample]) -> list[Verdict]:
        raise NotImplementedError(f"{self.name} only supports multiple-choice benchmarks")

    def score_choice_batch(self, samples: list[ChoiceSample]) -> list[ChoiceVerdict]:
        if not samples:
            return []
        try:
            return self.backend.score_choice_samples(samples)
        except Exception:
            verdicts: list[ChoiceVerdict] = []
            for sample in samples:
                try:
                    verdicts.extend(self.backend.score_choice_samples([sample]))
                except Exception as exc:
                    verdicts.append(
                        ChoiceVerdict(
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


@register_model("gemma_3_4b_it_choice")
class Gemma3ChoiceModel(_BaseTransformersChoiceModel):
    backend_cls = TransformersGemma3ChoiceBackend


@register_model("nemotron_cs_choice")
class NemotronContentSafetyChoiceModel(_BaseTransformersChoiceModel):
    backend_cls = TransformersGemma3ChoiceBackend


@register_model("qwen2_5_vl_3b_instruct_choice")
class Qwen25VL3BInstructChoiceModel(_BaseTransformersChoiceModel):
    backend_cls = TransformersQwen25VLChoiceBackend


@register_model("guardreasoner_vl_3b_choice")
class GuardReasonerVL3BChoiceModel(_BaseTransformersChoiceModel):
    backend_cls = TransformersQwen25VLChoiceBackend


@register_model("qwen2_5_omni_3b_choice")
class Qwen25Omni3BChoiceModel(_BaseTransformersChoiceModel):
    backend_cls = TransformersQwen25OmniChoiceBackend


@register_model("omniguard_3b_choice")
class OmniGuard3BChoiceModel(_BaseTransformersChoiceModel):
    backend_cls = TransformersQwen25OmniChoiceBackend
