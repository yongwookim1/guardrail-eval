from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import Sample
from .transformers_gemma3_backend import TransformersGemma3Backend, _encode_image_base64


class TransformersGemma3ClassifierBackend(TransformersGemma3Backend):
    """Gemma-3 backend specialized for prompted binary safety classification."""

    error_name = "gemma_3_4b_it"

    def __init__(
        self,
        model_ref: str,
        *,
        system_prompt: str,
        user_prompt_template: str,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.system_prompt = system_prompt.strip()
        self.user_prompt_template = user_prompt_template.strip()
        super().__init__(model_ref=model_ref, backend_kwargs=backend_kwargs)

    def _build_messages(self, sample: Sample) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

        content: list[dict[str, Any]] = []
        if sample.image_path:
            image_path = str(Path(sample.image_path))
            content.append({"type": "image", "image": _encode_image_base64(image_path)})

        user_text = (sample.text or "").strip() or "(no text provided)"
        prompt_text = self.user_prompt_template.format(user_text=user_text)
        content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": content})
        return messages
