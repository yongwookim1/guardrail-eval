from __future__ import annotations

import time
from typing import Any

from ..io import file_to_data_uri
from ..types import Sample


class VLLMBackend:
    """Thin wrapper around vLLM's offline engine, built for multimodal chat.

    Models handle their own message construction and output parsing. This class
    only owns the engine lifecycle and a single `chat(...)` entry point.
    """

    def __init__(self, model_ref: str, backend_kwargs: dict[str, Any] | None = None) -> None:
        from vllm import LLM  # imported lazily so unit tests don't need vLLM

        backend_kwargs = backend_kwargs or {}
        self.model_ref = model_ref
        self.llm = LLM(model=model_ref, **backend_kwargs)

    def chat(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        sampling: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Run a batch of chat conversations.

        Returns (text, batch_avg_latency_ms) per item. vLLM processes batch
        items concurrently, so there is no cheap true per-item latency; we
        report batch wall-time divided by batch size as a throughput proxy.
        """
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=sampling.get("temperature", 0.0),
            max_tokens=sampling.get("max_tokens", 128),
            top_p=sampling.get("top_p", 1.0),
        )

        kwargs: dict[str, Any] = {"sampling_params": sp, "use_tqdm": False}
        if chat_template_kwargs:
            kwargs["chat_template_kwargs"] = chat_template_kwargs

        t0 = time.perf_counter()
        outputs = self.llm.chat(conversations, **kwargs)
        total_ms = (time.perf_counter() - t0) * 1000.0
        batch_avg_ms = total_ms / max(len(outputs), 1)
        return [(o.outputs[0].text, batch_avg_ms) for o in outputs]

    @staticmethod
    def build_user_message(text: str | None, image_path: str | None) -> dict[str, Any]:
        """Build an OpenAI-style user message with optional image."""
        content: list[dict[str, Any]] = []
        if image_path:
            content.append({"type": "image_url", "image_url": {"url": file_to_data_uri(image_path)}})
        if text:
            content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def close(self) -> None:
        # vLLM's LLM releases GPU memory when the object is garbage-collected.
        self.llm = None  # type: ignore[assignment]


def build_user_messages(samples: list[Sample]) -> list[list[dict[str, Any]]]:
    """Convenience: one-user-turn conversations for a batch of samples."""
    return [[VLLMBackend.build_user_message(s.text, s.image_path)] for s in samples]


def chat_samples(
    backend: VLLMBackend,
    samples: list[Sample],
    *,
    sampling: dict[str, Any],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[tuple[str, float]]:
    return backend.chat(
        build_user_messages(samples),
        sampling=sampling,
        chat_template_kwargs=chat_template_kwargs,
    )
