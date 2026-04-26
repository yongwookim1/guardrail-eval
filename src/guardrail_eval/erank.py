from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .benchmarks.base import Benchmark
from .erank_backends import (
    TransformersGemma3ERankBackend,
    TransformersQwen25VLERankBackend,
)
from .io import JsonlWriter, write_json
from .models.base import resolve_model_source


def effective_rank(matrix: torch.Tensor, *, metric_device: str = "cpu", eps: float = 1e-12) -> float:
    if matrix.ndim != 2:
        raise ValueError(f"effective_rank() expects a 2D matrix, got shape={tuple(matrix.shape)!r}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("effective_rank() requires a non-empty matrix")

    features = matrix.to(device=metric_device, dtype=torch.float32)
    singular_values = torch.linalg.svdvals(features)
    singular_values = singular_values[singular_values > eps]
    if singular_values.numel() == 0:
        return 0.0

    probabilities = singular_values / singular_values.sum()
    entropy = -(probabilities * torch.log(probabilities)).sum()
    return float(torch.exp(entropy).item())


def infer_erank_family(model_config: dict[str, Any]) -> str:
    class_path = str(model_config.get("class", ""))
    model_name = str(model_config.get("name", ""))
    if "qwen2_5_vl" in class_path or model_name.startswith(
        ("qwen2_5_vl", "guardreasoner_vl", "safeqwen2_5_vl")
    ):
        return "qwen2_5_vl"
    if "gemma_3" in class_path or "nemotron" in class_path or model_name.startswith(
        ("gemma_3", "nemotron")
    ):
        return "gemma3"
    raise ValueError(f"Model {model_name!r} is not effective-rank-supported yet")


def build_erank_backend(model_config: dict[str, Any]):
    model_ref = resolve_model_source(model_config)
    backend_kwargs = dict(model_config.get("backend_kwargs", {}))
    family = infer_erank_family(model_config)

    if family == "gemma3":
        backend = TransformersGemma3ERankBackend(
            model_ref=model_ref,
            backend_kwargs=backend_kwargs,
        )
        backend.error_name = str(model_config["name"])
        return family, backend

    if family == "qwen2_5_vl":
        backend = TransformersQwen25VLERankBackend(
            model_ref=model_ref,
            backend_kwargs=backend_kwargs,
        )
        backend.error_name = str(model_config["name"])
        return family, backend

    raise ValueError(f"Unsupported effective-rank family: {family}")


def resolve_metric_device(requested: str, model_device: str) -> str:
    if requested == "auto":
        if model_device.startswith("cuda") and torch.cuda.is_available():
            return model_device
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested metric_device={requested!r}, but CUDA is not available")
    return requested


@dataclass
class ERankRunArtifacts:
    summary: dict[str, Any]
    per_position: list[dict[str, Any]]
    top_positions: list[dict[str, Any]]
    debug_records: list[dict[str, Any]]
    samples: list[dict[str, Any]]
    config: dict[str, Any]


def run_erank_evaluation(
    *,
    model_config: dict[str, Any],
    benchmark: Benchmark,
    limit: int | None,
    top_k: int,
    metric_device: str,
    debug_limit: int = 5,
) -> ERankRunArtifacts:
    if benchmark.task_type == "multiple_choice":
        raise ValueError(
            f"Effective-rank evaluation expects an open-ended multimodal benchmark, got multiple_choice benchmark {benchmark.name!r}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    family, backend = build_erank_backend(model_config)
    resolved_metric_device = resolve_metric_device(metric_device, str(getattr(backend, "device", "cpu")))

    image_token_rows: list[torch.Tensor] = []
    image_token_counts: list[int] = []
    debug_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []

    try:
        sample_iter = benchmark.iter_samples(limit=limit)
        total = benchmark.num_samples(limit=limit)
        for sample in tqdm(sample_iter, total=total, desc=f"eRank:{model_config['name']}/{benchmark.name}"):
            sample_layers = backend.collect_erank_sample(sample)
            image_token_rows.append(sample_layers.image_tokens)
            image_token_count = int(sample_layers.debug["vision_token_count"])
            image_token_counts.append(image_token_count)
            sample_records.append(
                {
                    "sample_id": sample.id,
                    "question_id": sample.meta.get("question_id"),
                    "question": sample.text,
                    "category": sample.category,
                    "image_path": sample.image_path,
                    "image_token_count": image_token_count,
                }
            )
            if len(debug_records) < debug_limit:
                debug_records.append(dict(sample_layers.debug))
    finally:
        backend.close()

    if not image_token_rows:
        raise RuntimeError("Effective-rank evaluation did not collect any image-token hidden states")

    shared_image_tokens = min(int(rows.shape[0]) for rows in image_token_rows)
    hidden_size = int(image_token_rows[0].shape[1])
    per_position: list[dict[str, Any]] = []
    for position in range(shared_image_tokens):
        matrix = torch.stack([rows[position] for rows in image_token_rows], dim=0)
        score = effective_rank(matrix, metric_device=resolved_metric_device)
        per_position.append(
            {
                "position": position,
                "position_1indexed": position + 1,
                "effective_rank": score,
                "rows": int(matrix.shape[0]),
                "hidden_size": int(matrix.shape[1]),
            }
        )

    top_positions = sorted(
        per_position,
        key=lambda row: (-float(row["effective_rank"]), int(row["position"])),
    )[:top_k]
    erank_values = [float(row["effective_rank"]) for row in per_position]

    summary = {
        "model_name": model_config["name"],
        "erank_family": family,
        "benchmark": benchmark.name,
        "n_samples": len(sample_records),
        "shared_image_tokens": shared_image_tokens,
        "hidden_size": hidden_size,
        "metric_device": resolved_metric_device,
        "top_k": top_k,
        "position_effective_rank_mean": statistics.fmean(erank_values),
        "position_effective_rank_median": statistics.median(erank_values),
        "position_effective_rank_min": min(erank_values),
        "position_effective_rank_max": max(erank_values),
        "position_effective_rank_std": statistics.pstdev(erank_values) if len(erank_values) > 1 else 0.0,
        "top_k_effective_rank_mean": statistics.fmean(
            [float(row["effective_rank"]) for row in top_positions]
        ),
        "image_token_count_mean": statistics.fmean(image_token_counts),
        "image_token_count_min": min(image_token_counts),
        "image_token_count_max": max(image_token_counts),
        "token_alignment_strategy": "shared_min_prefix",
    }
    config = {
        "model_config": model_config,
        "benchmark_config": benchmark.config,
        "runner": {
            "limit": limit,
            "top_k": top_k,
            "metric_device": resolved_metric_device,
            "debug_limit": debug_limit,
        },
    }
    return ERankRunArtifacts(
        summary=summary,
        per_position=per_position,
        top_positions=top_positions,
        debug_records=debug_records,
        samples=sample_records,
        config=config,
    )


def write_erank_artifacts(output_dir: str | Path, artifacts: ERankRunArtifacts) -> None:
    output_path = Path(output_dir)
    write_json(output_path / "results_summary.json", artifacts.summary)
    write_json(output_path / "per_position.json", {"positions": artifacts.per_position})
    write_json(output_path / "top_positions.json", {"positions": artifacts.top_positions})
    write_json(output_path / "samples.json", {"samples": artifacts.samples})
    write_json(output_path / "config.json", artifacts.config)
    if artifacts.debug_records:
        with JsonlWriter(output_path / "debug_spans.jsonl") as writer:
            writer.write_many(artifacts.debug_records, flush=True)
