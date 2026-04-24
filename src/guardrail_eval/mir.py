from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .io import JsonlWriter, write_json
from .mir_backends import (
    TransformersGemma3MIRBackend,
    TransformersQwen25VLMIRBackend,
)
from .mir_data import MIRInputPair, build_mir_input_pairs
from .models.base import resolve_model_source


class MatrixSquareRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        assert input_tensor.shape[0] == input_tensor.shape[1], "Input must be a square matrix"
        max_iter = 5
        eye = torch.eye(input_tensor.shape[0], device=input_tensor.device, dtype=input_tensor.dtype)
        y = input_tensor
        z = eye
        for _ in range(max_iter):
            y = 0.5 * (y + torch.inverse(z) @ input_tensor)
            z = 0.5 * (z + torch.inverse(y) @ input_tensor)
        ctx.save_for_backward(y, z)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, z = ctx.saved_tensors
        grad_input = grad_output @ torch.inverse(y).t() @ torch.inverse(y).t()
        return grad_input


def matrix_sqrt(input_tensor: torch.Tensor) -> torch.Tensor:
    return MatrixSquareRoot.apply(input_tensor)


def _covariance(features: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    centered = features - features.mean(dim=0)
    cov = centered.T @ centered / (features.size(0) - 1)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov + eps * eye


def _matrix_sqrt_trace_symmetric_psd(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvalsh(matrix)
    clipped = torch.clamp(eigenvalues, min=0.0)
    return torch.sqrt(clipped).sum()


def _trace_sqrt_product_torch(cov_a: torch.Tensor, cov_b: torch.Tensor) -> torch.Tensor:
    # For PSD covariances A and B, trace(sqrt(A B)) == trace(sqrt(A^(1/2) B A^(1/2))).
    # The inner matrix is symmetric PSD, so an eigendecomposition is stable and keeps
    # the "fast" mode independent of SciPy.
    eigvals_a, eigvecs_a = torch.linalg.eigh(cov_a)
    eigvals_a = torch.clamp(eigvals_a, min=0.0)
    sqrt_cov_a = eigvecs_a @ torch.diag(torch.sqrt(eigvals_a)) @ eigvecs_a.T
    sym_prod = sqrt_cov_a @ cov_b @ sqrt_cov_a
    sym_prod = 0.5 * (sym_prod + sym_prod.T)
    return _matrix_sqrt_trace_symmetric_psd(sym_prod)


def calculate_fid(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    try:
        from scipy.linalg import sqrtm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "accurate MIR mode requires scipy. Install it or run with --mode fast."
        ) from exc

    tensor_a = tensor_a.to(dtype=torch.float32)
    tensor_b = tensor_b.to(dtype=torch.float32)
    mu_a = tensor_a.mean(dim=0)
    mu_b = tensor_b.mean(dim=0)
    cov_a = _covariance(tensor_a)
    cov_b = _covariance(tensor_b)
    cov_a_np = cov_a.cpu().numpy()
    cov_b_np = cov_b.cpu().numpy()
    sqrt_cov_a_np = sqrtm(cov_a_np)
    if np.iscomplexobj(sqrt_cov_a_np):
        sqrt_cov_a_np = sqrt_cov_a_np.real
    sym_prod = sqrt_cov_a_np.dot(cov_b_np).dot(sqrt_cov_a_np)
    sym_prod = 0.5 * (sym_prod + sym_prod.T)
    covmean = sqrtm(sym_prod)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    mean_diff_np = (mu_a - mu_b).cpu().numpy()
    fid = (
        np.sum(mean_diff_np ** 2)
        + np.trace(cov_a_np)
        + np.trace(cov_b_np)
        - 2 * np.trace(covmean)
    )
    return float(fid)


@torch.no_grad()
def calculate_fid_pytorch(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    tensor_a = tensor_a.to(dtype=torch.float32)
    tensor_b = tensor_b.to(dtype=torch.float32)
    mu_a = tensor_a.mean(dim=0)
    mu_b = tensor_b.mean(dim=0)
    cov_a = _covariance(tensor_a)
    cov_b = _covariance(tensor_b)
    mean_diff = mu_a - mu_b
    trace_covmean = _trace_sqrt_product_torch(cov_a, cov_b)
    fid = torch.sum(mean_diff ** 2) + torch.trace(cov_a) + torch.trace(cov_b) - 2 * trace_covmean
    return float(fid.item())


def replace_outliers_with_median_l2(data: torch.Tensor) -> torch.Tensor:
    cleaned = data.clone()
    norms = torch.norm(cleaned, p=2, dim=-1)
    median_norm = torch.median(norms)
    std_dev = torch.std(norms)
    outliers = torch.abs(norms - median_norm) > 3 * std_dev
    median_values = torch.median(cleaned, dim=0).values
    cleaned[outliers, :] = median_values
    return cleaned


def _sanitize_non_negative(value: float) -> float:
    if value < 0.0 and abs(value) < 1e-6:
        return 0.0
    return value


def apply_text_centric_normalization(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_norm_mean = text_features.norm(p=2, dim=-1).mean(0)
    if torch.isclose(text_norm_mean, torch.tensor(0.0, device=text_features.device)):
        raise ValueError("Text feature norm mean is zero; cannot apply text-centric normalization")
    scale_factor = 1.0 / text_norm_mean
    return scale_factor * vision_features, scale_factor * text_features


def compute_layer_mir(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    mode: str,
    metric_device: str,
) -> float:
    vision_features = vision_features.to(device=metric_device, dtype=torch.float32)
    text_features = text_features.to(device=metric_device, dtype=torch.float32)
    vision_features, text_features = apply_text_centric_normalization(vision_features, text_features)
    vision_features = replace_outliers_with_median_l2(vision_features)
    text_features = replace_outliers_with_median_l2(text_features)

    if mode == "fast":
        score = calculate_fid_pytorch(vision_features, text_features)
    elif mode == "accurate":
        score = calculate_fid(vision_features, text_features)
    else:
        raise ValueError(f"Unsupported MIR mode: {mode}")
    return _sanitize_non_negative(float(score))


def infer_mir_family(model_config: dict[str, Any]) -> str:
    class_path = str(model_config.get("class", ""))
    model_name = str(model_config.get("name", ""))
    if "qwen2_5_vl" in class_path or model_name.startswith(("qwen2_5_vl", "guardreasoner_vl", "safeqwen2_5_vl")):
        return "qwen2_5_vl"
    if "gemma_3" in class_path or "nemotron" in class_path or model_name.startswith(("gemma_3", "nemotron")):
        return "gemma3"
    raise ValueError(f"Model {model_name!r} is not MIR-supported yet")


def build_mir_backend(model_config: dict[str, Any]):
    model_ref = resolve_model_source(model_config)
    backend_kwargs = dict(model_config.get("backend_kwargs", {}))
    family = infer_mir_family(model_config)
    if family == "gemma3":
        backend = TransformersGemma3MIRBackend(model_ref=model_ref, backend_kwargs=backend_kwargs)
        backend.error_name = str(model_config["name"])
        return family, backend
    if family == "qwen2_5_vl":
        backend = TransformersQwen25VLMIRBackend(model_ref=model_ref, backend_kwargs=backend_kwargs)
        backend.error_name = str(model_config["name"])
        return family, backend
    raise ValueError(f"Unsupported MIR family: {family}")


def resolve_metric_device(requested: str, model_device: str) -> str:
    if requested == "auto":
        if model_device.startswith("cuda") and torch.cuda.is_available():
            return model_device
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested metric_device={requested!r}, but CUDA is not available")
    return requested


@dataclass
class MIRRunArtifacts:
    summary: dict[str, Any]
    per_layer: list[dict[str, Any]]
    sample_pairs: list[dict[str, Any]]
    debug_records: list[dict[str, Any]]
    config: dict[str, Any]


def run_mir_evaluation(
    *,
    model_config: dict[str, Any],
    image_data_path: str | Path,
    text_data_path: str | Path,
    eval_num: int,
    mode: str,
    shuffle: bool,
    seed: int,
    metric_device: str,
    debug_limit: int = 5,
) -> MIRRunArtifacts:
    pairs = build_mir_input_pairs(
        image_data_path,
        text_data_path,
        eval_num=eval_num,
        shuffle=shuffle,
        seed=seed,
    )
    family, backend = build_mir_backend(model_config)
    resolved_metric_device = resolve_metric_device(metric_device, str(getattr(backend, "device", "cpu")))

    vision_layers_by_index: list[list[torch.Tensor]] = []
    text_layers_by_index: list[list[torch.Tensor]] = []
    debug_records: list[dict[str, Any]] = []
    sample_pairs: list[dict[str, Any]] = []
    prompt_token_counts: list[int] = []
    vision_token_counts: list[int] = []
    text_token_counts: list[int] = []

    try:
        for pair in tqdm(pairs, desc=f"MIR:{model_config['name']}"):
            sample_layers = backend.collect_mir_sample(
                sample_id=pair.sample_id,
                image_path=str(pair.image_path),
                text=pair.text,
            )
            if not vision_layers_by_index:
                layer_count = len(sample_layers.vision_layers)
                vision_layers_by_index = [[] for _ in range(layer_count)]
                text_layers_by_index = [[] for _ in range(layer_count)]

            for layer_idx, layer_tensor in enumerate(sample_layers.vision_layers):
                vision_layers_by_index[layer_idx].append(layer_tensor)
            for layer_idx, layer_tensor in enumerate(sample_layers.text_layers):
                text_layers_by_index[layer_idx].append(layer_tensor)

            prompt_token_counts.append(int(sample_layers.debug["prompt_token_count"]))
            vision_token_counts.append(int(sample_layers.debug["vision_token_count"]))
            text_token_counts.append(int(sample_layers.debug["text_token_count"]))
            sample_pairs.append(
                {
                    "sample_id": pair.sample_id,
                    "image_path": str(pair.image_path),
                    "text_path": str(pair.text_path),
                }
            )
            if len(debug_records) < debug_limit:
                debug_record = dict(sample_layers.debug)
                debug_record["text_path"] = str(pair.text_path)
                debug_records.append(debug_record)
    finally:
        backend.close()

    if not vision_layers_by_index:
        raise RuntimeError("MIR evaluation did not collect any hidden states")

    per_layer: list[dict[str, Any]] = []
    for layer_idx in range(1, len(vision_layers_by_index)):
        vision_features = torch.cat(vision_layers_by_index[layer_idx], dim=0)
        text_features = torch.cat(text_layers_by_index[layer_idx], dim=0)
        mir_score = compute_layer_mir(
            vision_features,
            text_features,
            mode=mode,
            metric_device=resolved_metric_device,
        )
        per_layer.append(
            {
                "layer": layer_idx,
                "mir": mir_score,
                "vision_rows": int(vision_features.shape[0]),
                "text_rows": int(text_features.shape[0]),
                "hidden_size": int(vision_features.shape[1]),
            }
        )

    total_mir = sum(layer_result["mir"] for layer_result in per_layer)
    overall_mir = float("-inf") if total_mir <= 0 else float(math.log10(total_mir))

    summary = {
        "model_name": model_config["name"],
        "mir_family": family,
        "mode": mode,
        "eval_num": eval_num,
        "overall_mir": overall_mir,
        "scored_layers": len(per_layer),
        "metric_device": resolved_metric_device,
        "image_data_path": str(Path(image_data_path).resolve()),
        "text_data_path": str(Path(text_data_path).resolve()),
        "prompt_token_count_mean": float(np.mean(prompt_token_counts)),
        "vision_token_count_mean": float(np.mean(vision_token_counts)),
        "text_token_count_mean": float(np.mean(text_token_counts)),
        "prompt_token_count_min": int(min(prompt_token_counts)),
        "prompt_token_count_max": int(max(prompt_token_counts)),
        "vision_token_count_min": int(min(vision_token_counts)),
        "vision_token_count_max": int(max(vision_token_counts)),
        "text_token_count_min": int(min(text_token_counts)),
        "text_token_count_max": int(max(text_token_counts)),
    }
    config = {
        "model_config": model_config,
        "runner": {
            "image_data_path": str(Path(image_data_path).resolve()),
            "text_data_path": str(Path(text_data_path).resolve()),
            "eval_num": eval_num,
            "mode": mode,
            "shuffle": shuffle,
            "seed": seed,
            "metric_device": resolved_metric_device,
            "debug_limit": debug_limit,
        },
    }
    return MIRRunArtifacts(
        summary=summary,
        per_layer=per_layer,
        sample_pairs=sample_pairs,
        debug_records=debug_records,
        config=config,
    )


def write_mir_artifacts(output_dir: str | Path, artifacts: MIRRunArtifacts) -> None:
    output_path = Path(output_dir)
    write_json(output_path / "results_summary.json", artifacts.summary)
    write_json(output_path / "per_layer.json", {"layers": artifacts.per_layer})
    write_json(output_path / "config.json", artifacts.config)
    write_json(output_path / "sample_pairs.json", {"samples": artifacts.sample_pairs})
    if artifacts.debug_records:
        with JsonlWriter(output_path / "debug_spans.jsonl") as writer:
            writer.write_many(artifacts.debug_records, flush=True)
