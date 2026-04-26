from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmarks.base import Benchmark
from .benchmarks.registry import load_benchmark
from .erank import infer_erank_family, run_erank_evaluation, write_erank_artifacts
from .io import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def _resolve_benchmark_config(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    candidate = CONFIGS_DIR / "benchmarks" / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No benchmark config found for {name_or_path!r}")


def _resolve_model_config_for_task(name_or_path: str, task_type: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path

    candidates: list[str]
    if task_type == "multiple_choice":
        candidates = [f"{name_or_path}_choice", name_or_path]
    else:
        candidates = [name_or_path, f"{name_or_path}_choice"]

    for candidate_name in candidates:
        candidate = CONFIGS_DIR / "models" / f"{candidate_name}.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model config found for {name_or_path!r} with task_type={task_type!r}")


def _public_model_names() -> list[str]:
    names: list[str] = []
    for config_path in sorted((CONFIGS_DIR / "models").glob("*.yaml")):
        if config_path.stem.endswith("_choice"):
            continue
        cfg = load_yaml(config_path)
        try:
            infer_erank_family(cfg)
        except ValueError:
            continue
        names.append(config_path.stem)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="guardrail-erank",
        description="Measure last-layer image-token effective rank on open-ended multimodal benchmarks.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help='One or more model config names/paths, or "all".',
    )
    parser.add_argument("--benchmark", default="okvqa_erank")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "erank"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metric-device", default="auto", help='Metric device: "auto", "cpu", or e.g. "cuda".')
    parser.add_argument("--debug-spans", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    benchmark_cfg_path = _resolve_benchmark_config(args.benchmark)
    benchmark_cfg = load_yaml(benchmark_cfg_path)
    benchmark = load_benchmark(str(benchmark_cfg_path))
    if not isinstance(benchmark, Benchmark):
        raise TypeError(f"Invalid benchmark instance loaded for {benchmark_cfg_path}")
    if benchmark.task_type == "multiple_choice":
        raise TypeError(
            f"Benchmark {benchmark.name!r} is multiple_choice; use an open-ended query benchmark such as okvqa_erank."
        )

    requested_models = _public_model_names() if "all" in args.model else args.model
    exit_code = 0

    for requested_model in requested_models:
        model_cfg_path = _resolve_model_config_for_task(requested_model, benchmark.task_type)
        model_cfg = load_yaml(model_cfg_path)
        model_name = str(model_cfg["name"])
        output_dir = Path(args.output_dir) / model_name / benchmark.name
        summary_path = output_dir / "results_summary.json"

        if args.skip_existing and summary_path.exists():
            print(f"[skip {model_name}/{benchmark.name}] {summary_path} already exists")
            continue

        try:
            artifacts = run_erank_evaluation(
                model_config=model_cfg,
                benchmark=benchmark,
                limit=args.limit,
                top_k=args.top_k,
                metric_device=args.metric_device,
                debug_limit=args.debug_spans,
            )
        except Exception as exc:
            exit_code = 1
            print(f"[error {requested_model}/{benchmark.name}] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        write_erank_artifacts(output_dir, artifacts)
        print(
            f"[{model_name}/{benchmark.name}] "
            f"n={artifacts.summary['n_samples']} "
            f"shared_tokens={artifacts.summary['shared_image_tokens']} "
            f"erank_mean={artifacts.summary['position_effective_rank_mean']:.4f} "
            f"top{artifacts.summary['top_k']}_mean={artifacts.summary['top_k_effective_rank_mean']:.4f}"
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
