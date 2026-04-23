from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmarks.registry import load_benchmark
from .evaluator import run, run_choice
from .io import load_yaml
from .models.registry import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def _resolve_config(kind: str, name_or_path: str) -> Path:
    """Accept either a config name (e.g. 'nemotron_cs') or a full path."""
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = CONFIGS_DIR / kind / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No {kind} config found for {name_or_path!r}")


def _names(kind: str) -> list[str]:
    d = CONFIGS_DIR / kind
    return sorted(p.stem for p in d.glob("*.yaml")) if d.exists() else []


def _public_model_names() -> list[str]:
    public_names: set[str] = set()
    for stem in _names("models"):
        if stem.endswith("_choice"):
            public_names.add(stem[:-7])
        else:
            public_names.add(stem)
    return sorted(public_names)


def _resolve_model_config_for_task(name_or_path: str, task_type: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p

    candidates: list[str]
    if task_type == "multiple_choice":
        candidates = [f"{name_or_path}_choice", name_or_path]
    else:
        candidates = [name_or_path, f"{name_or_path}_choice"]

    for candidate_name in candidates:
        candidate = CONFIGS_DIR / "models" / f"{candidate_name}.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No models config found for {name_or_path!r} with task_type={task_type!r}")


def _fmt_metric(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guardrail-eval", description="Evaluate guardrail models on multimodal safety benchmarks.")
    parser.add_argument("--model", required=True, help='Model config name or path, or "all".')
    parser.add_argument("--benchmark", required=True, help='Benchmark config name or path, or "all".')
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume from existing results.jsonl if present.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have a results_summary.json.")
    parser.add_argument("--flush-every-batches", type=int, default=16, help="Flush results.jsonl every N batches instead of every batch.")
    args = parser.parse_args(argv)

    model_names = _public_model_names() if args.model == "all" else [args.model]
    benchmark_names = _names("benchmarks") if args.benchmark == "all" else [args.benchmark]
    benchmark_cache: dict[str, tuple[dict[str, object], object]] = {}
    model_cache: dict[str, tuple[dict[str, object], object]] = {}

    try:
        for b in benchmark_names:
            bench_cfg_path = _resolve_config("benchmarks", b)
            bench_cfg = load_yaml(bench_cfg_path)
            benchmark = load_benchmark(str(bench_cfg_path))
            benchmark_cache[str(bench_cfg_path)] = (bench_cfg, benchmark)

            for requested_model_name in model_names:
                model_cfg_path = _resolve_model_config_for_task(requested_model_name, benchmark.task_type)
                model_cfg_key = str(model_cfg_path)
                if model_cfg_key not in model_cache:
                    model_cache[model_cfg_key] = (
                        load_yaml(model_cfg_path),
                        load_model(str(model_cfg_path)),
                    )
                model_cfg, model = model_cache[model_cfg_key]
                if not model.supports_task(benchmark.task_type):
                    print(
                        f"[skip {model.name}/{benchmark.name}] "
                        f"benchmark task_type={benchmark.task_type} is not supported by model task_types={sorted(model.task_types)}"
                    )
                    continue

                runner = run_choice if benchmark.task_type == "multiple_choice" else run
                summary = runner(
                    model,
                    benchmark,
                    output_dir=args.output_dir,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    resume=args.resume,
                    skip_existing=args.skip_existing,
                    flush_every_batches=args.flush_every_batches,
                    model_config=model_cfg,
                    benchmark_config=bench_cfg,
                    output_model_name=requested_model_name,
                )
                parts = [
                    f"[{requested_model_name}/{benchmark.name}]",
                    f"n={summary['n']}",
                    f"correct={summary.get('correct', 'na')}",
                    f"acc={_fmt_metric(summary.get('accuracy'))}",
                    f"errors={summary['errors']}",
                ]
                question_level = summary.get("question_level")
                if isinstance(question_level, dict) and question_level.get("questions_complete"):
                    parts.append(f"q_acc={_fmt_metric(question_level.get('question_accuracy'))}")
                permutation_bias = summary.get("permutation_bias")
                if isinstance(permutation_bias, dict) and permutation_bias.get("questions_with_complete_passes"):
                    parts.append(f"perm_inconsistency={_fmt_metric(permutation_bias.get('inconsistency_rate'))}")
                if benchmark.task_type != "multiple_choice" and summary.get("safe_total"):
                    parts.append(f"safe_recall={_fmt_metric(summary.get('safe_recall'))}")
                if benchmark.task_type != "multiple_choice" and summary.get("unsafe_total"):
                    parts.append(f"unsafe_recall={_fmt_metric(summary.get('unsafe_recall'))}")
                if benchmark.task_type != "multiple_choice" and summary.get("safe_total") and summary.get("unsafe_total"):
                    parts.append(f"balanced_acc={_fmt_metric(summary.get('balanced_accuracy'))}")
                print(" ".join(parts))
    finally:
        for _, model in model_cache.values():
            model.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
