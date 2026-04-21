from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmarks.registry import load_benchmark
from .evaluator import run
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guardrail-eval", description="Evaluate guardrail models on multimodal safety benchmarks.")
    parser.add_argument("--model", required=True, help='Model config name or path, or "all".')
    parser.add_argument("--benchmark", required=True, help='Benchmark config name or path, or "all".')
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume from existing results.jsonl if present.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have a results_summary.json.")
    args = parser.parse_args(argv)

    model_names = _names("models") if args.model == "all" else [args.model]
    benchmark_names = _names("benchmarks") if args.benchmark == "all" else [args.benchmark]

    for m in model_names:
        model_cfg_path = _resolve_config("models", m)
        model_cfg = load_yaml(model_cfg_path)
        model = load_model(str(model_cfg_path))
        try:
            for b in benchmark_names:
                bench_cfg_path = _resolve_config("benchmarks", b)
                bench_cfg = load_yaml(bench_cfg_path)
                benchmark = load_benchmark(str(bench_cfg_path))
                summary = run(
                    model,
                    benchmark,
                    output_dir=args.output_dir,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    resume=args.resume,
                    skip_existing=args.skip_existing,
                    model_config=model_cfg,
                    benchmark_config=bench_cfg,
                )
                print(
                    f"[{model.name}/{benchmark.name}] "
                    f"n={summary['n']} "
                    f"tp={summary['true_positives']} "
                    f"fn={summary['false_negatives']} "
                    f"errors={summary['errors']} "
                    f"unsafe_recall={summary['unsafe_recall']}"
                )
        finally:
            model.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
