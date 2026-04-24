from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .io import load_yaml
from .mir import run_mir_evaluation, write_mir_artifacts

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def _resolve_model_config(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    candidate = CONFIGS_DIR / "models" / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No model config found for {name_or_path!r}")


def _public_model_names() -> list[str]:
    names: set[str] = set()
    for config_path in (CONFIGS_DIR / "models").glob("*.yaml"):
        if config_path.stem.endswith("_choice"):
            names.add(config_path.stem[:-7])
        else:
            names.add(config_path.stem)
    return sorted(names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="guardrail-mir",
        description="Compute MIR for supported multimodal models in this repo.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help='One or more model config names/paths, or "all".',
    )
    parser.add_argument("--image-data-path", default=str(REPO_ROOT / "datasets" / "mir" / "images"))
    parser.add_argument("--text-data-path", default=str(REPO_ROOT / "datasets" / "mir" / "texts"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "mir"))
    parser.add_argument("--eval-num", type=int, default=100)
    parser.add_argument("--mode", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-device", default="auto", help='Metric device: "auto", "cpu", or e.g. "cuda".')
    parser.add_argument("--debug-spans", type=int, default=5, help="Write span-debug rows for the first N samples.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    requested_models = _public_model_names() if "all" in args.model else args.model
    exit_code = 0

    for requested_model in requested_models:
        model_cfg_path = _resolve_model_config(requested_model)
        model_cfg = load_yaml(model_cfg_path)
        model_name = str(model_cfg["name"])
        output_dir = Path(args.output_dir) / model_name
        summary_path = output_dir / "results_summary.json"

        if args.skip_existing and summary_path.exists():
            print(f"[skip {model_name}] {summary_path} already exists")
            continue

        try:
            artifacts = run_mir_evaluation(
                model_config=model_cfg,
                image_data_path=args.image_data_path,
                text_data_path=args.text_data_path,
                eval_num=args.eval_num,
                mode=args.mode,
                shuffle=args.shuffle,
                seed=args.seed,
                metric_device=args.metric_device,
                debug_limit=args.debug_spans,
            )
        except Exception as exc:
            exit_code = 1
            print(f"[error {requested_model}] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        write_mir_artifacts(output_dir, artifacts)
        print(
            f"[{model_name}] "
            f"overall_mir={artifacts.summary['overall_mir']:.6f} "
            f"layers={artifacts.summary['scored_layers']} "
            f"n={artifacts.summary['eval_num']} "
            f"vision_tokens_mean={artifacts.summary['vision_token_count_mean']:.2f} "
            f"text_tokens_mean={artifacts.summary['text_token_count_mean']:.2f}"
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
