#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_eval.analysis.permutation_bias import summarize_permutation_bias  # noqa: E402
from guardrail_eval.io import iter_records  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_results_path(args: argparse.Namespace) -> Path:
    if args.results:
        return Path(args.results)
    if not args.model or not args.benchmark:
        raise ValueError("Either --results or both --model and --benchmark are required")
    return Path(args.output_dir) / args.model / args.benchmark / "results.jsonl"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "na"
    return f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check permutation consistency for repeated multiple-choice runs.")
    parser.add_argument("--results", help="Path to a results.jsonl file.")
    parser.add_argument("--model", help="Model output directory name under results/.")
    parser.add_argument("--benchmark", help="Benchmark directory name under results/.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args(argv)

    results_path = _resolve_results_path(args)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    records = list(iter_records(results_path))
    summary = summarize_permutation_bias(records, max_examples=max(args.max_examples, 0))

    print(f"results={results_path}")
    print(f"questions_with_multiple_passes={summary['questions_with_multiple_passes']}")
    print(f"questions_with_complete_passes={summary['questions_with_complete_passes']}")
    print(f"questions_incomplete={summary['questions_incomplete']}")
    print(f"questions_inconsistent={summary['questions_inconsistent']}")
    print(f"inconsistency_rate={_fmt_ratio(summary['inconsistency_rate'])}")
    print(f"questions_invalid={summary['questions_invalid']}")
    print(f"invalid_rate={_fmt_ratio(summary['invalid_rate'])}")

    examples = summary["examples"]
    if examples:
        print("examples:")
        for example in examples:
            print(f"  base_sample_id={example['base_sample_id']} type={example['type']}")
            for row in example["passes"]:
                print(
                    "    "
                    f"pass={row['pass_index']} pred={row['pred_choice']} "
                    f"semantic={row['semantic_choice']} error={row['error_reason']}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
