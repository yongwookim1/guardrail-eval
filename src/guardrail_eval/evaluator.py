from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .benchmarks.base import Benchmark
from .io import JsonlWriter, write_json
from .metrics import summarize, verdict_to_record
from .models.base import GuardrailModel
from .types import Sample


def _chunks(items: list[Sample], size: int) -> list[list[Sample]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run(
    model: GuardrailModel,
    benchmark: Benchmark,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    batch_size: int = 8,
    model_config: dict[str, Any] | None = None,
    benchmark_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir) / model.name / benchmark.name
    out.mkdir(parents=True, exist_ok=True)

    samples = list(benchmark.iter_samples(limit=limit))
    records: list[dict[str, Any]] = []

    with JsonlWriter(out / "raw.jsonl") as writer:
        for batch in tqdm(_chunks(samples, batch_size), desc=f"{model.name}/{benchmark.name}"):
            verdicts = model.classify_batch(batch)
            for sample, verdict in zip(batch, verdicts):
                rec = verdict_to_record(
                    sample_id=sample.id,
                    expected=sample.expected_label,
                    expected_category=sample.category,
                    verdict=verdict,
                )
                records.append(rec)
                writer.write(rec)

    summary = {
        "model": model.name,
        "benchmark": benchmark.name,
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
        **summarize(records),
    }
    write_json(out / "summary.json", summary)

    write_json(
        out / "config.json",
        {"model": model_config, "benchmark": benchmark_config},
    )

    return summary
