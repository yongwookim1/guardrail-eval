from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm

from .benchmarks.base import Benchmark
from .io import JsonlWriter, write_json
from .metrics import summarize, verdict_to_record
from .models.base import GuardrailModel
from .types import Sample


def _batched(it: Iterable[Sample], size: int) -> Iterator[list[Sample]]:
    """Yield consecutive batches of up to `size` items from a sample iterator."""
    buf: list[Sample] = []
    for s in it:
        buf.append(s)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


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

    total = benchmark.num_samples(limit=limit)
    n_batches = math.ceil(total / batch_size) if total else None

    records: list[dict[str, Any]] = []
    with JsonlWriter(out / "raw.jsonl") as writer:
        sample_iter = benchmark.iter_samples(limit=limit)
        for batch in tqdm(
            _batched(sample_iter, batch_size),
            total=n_batches,
            desc=f"{model.name}/{benchmark.name}",
            unit="batch",
        ):
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
            # Drop image references for this batch so Python can GC them
            # before the next batch is loaded.
            for s in batch:
                s.image = None

    summary = {
        "model": model.name,
        "benchmark": benchmark.name,
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
        **summarize(records),
    }
    write_json(out / "summary.json", summary)
    write_json(out / "config.json", {"model": model_config, "benchmark": benchmark_config})
    return summary
