from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .benchmarks.base import Benchmark
from .io import JsonlWriter, iter_records, load_json, write_json
from .metrics import MetricsAccumulator, verdict_to_record
from .models.base import GuardrailModel
from .types import Sample

RAW_RESULTS_FILENAME = "results.jsonl"
LEGACY_RAW_RESULTS_FILENAME = "results.json"
SUMMARY_FILENAME = "results_summary.json"


def _take_batch(it: Iterator[Sample], size: int) -> list[Sample]:
    batch: list[Sample] = []
    for _ in range(size):
        try:
            batch.append(next(it))
        except StopIteration:
            break
    return batch


def run(
    model: GuardrailModel,
    benchmark: Benchmark,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    batch_size: int = 8,
    resume: bool = False,
    skip_existing: bool = False,
    flush_every_batches: int = 16,
    model_config: dict[str, Any] | None = None,
    benchmark_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir) / model.name / benchmark.name
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / SUMMARY_FILENAME
    raw_path = out / RAW_RESULTS_FILENAME
    legacy_raw_path = out / LEGACY_RAW_RESULTS_FILENAME

    if skip_existing and summary_path.exists():
        return load_json(summary_path)

    acc = MetricsAccumulator()
    existing_ids: set[str] = set()
    if resume:
        source_path = raw_path if raw_path.exists() else legacy_raw_path
        if source_path.exists():
            rewrite_legacy_results = source_path == legacy_raw_path and not raw_path.exists()
            if rewrite_legacy_results:
                with JsonlWriter(raw_path, mode="w") as writer:
                    for record in iter_records(source_path):
                        acc.update(record)
                        existing_ids.add(str(record["sample_id"]))
                        writer.write(record)
                    writer.flush()
            else:
                for record in iter_records(source_path):
                    acc.update(record)
                    existing_ids.add(str(record["sample_id"]))

    total = benchmark.num_samples(limit=limit)
    remaining_total = max(total - len(existing_ids), 0) if total is not None else None
    batch_size = max(batch_size, 1)
    flush_every_batches = max(flush_every_batches, 1)

    sample_iter = benchmark.iter_samples(limit=limit)
    if existing_ids:
        sample_iter = (sample for sample in sample_iter if sample.id not in existing_ids)

    with JsonlWriter(raw_path, mode="a" if existing_ids else "w") as writer, tqdm(
        total=remaining_total,
        desc=f"{model.name}/{benchmark.name}",
        unit="sample",
    ) as pbar:
        sample_iter = iter(sample_iter)
        pending_flush_batches = 0
        while True:
            batch = _take_batch(sample_iter, batch_size)
            if not batch:
                break
            verdicts = model.classify_batch(batch)
            if len(verdicts) != len(batch):
                raise ValueError(
                    f"{model.name} returned {len(verdicts)} verdicts for batch of size {len(batch)}"
                )
            batch_records: list[dict[str, Any]] = []
            for sample, verdict in zip(batch, verdicts):
                rec = verdict_to_record(
                    sample_id=sample.id,
                    expected=sample.expected_label,
                    expected_category=sample.category,
                    verdict=verdict,
                    expected_type=sample.meta.get("type"),
                )
                acc.update(rec)
                batch_records.append(rec)

            writer.write_many(batch_records, flush=False)
            pending_flush_batches += 1
            if pending_flush_batches >= flush_every_batches:
                writer.flush()
                pending_flush_batches = 0
            pbar.update(len(batch))
        if pending_flush_batches:
            writer.flush()

    summary = {
        "model": model.name,
        "benchmark": benchmark.name,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        **acc.summary(),
    }
    write_json(summary_path, summary)
    write_json(out / "config.json", {"model": model_config, "benchmark": benchmark_config})
    return summary
