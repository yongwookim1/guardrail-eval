from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .benchmarks.base import Benchmark
from .io import JsonlWriter, load_json, load_jsonl, write_json
from .metrics import MetricsAccumulator, verdict_to_record
from .models.base import GuardrailModel
from .types import Sample


def _take_batch(it: Iterator[Sample], size: int) -> list[Sample]:
    batch: list[Sample] = []
    for _ in range(size):
        try:
            batch.append(next(it))
        except StopIteration:
            break
    return batch


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def _clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _classify_with_backoff(model: GuardrailModel, batch: list[Sample]) -> tuple[list[Any], int]:
    try:
        return model.classify_batch(batch), len(batch)
    except RuntimeError as exc:
        if not _is_oom_error(exc) or len(batch) == 1:
            raise
        _clear_cuda_cache()
        split = max(1, len(batch) // 2)
        left, left_size = _classify_with_backoff(model, batch[:split])
        right, right_size = _classify_with_backoff(model, batch[split:])
        return left + right, min(left_size, right_size)


def run(
    model: GuardrailModel,
    benchmark: Benchmark,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    batch_size: int = 8,
    resume: bool = False,
    skip_existing: bool = False,
    model_config: dict[str, Any] | None = None,
    benchmark_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir) / model.name / benchmark.name
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "results_summary.json"
    raw_path = out / "results.json"

    if skip_existing and summary_path.exists():
        return load_json(summary_path)

    acc = MetricsAccumulator()
    existing_ids: set[str] = set()
    if resume and raw_path.exists():
        existing_records = load_jsonl(raw_path)
        acc.update_many(existing_records)
        existing_ids = {str(r["sample_id"]) for r in existing_records}

    total = benchmark.num_samples(limit=limit)
    remaining_total = max(total - len(existing_ids), 0) if total is not None else None
    target_batch_size = max(batch_size, 1)

    sample_iter = benchmark.iter_samples(limit=limit)
    if existing_ids:
        sample_iter = (sample for sample in sample_iter if sample.id not in existing_ids)

    with JsonlWriter(raw_path, mode="a" if existing_ids else "w") as writer, tqdm(
        total=remaining_total,
        desc=f"{model.name}/{benchmark.name}",
        unit="sample",
    ) as pbar:
        sample_iter = iter(sample_iter)
        while True:
            batch = _take_batch(sample_iter, target_batch_size)
            if not batch:
                break
            verdicts, stable_batch_size = _classify_with_backoff(model, batch)
            target_batch_size = min(target_batch_size, stable_batch_size)
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
                )
                acc.update(rec)
                batch_records.append(rec)

            writer.write_many(batch_records, flush=True)

            for s in batch:
                s.image = None
                s.image_data_uri = None
            pbar.update(len(batch))

    summary = {
        "model": model.name,
        "benchmark": benchmark.name,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "resumed": resume,
        "resumed_records": len(existing_ids),
        **acc.summary(),
    }
    write_json(summary_path, summary)
    write_json(out / "config.json", {"model": model_config, "benchmark": benchmark_config})
    return summary
