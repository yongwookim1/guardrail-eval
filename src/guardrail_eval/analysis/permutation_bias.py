from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..types import CHOICE_LETTERS


def _semantic_choice(record: dict[str, Any]) -> tuple[str | None, int | None]:
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return None, None
    base_options = meta.get("base_options")
    if not isinstance(base_options, list) or not base_options:
        return None, None

    raw_rotation = meta.get("rotation", 0)
    try:
        rotation = int(raw_rotation)
    except (TypeError, ValueError):
        return None, None

    pred_choice = str(record.get("pred_choice") or "").strip().upper()
    if pred_choice not in CHOICE_LETTERS[: len(base_options)]:
        return None, None

    pred_idx = CHOICE_LETTERS.index(pred_choice)
    semantic_idx = (rotation + pred_idx) % len(base_options)
    return str(base_options[semantic_idx]), semantic_idx


def summarize_permutation_bias(
    records: list[dict[str, Any]],
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        meta = record.get("meta")
        if not isinstance(meta, dict):
            continue
        base_sample_id = meta.get("base_sample_id")
        if base_sample_id is None:
            continue
        groups[str(base_sample_id)].append(record)

    analyzed = 0
    complete = 0
    incomplete = 0
    inconsistent = 0
    invalid = 0
    examples: list[dict[str, Any]] = []

    for base_sample_id, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        analyzed += 1
        sorted_group = sorted(group, key=lambda rec: int(rec.get("meta", {}).get("pass_index", 0)))

        expected_passes_raw = sorted_group[0].get("meta", {}).get("num_passes")
        try:
            expected_passes = int(expected_passes_raw)
        except (TypeError, ValueError):
            expected_passes = None
        is_complete = expected_passes is not None and len(sorted_group) == expected_passes
        if is_complete:
            complete += 1
        else:
            incomplete += 1

        semantic_predictions: list[tuple[int, str]] = []
        pass_rows: list[dict[str, Any]] = []
        has_invalid = False
        for record in sorted_group:
            semantic_choice, semantic_idx = _semantic_choice(record)
            if semantic_choice is None or semantic_idx is None:
                has_invalid = True
                pass_rows.append(
                    {
                        "pass_index": record.get("meta", {}).get("pass_index"),
                        "pred_choice": record.get("pred_choice"),
                        "semantic_choice": None,
                        "semantic_index": None,
                        "error_reason": record.get("error_reason"),
                    }
                )
                continue
            semantic_predictions.append((semantic_idx, semantic_choice))
            pass_rows.append(
                {
                    "pass_index": record.get("meta", {}).get("pass_index"),
                    "pred_choice": record.get("pred_choice"),
                    "semantic_choice": semantic_choice,
                    "semantic_index": semantic_idx,
                    "error_reason": record.get("error_reason"),
                }
            )

        if not is_complete:
            if len(examples) < max_examples:
                examples.append(
                    {
                        "base_sample_id": base_sample_id,
                        "type": "incomplete",
                        "passes": pass_rows,
                    }
                )
            continue

        if has_invalid:
            invalid += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "base_sample_id": base_sample_id,
                        "type": "invalid",
                        "passes": pass_rows,
                    }
                )
            continue

        if len(set(semantic_predictions)) > 1:
            inconsistent += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "base_sample_id": base_sample_id,
                        "type": "inconsistent",
                        "passes": pass_rows,
                    }
                )

    inconsistency_rate = (inconsistent / complete) if complete else None
    invalid_rate = (invalid / complete) if complete else None
    return {
        "questions_with_multiple_passes": analyzed,
        "questions_with_complete_passes": complete,
        "questions_incomplete": incomplete,
        "questions_inconsistent": inconsistent,
        "inconsistency_rate": inconsistency_rate,
        "questions_invalid": invalid,
        "invalid_rate": invalid_rate,
        "examples": examples,
    }
