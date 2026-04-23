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


def _group_records_by_base_sample(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        meta = record.get("meta")
        if not isinstance(meta, dict):
            continue
        base_sample_id = meta.get("base_sample_id")
        if base_sample_id is None:
            continue
        groups[str(base_sample_id)].append(record)
    return groups


def _sorted_group(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(group, key=lambda rec: int(rec.get("meta", {}).get("pass_index", 0)))


def _is_complete_group(group: list[dict[str, Any]]) -> bool:
    if not group:
        return False
    expected_passes_raw = group[0].get("meta", {}).get("num_passes")
    try:
        expected_passes = int(expected_passes_raw)
    except (TypeError, ValueError):
        return False
    return len(group) == expected_passes


def summarize_question_level_choice(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _group_records_by_base_sample(records)
    if not groups:
        return {}

    total = len(groups)
    complete = 0
    incomplete = 0
    correct = 0
    invalid = 0

    for group in groups.values():
        sorted_group = _sorted_group(group)
        if not _is_complete_group(sorted_group):
            incomplete += 1
            continue

        complete += 1
        has_invalid = False
        all_correct = True
        for record in sorted_group:
            semantic_choice, semantic_idx = _semantic_choice(record)
            if semantic_choice is None or semantic_idx is None or record.get("error_reason"):
                has_invalid = True
                all_correct = False
                break
            if not record.get("correct"):
                all_correct = False
        if has_invalid:
            invalid += 1
        if all_correct:
            correct += 1

    return {
        "questions_total": total,
        "questions_complete": complete,
        "questions_incomplete": incomplete,
        "questions_correct": correct,
        "question_accuracy": (correct / complete) if complete else None,
        "questions_invalid": invalid,
    }


def summarize_permutation_bias(
    records: list[dict[str, Any]],
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    groups = _group_records_by_base_sample(records)

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
        sorted_group = _sorted_group(group)
        is_complete = _is_complete_group(sorted_group)
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
