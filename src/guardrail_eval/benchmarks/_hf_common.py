"""Shared HF-datasets loader used by both SIUO and VLSBench.

Both benchmarks are PIL-image + text-prompt safety datasets. They differ only
in column names; the YAML `columns` mapping absorbs that difference.
"""
from __future__ import annotations

from typing import Any, Iterator

from PIL.Image import Image

from ..types import Sample


def iter_hf_samples(
    hf_id: str,
    split: str,
    columns: dict[str, str],
    expected_label: str,
    name: str,
    limit: int | None = None,
) -> Iterator[Sample]:
    from datasets import load_dataset

    ds = load_dataset(hf_id, split=split)

    id_col = columns.get("id", "id")
    text_col = columns.get("text", "question")
    image_col = columns.get("image", "image")
    category_col = columns.get("category", "category")

    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break

        sample_id = str(row.get(id_col, idx)) if id_col in row else str(idx)
        text = row.get(text_col)
        image = row.get(image_col) if image_col in row else None
        category = row.get(category_col) if category_col in row else None

        if image is not None and not isinstance(image, Image):
            image = _coerce_to_pil(image)

        yield Sample(
            id=f"{name}_{sample_id}",
            text=text,
            image=image,
            expected_label=expected_label,  # type: ignore[arg-type]
            category=str(category) if category is not None else None,
            meta={"dataset": hf_id, "split": split},
        )


def _coerce_to_pil(obj: Any) -> Image | None:
    """Best-effort conversion of common image encodings (bytes, path, dict) to PIL."""
    from io import BytesIO
    from PIL import Image as PILImage

    if isinstance(obj, PILImage.Image):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return PILImage.open(BytesIO(obj))
    if isinstance(obj, str):
        return PILImage.open(obj)
    if isinstance(obj, dict):
        if "bytes" in obj and obj["bytes"]:
            return PILImage.open(BytesIO(obj["bytes"]))
        if "path" in obj and obj["path"]:
            return PILImage.open(obj["path"])
    return None
