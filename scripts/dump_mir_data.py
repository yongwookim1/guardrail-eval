#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump MIR input pools from locally cloned Hugging Face dataset repos "
            "into mir_data/images and mir_data/texts."
        )
    )
    parser.add_argument(
        "--textvqa-root",
        type=Path,
        required=True,
        help="Local clone root for the TextVQA dataset repo.",
    )
    parser.add_argument(
        "--cnndm-root",
        type=Path,
        required=True,
        help="Local clone root for the CNN/DailyMail dataset repo.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root that will receive images/, texts/, and manifest.json.",
    )
    parser.add_argument(
        "--textvqa-split",
        default="validation",
        help="TextVQA split to dump. Default: validation.",
    )
    parser.add_argument(
        "--cnndm-version",
        default="3.0.0",
        help="CNN/DailyMail dataset version directory. Default: 3.0.0.",
    )
    parser.add_argument(
        "--cnndm-split",
        default="validation",
        help="CNN/DailyMail split to dump. Default: validation.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of unique images to export.",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Optional cap on the number of text files to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_args()


def find_parquet_shards(root: Path, split: str, version: str | None = None) -> list[Path]:
    if version:
        search_roots = [
            root / version,
            root / version / "data",
            root / version / "default" / split,
            root / version / split,
        ]
    else:
        search_roots = [
            root,
            root / "data",
            root / "default" / split,
            root / split,
        ]

    matches: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in search_root.rglob("*.parquet"):
            if candidate.name.startswith(f"{split}-") or candidate.parent.name == split:
                matches.add(candidate)

    shards = sorted(matches)
    if not shards:
        raise FileNotFoundError(
            f"Could not find parquet shards for split={split!r} under {root}"
            + (f" (version={version})" if version else "")
        )
    return shards


def load_local_parquet_dataset(parquet_paths: Iterable[Path]):
    from datasets import load_dataset

    data_files = [str(path) for path in parquet_paths]
    return load_dataset("parquet", data_files=data_files, split="train")


def infer_suffix_from_bytes(raw_bytes: bytes) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(raw_bytes)) as image:
        image_format = (image.format or "PNG").upper()
    return {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "GIF": ".gif",
        "TIFF": ".tiff",
    }.get(image_format, ".png")


def image_payload_to_bytes(image_value) -> tuple[bytes, str]:
    if isinstance(image_value, dict):
        path_value = image_value.get("path")
        suffix = Path(path_value).suffix.lower() if path_value else ""
        raw_bytes = image_value.get("bytes")
        if raw_bytes is not None:
            return raw_bytes, suffix or infer_suffix_from_bytes(raw_bytes)
        if path_value:
            raw_bytes = Path(path_value).read_bytes()
            return raw_bytes, suffix or infer_suffix_from_bytes(raw_bytes)

    if hasattr(image_value, "save"):
        buffer = io.BytesIO()
        image_format = getattr(image_value, "format", None) or "PNG"
        image_value.save(buffer, format=image_format)
        raw_bytes = buffer.getvalue()
        return raw_bytes, infer_suffix_from_bytes(raw_bytes)

    raise TypeError(f"Unsupported image payload type: {type(image_value)!r}")


def render_story(article: str, highlights: str) -> str:
    cleaned_article = article.strip()
    cleaned_highlights = [line.strip() for line in highlights.splitlines() if line.strip()]

    parts = [cleaned_article] if cleaned_article else []
    if cleaned_highlights:
        if parts:
            parts.append("")
        for highlight in cleaned_highlights:
            parts.append("@highlight")
            parts.append(highlight)
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def export_textvqa_images(
    dataset,
    output_dir: Path,
    *,
    max_items: int | None,
    overwrite: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_image_ids: set[str] = set()
    written = 0
    skipped_existing = 0

    for row_index, row in enumerate(dataset):
        image_id = str(row.get("image_id") or f"textvqa_{row_index:08d}")
        if image_id in seen_image_ids:
            continue
        if max_items is not None and len(seen_image_ids) >= max_items:
            break
        seen_image_ids.add(image_id)

        existing_matches = list(output_dir.glob(f"{image_id}.*"))
        if existing_matches and not overwrite:
            skipped_existing += 1
            continue

        image_bytes, suffix = image_payload_to_bytes(row["image"])
        output_path = output_dir / f"{image_id}{suffix}"
        if overwrite:
            for existing_match in existing_matches:
                if existing_match != output_path and existing_match.exists():
                    existing_match.unlink()
        output_path.write_bytes(image_bytes)
        written += 1

    return {
        "written": written,
        "skipped_existing": skipped_existing,
        "unique_rows_seen": len(seen_image_ids),
    }


def export_cnndm_stories(
    dataset,
    output_dir: Path,
    *,
    max_items: int | None,
    overwrite: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_existing = 0
    selected = 0

    for row_index, row in enumerate(dataset):
        if max_items is not None and selected >= max_items:
            break
        selected += 1
        story_id = str(row.get("id") or f"story_{row_index:08d}")
        output_path = output_dir / f"{story_id}.story"
        if output_path.exists() and not overwrite:
            skipped_existing += 1
        else:
            story_text = render_story(str(row["article"]), str(row["highlights"]))
            output_path.write_text(story_text, encoding="utf-8")
            written += 1

    return {
        "written": written,
        "skipped_existing": skipped_existing,
    }


def main() -> None:
    args = parse_args()

    textvqa_shards = find_parquet_shards(args.textvqa_root, args.textvqa_split)
    cnndm_shards = find_parquet_shards(
        args.cnndm_root,
        args.cnndm_split,
        version=args.cnndm_version,
    )

    textvqa_dataset = load_local_parquet_dataset(textvqa_shards)
    cnndm_dataset = load_local_parquet_dataset(cnndm_shards)

    images_dir = args.output_root / "images"
    texts_dir = args.output_root / "texts"
    args.output_root.mkdir(parents=True, exist_ok=True)

    image_stats = export_textvqa_images(
        textvqa_dataset,
        images_dir,
        max_items=args.max_images,
        overwrite=args.overwrite,
    )
    text_stats = export_cnndm_stories(
        cnndm_dataset,
        texts_dir,
        max_items=args.max_texts,
        overwrite=args.overwrite,
    )

    manifest = {
        "sources": {
            "textvqa": {
                "root": str(args.textvqa_root.resolve()),
                "split": args.textvqa_split,
                "parquet_shards": [str(path.resolve()) for path in textvqa_shards],
            },
            "cnn_dailymail": {
                "root": str(args.cnndm_root.resolve()),
                "version": args.cnndm_version,
                "split": args.cnndm_split,
                "parquet_shards": [str(path.resolve()) for path in cnndm_shards],
            },
        },
        "outputs": {
            "root": str(args.output_root.resolve()),
            "images_dir": str(images_dir.resolve()),
            "texts_dir": str(texts_dir.resolve()),
            "images": image_stats,
            "texts": text_stats,
        },
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
