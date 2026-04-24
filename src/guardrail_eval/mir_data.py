from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
TEXT_SUFFIXES = {".story", ".txt", ".text", ".md"}


@dataclass(frozen=True)
class MIRInputPair:
    sample_id: str
    image_path: Path
    text_path: Path
    text: str


def read_story_text(path: str | Path) -> tuple[str, list[str]]:
    content = Path(path).read_text(encoding="utf-8")
    parts = content.split("@highlight")
    story = parts[0].strip()
    highlights = [part.strip() for part in parts[1:] if part.strip()]
    return story, highlights


def read_text_payload(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() == ".story":
        story, _ = read_story_text(file_path)
        return story
    return file_path.read_text(encoding="utf-8").strip()


def list_image_files(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Image data path does not exist: {base}")
    files = sorted(
        path for path in base.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No image files found under {base}")
    return files


def list_text_files(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Text data path does not exist: {base}")
    files = sorted(
        path for path in base.iterdir()
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No text files found under {base}")
    return files


def build_mir_input_pairs(
    image_root: str | Path,
    text_root: str | Path,
    *,
    eval_num: int,
    shuffle: bool = False,
    seed: int = 0,
) -> list[MIRInputPair]:
    if eval_num <= 0:
        raise ValueError("eval_num must be a positive integer")

    image_files = list_image_files(image_root)
    text_files = list_text_files(text_root)
    if len(image_files) < eval_num:
        raise ValueError(f"Requested eval_num={eval_num}, but only found {len(image_files)} images")
    if len(text_files) < eval_num:
        raise ValueError(f"Requested eval_num={eval_num}, but only found {len(text_files)} text files")

    if shuffle:
        rng = random.Random(seed)
        image_files = image_files[:]
        text_files = text_files[:]
        rng.shuffle(image_files)
        rng.shuffle(text_files)

    pairs: list[MIRInputPair] = []
    for idx in range(eval_num):
        image_path = image_files[idx]
        text_path = text_files[idx]
        pairs.append(
            MIRInputPair(
                sample_id=f"mir_{idx:04d}",
                image_path=image_path,
                text_path=text_path,
                text=read_text_payload(text_path),
            )
        )
    return pairs
