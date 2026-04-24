from __future__ import annotations

from pathlib import Path

import torch

from guardrail_eval.mir import (
    compute_layer_mir,
    replace_outliers_with_median_l2,
    run_mir_evaluation,
)
from guardrail_eval.mir_backends import MIRSampleLayers
from guardrail_eval.mir_data import MIRInputPair, build_mir_input_pairs, read_story_text


def test_read_story_text_and_build_pairs(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    texts_dir = tmp_path / "texts"
    images_dir.mkdir()
    texts_dir.mkdir()

    (images_dir / "b.jpg").write_bytes(b"fake-jpg")
    (images_dir / "a.png").write_bytes(b"fake-png")
    (texts_dir / "b.story").write_text("Story body\n\n@highlight\nPoint one\n", encoding="utf-8")
    (texts_dir / "a.txt").write_text("Plain text payload", encoding="utf-8")

    story, highlights = read_story_text(texts_dir / "b.story")
    assert story == "Story body"
    assert highlights == ["Point one"]

    pairs = build_mir_input_pairs(images_dir, texts_dir, eval_num=2, shuffle=False, seed=0)
    assert [pair.image_path.name for pair in pairs] == ["a.png", "b.jpg"]
    assert [pair.text_path.name for pair in pairs] == ["a.txt", "b.story"]
    assert pairs[0].text == "Plain text payload"
    assert pairs[1].text == "Story body"


def test_replace_outliers_with_median_l2() -> None:
    rows = [[1.0, 1.0] for _ in range(12)]
    rows.append([100.0, 100.0])
    data = torch.tensor(rows)
    cleaned = replace_outliers_with_median_l2(data)
    assert torch.allclose(cleaned[-1], torch.tensor([1.0, 1.0]))


def test_compute_layer_mir_fast_cpu_returns_non_negative() -> None:
    vision = torch.tensor(
        [
            [1.0, 0.0],
            [0.1, 0.9],
            [0.9, 0.2],
            [0.2, 1.0],
        ]
    )
    text = torch.tensor(
        [
            [0.8, 0.2],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.3, 0.7],
        ]
    )
    score = compute_layer_mir(vision, text, mode="fast", metric_device="cpu")
    assert score >= 0.0


class _FakeMIRBackend:
    def __init__(self) -> None:
        self.device = "cpu"
        self.closed = False

    def collect_mir_sample(self, *, sample_id: str, image_path: str, text: str) -> MIRSampleLayers:
        del sample_id, image_path, text
        return MIRSampleLayers(
            vision_layers=[
                torch.zeros((2, 2)),
                torch.tensor([[1.0, 0.0], [0.1, 0.9]]),
                torch.tensor([[2.0, 0.1], [0.2, 1.8]]),
            ],
            text_layers=[
                torch.zeros((2, 2)),
                torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
                torch.tensor([[1.7, 0.3], [0.3, 1.7]]),
            ],
            debug={
                "sample_id": "fake",
                "image_path": "img.png",
                "prompt_token_count": 20,
                "vision_token_count": 2,
                "text_token_count": 2,
                "first_vision_position": 4,
                "last_vision_position": 5,
                "first_text_position": 6,
                "last_text_position": 7,
                "input_ids_length": 20,
                "input_keys": ["attention_mask", "input_ids"],
            },
        )

    def close(self) -> None:
        self.closed = True


def test_run_mir_evaluation_with_fake_backend(monkeypatch, tmp_path: Path) -> None:
    pairs = [
        MIRInputPair(
            sample_id="mir_0000",
            image_path=tmp_path / "0.png",
            text_path=tmp_path / "0.story",
            text="alpha",
        ),
        MIRInputPair(
            sample_id="mir_0001",
            image_path=tmp_path / "1.png",
            text_path=tmp_path / "1.story",
            text="beta",
        ),
    ]
    fake_backend = _FakeMIRBackend()

    monkeypatch.setattr("guardrail_eval.mir.build_mir_input_pairs", lambda *args, **kwargs: pairs)
    monkeypatch.setattr(
        "guardrail_eval.mir.build_mir_backend",
        lambda model_config: ("gemma3", fake_backend),
    )

    artifacts = run_mir_evaluation(
        model_config={"name": "gemma_3_4b_it"},
        image_data_path=tmp_path / "images",
        text_data_path=tmp_path / "texts",
        eval_num=2,
        mode="fast",
        shuffle=False,
        seed=0,
        metric_device="cpu",
        debug_limit=1,
    )

    assert fake_backend.closed is True
    assert artifacts.summary["model_name"] == "gemma_3_4b_it"
    assert artifacts.summary["eval_num"] == 2
    assert artifacts.summary["scored_layers"] == 2
    assert len(artifacts.per_layer) == 2
    assert len(artifacts.sample_pairs) == 2
    assert len(artifacts.debug_records) == 1
