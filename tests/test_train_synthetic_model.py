from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools import train_synthetic_model as trainer


def sample_fixture() -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    image = np.full((8, 8, 3), 0.15, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    image[2:6, 2:6] = np.asarray([0.75, 0.30, 0.10], dtype=np.float32)
    sample = SimpleNamespace(
        overlay_style=SimpleNamespace(fill_opacity=0.5, stroke_color="#ff0000"),
    )
    return image, mask, sample


def test_guidance_policy_cli_preserves_mixed_default_and_accepts_automatic_modes() -> None:
    parser = trainer.build_parser()

    assert parser.parse_args([]).guidance_policy == "mixed"
    assert parser.parse_args(["--guidance-policy", "automatic-heavy"]).guidance_policy == "automatic-heavy"
    assert parser.parse_args(["--guidance-policy", "none"]).guidance_policy == "none"


def test_automatic_heavy_validation_is_oracle_free() -> None:
    image, mask, sample = sample_fixture()

    with patch.object(trainer, "observed_overlay_color", side_effect=AssertionError("oracle color used")):
        guidance = trainer.training_guidance_channels(
            image,
            mask,
            sample,
            augment=False,
            guidance_policy="automatic-heavy",
        )

    assert np.count_nonzero(guidance) == 0
    assert trainer.validation_guidance_policy("automatic-heavy") == "none"


def test_automatic_heavy_training_drops_most_hints_but_retains_explicit_exposure() -> None:
    image, mask, sample = sample_fixture()

    with patch.object(trainer.random, "random", return_value=0.99):
        dropped = trainer.training_guidance_channels(
            image,
            mask,
            sample,
            augment=True,
            guidance_policy="automatic-heavy",
        )
    with (
        patch.object(trainer.random, "random", side_effect=[0.01, 0.01]),
        patch.object(trainer.random, "randrange", return_value=0),
    ):
        exposed = trainer.training_guidance_channels(
            image,
            mask,
            sample,
            augment=True,
            guidance_policy="automatic-heavy",
        )

    assert np.count_nonzero(dropped) == 0
    assert np.count_nonzero(exposed[0]) > 0
    assert np.count_nonzero(exposed[1]) > 0
    assert trainer.AUTOMATIC_HEAVY_GUIDANCE_EXPOSURE < 0.25


def test_none_guidance_policy_never_uses_oracle_channels() -> None:
    image, mask, sample = sample_fixture()

    with patch.object(trainer, "observed_overlay_color", side_effect=AssertionError("oracle color used")):
        guidance = trainer.training_guidance_channels(
            image,
            mask,
            sample,
            augment=True,
            guidance_policy="none",
        )

    assert np.count_nonzero(guidance) == 0


def test_mixed_validation_preserves_existing_seed_and_target_guidance() -> None:
    image, mask, sample = sample_fixture()

    guidance = trainer.training_guidance_channels(
        image,
        mask,
        sample,
        augment=False,
        guidance_policy="mixed",
    )

    assert np.count_nonzero(guidance[0]) > 0
    assert np.count_nonzero(guidance[1]) > 0


def test_tversky_loss_penalizes_false_negatives_more_than_false_positives() -> None:
    masks = torch.tensor([[[[1.0, 0.0]]]])
    false_negative_logits = torch.tensor([[[[-12.0, -12.0]]]])
    false_positive_logits = torch.tensor([[[[12.0, 12.0]]]])

    false_negative_loss = trainer.tversky_loss(false_negative_logits, masks, alpha=0.30, beta=0.70)
    false_positive_loss = trainer.tversky_loss(false_positive_logits, masks, alpha=0.30, beta=0.70)

    assert float(false_negative_loss) > float(false_positive_loss)


def test_zero_tversky_weight_recovers_legacy_segmentation_objective() -> None:
    logits = torch.tensor([[[[0.3, -0.7], [1.2, -0.1]]]])
    masks = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    actual = trainer.segmentation_loss(logits, masks, tversky_weight=0.0)
    expected = (
        0.55 * torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
        + 0.35 * trainer.dice_loss(logits, masks)
        + 0.10 * trainer.boundary_loss(logits, masks)
    )

    assert torch.allclose(actual, expected)


def test_validation_summary_and_checkpoint_metadata_include_tail_metrics() -> None:
    metrics = trainer.summarize_validation_scores(
        [0.0, 0.5, 1.0],
        [0.2, 0.6, 1.0],
    )
    checkpoint = trainer.validation_checkpoint_metadata(metrics)

    assert metrics["p05_iou"] == pytest.approx(0.05)
    assert metrics["p05_boundary_iou_2px"] == pytest.approx(0.24)
    assert metrics["tail_score"] == pytest.approx(
        0.20 * metrics["iou"]
        + 0.30 * metrics["boundary_iou_2px"]
        + 0.20 * metrics["p05_iou"]
        + 0.30 * metrics["p05_boundary_iou_2px"]
    )
    assert metrics["score"] == metrics["tail_score"]
    assert checkpoint == {
        "validation_iou": metrics["iou"],
        "validation_p05_iou": metrics["p05_iou"],
        "validation_boundary_iou_2px": metrics["boundary_iou_2px"],
        "validation_p05_boundary_iou_2px": metrics["p05_boundary_iou_2px"],
        "validation_tail_score": metrics["tail_score"],
        "validation_score": metrics["score"],
    }


def test_loss_argument_validation_rejects_invalid_tversky_configuration() -> None:
    args = trainer.build_parser().parse_args(["--tversky-weight", "1.1"])
    with pytest.raises(ValueError, match="between zero and one"):
        trainer.validate_loss_args(args)


def test_selector_release_metadata_binds_training_and_checkpoint_artifacts(
    tmp_path: Path,
) -> None:
    manifest = {
        "name": "selector-training",
        "version": "synthetic-generator-v20-centered-stroke-raster-v2",
        "properties": {},
        "samples": [],
    }
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    resume = tmp_path / "resume.pt"
    resume.write_bytes(b"resume-weights")
    selected = tmp_path / "epoch-004.pt"
    selected.write_bytes(b"selected-weights")
    output = tmp_path / "selector.onnx"
    output.write_bytes(b"selector")
    args = trainer.build_parser().parse_args(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output",
            str(output),
            "--seed",
            "404000",
            "--learning-rate",
            "0.0002",
            "--batch-size",
            "2",
            "--epochs",
            "8",
            "--validation-count",
            "256",
            "--guidance-policy",
            "automatic-heavy",
            "--resume-checkpoint",
            str(resume),
        ]
    )
    checkpoint = {
        "epoch": 4,
        "arch": "resunet",
        "base_channels": 24,
        "image_size": 320,
        "input_channels": 5,
        "guidance_policy": "automatic-heavy",
        "validation_guidance_policy": "none",
        "validation_iou": 0.986,
        "validation_p05_iou": 0.969,
        "validation_boundary_iou_2px": 0.894,
        "validation_p05_boundary_iou_2px": 0.736,
        "validation_tail_score": 0.880,
        "validation_score": 0.880,
    }

    trainer.write_selector_metadata(
        output,
        checkpoint=checkpoint,
        selected_checkpoint=selected,
        args=args,
    )

    metadata = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == trainer.SELECTOR_METADATA_SCHEMA_VERSION
    assert metadata["training_dataset"]["version"] == manifest["version"]
    assert len(metadata["training_dataset"]["manifest_sha256"]) == 64
    assert metadata["resume_artifact"]["bytes"] == len(b"resume-weights")
    assert metadata["selected_checkpoint"]["artifact"]["bytes"] == len(
        b"selected-weights"
    )
    assert metadata["selected_checkpoint"]["epoch"] == 4
    assert metadata["production"] == {
        "threshold": 0.45,
        "output_activation": "logits",
        "input_width": 320,
        "input_height": 320,
        "input_channels": 5,
    }
    assert metadata["guidance"] == {
        "training_policy": "automatic-heavy",
        "validation_policy": "none",
    }
