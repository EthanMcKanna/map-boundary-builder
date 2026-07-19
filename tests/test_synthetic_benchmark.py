from pathlib import Path

import numpy as np
from PIL import Image

import map_boundary_builder.synthetic_benchmark as synthetic_benchmark
from map_boundary_builder.extract import ExtractionResult
from map_boundary_builder.synthetic import SyntheticSceneConfig, generate_synthetic_dataset


def test_score_synthetic_manifest_reports_raw_mask_metrics(monkeypatch, tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=2, seed=11, width=180, height=120)

    def fake_extract(image_path, **_kwargs):
        sample = next(
            sample for sample in manifest.samples if str(image_path).endswith(sample.artifacts.screenshot)
        )
        mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
        return ExtractionResult(
            mask=mask,
            style="synthetic-oracle",
            pixel_geometry=sample_polygon(),
            coverage_ratio=float(mask.mean()),
            contour_count=1,
            confidence=1.0,
        )

    monkeypatch.setattr(synthetic_benchmark, "extract_service_area", fake_extract)

    report = synthetic_benchmark.score_synthetic_manifest(manifest, tmp_path)

    assert report["summary"]["sample_count"] == 2
    assert report["summary"]["failure_count"] == 0
    assert report["summary"]["mean_iou"] == 1.0
    assert report["rows"][0]["metrics"]["boundary_iou_2px"] == 1.0
    assert report["rows"][0]["geometry"]["is_valid"] is True


def test_score_synthetic_manifest_records_extraction_failures(monkeypatch, tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=1, width=120, height=90)

    def fake_extract(*_args, **_kwargs):
        raise RuntimeError("synthetic extraction failed")

    monkeypatch.setattr(synthetic_benchmark, "extract_service_area", fake_extract)

    report = synthetic_benchmark.score_synthetic_manifest(manifest, tmp_path)

    assert report["summary"]["failure_count"] == 1
    assert report["summary"]["scored_count"] == 0
    assert report["rows"][0]["status"] == "failed"
    assert "synthetic extraction failed" in report["rows"][0]["error"]


def test_cli_can_generate_and_score_with_lenient_thresholds(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_extract(image_path, **_kwargs):
        mask_path = Path(str(image_path)).with_name("mask.png")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        return ExtractionResult(
            mask=mask,
            style="synthetic-oracle",
            pixel_geometry=sample_polygon(),
            coverage_ratio=float(mask.mean()),
            contour_count=1,
            confidence=1.0,
        )

    monkeypatch.setattr(synthetic_benchmark, "extract_service_area", fake_extract)

    exit_code = synthetic_benchmark.main(
        [
            "--dataset-dir",
            str(tmp_path),
            "--generate",
            "--count",
            "2",
            "--width",
            "120",
            "--height",
            "90",
            "--mean-iou",
            "0.99",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthetic-benchmark-report.json").exists()
    assert "mean_iou" in capsys.readouterr().out


def test_synthetic_guidance_uses_mask_seed_and_overlay_color(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=23, width=120, height=90)
    sample = manifest.samples[0]
    mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)

    hints = synthetic_benchmark.synthetic_guidance(sample, mask)

    assert hints["seed_point"] is not None
    assert mask[round(hints["seed_point"][1]), round(hints["seed_point"][0])]
    assert len(hints["target_rgb"]) == 3


def sample_polygon():
    from shapely.geometry import Polygon

    return Polygon([(10, 10), (60, 10), (60, 50), (10, 50)])
