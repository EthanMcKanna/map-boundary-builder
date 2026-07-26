import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import shape

import map_boundary_builder.synthetic_benchmark as synthetic_benchmark
from map_boundary_builder.evaluation import rasterize_geometry_mask
from map_boundary_builder.extract import ExtractionResult
from map_boundary_builder.synthetic import generate_synthetic_dataset


def test_score_synthetic_manifest_reports_raw_mask_metrics(monkeypatch, tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=2, seed=11, width=180, height=120)
    pending_masks = [
        synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
        for sample in manifest.samples
    ]

    def fake_segment(rgb, **_kwargs):
        mask = pending_masks.pop(0)
        return ExtractionResult(
            mask=mask,
            style="synthetic-oracle",
            pixel_geometry=sample_polygon(),
            coverage_ratio=float(mask.mean()),
            contour_count=1,
            confidence=1.0,
        )

    monkeypatch.setattr(synthetic_benchmark, "segment_image", fake_segment)

    report = synthetic_benchmark.score_synthetic_manifest(manifest, tmp_path)

    assert report["summary"]["sample_count"] == 2
    assert report["summary"]["failure_count"] == 0
    assert report["summary"]["mean_iou"] == 1.0
    assert report["rows"][0]["metrics"]["boundary_iou_2px"] == 1.0
    assert report["rows"][0]["geometry"]["is_valid"] is True
    assert report["schema_version"] == synthetic_benchmark.REPORT_SCHEMA_VERSION
    assert report["metadata"]["guidance_mode"] == "automatic"
    assert report["metadata"]["guided"] is False
    assert report["metadata"]["oracle_hints"] is False


def test_score_synthetic_manifest_records_extraction_failures(monkeypatch, tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=1, width=120, height=90)

    def fake_segment(*_args, **_kwargs):
        raise RuntimeError("synthetic extraction failed")

    monkeypatch.setattr(synthetic_benchmark, "segment_image", fake_segment)

    report = synthetic_benchmark.score_synthetic_manifest(manifest, tmp_path)

    assert report["summary"]["failure_count"] == 1
    assert report["summary"]["scored_count"] == 0
    assert report["rows"][0]["status"] == "failed"
    assert "synthetic extraction failed" in report["rows"][0]["error"]


def test_cli_can_generate_and_score_with_lenient_thresholds(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_load_rgb(image_path):
        mask_path = Path(str(image_path)).with_name("mask.png")
        return np.asarray(Image.open(mask_path).convert("L"))

    def fake_segment(rgb, **_kwargs):
        mask = np.asarray(rgb) > 0
        return ExtractionResult(
            mask=mask,
            style="synthetic-oracle",
            pixel_geometry=sample_polygon(),
            coverage_ratio=float(mask.mean()),
            contour_count=1,
            confidence=1.0,
        )

    monkeypatch.setattr(synthetic_benchmark, "load_rgb", fake_load_rgb)
    monkeypatch.setattr(synthetic_benchmark, "segment_image", fake_segment)

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
    report = json.loads(
        (tmp_path / "synthetic-benchmark-report.json").read_text(encoding="utf-8")
    )
    assert len(report["metadata"]["manifest_sha256"]) == 64
    assert "mean_iou" in capsys.readouterr().out


def test_synthetic_guidance_uses_mask_seed_and_overlay_color(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=23, width=120, height=90)
    sample = manifest.samples[0]
    mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)

    hints = synthetic_benchmark.synthetic_guidance(sample, mask)

    assert hints["seed_point"] is not None
    assert mask[round(hints["seed_point"][1]), round(hints["seed_point"][0])]
    assert len(hints["target_rgb"]) == 3


def test_guided_scoring_passes_manifest_hints_to_segment_image(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=37, width=120, height=90)
    sample = manifest.samples[0]
    expected_mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
    reference_geometry = _reference_geometry(tmp_path, sample)
    observed: dict[str, object] = {}

    def fake_segment(rgb, *, model_path=None, threshold=None, hints=None, **_kwargs):
        observed["model_path"] = model_path
        observed["threshold"] = threshold
        observed["hints"] = hints
        return ExtractionResult(
            mask=expected_mask,
            style="model-mask",
            pixel_geometry=reference_geometry,
            coverage_ratio=float(expected_mask.mean()),
            contour_count=1,
            confidence=0.99,
        )

    monkeypatch.setattr(synthetic_benchmark, "segment_image", fake_segment)

    row = synthetic_benchmark.score_synthetic_sample(
        sample,
        tmp_path,
        model_path=tmp_path / "boundary.onnx",
        model_threshold=0.4,
        guided=True,
    )

    assert row["status"] == "scored"
    assert observed["model_path"] == tmp_path / "boundary.onnx"
    assert observed["threshold"] == 0.4
    hints = observed["hints"]
    assert hints is not None
    assert expected_mask[round(hints["seed_point"][1]), round(hints["seed_point"][0])]
    assert len(hints["target_rgb"]) == 3


def test_unguided_scoring_passes_no_hints(monkeypatch, tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=31, width=120, height=90)
    sample = manifest.samples[0]
    expected_mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
    observed: dict[str, object] = {"hints": "unset"}

    def fake_segment(rgb, *, hints=None, **_kwargs):
        observed["hints"] = hints
        return ExtractionResult(
            mask=expected_mask,
            style="model-mask",
            pixel_geometry=_reference_geometry(tmp_path, sample),
            coverage_ratio=float(expected_mask.mean()),
            contour_count=1,
            confidence=0.99,
        )

    monkeypatch.setattr(synthetic_benchmark, "segment_image", fake_segment)

    row = synthetic_benchmark.score_synthetic_sample(sample, tmp_path)

    assert row["status"] == "scored"
    assert observed["hints"] is None
    assert row["metrics"]["iou"] == 1.0


def test_report_extractor_and_model_artifact_follow_model_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=41, width=120, height=90)
    sample = manifest.samples[0]
    expected_mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
    model_file = tmp_path / "boundary.onnx"
    model_file.write_bytes(b"onnx-graph")

    monkeypatch.setattr(
        synthetic_benchmark,
        "segment_image",
        lambda rgb, **_kwargs: ExtractionResult(
            mask=expected_mask,
            style="model-mask",
            pixel_geometry=_reference_geometry(tmp_path, sample),
            coverage_ratio=float(expected_mask.mean()),
            contour_count=1,
            confidence=0.99,
        ),
    )

    with_model = synthetic_benchmark.score_synthetic_manifest(
        manifest,
        tmp_path,
        model_path=model_file,
    )
    assert with_model["extractor"] == "model"
    assert with_model["model_path"] == str(model_file)
    assert with_model["metadata"]["model_artifact"]["path"] == str(model_file.resolve())
    assert len(with_model["metadata"]["model_artifact"]["sha256"]) == 64

    missing_packaged = tmp_path / "missing" / "boundary_v1.onnx"
    monkeypatch.setattr(
        synthetic_benchmark,
        "default_model_path",
        lambda: missing_packaged,
    )
    without_model = synthetic_benchmark.score_synthetic_manifest(manifest, tmp_path)
    assert without_model["extractor"] == "auto-fill"
    assert without_model["model_path"] is None
    assert without_model["metadata"]["model_artifact"] is None

    packaged = tmp_path / "packaged.onnx"
    packaged.write_bytes(b"packaged-onnx-graph")
    monkeypatch.setattr(synthetic_benchmark, "default_model_path", lambda: packaged)
    with_packaged = synthetic_benchmark.score_synthetic_manifest(manifest, tmp_path)
    assert with_packaged["extractor"] == "model"
    assert with_packaged["model_path"] is None
    assert with_packaged["metadata"]["model_artifact"]["path"] == str(packaged.resolve())


def test_artifact_identity_binds_external_data_and_metadata_bytes(
    tmp_path: Path,
) -> None:
    selector = tmp_path / "selector.onnx"
    external_data = tmp_path / "selector.onnx.data"
    metadata = tmp_path / "selector.onnx.json"
    selector.write_bytes(b"onnx-graph")
    external_data.write_bytes(b"external-weights")
    metadata.write_text('{"schema_version":"selector-test-v1"}', encoding="utf-8")

    identity = synthetic_benchmark._artifact_identity(selector)

    assert identity["schema_version"] == (
        synthetic_benchmark.ARTIFACT_BUNDLE_SCHEMA_VERSION
    )
    assert identity["total_bytes"] == sum(
        path.stat().st_size for path in (selector, external_data, metadata)
    )
    assert len(identity["bundle_sha256"]) == 64
    assert {
        (record["relative_path"], record["role"])
        for record in identity["files"]
    } == {
        ("selector.onnx", "onnx"),
        ("selector.onnx.data", "external_data"),
        ("selector.onnx.json", "metadata"),
    }
    assert all(record["exists"] is True for record in identity["files"])
    assert all(len(record["sha256"]) == 64 for record in identity["files"])


def test_score_masks_gates_reference_normalized_roughness() -> None:
    from shapely.geometry import Polygon

    reference = Polygon(
        [(4, 4), (8, 5), (12, 3), (16, 5), (20, 3), (24, 4), (24, 20), (4, 20)]
    )
    mask = rasterize_geometry_mask(reference, width=32, height=28)

    metrics = synthetic_benchmark._score_masks(
        mask,
        mask,
        predicted_geometry=reference,
        reference_geometry=reference,
    )

    assert metrics["straight_run_rms_px"] > 0.35
    assert metrics["straight_run_rms_excess_px"] == 0.0
    assert metrics["canonical_complexity_ratio"] == 1.0
    assert metrics["topology_matches"] is True
    assert metrics["raster_topology_matches"] is True


def test_score_masks_separates_vector_gate_from_raster_topology_diagnostic() -> None:
    from shapely.geometry import Polygon

    plain = np.zeros((24, 24), dtype=bool)
    plain[2:22, 2:22] = True
    holed = plain.copy()
    holed[8:16, 8:16] = False
    simple = Polygon([(2, 2), (21, 2), (21, 21), (2, 21)])
    with_hole = Polygon(
        [(2, 2), (21, 2), (21, 21), (2, 21)],
        [[(8, 8), (15, 8), (15, 15), (8, 15)]],
    )

    raster_only_error = synthetic_benchmark._score_masks(
        holed,
        plain,
        predicted_geometry=simple,
        reference_geometry=simple,
    )
    vector_error = synthetic_benchmark._score_masks(
        plain,
        plain,
        predicted_geometry=with_hole,
        reference_geometry=simple,
    )

    assert raster_only_error["topology_matches"] is True
    assert raster_only_error["raster_topology_matches"] is False
    assert raster_only_error["predicted_holes"] == 0
    assert raster_only_error["predicted_raster_holes"] == 1
    assert vector_error["topology_matches"] is False
    assert vector_error["raster_topology_matches"] is True


def sample_polygon():
    from shapely.geometry import Polygon

    return Polygon([(10, 10), (60, 10), (60, 50), (10, 50)])


def _reference_geometry(tmp_path: Path, sample):
    reference_geojson = json.loads(
        (tmp_path / sample.artifacts.geojson).read_text(encoding="utf-8")
    )
    return shape(reference_geojson["metadata"]["pixel_geometry"])
