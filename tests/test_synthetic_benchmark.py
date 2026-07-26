import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from shapely.geometry import shape

import map_boundary_builder.synthetic_benchmark as synthetic_benchmark
import map_boundary_builder.edgegraph as edgegraph
import map_boundary_builder.model_extract as model_extract
from map_boundary_builder.evaluation import rasterize_geometry_mask
from map_boundary_builder.extract import ExtractionHints, ExtractionResult
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
    assert report["schema_version"] == synthetic_benchmark.REPORT_SCHEMA_VERSION
    assert report["metadata"]["guidance_mode"] == "automatic"
    assert report["metadata"]["guided"] is False
    assert report["metadata"]["oracle_hints"] is False


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


def test_automatic_edgegraph_scoring_uses_selector_bootstrap_guidance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=31, width=120, height=90)
    sample = manifest.samples[0]
    expected_mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
    reference_geometry = _reference_geometry(tmp_path, sample)
    deterministic = ExtractionResult(
        mask=expected_mask,
        style="auto-fill",
        pixel_geometry=reference_geometry,
        coverage_ratio=float(expected_mask.mean()),
        contour_count=1,
        confidence=0.98,
        diagnostics={"candidate": True},
    )
    automatic_hints = ExtractionHints(
        seed_point=(42.0, 36.0),
        target_rgb=(88, 177, 211),
    )
    observed: dict[str, object] = {}

    def fake_extract(image_path, **kwargs):
        observed["deterministic_image_path"] = image_path
        observed["deterministic_kwargs"] = kwargs
        return deterministic

    def fake_hints(rgb, coarse, *, threshold):
        observed["hint_rgb"] = rgb
        observed["hint_coarse"] = coarse
        observed["hint_threshold"] = threshold
        return automatic_hints, {"route": "selector-bootstrap-v1"}

    def fake_predict(rgb, session, *, config, hints):
        observed["selector_hints"] = hints
        return expected_mask.astype(np.float32)

    def fake_refine(rgb, coarse, session, *, hints, config):
        observed["refiner_hints"] = hints
        return SimpleNamespace(
            mask=expected_mask,
            pixel_geometry=reference_geometry,
            contour_count=1,
            confidence=0.99,
            diagnostics={
                "source_native_proposals": True,
                "source_native_proposal": {"reason": "accepted"},
            },
        )

    monkeypatch.setattr(synthetic_benchmark, "extract_service_area", fake_extract)
    monkeypatch.setattr(edgegraph, "automatic_edgegraph_hints", fake_hints)
    monkeypatch.setattr(
        synthetic_benchmark,
        "should_fallback_to_deterministic_result",
        lambda *_args: False,
    )
    monkeypatch.setattr(model_extract, "load_onnx_session", lambda path: path)
    monkeypatch.setattr(model_extract, "predict_mask_probabilities", fake_predict)
    monkeypatch.setattr(edgegraph, "refine_boundary_with_edgegraph", fake_refine)

    row = synthetic_benchmark.score_synthetic_sample(
        sample,
        tmp_path,
        model_path=tmp_path / "selector.onnx",
        edgegraph_refiner_path=tmp_path / "refiner.onnx",
    )

    assert row["status"] == "scored"
    assert observed["selector_hints"] is None
    assert observed["refiner_hints"] is automatic_hints
    deterministic_kwargs = observed["deterministic_kwargs"]
    assert deterministic_kwargs["cache"] is False
    assert deterministic_kwargs["use_model"] is False
    assert row["extraction"]["diagnostics"]["automatic_guidance"] == "selector_bootstrap"
    assert row["extraction"]["diagnostics"]["selector_bootstrap"]["route"] == (
        "selector-bootstrap-v1"
    )
    assert row["extraction"]["diagnostics"]["deterministic_review"][
        "candidate_agreement_iou"
    ] == 1.0
    assert row["extraction"]["diagnostics"]["source_native_proposals"] is True
    assert row["extraction"]["diagnostics"]["model_guidance"] == {
        "seed_point": True,
        "target_rgb": True,
    }


def test_automatic_edgegraph_scoring_preflights_verified_source_native_geometry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=35, width=120, height=90)
    sample = manifest.samples[0]
    expected_mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
    exact_result = ExtractionResult(
        mask=expected_mask,
        style="gray-fill",
        pixel_geometry=_reference_geometry(tmp_path, sample),
        coverage_ratio=float(expected_mask.mean()),
        contour_count=1,
        confidence=1.0,
        diagnostics={
            "gray_outline": {
                "verified_source_native": True,
            },
        },
    )

    monkeypatch.setattr(
        synthetic_benchmark,
        "gray_outline_extraction_result",
        lambda *_args, **_kwargs: exact_result,
    )
    monkeypatch.setattr(
        model_extract,
        "load_onnx_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("verified source-native geometry must bypass model inference")
        ),
    )

    row = synthetic_benchmark.score_synthetic_sample(
        sample,
        tmp_path,
        model_path=tmp_path / "selector.onnx",
        edgegraph_refiner_path=tmp_path / "refiner.onnx",
    )

    assert row["status"] == "scored"
    assert row["metrics"]["iou"] == 1.0
    assert row["extraction"]["diagnostics"]["automatic_route"] == (
        "verified-source-native-preflight-v1"
    )


def test_guided_edgegraph_scoring_uses_oracle_without_deterministic_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=37, width=120, height=90)
    sample = manifest.samples[0]
    expected_mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
    reference_geometry = _reference_geometry(tmp_path, sample)
    oracle_hints = {"seed_point": (20.0, 20.0), "target_rgb": (20, 120, 220)}
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        synthetic_benchmark,
        "extract_service_area",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("guided scoring must not run deterministic extraction")
        ),
    )
    monkeypatch.setattr(
        synthetic_benchmark,
        "synthetic_guidance",
        lambda scored_sample, mask, rgb: oracle_hints,
    )
    monkeypatch.setattr(model_extract, "load_onnx_session", lambda path: path)
    monkeypatch.setattr(
        model_extract,
        "predict_mask_probabilities",
        lambda rgb, session, *, config, hints: (
            observed.setdefault("selector_hints", hints),
            expected_mask.astype(np.float32),
        )[1],
    )
    monkeypatch.setattr(
        edgegraph,
        "refine_boundary_with_edgegraph",
        lambda rgb, coarse, session, *, hints, config: (
            observed.setdefault("refiner_hints", hints),
            SimpleNamespace(
                mask=expected_mask,
                pixel_geometry=reference_geometry,
                contour_count=1,
                confidence=0.99,
                diagnostics={"source_native_proposals": True},
            ),
        )[1],
    )

    row = synthetic_benchmark.score_synthetic_sample(
        sample,
        tmp_path,
        model_path=tmp_path / "selector.onnx",
        edgegraph_refiner_path=tmp_path / "refiner.onnx",
        guided=True,
    )

    assert row["status"] == "scored"
    assert observed["selector_hints"] is oracle_hints
    assert observed["refiner_hints"] is oracle_hints
    assert "automatic_guidance" not in row["extraction"]["diagnostics"]


def test_automatic_edgegraph_scoring_preserves_production_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=41, width=120, height=90)
    sample = manifest.samples[0]
    expected_mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
    reference_geometry = _reference_geometry(tmp_path, sample)
    deterministic = ExtractionResult(
        mask=expected_mask,
        style="auto-fill",
        pixel_geometry=reference_geometry,
        coverage_ratio=float(expected_mask.mean()),
        contour_count=1,
        confidence=0.99,
        diagnostics={"deterministic": True},
    )
    model_mask = np.zeros_like(expected_mask)
    model_mask[10:14, 10:14] = True

    monkeypatch.setattr(
        synthetic_benchmark,
        "extract_service_area",
        lambda *_args, **_kwargs: deterministic,
    )
    monkeypatch.setattr(
        synthetic_benchmark,
        "should_fallback_to_deterministic_result",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        edgegraph,
        "automatic_edgegraph_hints",
        lambda *_args, **_kwargs: (
            ExtractionHints(
                seed_point=(20.0, 20.0),
                target_rgb=(30, 120, 210),
            ),
            {"route": "selector-bootstrap-v1"},
        ),
    )
    monkeypatch.setattr(
        synthetic_benchmark,
        "model_fallback_reason",
        lambda *_args: "test_fallback",
    )
    monkeypatch.setattr(model_extract, "load_onnx_session", lambda path: path)
    monkeypatch.setattr(
        model_extract,
        "predict_mask_probabilities",
        lambda *_args, **_kwargs: model_mask.astype(np.float32),
    )
    monkeypatch.setattr(
        edgegraph,
        "refine_boundary_with_edgegraph",
        lambda *_args, **_kwargs: SimpleNamespace(
            mask=model_mask,
            pixel_geometry=sample_polygon(),
            contour_count=1,
            confidence=0.6,
            diagnostics={"edgegraph": True},
        ),
    )

    row = synthetic_benchmark.score_synthetic_sample(
        sample,
        tmp_path,
        model_path=tmp_path / "selector.onnx",
        edgegraph_refiner_path=tmp_path / "refiner.onnx",
    )

    assert row["status"] == "scored"
    assert row["metrics"]["iou"] == 1.0
    assert row["extraction"]["diagnostics"]["deterministic"] is True
    fallback = row["extraction"]["diagnostics"]["model_fallback"]
    assert fallback["reason"] == "test_fallback"
    assert fallback["model_coverage_ratio"] == float(model_mask.mean())


def test_edgegraph_selector_config_rejects_nonproduction_contract() -> None:
    incompatible = synthetic_benchmark.ModelExtractionConfig(
        input_width=320,
        input_height=320,
        threshold=0.45,
        output_activation="logits",
        input_channels=3,
    )

    try:
        synthetic_benchmark._edgegraph_selector_config(incompatible)
    except ValueError as exc:
        assert "five input channels" in str(exc)
    else:
        raise AssertionError("nonproduction EdgeGraph selector config was accepted")


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


def test_automatic_edgegraph_report_binds_route_and_selector_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=1, seed=43, width=120, height=90)
    sample = manifest.samples[0]
    expected_mask = synthetic_benchmark._load_mask(tmp_path / sample.artifacts.mask)
    reference_geometry = _reference_geometry(tmp_path, sample)
    hints = ExtractionHints(seed_point=(30.0, 30.0), target_rgb=(40, 100, 180))

    monkeypatch.setattr(
        synthetic_benchmark,
        "extract_service_area",
        lambda *_args, **_kwargs: ExtractionResult(
            mask=expected_mask,
            style="auto-fill",
            pixel_geometry=reference_geometry,
            coverage_ratio=float(expected_mask.mean()),
            contour_count=1,
            confidence=0.99,
        ),
    )
    monkeypatch.setattr(
        synthetic_benchmark,
        "should_fallback_to_deterministic_result",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        edgegraph,
        "automatic_edgegraph_hints",
        lambda *_args, **_kwargs: (hints, {"route": "selector-bootstrap-v1"}),
    )
    monkeypatch.setattr(model_extract, "load_onnx_session", lambda path: path)
    monkeypatch.setattr(
        model_extract,
        "predict_mask_probabilities",
        lambda *_args, **_kwargs: expected_mask.astype(np.float32),
    )
    monkeypatch.setattr(
        edgegraph,
        "refine_boundary_with_edgegraph",
        lambda *_args, **_kwargs: SimpleNamespace(
            mask=expected_mask,
            pixel_geometry=reference_geometry,
            contour_count=1,
            confidence=0.99,
            diagnostics={"edgegraph": True},
        ),
    )

    report = synthetic_benchmark.score_synthetic_manifest(
        manifest,
        tmp_path,
        model_path=tmp_path / "selector.onnx",
        edgegraph_refiner_path=tmp_path / "refiner.onnx",
    )

    assert report["schema_version"] == synthetic_benchmark.REPORT_SCHEMA_VERSION
    assert report["metadata"]["automatic_inference_route"] == (
        synthetic_benchmark.AUTOMATIC_EDGEGRAPH_ROUTE
    )
    assert report["metadata"]["selector_config"] == {
        "input_width": 320,
        "input_height": 320,
        "input_channels": 5,
        "threshold": 0.45,
        "output_activation": "logits",
    }


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
