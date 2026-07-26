import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from map_boundary_builder import synthetic_benchmark
from tools import benchmark_v20_native
from tools import check_v20_promotion


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_refiner_onnx(
    path: Path,
    *,
    input_name: str = check_v20_promotion.VECTOR_V3_ONNX_INPUT_NAME,
    input_channels: int = len(check_v20_promotion.VECTOR_V3_INPUT_CHANNELS),
) -> None:
    input_value = helper.make_tensor_value_info(
        input_name,
        TensorProto.FLOAT,
        ["batch", input_channels, "contour_length", 49],
    )
    outputs = [
        helper.make_tensor_value_info(
            "offset_logits",
            TensorProto.FLOAT,
            ["batch", "contour_length", 49],
        ),
        helper.make_tensor_value_info(
            "corner_logits",
            TensorProto.FLOAT,
            ["batch", "contour_length"],
        ),
        helper.make_tensor_value_info(
            "reliability_logits",
            TensorProto.FLOAT,
            ["batch", "contour_length"],
        ),
    ]
    nodes = [
        helper.make_node(
            "ReduceMean",
            [input_name],
            ["offset_logits"],
            axes=[1],
            keepdims=0,
        ),
        helper.make_node(
            "ReduceMean",
            [input_name],
            ["corner_logits"],
            axes=[1, 3],
            keepdims=0,
        ),
        helper.make_node("Identity", ["corner_logits"], ["reliability_logits"]),
    ]
    graph = helper.make_graph(nodes, "promotion-test-refiner", [input_value], outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _write_selector_onnx(
    path: Path,
    *,
    input_channels: int = 5,
    image_size: int = 320,
    output_name: str = "mask_logits",
) -> None:
    input_value = helper.make_tensor_value_info(
        "image",
        TensorProto.FLOAT,
        [1, input_channels, image_size, image_size],
    )
    output_value = helper.make_tensor_value_info(
        output_name,
        TensorProto.FLOAT,
        [1, 1, image_size, image_size],
    )
    weights = numpy_helper.from_array(
        np.full((1, input_channels, 1, 1), 0.2, dtype=np.float32),
        "weights",
    )
    node = helper.make_node(
        "Conv",
        ["image", "weights"],
        [output_name],
        kernel_shape=[1, 1],
    )
    graph = helper.make_graph(
        [node],
        "promotion-test-selector",
        [input_value],
        [output_value],
        [weights],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=0,
    )


def _source_artifact(path: str, byte_count: int, sha256: str) -> dict[str, object]:
    return {
        "path": path,
        "bytes": byte_count,
        "sha256": sha256,
    }


def _promotion_inputs(artifact: Path):
    _write_refiner_onnx(artifact)
    selector = artifact.with_name("selector.onnx")
    _write_selector_onnx(selector)
    training_manifest = artifact.with_name("training-manifest.json")
    training_manifest.write_text(
        json.dumps(
            {
                "version": "synthetic-generator-v20-centered-stroke-raster-v2",
                "samples": [
                    {"sample_id": f"train-{index:04d}"}
                    for index in range(512)
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    Path(str(selector) + ".json").write_text(
        json.dumps(
            {
                "schema_version": (
                    check_v20_promotion.SELECTOR_METADATA_SCHEMA_VERSION
                ),
                "architecture": "generalized-v20-edgegraph-selector-resunet-v1",
                "onnx_input_name": "image",
                "onnx_output_name": "mask_logits",
                "training_dataset": {
                    "version": "synthetic-generator-v20-centered-stroke-raster-v2",
                    "manifest_sha256": _sha256(training_manifest),
                    "sample_count": 512,
                },
                "seed": 101,
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": 0.0004,
                    "weight_decay": 0.0001,
                },
                "run_config": {
                    "architecture": "resunet",
                    "image_size": 320,
                    "input_channels": 5,
                    "base_channels": 24,
                    "batch_size": 2,
                    "epochs": 8,
                    "validation_count": 256,
                },
                "resume_artifact": _source_artifact(
                    "selector-bootstrap-best.pt",
                    1_234_567,
                    "c" * 64,
                ),
                "production": {
                    "threshold": 0.45,
                    "output_activation": "logits",
                    "input_width": 320,
                    "input_height": 320,
                    "input_channels": 5,
                },
                "guidance": {
                    "training_policy": "automatic-heavy",
                    "validation_policy": "none",
                },
                "selected_checkpoint": {
                    "artifact": _source_artifact(
                        "selector-auto-best.pt",
                        1_345_678,
                        "d" * 64,
                    ),
                    "epoch": 8,
                    "metrics": {
                        "validation_iou": 0.99,
                        "validation_p05_iou": 0.97,
                        "validation_boundary_iou_2px": 0.98,
                        "validation_p05_boundary_iou_2px": 0.95,
                        "validation_tail_score": 0.971,
                        "validation_score": 0.971,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    Path(str(artifact) + ".json").write_text(
        json.dumps(
            {
                "architecture": "generalized-v20-edgegraph-stripnet-vector-v3",
                "channels": len(check_v20_promotion.VECTOR_V3_INPUT_CHANNELS),
                "input_channels": list(check_v20_promotion.VECTOR_V3_INPUT_CHANNELS),
                "onnx_input_name": check_v20_promotion.VECTOR_V3_ONNX_INPUT_NAME,
                "coarse_probability_input": False,
                "conservative_offset_head": False,
                "learned_offset_logits_preserved": True,
                "profile_dilations": [1, 2, 4],
                "profile_receptive_field_px": 14.0,
                "training_augmentation": {
                    "profile_center_shift_distribution": "uniform",
                    "profile_center_shift_max_abs_px": 6.0,
                    "profile_center_shift_scope": "global_per_sample",
                    "inside_direction_reference": "unshifted-contour-center-v1",
                },
                "training_dataset": {
                    "version": "synthetic-generator-v20-centered-stroke-raster-v2",
                    "manifest_sha256": _sha256(training_manifest),
                    "sample_count": 512,
                },
                "supervision": {"target_version": "geojson-pixel-ray-segment-v1"},
            }
        ),
        encoding="utf-8",
    )
    families = ("rectilinear", "angular", "road-following", "radial")
    rows = []
    samples = []
    for index in range(96):
        sample_id = f"sample-{index:03d}"
        family = families[index % len(families)]
        rows.append(
            {
                "sample_id": sample_id,
                "stroke_width_px": float((index // 4) % 4),
                "status": "scored",
                "metrics": {
                    "iou": 0.99,
                    "boundary_f1_1px": 0.95,
                    "boundary_distance_p95_px": 0.4,
                    "boundary_distance_p99_px": 0.7,
                    "boundary_distance_max_px": 1.0,
                    "corner_f1": 0.96,
                    "corner_angular_error_p95_degrees": 1.5,
                    "straight_run_rms_px": 0.1,
                    "straight_run_rms_excess_px": 0.1,
                    "excess_vertex_ratio": 1.05,
                    "canonical_complexity_ratio": 1.05,
                    "topology_matches": True,
                    "raster_topology_matches": True,
                },
            }
        )
        samples.append({"sample_id": sample_id, "properties": {"shape_family": family}})
        rows[-1]["geometry"] = {
            "is_present": True,
            "geometry_type": "Polygon",
            "is_empty": False,
            "is_valid": True,
            "area": 10_000.0,
        }
    manifest = {"samples": samples}
    manifest_path = artifact.with_name("holdout-manifest.json")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    synthetic = {
        "schema_version": check_v20_promotion.SYNTHETIC_REPORT_SCHEMA_VERSION,
        "extractor": "edgegraph",
        "passed": True,
        "metadata": {
            "guidance_mode": "automatic",
            "guided": False,
            "oracle_hints": False,
            "automatic_inference_route": check_v20_promotion.AUTOMATIC_EDGEGRAPH_ROUTE,
            "selector_config": {
                "input_width": 320,
                "input_height": 320,
                "input_channels": 5,
                "threshold": 0.45,
                "output_activation": "logits",
            },
            "manifest_sha256": _sha256(manifest_path),
            "model_artifacts": {
                "selector": synthetic_benchmark._artifact_identity(selector),
                "refiner": synthetic_benchmark._artifact_identity(artifact),
            },
        },
        "summary": {"sample_count": 96, "scored_count": 96, "failure_count": 0},
        "rows": rows,
    }
    source_sha256 = "b" * 64

    def real_report(model_variant: str, row_iou: float) -> dict[str, object]:
        scores = []
        for index in range(12):
            condition = {
                "slug": f"real-{index:02d}",
                "width": 1024 - index,
                "height": 768 - index,
                "encoding": ("png", "jpeg", "webp")[index % 3],
                "quality": None if index % 3 == 0 else 90 - index,
            }
            scores.append(
                {
                    "slug": condition["slug"],
                    "status": "active",
                    "duration_s": 0.2,
                    "condition": condition,
                    "raster_sha256": f"{index:064x}",
                    "reference": {
                        "space": "native_pixels",
                        "source": "exact_svg_vector_path",
                        "width": condition["width"],
                        "height": condition["height"],
                        "flatten_tolerance_px": 0.15,
                    },
                    "metrics": {
                        "iou": row_iou,
                        "boundary_f1_1px": 0.95,
                        "boundary_distance_p95_px": 0.8,
                        "boundary_distance_p99_px": 1.2,
                        "boundary_distance_max_px": 1.5,
                        "corner_f1": 0.95,
                        "topology_matches": True,
                        "raster_topology_matches": True,
                        "geometry_valid": True,
                        "geometry_empty": False,
                        "reference_geometry_valid": True,
                    },
                    "result": {"selected_model_variant": model_variant},
                }
            )
        return {
            "schema_version": check_v20_promotion.NATIVE_REPORT_SCHEMA_VERSION,
            "reference_space": "native_pixels",
            "model_variant": model_variant,
            "metadata": {
                "source_svg": "/fixtures/observed.svg",
                "source_sha256": source_sha256,
                "reference_space": "native_pixels",
                "truth_source": "exact_svg_vector_path",
                "automatic_production_extraction": True,
                "oracle_hints": False,
                "cache_enabled": False,
                "timing_profile": "warm",
                "rasterizer": "resvg-py",
            },
            "summary": benchmark_v20_native.summarize_scores(scores),
            "scores": scores,
        }

    real = real_report(check_v20_promotion.EDGEGRAPH_MODEL_VARIANT, 0.985)
    baseline = real_report(check_v20_promotion.BOUNDARYFIELD_MODEL_VARIANT, 0.96)
    svg = {
        "summary": {
            "fixture_count": 3,
            "failure_count": 0,
            "topology_error_count": 0,
            "min_corner_f1": 1.0,
            "max_path_deviation_px": 0.0,
        },
        "observed_svg": {
            "source_sha256": source_sha256,
            "geometry_type": "Polygon",
            "valid": True,
            "is_empty": False,
            "area": 100_000.0,
            "vertices": 24,
            "line_segment_count": 23,
            "curve_segment_count": 0,
            "flatten_tolerance_px": 0.15,
        },
    }
    return synthetic, manifest, real, baseline, svg, [selector, artifact]


def _build_report(
    synthetic,
    manifest,
    real,
    baseline,
    svg,
    artifacts,
):
    root = artifacts[-1].parent
    manifest_path = root / "holdout-manifest.json"
    return check_v20_promotion.build_promotion_report(
        synthetic=synthetic,
        manifest=manifest,
        real=real,
        baseline=baseline,
        svg=svg,
        artifacts=artifacts,
        manifest_path=manifest_path,
        training_manifest_path=root / "training-manifest.json",
        expected_holdout_manifest_sha256=_sha256(manifest_path),
    )


def test_v20_promotion_requires_native_edge_and_svg_quality(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is True
    assert all(report["gates"].values())
    assert report["edge_pressure"]["sample_count"] == 72


def test_v20_promotion_fails_closed_when_edge_metric_is_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    del synthetic["rows"][0]["metrics"]["boundary_distance_p99_px"]

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["synthetic_complete_edge_metrics"] is False
    assert report["synthetic"]["missing_metrics"]["boundary_distance_p99_px"] == 1


def test_v20_promotion_gates_raster_topology_and_not_absolute_roughness(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    for row in synthetic["rows"]:
        row["metrics"]["straight_run_rms_px"] = 9.0
        row["metrics"]["excess_vertex_ratio"] = 9.0
    synthetic["rows"][0]["metrics"]["raster_topology_matches"] = False

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["synthetic_zero_raster_topology_errors"] is False
    assert report["edge_pressure"]["p95_straight_run_rms_px"] == 9.0
    assert report["edge_pressure"]["p95_excess_vertex_ratio"] == 9.0
    assert report["synthetic"]["raster_topology_error_count"] == 1


def test_v20_promotion_rejects_reference_normalized_roughness(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    for row in synthetic["rows"]:
        row["metrics"]["straight_run_rms_excess_px"] = 0.36
        row["metrics"]["canonical_complexity_ratio"] = 1.36

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["edge_straight_run_rms_excess"] is False
    assert report["gates"]["edge_canonical_complexity_ratio"] is False


def test_v20_promotion_blocks_vector_topology_error(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    synthetic["rows"][0]["metrics"]["topology_matches"] = False

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["synthetic_zero_topology_errors"] is False
    assert report["gates"]["synthetic_zero_topology_errors_per_family"] is False


def test_v20_promotion_binds_exact_locked_manifest_bytes_and_unique_rows(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    manifest_path = tmp_path / "holdout-manifest.json"

    wrong_lock = check_v20_promotion.build_promotion_report(
        synthetic=synthetic,
        manifest=manifest,
        real=real,
        baseline=baseline,
        svg=svg,
        artifacts=artifacts,
        manifest_path=manifest_path,
        training_manifest_path=tmp_path / "training-manifest.json",
        expected_holdout_manifest_sha256="0" * 64,
    )
    assert wrong_lock["gates"]["holdout_manifest_locked_sha256"] is False

    manifest["samples"][1]["sample_id"] = manifest["samples"][0]["sample_id"]
    duplicate = _build_report(synthetic, manifest, real, baseline, svg, artifacts)
    assert duplicate["gates"]["holdout_manifest_unique_valid_ids"] is False
    assert duplicate["gates"]["synthetic_rows_match_locked_manifest"] is False


def test_v20_promotion_rejects_guided_or_unbound_synthetic_evidence(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    synthetic["metadata"]["guidance_mode"] = "oracle_guided"
    synthetic["metadata"]["guided"] = True
    synthetic["metadata"]["oracle_hints"] = True
    synthetic["metadata"]["model_artifacts"]["selector"]["sha256"] = "f" * 64

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["synthetic_automatic_no_oracle_guidance"] is False
    assert report["gates"]["synthetic_selector_artifact_bound"] is False


def test_v20_promotion_rejects_stale_automatic_route_or_selector_contract(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    synthetic["metadata"]["automatic_inference_route"] = "legacy-unguided"
    synthetic["metadata"]["selector_config"]["input_channels"] = 3

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["synthetic_production_inference_route"] is False
    assert report["gates"]["synthetic_production_selector_config"] is False


def test_v20_promotion_binds_selector_external_data_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    selector_external_data = tmp_path / "selector.onnx.data"
    selector_external_data.write_bytes(
        selector_external_data.read_bytes() + b"\x00"
    )

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["synthetic_selector_artifact_bound"] is False


def test_v20_promotion_inspects_selector_graph_instead_of_trusting_report(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    selector = tmp_path / "selector.onnx"
    Path(str(selector) + ".data").unlink()
    _write_selector_onnx(selector, input_channels=3)
    synthetic["metadata"]["model_artifacts"]["selector"] = (
        synthetic_benchmark._artifact_identity(selector)
    )

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["synthetic_selector_artifact_bound"] is True
    assert report["gates"]["selector_onnx_graph_contract"] is False


def test_v20_promotion_rejects_unbound_selector_metadata_bytes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    metadata_path = tmp_path / "selector.onnx.json"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["synthetic_selector_artifact_bound"] is False
    assert report["gates"]["selector_metadata_bytes_bound"] is False


def test_v20_promotion_rejects_invalid_selector_training_provenance(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    selector = tmp_path / "selector.onnx"
    metadata_path = Path(str(selector) + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_dataset"]["manifest_sha256"] = "e" * 64
    metadata["seed"] = "101"
    metadata["optimizer"]["name"] = "SGD"
    metadata["resume_artifact"]["bytes"] = 0
    metadata["production"]["threshold"] = 0.44
    metadata["guidance"]["validation_policy"] = "mixed"
    del metadata["selected_checkpoint"]["metrics"]["validation_p05_iou"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    synthetic["metadata"]["model_artifacts"]["selector"] = (
        synthetic_benchmark._artifact_identity(selector)
    )

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["synthetic_selector_artifact_bound"] is True
    assert report["gates"]["selector_metadata_bytes_bound"] is True
    assert report["gates"]["selector_training_manifest_bytes_bound"] is False
    assert report["gates"]["selector_training_seed"] is False
    assert report["gates"]["selector_optimizer_run_config"] is False
    assert report["gates"]["selector_resume_artifact_identity"] is False
    assert report["gates"]["selector_production_contract"] is False
    assert report["gates"]["selector_automatic_guidance_contract"] is False
    assert report["gates"]["selector_selected_checkpoint_contract"] is False


def test_v20_promotion_inspects_refiner_graph_instead_of_trusting_metadata(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    _write_refiner_onnx(artifact, input_name="mislabeled_strip", input_channels=9)
    synthetic["metadata"]["model_artifacts"]["refiner"] = (
        synthetic_benchmark._artifact_identity(artifact)
    )

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["synthetic_refiner_artifact_bound"] is True
    assert report["gates"]["refiner_vector_v3_contract"] is True
    assert report["gates"]["refiner_onnx_graph_contract"] is False


def test_v20_promotion_binds_training_manifest_bytes_version_and_count(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    training_manifest = tmp_path / "training-manifest.json"
    training_manifest.write_text(
        training_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["refiner_training_manifest_bytes_bound"] is False
    assert report["training_manifest"]["sha256_matches"] is False


def test_v20_promotion_rejects_invalid_synthetic_geometry(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    synthetic["rows"][0]["geometry"]["is_valid"] = False

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["synthetic_geometry_valid"] is False


def test_v20_promotion_requires_exact_paired_warm_real_conditions(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    baseline["scores"][0]["condition"]["width"] += 1
    baseline["metadata"]["timing_profile"] = "cold"
    baseline["metadata"]["source_sha256"] = "c" * 64

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["real_baseline_contract"] is False
    assert report["gates"]["real_candidate_baseline_pairing"] is False
    assert report["gates"]["real_candidate_baseline_source_match"] is False


def test_v20_promotion_rejects_failed_or_wrong_variant_real_rows(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    real["scores"][0]["status"] = "failed"
    real["scores"][0]["error"] = "forced failure"
    real["model_variant"] = check_v20_promotion.BOUNDARYFIELD_MODEL_VARIANT

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["real_candidate_contract"] is False
    assert report["gates"]["real_all_active_pass"] is False
    assert report["gates"]["real_candidate_summary_matches_rows"] is False


def test_v20_promotion_gates_observed_svg_source_geometry_and_tolerance(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    svg["observed_svg"].update(
        {
            "source_sha256": "d" * 64,
            "valid": False,
            "vertices": 3,
            "flatten_tolerance_px": 0.151,
        }
    )

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["gates"]["svg_observed_source_bound"] is False
    assert report["gates"]["svg_observed_valid_geometry"] is False
    assert report["gates"]["svg_observed_tolerance"] is False
    assert report["gates"]["svg_observed_vertices"] is False


def test_v20_promotion_rejects_stabilized_legacy_offset_head(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    metadata_path = Path(str(artifact) + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["conservative_offset_head"] = True
    metadata["learned_offset_logits_preserved"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["refiner_learned_offsets_preserved"] is False


def test_v20_promotion_rejects_legacy_or_mislabeled_vector_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    metadata_path = Path(str(artifact) + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["architecture"] = "generalized-v20-edgegraph-stripnet-vector-v2"
    metadata["input_channels"][-1] = "raw_coarse_probability"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["refiner_vector_v3_contract"] is False


def test_v20_promotion_rejects_missing_shifted_center_augmentation(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    metadata_path = Path(str(artifact) + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["training_augmentation"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["refiner_shifted_center_training_contract"] is False


def test_v20_promotion_rejects_profile_tower_that_cannot_span_centered_stroke(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    metadata_path = Path(str(artifact) + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["profile_dilations"] = [1, 1, 1]
    metadata["profile_receptive_field_px"] = 6.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["refiner_full_stroke_receptive_field"] is False


def test_v20_promotion_rejects_pre_raster_v2_training_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    synthetic, manifest, real, baseline, svg, artifacts = _promotion_inputs(artifact)
    metadata_path = Path(str(artifact) + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_dataset"]["version"] = "synthetic-generator-v20-centered-stroke"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _build_report(synthetic, manifest, real, baseline, svg, artifacts)

    assert report["passed"] is False
    assert report["gates"]["refiner_raster_v2_training_contract"] is False


def test_v20_promotion_cli_writes_machine_readable_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "edgefield.onnx"
    artifact.write_bytes(b"small-model")
    inputs = _promotion_inputs(artifact)
    paths = []
    for name, payload in zip(
        ("synthetic", "real", "baseline", "svg"),
        (inputs[0], inputs[2], inputs[3], inputs[4]),
        strict=True,
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    manifest_path = tmp_path / "holdout-manifest.json"
    monkeypatch.setattr(
        check_v20_promotion,
        "LOCKED_HOLDOUT_MANIFEST_SHA256",
        _sha256(manifest_path),
    )
    output = tmp_path / "promotion.json"

    exit_code = check_v20_promotion.main(
        [
            "--synthetic-report",
            str(paths[0]),
            "--manifest",
            str(manifest_path),
            "--training-manifest",
            str(tmp_path / "training-manifest.json"),
            "--real-report",
            str(paths[1]),
            "--baseline-real-report",
            str(paths[2]),
            "--svg-report",
            str(paths[3]),
            "--artifact",
            str(inputs[5][0]),
            "--artifact",
            str(artifact),
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
