from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort


SHAPE_FAMILIES = ("rectilinear", "angular", "road-following", "radial")
EDGE_FAMILIES = frozenset(("rectilinear", "angular", "road-following"))
VECTOR_V3_INPUT_CHANNELS = (
    "rgb_red",
    "rgb_green",
    "rgb_blue",
    "luminance",
    "scharr_magnitude_p99",
    "normal_projected_scharr_p99",
    "target_rgb_similarity",
    "source_sample_valid",
    "normalized_native_offset",
    "inside_direction_sign",
)
VECTOR_V3_ONNX_INPUT_NAME = "edge_strip_vector_v3"
LOCKED_HOLDOUT_MANIFEST_SHA256 = (
    "1badc6705ec89c9ad6c9566c30a8cd3f93b4b0fbf6320ca7e3d0ab2d9c11271c"
)
SYNTHETIC_REPORT_SCHEMA_VERSION = "synthetic-benchmark-v3"
ARTIFACT_BUNDLE_SCHEMA_VERSION = "onnx-artifact-bundle-v1"
SELECTOR_METADATA_SCHEMA_VERSION = "generalized-v20-edgegraph-selector-metadata-v1"
AUTOMATIC_EDGEGRAPH_ROUTE = (
    "verified-source-native-preflight-else-selector-bootstrap-then-edgegraph-with-guarded-deterministic-review-v1"
)
NATIVE_REPORT_SCHEMA_VERSION = "v20-native-raster-v1"
EDGEGRAPH_MODEL_VARIANT = "generalized_v20_edgegraph"
BOUNDARYFIELD_MODEL_VARIANT = "generalized_v12_boundaryfield"
EXPECTED_REAL_CONDITION_COUNT = 12
EXPECTED_REFINER_OUTPUTS = (
    ("offset_logits", 3),
    ("corner_logits", 2),
    ("reliability_logits", 2),
)
EXPECTED_SELECTOR_CHECKPOINT_METRICS = (
    "validation_iou",
    "validation_p05_iou",
    "validation_boundary_iou_2px",
    "validation_p05_boundary_iou_2px",
    "validation_tail_score",
    "validation_score",
)

THRESHOLDS: dict[str, float | int] = {
    "synthetic_sample_count": 96,
    "synthetic_family_count": 24,
    "synthetic_stroke_group_count": 24,
    "synthetic_mean_iou": 0.965,
    "synthetic_p05_iou": 0.900,
    "synthetic_p05_boundary_f1_1px": 0.720,
    "synthetic_p95_boundary_distance_px": 2.000,
    "synthetic_stroke_group_p05_boundary_f1_1px": 0.720,
    "synthetic_stroke_group_mean_corner_f1": 0.900,
    "edge_sample_count": 72,
    "edge_p05_boundary_f1_1px": 0.820,
    "edge_p95_boundary_distance_px": 1.250,
    "edge_p95_boundary_distance_p99_px": 2.000,
    "edge_max_boundary_distance_px": 4.000,
    "edge_mean_corner_f1": 0.900,
    "edge_p05_corner_f1": 0.820,
    "edge_p95_corner_angular_error_degrees": 5.000,
    "edge_p95_straight_run_rms_excess_px": 0.350,
    "edge_p95_canonical_complexity_ratio": 1.350,
    "real_fixture_count": EXPECTED_REAL_CONDITION_COUNT,
    "real_average_iou": 0.970,
    "real_min_iou": 0.920,
    "real_improvement_over_baseline": 0.002,
    "real_p95_boundary_distance_px": 1.250,
    "real_p99_boundary_distance_px": 2.000,
    "real_mean_corner_f1": 0.900,
    "warm_p95_duration_s": 0.400,
    "max_duration_s": 0.650,
    "svg_fixture_count": 3,
    "svg_min_corner_f1": 0.980,
    "svg_max_path_deviation_px": 0.250,
    "svg_observed_max_flatten_tolerance_px": 0.150,
    "svg_observed_min_vertices": 4,
    "artifact_bytes": 15 * 1024 * 1024,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check immutable v20 EdgeField promotion gates.",
    )
    parser.add_argument("--synthetic-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        required=True,
        help="Exact training manifest whose bytes are bound into the refiner metadata.",
    )
    parser.add_argument("--real-report", type=Path, required=True)
    parser.add_argument("--baseline-real-report", type=Path, required=True)
    parser.add_argument("--svg-report", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        required=True,
        help="Model artifact to include in the total size budget. Repeat for each artifact.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _numeric_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    return (
        _is_finite_number(left)
        and _is_finite_number(right)
        and abs(float(left) - float(right)) <= tolerance
    )


def _quantile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    return round(float(np.quantile(values, value)), 6)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 6)


def _metric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get("metrics", {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def profile_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_metric_names = (
        "iou",
        "boundary_f1_1px",
        "boundary_distance_p95_px",
        "boundary_distance_p99_px",
        "boundary_distance_max_px",
        "corner_f1",
        "corner_angular_error_p95_degrees",
        "straight_run_rms_excess_px",
        "canonical_complexity_ratio",
    )
    diagnostic_metric_names = (
        "straight_run_rms_px",
        "excess_vertex_ratio",
    )
    values = {
        name: _metric_values(rows, name)
        for name in (*required_metric_names, *diagnostic_metric_names)
    }
    missing = {
        name: len(rows) - len(metric_values)
        for name, metric_values in (
            (name, values[name]) for name in required_metric_names
        )
        if len(metric_values) != len(rows)
    }
    return {
        "sample_count": len(rows),
        "mean_iou": _mean(values["iou"]),
        "p05_iou": _quantile(values["iou"], 0.05),
        "mean_boundary_f1_1px": _mean(values["boundary_f1_1px"]),
        "p05_boundary_f1_1px": _quantile(values["boundary_f1_1px"], 0.05),
        "p95_boundary_distance_px": _quantile(values["boundary_distance_p95_px"], 0.95),
        "p95_boundary_distance_p99_px": _quantile(values["boundary_distance_p99_px"], 0.95),
        "max_boundary_distance_px": round(max(values["boundary_distance_max_px"]), 6)
        if values["boundary_distance_max_px"]
        else None,
        "mean_corner_f1": _mean(values["corner_f1"]),
        "p05_corner_f1": _quantile(values["corner_f1"], 0.05),
        "p95_corner_angular_error_degrees": _quantile(
            values["corner_angular_error_p95_degrees"], 0.95
        ),
        "p95_straight_run_rms_excess_px": _quantile(
            values["straight_run_rms_excess_px"], 0.95
        ),
        "p95_canonical_complexity_ratio": _quantile(
            values["canonical_complexity_ratio"], 0.95
        ),
        # Absolute self-roughness is retained for diagnosis but is not gated;
        # authored reference geometry can have a non-zero self-simplification floor.
        "p95_straight_run_rms_px": _quantile(values["straight_run_rms_px"], 0.95),
        "p95_excess_vertex_ratio": _quantile(values["excess_vertex_ratio"], 0.95),
        "topology_error_count": sum(
            row.get("metrics", {}).get("topology_matches") is not True for row in rows
        ),
        "raster_topology_error_count": sum(
            row.get("metrics", {}).get("raster_topology_matches") is False
            for row in rows
        ),
        "raster_topology_missing_count": sum(
            not isinstance(
                row.get("metrics", {}).get("raster_topology_matches"),
                bool,
            )
            for row in rows
        ),
        "missing_metrics": missing,
    }


def _is_at_least(value: Any, threshold: float | int) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= threshold


def _is_at_most(value: Any, threshold: float | int) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value <= threshold


def _artifact_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        expanded.append(path.resolve())
        if path.suffix == ".onnx":
            expanded.extend(_onnx_companion_paths(path.resolve()))
    return list(dict.fromkeys(expanded))


def _onnx_companion_paths(path: Path) -> list[Path]:
    companions: list[Path] = []
    conventional_data = Path(str(path) + ".data")
    metadata = Path(str(path) + ".json")
    if conventional_data.exists():
        companions.append(conventional_data.resolve())
    companions.extend(_onnx_external_data_paths(path))
    if metadata.exists():
        companions.append(metadata.resolve())
    return list(dict.fromkeys(companions))


def _onnx_external_data_paths(path: Path) -> list[Path]:
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
    except Exception:
        return []

    parent = path.parent.resolve()
    external_paths: list[Path] = []
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key != "location" or not entry.value:
                continue
            candidate = (parent / entry.value).resolve()
            try:
                candidate.relative_to(parent)
            except ValueError:
                continue
            external_paths.append(candidate)
    return list(dict.fromkeys(external_paths))


def _artifact_bundle_files(path: Path) -> list[Path]:
    return [path.resolve(), *_onnx_companion_paths(path.resolve())]


def _artifact_file_identity(path: Path, *, artifact: Path) -> dict[str, Any]:
    resolved = path.resolve()
    artifact = artifact.resolve()
    try:
        relative_path = str(resolved.relative_to(artifact.parent))
    except ValueError:
        relative_path = resolved.name
    if resolved == artifact:
        role = "onnx"
    elif resolved == Path(str(artifact) + ".json"):
        role = "metadata"
    else:
        role = "external_data"
    exists = resolved.is_file()
    return {
        "relative_path": relative_path,
        "role": role,
        "path": str(resolved),
        "exists": exists,
        "bytes": resolved.stat().st_size if exists else None,
        "sha256": _sha256(resolved) if exists else None,
    }


def _artifact_bundle_sha256(records: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "relative_path": record["relative_path"],
            "role": record["role"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in sorted(records, key=lambda item: str(item["relative_path"]))
    ]
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _artifact_bundle_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        return {
            "schema_version": ARTIFACT_BUNDLE_SCHEMA_VERSION,
            "path": str(path),
            "bytes": None,
            "sha256": None,
            "total_bytes": None,
            "bundle_sha256": None,
            "files": [],
        }
    records = [
        _artifact_file_identity(item, artifact=path)
        for item in _artifact_bundle_files(path)
    ]
    complete = all(record["exists"] is True for record in records)
    return {
        "schema_version": ARTIFACT_BUNDLE_SCHEMA_VERSION,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "total_bytes": (
            sum(int(record["bytes"]) for record in records)
            if complete
            else None
        ),
        "bundle_sha256": (
            _artifact_bundle_sha256(records)
            if complete
            else None
        ),
        "files": records,
    }


def _refiner_contract(
    paths: list[Path],
) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    for artifact in paths:
        if artifact.suffix != ".onnx":
            continue
        metadata_path = Path(str(artifact) + ".json")
        if not metadata_path.is_file():
            continue
        try:
            payload = _load_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if str(payload.get("architecture", "")).startswith("generalized-v20-edgegraph-stripnet"):
            return artifact, metadata_path, payload
    return None, None, None


def _artifact_path_for_identity(
    identity: Any,
    paths: list[Path],
) -> Path | None:
    if not isinstance(identity, dict):
        return None
    sha256 = identity.get("sha256")
    size = identity.get("bytes")
    if not _is_sha256(sha256) or not isinstance(size, int):
        return None
    for path in paths:
        if (
            path.suffix == ".onnx"
            and path.is_file()
            and path.stat().st_size == size
            and _sha256(path) == sha256
        ):
            return path.resolve()
    return None


def _selector_contract(
    paths: list[Path],
    identity: Any,
) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    artifact_path = _artifact_path_for_identity(identity, paths)
    if artifact_path is None:
        return None, None, None
    metadata_path = Path(str(artifact_path) + ".json")
    if not metadata_path.is_file():
        return artifact_path, None, None
    try:
        metadata = _load_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return artifact_path, metadata_path, None
    return artifact_path, metadata_path, metadata


def _artifact_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        record: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": None,
            "sha256": None,
        }
        if path.is_file():
            record["bytes"] = path.stat().st_size
            record["sha256"] = _sha256(path)
        records.append(record)
    return records


def _inspect_refiner_onnx(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"loaded": False, "error": "missing refiner ONNX artifact"}
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = [
            {"name": value.name, "shape": list(value.shape), "type": value.type}
            for value in session.get_inputs()
        ]
        outputs = [
            {"name": value.name, "shape": list(value.shape), "type": value.type}
            for value in session.get_outputs()
        ]
    except Exception as exc:
        return {
            "loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"loaded": True, "inputs": inputs, "outputs": outputs}


def _inspect_selector_onnx(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"loaded": False, "error": "missing selector ONNX artifact"}
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = [
            {"name": value.name, "shape": list(value.shape), "type": value.type}
            for value in session.get_inputs()
        ]
        outputs = [
            {"name": value.name, "shape": list(value.shape), "type": value.type}
            for value in session.get_outputs()
        ]
    except Exception as exc:
        return {
            "loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"loaded": True, "inputs": inputs, "outputs": outputs}


def _onnx_shape_matches_selector(
    contract: dict[str, Any],
    selector_config: dict[str, Any],
) -> bool:
    if contract.get("loaded") is not True:
        return False
    inputs = contract.get("inputs")
    outputs = contract.get("outputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) != 1
        or not isinstance(outputs, list)
        or len(outputs) != 1
    ):
        return False
    width = selector_config.get("input_width")
    height = selector_config.get("input_height")
    channels = selector_config.get("input_channels")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or not isinstance(channels, int)
        or width != height
        or width <= 0
    ):
        return False
    model_input = inputs[0]
    model_output = outputs[0]
    return (
        model_input.get("name") == "image"
        and model_input.get("type") == "tensor(float)"
        and model_input.get("shape") == [1, channels, height, width]
        and model_output.get("name") == "mask_logits"
        and model_output.get("type") == "tensor(float)"
        and model_output.get("shape") == [1, 1, height, width]
    )


def _onnx_shape_matches_vector_v3(contract: dict[str, Any]) -> bool:
    if contract.get("loaded") is not True:
        return False
    inputs = contract.get("inputs")
    outputs = contract.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(outputs, list):
        return False
    model_input = inputs[0]
    shape = model_input.get("shape")
    if (
        model_input.get("name") != VECTOR_V3_ONNX_INPUT_NAME
        or model_input.get("type") != "tensor(float)"
        or shape
        != [
            "batch",
            len(VECTOR_V3_INPUT_CHANNELS),
            "contour_length",
            49,
        ]
    ):
        return False
    output_by_name = {
        output.get("name"): output
        for output in outputs
        if isinstance(output, dict)
    }
    expected_output_shapes = {
        "offset_logits": ["batch", "contour_length", 49],
        "corner_logits": ["batch", "contour_length"],
        "reliability_logits": ["batch", "contour_length"],
    }
    for output_name, rank in EXPECTED_REFINER_OUTPUTS:
        output = output_by_name.get(output_name)
        if (
            not isinstance(output, dict)
            or output.get("type") != "tensor(float)"
            or not isinstance(output.get("shape"), list)
            or len(output["shape"]) != rank
            or output["shape"] != expected_output_shapes[output_name]
        ):
            return False
    return len(output_by_name) == len(EXPECTED_REFINER_OUTPUTS)


def _artifact_identity_matches(
    identity: Any,
    records: list[dict[str, Any]],
    artifact_paths: list[Path],
    *,
    expected_sha256: str | None = None,
) -> bool:
    if (
        not isinstance(identity, dict)
        or identity.get("schema_version") != ARTIFACT_BUNDLE_SCHEMA_VERSION
        or not _is_sha256(identity.get("sha256"))
        or not _is_sha256(identity.get("bundle_sha256"))
        or not isinstance(identity.get("bytes"), int)
        or not isinstance(identity.get("total_bytes"), int)
        or not isinstance(identity.get("files"), list)
        or not identity["files"]
    ):
        return False
    sha256 = identity["sha256"]
    if expected_sha256 is not None and sha256 != expected_sha256:
        return False
    main_matches = any(
        record.get("exists") is True
        and record.get("sha256") == sha256
        and record.get("bytes") == identity.get("bytes")
        for record in records
    )
    if not main_matches:
        return False
    artifact_path = _artifact_path_for_identity(identity, artifact_paths)
    if artifact_path is None:
        return False
    actual = _artifact_bundle_identity(artifact_path)
    return (
        identity.get("bytes") == actual.get("bytes")
        and identity.get("sha256") == actual.get("sha256")
        and identity.get("total_bytes") == actual.get("total_bytes")
        and identity.get("bundle_sha256") == actual.get("bundle_sha256")
        and _canonical_bundle_files(identity.get("files"))
        == _canonical_bundle_files(actual.get("files"))
    )


def _canonical_bundle_files(files: Any) -> list[dict[str, Any]] | None:
    if not isinstance(files, list):
        return None
    canonical: list[dict[str, Any]] = []
    relative_paths: list[str] = []
    for record in files:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("relative_path"), str)
            or not record["relative_path"]
            or record.get("role") not in {"onnx", "external_data", "metadata"}
            or record.get("exists") is not True
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or not _is_sha256(record.get("sha256"))
        ):
            return None
        relative_paths.append(record["relative_path"])
        canonical.append(
            {
                "relative_path": record["relative_path"],
                "role": record["role"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    if len(relative_paths) != len(set(relative_paths)):
        return None
    return sorted(canonical, key=lambda record: record["relative_path"])


def _metadata_file_bound(
    identity: Any,
    metadata_path: Path | None,
) -> bool:
    if not isinstance(identity, dict) or metadata_path is None or not metadata_path.is_file():
        return False
    metadata_record = {
        "relative_path": metadata_path.name,
        "role": "metadata",
        "bytes": metadata_path.stat().st_size,
        "sha256": _sha256(metadata_path),
    }
    files = _canonical_bundle_files(identity.get("files"))
    return isinstance(files, list) and metadata_record in files


def _source_artifact_identity_valid(identity: Any) -> bool:
    return (
        isinstance(identity, dict)
        and isinstance(identity.get("path"), str)
        and bool(identity["path"])
        and isinstance(identity.get("bytes"), int)
        and identity["bytes"] > 0
        and _is_sha256(identity.get("sha256"))
    )


def _selector_metadata_contract(
    metadata: dict[str, Any] | None,
    selector_config: dict[str, Any],
) -> dict[str, bool]:
    payload = metadata if isinstance(metadata, dict) else {}
    optimizer = payload.get("optimizer", {})
    run_config = payload.get("run_config", {})
    production = payload.get("production", {})
    guidance = payload.get("guidance", {})
    selected = payload.get("selected_checkpoint", {})
    metrics = selected.get("metrics", {}) if isinstance(selected, dict) else {}
    selected_epoch = selected.get("epoch") if isinstance(selected, dict) else None
    epochs = run_config.get("epochs") if isinstance(run_config, dict) else None
    metrics_valid = (
        isinstance(metrics, dict)
        and all(
            _is_finite_number(metrics.get(name))
            and 0.0 <= float(metrics[name]) <= 1.0
            for name in EXPECTED_SELECTOR_CHECKPOINT_METRICS
        )
        and _numeric_equal(
            metrics.get("validation_score"),
            metrics.get("validation_tail_score"),
        )
    )
    return {
        "schema": (
            payload.get("schema_version") == SELECTOR_METADATA_SCHEMA_VERSION
            and isinstance(payload.get("architecture"), str)
            and payload["architecture"].startswith(
                "generalized-v20-edgegraph-selector-"
            )
            and payload.get("onnx_input_name") == "image"
            and payload.get("onnx_output_name") == "mask_logits"
        ),
        "seed": (
            isinstance(payload.get("seed"), int)
            and not isinstance(payload.get("seed"), bool)
            and payload["seed"] >= 0
        ),
        "optimizer_run_config": (
            isinstance(optimizer, dict)
            and str(optimizer.get("name", "")).lower() == "adamw"
            and _is_finite_number(optimizer.get("learning_rate"))
            and float(optimizer["learning_rate"]) > 0.0
            and _is_finite_number(optimizer.get("weight_decay"))
            and float(optimizer["weight_decay"]) >= 0.0
            and isinstance(run_config, dict)
            and isinstance(run_config.get("architecture"), str)
            and bool(run_config["architecture"])
            and run_config.get("image_size") == selector_config.get("input_width")
            and run_config.get("input_channels")
            == selector_config.get("input_channels")
            and isinstance(run_config.get("base_channels"), int)
            and run_config["base_channels"] > 0
            and isinstance(run_config.get("batch_size"), int)
            and run_config["batch_size"] > 0
            and isinstance(epochs, int)
            and epochs > 0
            and isinstance(run_config.get("validation_count"), int)
            and run_config["validation_count"] > 0
        ),
        "resume_artifact": _source_artifact_identity_valid(
            payload.get("resume_artifact")
        ),
        "production": (
            isinstance(production, dict)
            and _numeric_equal(production.get("threshold"), 0.45)
            and production.get("output_activation") == "logits"
            and production.get("input_width") == selector_config.get("input_width")
            and production.get("input_height") == selector_config.get("input_height")
            and production.get("input_channels")
            == selector_config.get("input_channels")
        ),
        "guidance": (
            isinstance(guidance, dict)
            and guidance.get("training_policy") == "automatic-heavy"
            and guidance.get("validation_policy") == "none"
        ),
        "selected_checkpoint": (
            isinstance(selected, dict)
            and _source_artifact_identity_valid(selected.get("artifact"))
            and isinstance(selected_epoch, int)
            and not isinstance(selected_epoch, bool)
            and selected_epoch >= 0
            and isinstance(epochs, int)
            and selected_epoch <= epochs
            and metrics_valid
        ),
    }


def _manifest_evidence(
    manifest: dict[str, Any],
    manifest_path: Path | None,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    manifest_from_file: dict[str, Any] | None = None
    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest_from_file = _load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            manifest_from_file = None
    samples = manifest.get("samples")
    sample_ids = [
        sample.get("sample_id")
        for sample in samples
        if isinstance(sample, dict)
    ] if isinstance(samples, list) else []
    actual_sha256 = (
        _sha256(manifest_path)
        if manifest_path is not None and manifest_path.is_file()
        else None
    )
    return {
        "path": str(manifest_path) if manifest_path is not None else None,
        "actual_sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "sha256_matches": actual_sha256 == expected_sha256,
        "parsed_content_matches": manifest_from_file == manifest,
        "sample_count": len(samples) if isinstance(samples, list) else None,
        "sample_ids": sample_ids,
        "sample_ids_valid": (
            isinstance(samples, list)
            and len(sample_ids) == len(samples)
            and all(isinstance(sample_id, str) and bool(sample_id) for sample_id in sample_ids)
        ),
        "sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
    }


def _training_manifest_evidence(
    manifest_path: Path | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    dataset = metadata.get("training_dataset", {}) if isinstance(metadata, dict) else {}
    exists = manifest_path is not None and manifest_path.is_file()
    payload: dict[str, Any] | None = None
    error: str | None = None
    if exists:
        try:
            payload = _load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    samples = payload.get("samples") if isinstance(payload, dict) else None
    actual_sha256 = _sha256(manifest_path) if exists else None
    return {
        "path": str(manifest_path) if manifest_path is not None else None,
        "exists": exists,
        "error": error,
        "actual_sha256": actual_sha256,
        "metadata_sha256": dataset.get("manifest_sha256"),
        "sha256_matches": (
            _is_sha256(actual_sha256)
            and actual_sha256 == dataset.get("manifest_sha256")
        ),
        "actual_version": payload.get("version") if isinstance(payload, dict) else None,
        "metadata_version": dataset.get("version"),
        "version_matches": (
            isinstance(payload, dict)
            and payload.get("version") == dataset.get("version")
        ),
        "actual_sample_count": len(samples) if isinstance(samples, list) else None,
        "metadata_sample_count": dataset.get("sample_count"),
        "sample_count_matches": (
            isinstance(samples, list)
            and len(samples) == dataset.get("sample_count")
        ),
    }


def _native_pixel_references(report: dict[str, Any]) -> bool:
    return (
        report.get("reference_space") == "native_pixels"
        or report.get("metadata", {}).get("reference_space") == "native_pixels"
        or report.get("summary", {}).get("native_pixel_references") is True
    )


REAL_REQUIRED_METRICS = (
    "iou",
    "boundary_f1_1px",
    "boundary_distance_p95_px",
    "boundary_distance_p99_px",
    "boundary_distance_max_px",
    "corner_f1",
)


def _real_rows_profile(report: dict[str, Any]) -> dict[str, Any]:
    scores = report.get("scores")
    if not isinstance(scores, list):
        scores = []
    active = [
        score
        for score in scores
        if isinstance(score, dict) and score.get("status") == "active"
    ]
    metric_values = {
        name: _metric_values(active, name)
        for name in REAL_REQUIRED_METRICS
    }
    durations = [
        float(score["duration_s"])
        for score in active
        if _is_finite_number(score.get("duration_s"))
        and float(score["duration_s"]) >= 0.0
    ]
    slugs = [
        score.get("slug")
        for score in scores
        if isinstance(score, dict)
    ]
    return {
        "rows": scores,
        "active_rows": active,
        "fixture_count": len(scores),
        "scored_fixtures": len(active),
        "failed_fixtures": len(scores) - len(active),
        "all_active": len(active) == len(scores),
        "slugs": slugs,
        "slugs_valid": (
            len(slugs) == len(scores)
            and all(isinstance(slug, str) and bool(slug) for slug in slugs)
        ),
        "slugs_unique": len(slugs) == len(set(slugs)),
        "complete_metrics": all(
            len(values) == len(active)
            for values in metric_values.values()
        ),
        "complete_durations": len(durations) == len(active),
        "average_iou": _mean(metric_values["iou"]),
        "min_iou": (
            round(min(metric_values["iou"]), 6)
            if metric_values["iou"]
            else None
        ),
        "p95_boundary_distance_px": _quantile(
            metric_values["boundary_distance_p95_px"], 0.95
        ),
        "p99_boundary_distance_px": _quantile(
            metric_values["boundary_distance_p99_px"], 0.95
        ),
        "mean_corner_f1": _mean(metric_values["corner_f1"]),
        "p95_duration_s": _quantile(durations, 0.95),
        "max_duration_s": max(durations) if durations else None,
        "topology_error_count": sum(
            score.get("metrics", {}).get("topology_matches") is not True
            for score in active
        ),
        "raster_topology_error_count": sum(
            score.get("metrics", {}).get("raster_topology_matches") is not True
            for score in active
        ),
        "geometry_invalid_count": sum(
            score.get("metrics", {}).get("geometry_valid") is not True
            or score.get("metrics", {}).get("geometry_empty") is not False
            or score.get("metrics", {}).get("reference_geometry_valid") is not True
            for score in active
        ),
    }


def _real_summary_matches_rows(
    summary: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    exact_fields = (
        "fixture_count",
        "scored_fixtures",
        "failed_fixtures",
        "topology_error_count",
        "raster_topology_error_count",
        "geometry_invalid_count",
    )
    numeric_fields = (
        "average_iou",
        "min_iou",
        "p95_boundary_distance_px",
        "p99_boundary_distance_px",
        "mean_corner_f1",
        "p95_duration_s",
        "max_duration_s",
    )
    return (
        summary.get("complete_metric_rows") is profile["complete_metrics"]
        and all(summary.get(field) == profile.get(field) for field in exact_fields)
        and all(
            _numeric_equal(summary.get(field), profile.get(field))
            for field in numeric_fields
        )
    )


def _real_report_contract(
    report: dict[str, Any],
    profile: dict[str, Any],
    *,
    expected_model_variant: str,
) -> bool:
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        return False
    rows = profile["rows"]
    return (
        report.get("schema_version") == NATIVE_REPORT_SCHEMA_VERSION
        and report.get("model_variant") == expected_model_variant
        and report.get("reference_space") == "native_pixels"
        and metadata.get("reference_space") == "native_pixels"
        and metadata.get("truth_source") == "exact_svg_vector_path"
        and metadata.get("automatic_production_extraction") is True
        and metadata.get("oracle_hints") is False
        and metadata.get("cache_enabled") is False
        and metadata.get("timing_profile") == "warm"
        and _is_sha256(metadata.get("source_sha256"))
        and profile["fixture_count"] == EXPECTED_REAL_CONDITION_COUNT
        and profile["scored_fixtures"] == EXPECTED_REAL_CONDITION_COUNT
        and profile["failed_fixtures"] == 0
        and profile["all_active"]
        and profile["slugs_valid"]
        and profile["slugs_unique"]
        and profile["complete_metrics"]
        and profile["complete_durations"]
        and profile["topology_error_count"] == 0
        and profile["raster_topology_error_count"] == 0
        and profile["geometry_invalid_count"] == 0
        and all(
            isinstance(score.get("condition"), dict)
            and _is_sha256(score.get("raster_sha256"))
            and score.get("reference", {}).get("space") == "native_pixels"
            and score.get("reference", {}).get("source") == "exact_svg_vector_path"
            and score.get("result", {}).get("selected_model_variant")
            == expected_model_variant
            for score in rows
        )
    )


def _real_pairing_matches(
    candidate_profile: dict[str, Any],
    baseline_profile: dict[str, Any],
) -> bool:
    candidate_rows = {
        score.get("slug"): score
        for score in candidate_profile["rows"]
        if isinstance(score, dict) and isinstance(score.get("slug"), str)
    }
    baseline_rows = {
        score.get("slug"): score
        for score in baseline_profile["rows"]
        if isinstance(score, dict) and isinstance(score.get("slug"), str)
    }
    if set(candidate_rows) != set(baseline_rows):
        return False
    return all(
        candidate_rows[slug].get("condition") == baseline_rows[slug].get("condition")
        and candidate_rows[slug].get("raster_sha256")
        == baseline_rows[slug].get("raster_sha256")
        and candidate_rows[slug].get("reference")
        == baseline_rows[slug].get("reference")
        for slug in candidate_rows
    )


def _synthetic_geometry_valid(rows: list[dict[str, Any]]) -> bool:
    return all(
        row.get("geometry", {}).get("is_present") is True
        and row.get("geometry", {}).get("is_empty") is False
        and row.get("geometry", {}).get("is_valid") is True
        and row.get("geometry", {}).get("geometry_type") in {"Polygon", "MultiPolygon"}
        and _is_finite_number(row.get("geometry", {}).get("area"))
        and float(row["geometry"]["area"]) > 0.0
        for row in rows
    )


def build_promotion_report(
    *,
    synthetic: dict[str, Any],
    manifest: dict[str, Any],
    real: dict[str, Any],
    baseline: dict[str, Any],
    svg: dict[str, Any],
    artifacts: list[Path],
    manifest_path: Path | None = None,
    training_manifest_path: Path | None = None,
    expected_holdout_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_holdout_manifest_sha256 is None:
        expected_holdout_manifest_sha256 = LOCKED_HOLDOUT_MANIFEST_SHA256
    manifest_evidence = _manifest_evidence(
        manifest,
        manifest_path,
        expected_sha256=expected_holdout_manifest_sha256,
    )
    manifest_samples = manifest.get("samples", [])
    sample_metadata = {
        sample["sample_id"]: sample
        for sample in manifest_samples
        if isinstance(sample, dict) and isinstance(sample.get("sample_id"), str)
    }
    synthetic_rows = [
        row for row in synthetic.get("rows", []) if isinstance(row, dict)
    ]
    scored_rows = [row for row in synthetic_rows if row.get("status") == "scored"]
    scored_ids = [row.get("sample_id") for row in scored_rows]
    manifest_ids = manifest_evidence["sample_ids"]
    synthetic_row_identity_matches = (
        len(synthetic_rows) == THRESHOLDS["synthetic_sample_count"]
        and len(scored_rows) == THRESHOLDS["synthetic_sample_count"]
        and all(isinstance(sample_id, str) and bool(sample_id) for sample_id in scored_ids)
        and len(scored_ids) == len(set(scored_ids))
        and set(scored_ids) == set(manifest_ids)
    )
    edge_rows = [
        row
        for row in scored_rows
        if sample_metadata.get(row.get("sample_id"), {}).get("properties", {}).get("shape_family")
        in EDGE_FAMILIES
    ]
    family_rows = {
        family: [
            row
            for row in scored_rows
            if sample_metadata.get(row.get("sample_id"), {}).get("properties", {}).get("shape_family")
            == family
        ]
        for family in SHAPE_FAMILIES
    }
    synthetic_profile = profile_summary(scored_rows)
    edge_profile = profile_summary(edge_rows)
    family_profiles = {family: profile_summary(rows) for family, rows in family_rows.items()}
    stroke_width_rows = {
        width: [
            row
            for row in scored_rows
            if float(row.get("stroke_width_px", 0.0)) == width
        ]
        for width in (0.0, 1.0, 2.0, 3.0)
    }
    stroke_width_profiles = {
        f"{width:g}px": profile_summary(rows)
        for width, rows in stroke_width_rows.items()
    }

    real_summary = real.get("summary", {})
    baseline_summary = baseline.get("summary", {})
    real_profile = _real_rows_profile(real)
    baseline_profile = _real_rows_profile(baseline)
    real_scores = real_profile["active_rows"]
    durations = [
        float(score["duration_s"])
        for score in real_scores
        if _is_finite_number(score.get("duration_s"))
    ]
    real_average_iou = real_profile.get("average_iou")
    baseline_average_iou = baseline_profile.get("average_iou")
    real_improvement = (
        float(real_average_iou) - float(baseline_average_iou)
        if isinstance(real_average_iou, (int, float))
        and isinstance(baseline_average_iou, (int, float))
        else None
    )

    svg_summary = svg.get("summary", {})
    observed_svg = svg.get("observed_svg")
    expanded_artifacts = _artifact_paths(artifacts)
    missing_artifacts = [str(path) for path in expanded_artifacts if not path.is_file()]
    artifact_records = _artifact_records(expanded_artifacts)
    artifact_bytes = (
        sum(path.stat().st_size for path in expanded_artifacts) if not missing_artifacts else None
    )
    refiner_artifact_path, refiner_metadata_path, refiner_metadata = _refiner_contract(
        expanded_artifacts
    )
    refiner_onnx_contract = _inspect_refiner_onnx(refiner_artifact_path)
    refiner_sha256 = (
        _sha256(refiner_artifact_path)
        if refiner_artifact_path is not None and refiner_artifact_path.is_file()
        else None
    )
    refiner_channels = (
        refiner_metadata.get("input_channels")
        if isinstance(refiner_metadata, dict)
        else None
    )
    refiner_supervision = (
        refiner_metadata.get("supervision", {})
        if isinstance(refiner_metadata, dict)
        else {}
    )
    refiner_training_dataset = (
        refiner_metadata.get("training_dataset", {})
        if isinstance(refiner_metadata, dict)
        else {}
    )
    refiner_training_augmentation = (
        refiner_metadata.get("training_augmentation", {})
        if isinstance(refiner_metadata, dict)
        else {}
    )
    training_manifest_evidence = _training_manifest_evidence(
        training_manifest_path,
        refiner_metadata or {},
    )
    synthetic_metadata = synthetic.get("metadata", {})
    synthetic_model_artifacts = (
        synthetic_metadata.get("model_artifacts", {})
        if isinstance(synthetic_metadata, dict)
        else {}
    )
    synthetic_selector_config = (
        synthetic_metadata.get("selector_config", {})
        if isinstance(synthetic_metadata, dict)
        else {}
    )
    selector_identity = (
        synthetic_model_artifacts.get("selector")
        if isinstance(synthetic_model_artifacts, dict)
        else None
    )
    selector_artifact_path, selector_metadata_path, selector_metadata = (
        _selector_contract(expanded_artifacts, selector_identity)
    )
    selector_onnx_contract = _inspect_selector_onnx(selector_artifact_path)
    selector_metadata_checks = _selector_metadata_contract(
        selector_metadata,
        synthetic_selector_config
        if isinstance(synthetic_selector_config, dict)
        else {},
    )
    selector_training_manifest_evidence = _training_manifest_evidence(
        training_manifest_path,
        selector_metadata or {},
    )

    family_counts_pass = all(
        profile["sample_count"] == THRESHOLDS["synthetic_family_count"]
        for profile in family_profiles.values()
    )
    family_topology_pass = all(
        profile["topology_error_count"] == 0 for profile in family_profiles.values()
    )
    stroke_width_balance_pass = all(
        profile["sample_count"] == THRESHOLDS["synthetic_stroke_group_count"]
        for profile in stroke_width_profiles.values()
    )
    stroke_width_quality_pass = all(
        _is_at_least(
            profile["p05_boundary_f1_1px"],
            THRESHOLDS["synthetic_stroke_group_p05_boundary_f1_1px"],
        )
        and _is_at_least(
            profile["mean_corner_f1"],
            THRESHOLDS["synthetic_stroke_group_mean_corner_f1"],
        )
        and profile["topology_error_count"] == 0
        for profile in stroke_width_profiles.values()
    )
    gates = {
        "holdout_manifest_locked_sha256": manifest_evidence["sha256_matches"],
        "holdout_manifest_parsed_content_matches": manifest_evidence[
            "parsed_content_matches"
        ],
        "holdout_manifest_exact_count": (
            manifest_evidence["sample_count"] == THRESHOLDS["synthetic_sample_count"]
        ),
        "holdout_manifest_unique_valid_ids": (
            manifest_evidence["sample_ids_valid"]
            and manifest_evidence["sample_ids_unique"]
        ),
        "synthetic_rows_match_locked_manifest": synthetic_row_identity_matches,
        "synthetic_report_schema": (
            synthetic.get("schema_version") == SYNTHETIC_REPORT_SCHEMA_VERSION
        ),
        "synthetic_edgegraph_extractor": synthetic.get("extractor") == "edgegraph",
        "synthetic_automatic_no_oracle_guidance": (
            isinstance(synthetic_metadata, dict)
            and synthetic_metadata.get("guidance_mode") == "automatic"
            and synthetic_metadata.get("guided") is False
            and synthetic_metadata.get("oracle_hints") is False
        ),
        "synthetic_production_inference_route": (
            isinstance(synthetic_metadata, dict)
            and synthetic_metadata.get("automatic_inference_route")
            == AUTOMATIC_EDGEGRAPH_ROUTE
        ),
        "synthetic_production_selector_config": (
            isinstance(synthetic_selector_config, dict)
            and synthetic_selector_config.get("input_channels") == 5
            and synthetic_selector_config.get("output_activation") == "logits"
            and _numeric_equal(
                synthetic_selector_config.get("threshold"),
                0.45,
            )
            and synthetic_selector_config.get("input_width") == 320
            and synthetic_selector_config.get("input_height") == 320
        ),
        "synthetic_report_manifest_sha256": (
            isinstance(synthetic_metadata, dict)
            and synthetic_metadata.get("manifest_sha256")
            == manifest_evidence["actual_sha256"]
        ),
        "synthetic_selector_artifact_bound": _artifact_identity_matches(
            synthetic_model_artifacts.get("selector"),
            artifact_records,
            expanded_artifacts,
        ),
        "synthetic_refiner_artifact_bound": _artifact_identity_matches(
            synthetic_model_artifacts.get("refiner"),
            artifact_records,
            expanded_artifacts,
            expected_sha256=refiner_sha256,
        ),
        "selector_onnx_graph_contract": _onnx_shape_matches_selector(
            selector_onnx_contract,
            synthetic_selector_config,
        ),
        "selector_metadata_bytes_bound": _metadata_file_bound(
            selector_identity,
            selector_metadata_path,
        ),
        "selector_metadata_schema": selector_metadata_checks["schema"],
        "selector_training_seed": selector_metadata_checks["seed"],
        "selector_optimizer_run_config": selector_metadata_checks[
            "optimizer_run_config"
        ],
        "selector_resume_artifact_identity": selector_metadata_checks[
            "resume_artifact"
        ],
        "selector_production_contract": selector_metadata_checks["production"],
        "selector_automatic_guidance_contract": selector_metadata_checks[
            "guidance"
        ],
        "selector_selected_checkpoint_contract": selector_metadata_checks[
            "selected_checkpoint"
        ],
        "selector_training_manifest_bytes_bound": (
            selector_training_manifest_evidence["sha256_matches"]
            and selector_training_manifest_evidence["version_matches"]
            and selector_training_manifest_evidence["sample_count_matches"]
        ),
        "synthetic_report_declared_pass": synthetic.get("passed") is True,
        "synthetic_no_failures": synthetic.get("summary", {}).get("failure_count") == 0,
        "synthetic_summary_exact_counts": (
            synthetic.get("summary", {}).get("sample_count")
            == THRESHOLDS["synthetic_sample_count"]
            and synthetic.get("summary", {}).get("scored_count")
            == THRESHOLDS["synthetic_sample_count"]
            and synthetic.get("summary", {}).get("failure_count") == 0
        ),
        "synthetic_sample_count": synthetic_profile["sample_count"]
        == THRESHOLDS["synthetic_sample_count"],
        "synthetic_family_balance": family_counts_pass,
        "synthetic_stroke_width_balance": stroke_width_balance_pass,
        "synthetic_stroke_width_invariance": stroke_width_quality_pass,
        "synthetic_complete_edge_metrics": not synthetic_profile["missing_metrics"],
        "synthetic_geometry_valid": _synthetic_geometry_valid(scored_rows),
        "synthetic_mean_iou": _is_at_least(
            synthetic_profile["mean_iou"], THRESHOLDS["synthetic_mean_iou"]
        ),
        "synthetic_p05_iou": _is_at_least(
            synthetic_profile["p05_iou"], THRESHOLDS["synthetic_p05_iou"]
        ),
        "synthetic_p05_boundary_f1_1px": _is_at_least(
            synthetic_profile["p05_boundary_f1_1px"],
            THRESHOLDS["synthetic_p05_boundary_f1_1px"],
        ),
        "synthetic_p95_boundary_distance_px": _is_at_most(
            synthetic_profile["p95_boundary_distance_px"],
            THRESHOLDS["synthetic_p95_boundary_distance_px"],
        ),
        "synthetic_zero_topology_errors": synthetic_profile["topology_error_count"] == 0,
        "synthetic_zero_topology_errors_per_family": family_topology_pass,
        "synthetic_zero_raster_topology_errors": (
            synthetic_profile["raster_topology_error_count"] == 0
            and synthetic_profile["raster_topology_missing_count"] == 0
        ),
        "edge_sample_count": edge_profile["sample_count"] == THRESHOLDS["edge_sample_count"],
        "edge_complete_metrics": not edge_profile["missing_metrics"],
        "edge_p05_boundary_f1_1px": _is_at_least(
            edge_profile["p05_boundary_f1_1px"], THRESHOLDS["edge_p05_boundary_f1_1px"]
        ),
        "edge_p95_boundary_distance_px": _is_at_most(
            edge_profile["p95_boundary_distance_px"],
            THRESHOLDS["edge_p95_boundary_distance_px"],
        ),
        "edge_p95_boundary_distance_p99_px": _is_at_most(
            edge_profile["p95_boundary_distance_p99_px"],
            THRESHOLDS["edge_p95_boundary_distance_p99_px"],
        ),
        "edge_max_boundary_distance_px": _is_at_most(
            edge_profile["max_boundary_distance_px"],
            THRESHOLDS["edge_max_boundary_distance_px"],
        ),
        "edge_mean_corner_f1": _is_at_least(
            edge_profile["mean_corner_f1"], THRESHOLDS["edge_mean_corner_f1"]
        ),
        "edge_p05_corner_f1": _is_at_least(
            edge_profile["p05_corner_f1"], THRESHOLDS["edge_p05_corner_f1"]
        ),
        "edge_corner_angular_error": _is_at_most(
            edge_profile["p95_corner_angular_error_degrees"],
            THRESHOLDS["edge_p95_corner_angular_error_degrees"],
        ),
        "edge_straight_run_rms_excess": _is_at_most(
            edge_profile["p95_straight_run_rms_excess_px"],
            THRESHOLDS["edge_p95_straight_run_rms_excess_px"],
        ),
        "edge_canonical_complexity_ratio": _is_at_most(
            edge_profile["p95_canonical_complexity_ratio"],
            THRESHOLDS["edge_p95_canonical_complexity_ratio"],
        ),
        "real_native_pixel_references": _native_pixel_references(real),
        "real_candidate_contract": _real_report_contract(
            real,
            real_profile,
            expected_model_variant=EDGEGRAPH_MODEL_VARIANT,
        ),
        "real_baseline_contract": _real_report_contract(
            baseline,
            baseline_profile,
            expected_model_variant=BOUNDARYFIELD_MODEL_VARIANT,
        ),
        "real_candidate_summary_matches_rows": _real_summary_matches_rows(
            real_summary,
            real_profile,
        ),
        "real_baseline_summary_matches_rows": _real_summary_matches_rows(
            baseline_summary,
            baseline_profile,
        ),
        "real_candidate_baseline_pairing": _real_pairing_matches(
            real_profile,
            baseline_profile,
        ),
        "real_candidate_baseline_source_match": (
            real.get("metadata", {}).get("source_sha256")
            == baseline.get("metadata", {}).get("source_sha256")
            and real.get("metadata", {}).get("source_svg")
            == baseline.get("metadata", {}).get("source_svg")
            and real.get("metadata", {}).get("rasterizer")
            == baseline.get("metadata", {}).get("rasterizer")
        ),
        "real_all_active_pass": real_profile["failed_fixtures"] == 0
        and real_profile["scored_fixtures"] == THRESHOLDS["real_fixture_count"],
        "real_average_iou": _is_at_least(real_average_iou, THRESHOLDS["real_average_iou"]),
        "real_min_iou": _is_at_least(real_profile.get("min_iou"), THRESHOLDS["real_min_iou"]),
        "real_improves_baseline": _is_at_least(
            real_improvement, THRESHOLDS["real_improvement_over_baseline"]
        ),
        "real_p95_boundary_distance_px": _is_at_most(
            real_profile.get("p95_boundary_distance_px"),
            THRESHOLDS["real_p95_boundary_distance_px"],
        ),
        "real_p99_boundary_distance_px": _is_at_most(
            real_profile.get("p99_boundary_distance_px"),
            THRESHOLDS["real_p99_boundary_distance_px"],
        ),
        "real_mean_corner_f1": _is_at_least(
            real_profile.get("mean_corner_f1"), THRESHOLDS["real_mean_corner_f1"]
        ),
        "warm_p95_duration_s": real_profile["complete_durations"]
        and _is_at_most(real_profile["p95_duration_s"], THRESHOLDS["warm_p95_duration_s"]),
        "max_duration_s": real_profile["complete_durations"]
        and bool(durations)
        and _is_at_most(real_profile["max_duration_s"], THRESHOLDS["max_duration_s"]),
        "svg_fixture_count": _is_at_least(
            svg_summary.get("fixture_count"), THRESHOLDS["svg_fixture_count"]
        ),
        "svg_no_failures": svg_summary.get("failure_count") == 0,
        "svg_exact_topology": svg_summary.get("topology_error_count") == 0,
        "svg_corner_f1": _is_at_least(
            svg_summary.get("min_corner_f1"), THRESHOLDS["svg_min_corner_f1"]
        ),
        "svg_path_deviation": _is_at_most(
            svg_summary.get("max_path_deviation_px"),
            THRESHOLDS["svg_max_path_deviation_px"],
        ),
        "svg_observed_source_bound": (
            isinstance(observed_svg, dict)
            and _is_sha256(observed_svg.get("source_sha256"))
            and observed_svg.get("source_sha256")
            == real.get("metadata", {}).get("source_sha256")
        ),
        "svg_observed_valid_geometry": (
            isinstance(observed_svg, dict)
            and observed_svg.get("valid") is True
            and observed_svg.get("is_empty") is False
            and observed_svg.get("geometry_type") == "Polygon"
            and _is_finite_number(observed_svg.get("area"))
            and float(observed_svg["area"]) > 0.0
        ),
        "svg_observed_tolerance": (
            isinstance(observed_svg, dict)
            and _is_at_most(
                observed_svg.get("flatten_tolerance_px"),
                THRESHOLDS["svg_observed_max_flatten_tolerance_px"],
            )
        ),
        "svg_observed_vertices": (
            isinstance(observed_svg, dict)
            and _is_at_least(
                observed_svg.get("vertices"),
                THRESHOLDS["svg_observed_min_vertices"],
            )
            and _is_at_least(
                (
                    observed_svg.get("line_segment_count", 0)
                    + observed_svg.get("curve_segment_count", 0)
                )
                if isinstance(observed_svg.get("line_segment_count"), int)
                and isinstance(observed_svg.get("curve_segment_count"), int)
                else None,
                1,
            )
        ),
        "artifacts_present": bool(expanded_artifacts) and not missing_artifacts,
        "artifact_budget": _is_at_most(artifact_bytes, THRESHOLDS["artifact_bytes"]),
        "refiner_onnx_graph_contract": _onnx_shape_matches_vector_v3(
            refiner_onnx_contract
        ),
        "refiner_vector_v3_contract": bool(refiner_metadata)
        and refiner_metadata.get("architecture")
        == "generalized-v20-edgegraph-stripnet-vector-v3"
        and refiner_metadata.get("channels") == len(VECTOR_V3_INPUT_CHANNELS)
        and refiner_channels == list(VECTOR_V3_INPUT_CHANNELS)
        and refiner_metadata.get("onnx_input_name") == VECTOR_V3_ONNX_INPUT_NAME
        and refiner_metadata.get("coarse_probability_input") is False,
        "refiner_learned_offsets_preserved": bool(refiner_metadata)
        and refiner_metadata.get("conservative_offset_head") is False
        and refiner_metadata.get("learned_offset_logits_preserved") is True,
        "refiner_exact_vector_supervision": bool(refiner_metadata)
        and refiner_supervision.get("target_version")
        == "geojson-pixel-ray-segment-v1",
        "refiner_full_stroke_receptive_field": bool(refiner_metadata)
        and refiner_metadata.get("profile_dilations") == [1, 2, 4]
        and _is_at_least(refiner_metadata.get("profile_receptive_field_px"), 12.0),
        "refiner_shifted_center_training_contract": bool(refiner_metadata)
        and refiner_training_augmentation.get("profile_center_shift_distribution")
        == "uniform"
        and refiner_training_augmentation.get("profile_center_shift_max_abs_px")
        == 6.0
        and refiner_training_augmentation.get("profile_center_shift_scope")
        == "global_per_sample"
        and refiner_training_augmentation.get("inside_direction_reference")
        == "unshifted-contour-center-v1",
        "refiner_raster_v2_training_contract": bool(refiner_metadata)
        and refiner_training_dataset.get("version")
        == "synthetic-generator-v20-centered-stroke-raster-v2"
        and _is_at_least(refiner_training_dataset.get("sample_count"), 512)
        and isinstance(refiner_training_dataset.get("manifest_sha256"), str)
        and len(refiner_training_dataset["manifest_sha256"]) == 64,
        "refiner_training_manifest_bytes_bound": (
            training_manifest_evidence["sha256_matches"]
            and training_manifest_evidence["version_matches"]
            and training_manifest_evidence["sample_count_matches"]
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "thresholds": THRESHOLDS,
        "synthetic": synthetic_profile,
        "edge_pressure": edge_profile,
        "shape_families": family_profiles,
        "stroke_widths": stroke_width_profiles,
        "real": {
            **{
                key: value
                for key, value in real_profile.items()
                if key not in {"rows", "active_rows"}
            },
            "native_pixel_references": _native_pixel_references(real),
            "improvement_over_baseline": real_improvement,
        },
        "baseline_real": {
            key: value
            for key, value in baseline_profile.items()
            if key not in {"rows", "active_rows"}
        },
        "svg": svg_summary,
        "observed_svg": observed_svg,
        "holdout_manifest": manifest_evidence,
        "training_manifest": training_manifest_evidence,
        "selector_training_manifest": selector_training_manifest_evidence,
        "artifacts": {
            "paths": [str(path) for path in expanded_artifacts],
            "missing": missing_artifacts,
            "total_bytes": artifact_bytes,
            "records": artifact_records,
            "selector_artifact_path": (
                str(selector_artifact_path)
                if selector_artifact_path is not None
                else None
            ),
            "selector_metadata_path": (
                str(selector_metadata_path)
                if selector_metadata_path is not None
                else None
            ),
            "selector_metadata": selector_metadata,
            "selector_onnx_contract": selector_onnx_contract,
            "refiner_artifact_path": (
                str(refiner_artifact_path) if refiner_artifact_path is not None else None
            ),
            "refiner_sha256": refiner_sha256,
            "refiner_metadata_path": (
                str(refiner_metadata_path) if refiner_metadata_path is not None else None
            ),
            "refiner_metadata": refiner_metadata,
            "refiner_onnx_contract": refiner_onnx_contract,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_promotion_report(
        synthetic=_load_json(args.synthetic_report),
        manifest=_load_json(args.manifest),
        real=_load_json(args.real_report),
        baseline=_load_json(args.baseline_real_report),
        svg=_load_json(args.svg_report),
        artifacts=args.artifact,
        manifest_path=args.manifest,
        training_manifest_path=args.training_manifest,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
