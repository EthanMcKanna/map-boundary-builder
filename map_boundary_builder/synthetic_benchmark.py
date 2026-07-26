"""Benchmark the unified segmentation path against synthetic image/mask artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import shape

from .evaluation import (
    area_ratio,
    boundary_distance_summary_px,
    boundary_f1,
    boundary_iou,
    centroid_distance_px,
    corner_f1,
    dice,
    geometry_validity_summary,
    geometry_corner_f1,
    geometry_complexity_comparison,
    geometry_complexity_summary,
    geometry_topology_signature,
    iou,
    precision,
    recall,
    topology_signature,
)
from .extract import ExtractionResult, load_rgb
from .segment import default_model_path, segment_image
from .synthetic import SyntheticDatasetManifest, generate_synthetic_dataset


REPORT_SCHEMA_VERSION = "synthetic-benchmark-v4"
ARTIFACT_BUNDLE_SCHEMA_VERSION = "onnx-artifact-bundle-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="map-boundary-synthetic-benchmark",
        description="Generate or score synthetic boundary fixtures with exact mask labels.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Output report path.")
    parser.add_argument("--generate", action="store_true", help="Generate a synthetic dataset before scoring.")
    parser.add_argument("--count", type=int, default=24, help="Sample count for --generate.")
    parser.add_argument("--seed", type=int, default=1, help="Dataset seed for --generate.")
    parser.add_argument("--width", type=int, default=960, help="Generated sample width.")
    parser.add_argument("--height", type=int, default=640, help="Generated sample height.")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N manifest samples.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="ONNX mask model to score. Defaults to the packaged model.",
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=None,
        help="Override the model probability threshold.",
    )
    parser.add_argument("--guided", action="store_true", help="Score with manifest-derived seed and color guidance.")
    parser.add_argument("--min-iou", type=float, default=0.70, help="Hard gate for every scored row.")
    parser.add_argument("--mean-iou", type=float, default=0.85, help="Hard gate for report mean IoU.")
    parser.add_argument("--p05-iou", type=float, default=0.0, help="Hard gate for fifth-percentile IoU.")
    parser.add_argument("--p05-boundary-iou-2px", type=float, default=0.0)
    parser.add_argument("--max-p95-boundary-distance-px", type=float, default=float("inf"))
    parser.add_argument("--min-corner-f1", type=float, default=0.0)
    parser.add_argument("--max-topology-errors", type=int, default=2**31 - 1)
    parser.add_argument("--max-p95-duration-s", type=float, default=float("inf"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir
    if args.generate:
        manifest = generate_synthetic_dataset(
            dataset_dir,
            count=args.count,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
        manifest_path = dataset_dir / "manifest.json"
    else:
        manifest_path = args.manifest or dataset_dir / "manifest.json"
        manifest = SyntheticDatasetManifest.read_json(manifest_path)
    if args.limit > 0:
        manifest = SyntheticDatasetManifest(
            name=manifest.name,
            version=manifest.version,
            samples=list(manifest.samples)[: args.limit],
            properties={**manifest.properties, "score_limit": args.limit},
        )

    report = score_synthetic_manifest(
        manifest,
        dataset_dir,
        model_path=args.model_path,
        model_threshold=args.model_threshold or None,
        guided=args.guided,
        manifest_path=manifest_path,
    )
    report["thresholds"] = {
        "min_iou": args.min_iou,
        "mean_iou": args.mean_iou,
        "p05_iou": args.p05_iou,
        "p05_boundary_iou_2px": args.p05_boundary_iou_2px,
        "max_p95_boundary_distance_px": args.max_p95_boundary_distance_px,
        "min_corner_f1": args.min_corner_f1,
        "max_topology_errors": args.max_topology_errors,
        "max_p95_duration_s": args.max_p95_duration_s,
    }
    report["passed"] = _passes_thresholds(
        report,
        min_iou=args.min_iou,
        mean_iou=args.mean_iou,
        p05_iou=args.p05_iou,
        p05_boundary_iou_2px=args.p05_boundary_iou_2px,
        max_p95_boundary_distance_px=args.max_p95_boundary_distance_px,
        min_corner_f1=args.min_corner_f1,
        max_topology_errors=args.max_topology_errors,
        max_p95_duration_s=args.max_p95_duration_s,
    )

    out_path = args.out or dataset_dir / "synthetic-benchmark-report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def score_synthetic_manifest(
    manifest: SyntheticDatasetManifest,
    dataset_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    model_threshold: float | None = None,
    guided: bool = False,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    rows = [
        score_synthetic_sample(
            sample,
            root,
            model_path=model_path,
            model_threshold=model_threshold,
            guided=guided,
        )
        for sample in manifest.samples
    ]
    scored_rows = [row for row in rows if row["status"] == "scored"]
    ious = [row["metrics"]["iou"] for row in scored_rows]
    boundary_ious = [row["metrics"]["boundary_iou_2px"] for row in scored_rows]
    boundary_distances = [row["metrics"]["boundary_distance_p95_px"] for row in scored_rows]
    boundary_f1_scores = [row["metrics"]["boundary_f1_1px"] for row in scored_rows]
    boundary_distance_p99 = [row["metrics"]["boundary_distance_p99_px"] for row in scored_rows]
    boundary_distance_max = [row["metrics"]["boundary_distance_max_px"] for row in scored_rows]
    corner_scores = [row["metrics"]["corner_f1"] for row in scored_rows]
    durations = [row["duration_s"] for row in scored_rows]
    failures = [row for row in rows if row["status"] != "scored"]
    summary = {
        "sample_count": len(rows),
        "scored_count": len(scored_rows),
        "failure_count": len(failures),
        "mean_iou": round(mean(ious), 6) if ious else 0.0,
        "min_iou": round(min(ious), 6) if ious else 0.0,
        "p05_iou": round(float(np.quantile(ious, 0.05)), 6) if ious else 0.0,
        "mean_boundary_iou_2px": round(mean(boundary_ious), 6) if boundary_ious else 0.0,
        "p05_boundary_iou_2px": round(float(np.quantile(boundary_ious, 0.05)), 6) if boundary_ious else 0.0,
        "p95_boundary_distance_px": round(float(np.quantile(boundary_distances, 0.95)), 6) if boundary_distances else float("inf"),
        "mean_boundary_f1_1px": round(mean(boundary_f1_scores), 6) if boundary_f1_scores else 0.0,
        "p05_boundary_f1_1px": round(float(np.quantile(boundary_f1_scores, 0.05)), 6) if boundary_f1_scores else 0.0,
        "p95_boundary_distance_p99_px": round(float(np.quantile(boundary_distance_p99, 0.95)), 6) if boundary_distance_p99 else float("inf"),
        "max_boundary_distance_px": round(max(boundary_distance_max), 6) if boundary_distance_max else float("inf"),
        "mean_corner_f1": round(mean(corner_scores), 6) if corner_scores else 0.0,
        "topology_error_count": sum(not row["metrics"]["topology_matches"] for row in scored_rows),
        "mean_duration_s": round(mean(durations), 6) if durations else 0.0,
        "p95_duration_s": round(float(np.quantile(durations, 0.95)), 6) if durations else float("inf"),
    }
    stroke_groups: dict[str, dict[str, float | int]] = {}
    for width in sorted({float(row.get("stroke_width_px", 0.0)) for row in scored_rows}):
        grouped = [row for row in scored_rows if float(row.get("stroke_width_px", 0.0)) == width]
        grouped_f1 = [float(row["metrics"]["boundary_f1_1px"]) for row in grouped]
        grouped_corners = [float(row["metrics"]["corner_f1"]) for row in grouped]
        stroke_groups[f"{width:g}px"] = {
            "sample_count": len(grouped),
            "mean_boundary_f1_1px": round(mean(grouped_f1), 6),
            "p05_boundary_f1_1px": round(float(np.quantile(grouped_f1, 0.05)), 6),
            "mean_corner_f1": round(mean(grouped_corners), 6),
        }
    summary["stroke_width_groups"] = stroke_groups
    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else None
    resolved_model_path = Path(model_path) if model_path is not None else default_model_path()
    model_resolves = model_path is not None or resolved_model_path.is_file()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest": {
            "name": manifest.name,
            "version": manifest.version,
            "properties": dict(manifest.properties),
        },
        "metadata": {
            "guidance_mode": "oracle_guided" if guided else "automatic",
            "guided": bool(guided),
            "oracle_hints": bool(guided),
            "manifest_sha256": (
                _file_sha256(resolved_manifest_path)
                if resolved_manifest_path is not None and resolved_manifest_path.is_file()
                else None
            ),
            "model_artifact": _artifact_identity(resolved_model_path) if model_resolves else None,
        },
        "extractor": "model" if model_resolves else "auto-fill",
        "model_path": str(model_path) if model_path is not None else None,
        "summary": summary,
        "rows": rows,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    artifact = Path(path)
    if not artifact.is_file():
        return {
            "schema_version": ARTIFACT_BUNDLE_SCHEMA_VERSION,
            "path": str(artifact),
            "bytes": None,
            "sha256": None,
            "total_bytes": None,
            "bundle_sha256": None,
            "files": [],
        }
    files = _artifact_bundle_files(artifact)
    records = [_artifact_file_identity(item, artifact=artifact) for item in files]
    complete = all(record["exists"] is True for record in records)
    total_bytes = (
        sum(int(record["bytes"]) for record in records)
        if complete
        else None
    )
    return {
        "schema_version": ARTIFACT_BUNDLE_SCHEMA_VERSION,
        "path": str(artifact.resolve()),
        "bytes": artifact.stat().st_size,
        "sha256": _file_sha256(artifact),
        "total_bytes": total_bytes,
        "bundle_sha256": _artifact_bundle_sha256(records) if complete else None,
        "files": records,
    }


def _artifact_bundle_files(artifact: Path) -> list[Path]:
    """Return the ONNX plus every byte-bearing companion that affects it."""

    resolved_artifact = artifact.resolve()
    companions: list[Path] = [resolved_artifact]
    metadata_path = Path(str(resolved_artifact) + ".json")
    conventional_data_path = Path(str(resolved_artifact) + ".data")
    if conventional_data_path.exists():
        companions.append(conventional_data_path)
    for external_path in _onnx_external_data_paths(resolved_artifact):
        companions.append(external_path)
    if metadata_path.exists():
        companions.append(metadata_path)
    return list(dict.fromkeys(companions))


def _onnx_external_data_paths(artifact: Path) -> list[Path]:
    try:
        import onnx

        model = onnx.load(str(artifact), load_external_data=False)
    except Exception:
        return []

    parent = artifact.parent.resolve()
    paths: list[Path] = []
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key != "location" or not entry.value:
                continue
            candidate = (parent / entry.value).resolve()
            try:
                candidate.relative_to(parent)
            except ValueError:
                # Refuse to let a model make the benchmark hash arbitrary files.
                continue
            paths.append(candidate)
    return list(dict.fromkeys(paths))


def _artifact_file_identity(path: Path, *, artifact: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative_path = str(resolved.relative_to(artifact.parent.resolve()))
    except ValueError:
        relative_path = resolved.name
    if resolved == artifact.resolve():
        role = "onnx"
    elif resolved == Path(str(artifact.resolve()) + ".json"):
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
        "sha256": _file_sha256(resolved) if exists else None,
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
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score_synthetic_sample(
    sample,
    dataset_dir: Path,
    *,
    model_path: str | Path | None = None,
    model_threshold: float | None = None,
    guided: bool = False,
) -> dict[str, Any]:
    image_path = dataset_dir / sample.artifacts.screenshot
    mask_path = dataset_dir / sample.artifacts.mask
    started = time.perf_counter()
    try:
        expected_mask = _load_mask(mask_path)
        rgb = load_rgb(image_path)
        hints = synthetic_guidance(sample, expected_mask, rgb) if guided else None
        result = segment_image(
            rgb,
            model_path=model_path,
            threshold=model_threshold,
            hints=hints,
        )
        return _scored_synthetic_row(
            sample,
            dataset_dir,
            expected_mask,
            result,
            started,
        )
    except Exception as exc:
        return {
            "sample_id": sample.sample_id,
            "variant": sample.variant,
            "overlay_style": sample.overlay_style.name,
            "status": "failed",
            "duration_s": round(time.perf_counter() - started, 6),
            "error": str(exc),
        }


def _scored_synthetic_row(
    sample,
    dataset_dir: Path,
    expected_mask: np.ndarray,
    result: ExtractionResult,
    started: float,
) -> dict[str, Any]:
    predicted_mask = result.mask.astype(bool, copy=False)
    reference_geojson = json.loads(
        (dataset_dir / sample.artifacts.geojson).read_text(encoding="utf-8")
    )
    reference_geometry = shape(reference_geojson["metadata"]["pixel_geometry"])
    return {
        "sample_id": sample.sample_id,
        "variant": sample.variant,
        "overlay_style": sample.overlay_style.name,
        "stroke_width_px": float(sample.overlay_style.stroke_width_px or 0.0),
        "status": "scored",
        "duration_s": round(time.perf_counter() - started, 6),
        "extraction": {
            "style": result.style,
            "coverage_ratio": round(result.coverage_ratio, 6),
            "confidence": round(result.confidence, 6),
            "contour_count": result.contour_count,
            "diagnostics": result.diagnostics or {},
        },
        "metrics": _score_masks(
            predicted_mask,
            expected_mask,
            predicted_geometry=result.pixel_geometry,
            reference_geometry=reference_geometry,
        ),
        "geometry": geometry_validity_summary(result.pixel_geometry),
    }


def synthetic_guidance(sample, expected_mask: np.ndarray, rgb: np.ndarray | None = None) -> dict[str, object]:
    ys, xs = np.where(expected_mask)
    hints: dict[str, object] = {}
    if len(xs):
        pick = len(xs) // 2
        hints["seed_point"] = (float(xs[pick]), float(ys[pick]))
    if rgb is not None and len(xs):
        if float(sample.overlay_style.fill_opacity) == 0.0 and sample.overlay_style.stroke_color:
            binary = expected_mask.astype(np.uint8)
            kernel = np.ones((9, 9), np.uint8)
            ring = cv2.dilate(binary, kernel, iterations=1) != cv2.erode(binary, kernel, iterations=1)
            candidates = rgb[ring].astype(np.float32)
            color = sample.overlay_style.stroke_color.lstrip("#")
            declared = np.asarray([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)
            if len(candidates):
                distances = np.linalg.norm(candidates - declared, axis=1)
                keep = max(8, len(candidates) // 5)
                target = np.median(candidates[np.argpartition(distances, keep - 1)[:keep]], axis=0)
            else:
                target = declared
        else:
            target = np.median(rgb[expected_mask], axis=0)
        hints["target_rgb"] = tuple(int(round(value)) for value in target)
    else:
        color = sample.overlay_style.fill_color.lstrip("#")
        if len(color) == 6:
            hints["target_rgb"] = tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))
    return hints


def _score_masks(
    predicted_mask: np.ndarray,
    expected_mask: np.ndarray,
    *,
    predicted_geometry=None,
    reference_geometry=None,
) -> dict[str, float]:
    raster_corners = corner_f1(predicted_mask, expected_mask, tolerance_px=2.0)
    one_pixel_boundary = boundary_f1(predicted_mask, expected_mask, tolerance_px=1.0)
    boundary_distances = boundary_distance_summary_px(predicted_mask, expected_mask)
    corners = (
        geometry_corner_f1(predicted_geometry, reference_geometry, tolerance_px=2.0)
        if predicted_geometry is not None and reference_geometry is not None
        else raster_corners
    )
    predicted_raster_topology = topology_signature(predicted_mask)
    expected_raster_topology = topology_signature(expected_mask)
    if predicted_geometry is not None and reference_geometry is not None:
        predicted_topology = geometry_topology_signature(predicted_geometry)
        expected_topology = geometry_topology_signature(reference_geometry)
    else:
        predicted_topology = predicted_raster_topology
        expected_topology = expected_raster_topology
    if predicted_geometry is not None and reference_geometry is not None:
        complexity = geometry_complexity_comparison(
            predicted_geometry,
            reference_geometry,
            max_deviation_px=0.35,
        )
    elif predicted_geometry is not None:
        absolute_complexity = geometry_complexity_summary(
            predicted_geometry,
            max_deviation_px=0.35,
        )
        complexity = {
            "predicted_straight_run_rms_px": absolute_complexity["straight_run_rms_px"],
            "reference_straight_run_rms_px": float("inf"),
            "straight_run_rms_excess_px": float("inf"),
            "predicted_excess_vertex_ratio": absolute_complexity["excess_vertex_ratio"],
            "reference_excess_vertex_ratio": float("inf"),
            "predicted_canonical_vertex_count": absolute_complexity["canonical_vertex_count"],
            "reference_canonical_vertex_count": 0,
            "canonical_complexity_ratio": float("inf"),
        }
    else:
        complexity = {
            "predicted_straight_run_rms_px": float("inf"),
            "reference_straight_run_rms_px": float("inf"),
            "straight_run_rms_excess_px": float("inf"),
            "predicted_excess_vertex_ratio": float("inf"),
            "reference_excess_vertex_ratio": float("inf"),
            "predicted_canonical_vertex_count": 0,
            "reference_canonical_vertex_count": 0,
            "canonical_complexity_ratio": float("inf"),
        }
    return {
        "iou": round(iou(predicted_mask, expected_mask), 6),
        "dice": round(dice(predicted_mask, expected_mask), 6),
        "precision": round(precision(predicted_mask, expected_mask), 6),
        "recall": round(recall(predicted_mask, expected_mask), 6),
        "area_ratio": round(area_ratio(predicted_mask, expected_mask), 6),
        "centroid_distance_px": round(centroid_distance_px(predicted_mask, expected_mask), 3),
        "boundary_iou_0px": round(boundary_iou(predicted_mask, expected_mask, tolerance_px=0), 6),
        "boundary_iou_2px": round(boundary_iou(predicted_mask, expected_mask, tolerance_px=2), 6),
        "boundary_iou_5px": round(boundary_iou(predicted_mask, expected_mask, tolerance_px=5), 6),
        "boundary_f1_1px": round(float(one_pixel_boundary["f1"]), 6),
        "boundary_precision_1px": round(float(one_pixel_boundary["precision"]), 6),
        "boundary_recall_1px": round(float(one_pixel_boundary["recall"]), 6),
        "boundary_distance_mean_px": round(boundary_distances["mean_px"], 6),
        "boundary_distance_p95_px": round(boundary_distances["p95_px"], 6),
        "boundary_distance_p99_px": round(boundary_distances["p99_px"], 6),
        "boundary_distance_max_px": round(boundary_distances["max_px"], 6),
        "corner_precision": round(corners["precision"], 6),
        "corner_recall": round(corners["recall"], 6),
        "corner_f1": round(corners["f1"], 6),
        "corner_angular_error_p95_degrees": round(float(corners["p95_angular_error_degrees"]), 6),
        "raster_corner_f1": round(raster_corners["f1"], 6),
        # Absolute self-roughness remains diagnostic. Promotion gates use the
        # reference-normalized fields below so exact authored geometry has a
        # zero/one oracle floor even when the reference itself simplifies.
        "straight_run_rms_px": round(
            float(complexity["predicted_straight_run_rms_px"]), 6
        ),
        "reference_straight_run_rms_px": round(
            float(complexity["reference_straight_run_rms_px"]), 6
        ),
        "straight_run_rms_excess_px": round(
            float(complexity["straight_run_rms_excess_px"]), 6
        ),
        "excess_vertex_ratio": round(
            float(complexity["predicted_excess_vertex_ratio"]), 6
        ),
        "reference_excess_vertex_ratio": round(
            float(complexity["reference_excess_vertex_ratio"]), 6
        ),
        "predicted_canonical_vertex_count": int(
            complexity["predicted_canonical_vertex_count"]
        ),
        "reference_canonical_vertex_count": int(
            complexity["reference_canonical_vertex_count"]
        ),
        "canonical_complexity_ratio": round(
            float(complexity["canonical_complexity_ratio"]), 6
        ),
        "topology_matches": predicted_topology == expected_topology,
        "predicted_components": predicted_topology["components"],
        "predicted_holes": predicted_topology["holes"],
        "expected_components": expected_topology["components"],
        "expected_holes": expected_topology["holes"],
        "raster_topology_matches": predicted_raster_topology == expected_raster_topology,
        "predicted_raster_components": predicted_raster_topology["components"],
        "predicted_raster_holes": predicted_raster_topology["holes"],
        "expected_raster_components": expected_raster_topology["components"],
        "expected_raster_holes": expected_raster_topology["holes"],
    }


def _load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _passes_thresholds(
    report: dict[str, Any], *, min_iou: float, mean_iou: float, p05_iou: float = 0.0,
    p05_boundary_iou_2px: float = 0.0,
    max_p95_boundary_distance_px: float = float("inf"),
    min_corner_f1: float = 0.0,
    max_topology_errors: int = 2**31 - 1,
    max_p95_duration_s: float = float("inf"),
) -> bool:
    summary = report["summary"]
    if summary["failure_count"]:
        return False
    if summary["min_iou"] < min_iou:
        return False
    if summary["mean_iou"] < mean_iou:
        return False
    if summary["p05_iou"] < p05_iou:
        return False
    if summary["p05_boundary_iou_2px"] < p05_boundary_iou_2px:
        return False
    if summary["p95_boundary_distance_px"] > max_p95_boundary_distance_px:
        return False
    if summary["mean_corner_f1"] < min_corner_f1:
        return False
    if summary["topology_error_count"] > max_topology_errors:
        return False
    if summary["p95_duration_s"] > max_p95_duration_s:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
