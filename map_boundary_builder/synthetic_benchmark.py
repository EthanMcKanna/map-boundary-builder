"""Benchmark extraction against synthetic image/mask artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import time
from typing import Any

import numpy as np
from PIL import Image

from .evaluation import (
    area_ratio,
    boundary_iou,
    centroid_distance_px,
    dice,
    geometry_validity_summary,
    iou,
    precision,
    recall,
)
from .extract import extract_service_area
from .model_extract import ModelExtractionConfig, extract_service_area_with_model
from .synthetic import SyntheticDatasetManifest, generate_synthetic_dataset


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
    parser.add_argument("--model-path", type=Path, default=None, help="Optional ONNX mask model to score.")
    parser.add_argument("--model-input-size", type=int, default=256)
    parser.add_argument("--model-input-channels", type=int, choices=(3, 5), default=3)
    parser.add_argument("--guided", action="store_true", help="Score with manifest-derived seed and color guidance.")
    parser.add_argument("--model-threshold", type=float, default=0.25)
    parser.add_argument("--min-iou", type=float, default=0.70, help="Hard gate for every scored row.")
    parser.add_argument("--mean-iou", type=float, default=0.85, help="Hard gate for report mean IoU.")
    parser.add_argument("--p05-iou", type=float, default=0.0, help="Hard gate for fifth-percentile IoU.")
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

    model_config = None
    if args.model_path is not None:
        model_config = ModelExtractionConfig(
            input_width=args.model_input_size,
            input_height=args.model_input_size,
            threshold=args.model_threshold,
            output_activation="logits",
            input_channels=args.model_input_channels,
        )
    report = score_synthetic_manifest(
        manifest,
        dataset_dir,
        model_path=args.model_path,
        model_config=model_config,
        guided=args.guided,
    )
    report["thresholds"] = {
        "min_iou": args.min_iou,
        "mean_iou": args.mean_iou,
        "p05_iou": args.p05_iou,
    }
    report["passed"] = _passes_thresholds(
        report, min_iou=args.min_iou, mean_iou=args.mean_iou, p05_iou=args.p05_iou
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
    model_config: ModelExtractionConfig | None = None,
    guided: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    rows = [
        score_synthetic_sample(sample, root, model_path=model_path, model_config=model_config, guided=guided)
        for sample in manifest.samples
    ]
    scored_rows = [row for row in rows if row["status"] == "scored"]
    ious = [row["metrics"]["iou"] for row in scored_rows]
    boundary_ious = [row["metrics"]["boundary_iou_2px"] for row in scored_rows]
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
        "mean_duration_s": round(mean(durations), 6) if durations else 0.0,
    }
    return {
        "manifest": {
            "name": manifest.name,
            "version": manifest.version,
            "properties": dict(manifest.properties),
        },
        "extractor": "model" if model_path is not None else "deterministic",
        "model_path": str(model_path) if model_path is not None else None,
        "summary": summary,
        "rows": rows,
    }


def score_synthetic_sample(
    sample,
    dataset_dir: Path,
    *,
    model_path: str | Path | None = None,
    model_config: ModelExtractionConfig | None = None,
    guided: bool = False,
) -> dict[str, Any]:
    image_path = dataset_dir / sample.artifacts.screenshot
    mask_path = dataset_dir / sample.artifacts.mask
    started = time.perf_counter()
    try:
        expected_mask = _load_mask(mask_path)
        if model_path is None:
            result = extract_service_area(image_path, cache=False)
        else:
            hints = (
                synthetic_guidance(sample, expected_mask, _load_rgb(image_path))
                if guided
                else None
            )
            result = extract_service_area_with_model(image_path, model_path, config=model_config, hints=hints)
        predicted_mask = result.mask.astype(bool, copy=False)
        row = {
            "sample_id": sample.sample_id,
            "variant": sample.variant,
            "overlay_style": sample.overlay_style.name,
            "status": "scored",
            "duration_s": round(time.perf_counter() - started, 6),
            "extraction": {
                "style": result.style,
                "coverage_ratio": round(result.coverage_ratio, 6),
                "confidence": round(result.confidence, 6),
                "contour_count": result.contour_count,
            },
            "metrics": _score_masks(predicted_mask, expected_mask),
            "geometry": geometry_validity_summary(result.pixel_geometry),
        }
        return row
    except Exception as exc:
        return {
            "sample_id": sample.sample_id,
            "variant": sample.variant,
            "overlay_style": sample.overlay_style.name,
            "status": "failed",
            "duration_s": round(time.perf_counter() - started, 6),
            "error": str(exc),
        }


def synthetic_guidance(sample, expected_mask: np.ndarray, rgb: np.ndarray | None = None) -> dict[str, object]:
    ys, xs = np.where(expected_mask)
    hints: dict[str, object] = {}
    if len(xs):
        pick = len(xs) // 2
        hints["seed_point"] = (float(xs[pick]), float(ys[pick]))
    if rgb is not None and len(xs):
        hints["target_rgb"] = tuple(int(round(value)) for value in np.median(rgb[expected_mask], axis=0))
    else:
        color = sample.overlay_style.fill_color.lstrip("#")
        if len(color) == 6:
            hints["target_rgb"] = tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))
    return hints


def _score_masks(predicted_mask: np.ndarray, expected_mask: np.ndarray) -> dict[str, float]:
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
    }


def _load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _passes_thresholds(
    report: dict[str, Any], *, min_iou: float, mean_iou: float, p05_iou: float = 0.0
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
    return True


if __name__ == "__main__":
    raise SystemExit(main())
