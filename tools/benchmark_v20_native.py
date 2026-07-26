"""Benchmark v20 and v12 on native-pixel rasters of a real SVG map.

The SVG service-area path is the pixel-space ground truth.  It is extracted
directly for every output size, while the model only sees a rasterized and
optionally lossy-encoded map.  Both model variants run through the same
automatic production extraction entry point (no oracle seed or color hints).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from math import ceil, isfinite
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image

from map_boundary_builder.evaluation import (
    boundary_distance_summary_px,
    boundary_f1,
    geometry_corner_f1,
    geometry_topology_signature,
    iou,
    topology_signature,
)
from map_boundary_builder.extract import (
    BOUNDARYFIELD_MODEL_VARIANT,
    EDGEGRAPH_MODEL_VARIANT,
    ExtractionResult,
    extract_service_area,
)
from map_boundary_builder.image_io import (
    rasterize_svg_with_cairosvg,
    rasterize_svg_with_resvg,
    read_svg_bytes,
    svg_intrinsic_size,
)
from map_boundary_builder.svg_vector import SvgVectorPath, extract_svg_service_path


MINIMUM_FIXTURE_COUNT = 12
REPORT_SCHEMA_VERSION = "v20-native-raster-v1"


@dataclass(frozen=True)
class RasterCondition:
    slug: str
    width: int
    height: int
    encoding: str
    quality: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare generalized v20 and v12 on real SVG-derived native-pixel "
            "rasters using automatic production extraction."
        )
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("/Users/ethanmckanna/Downloads/waymoaus.svg"),
        help="Real SVG map containing the #07f service-area path.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Candidate v20 report path.")
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=None,
        help="Baseline v12 report path (default: <out stem>-v12-baseline.json).",
    )
    parser.add_argument(
        "--candidate-model",
        default=EDGEGRAPH_MODEL_VARIANT,
        choices=(EDGEGRAPH_MODEL_VARIANT, BOUNDARYFIELD_MODEL_VARIANT),
    )
    parser.add_argument(
        "--baseline-model",
        default=BOUNDARYFIELD_MODEL_VARIANT,
        choices=(EDGEGRAPH_MODEL_VARIANT, BOUNDARYFIELD_MODEL_VARIANT),
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Include first-use session setup in timings (promotion runs should not use this).",
    )
    return parser


def default_baseline_path(candidate_path: Path) -> Path:
    suffix = candidate_path.suffix or ".json"
    return candidate_path.with_name(f"{candidate_path.stem}-v12-baseline{suffix}")


def raster_conditions(native_width: int, native_height: int) -> list[RasterCondition]:
    """Return twelve distinct, representative raster/codec conditions."""

    if native_width <= 0 or native_height <= 0:
        raise ValueError("Native SVG dimensions must be positive")
    target_widths = (
        native_width,
        min(native_width, 960),
        min(native_width, 768),
        min(native_width, 640),
    )
    # Small custom fixtures still need four distinct resolution tiers.
    if len(set(target_widths)) != len(target_widths):
        target_widths = tuple(
            max(1, min(native_width, round(native_width * scale)))
            for scale in (1.0, 0.82, 0.65, 0.54)
        )
    if len(set(target_widths)) != len(target_widths):
        raise ValueError("SVG is too small to create four distinct native-pixel resolutions")

    codec_profiles = (
        (("png", None), ("jpeg", 95), ("jpeg", 85)),
        (("png", None), ("jpeg", 92), ("webp", 90)),
        (("png", None), ("jpeg", 88), ("webp", 82)),
        (("png", None), ("jpeg", 84), ("webp", 76)),
    )
    level_names = ("native", "large", "medium", "compact")
    conditions: list[RasterCondition] = []
    for level_name, target_width, profiles in zip(
        level_names, target_widths, codec_profiles, strict=True
    ):
        scale = target_width / native_width
        # resvg preserves the SVG aspect ratio and rounds the unconstrained
        # dimension upward. Matching that policy avoids a hidden resize between
        # vector rendering and codec pressure.
        target_height = max(1, ceil(native_height * scale))
        for encoding, quality in profiles:
            quality_slug = f"-q{quality}" if quality is not None else ""
            conditions.append(
                RasterCondition(
                    slug=f"{level_name}-{target_width}x{target_height}-{encoding}{quality_slug}",
                    width=target_width,
                    height=target_height,
                    encoding=encoding,
                    quality=quality,
                )
            )
    if len(conditions) != MINIMUM_FIXTURE_COUNT:
        raise RuntimeError("Native benchmark must contain exactly twelve fixture conditions")
    return conditions


def _render_svg_png(
    svg_bytes: bytes,
    target_path: Path,
    *,
    width: int,
    height: int,
    source_path: Path,
) -> str:
    errors: list[str] = []
    for name, rasterizer in (
        ("resvg-py", rasterize_svg_with_resvg),
        ("CairoSVG", rasterize_svg_with_cairosvg),
    ):
        try:
            rasterizer(
                svg_bytes,
                target_path,
                output_size=(width, height),
                source_path=source_path,
            )
            return name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            target_path.unlink(missing_ok=True)
    raise RuntimeError("No SVG rasterizer succeeded: " + "; ".join(errors))


def _encode_fixture(source_png: Path, target_path: Path, condition: RasterCondition) -> None:
    with Image.open(source_png) as source:
        image = source.convert("RGB")
        if image.size != (condition.width, condition.height):
            raise RuntimeError(
                f"Rasterizer returned {image.size}; expected {(condition.width, condition.height)}"
            )
        if condition.encoding == "png":
            image.save(target_path, format="PNG", compress_level=6)
        elif condition.encoding == "jpeg":
            image.save(
                target_path,
                format="JPEG",
                quality=condition.quality,
                subsampling=2,
                optimize=False,
            )
        elif condition.encoding == "webp":
            image.save(target_path, format="WEBP", quality=condition.quality, method=4)
        else:  # pragma: no cover - conditions are defined above.
            raise ValueError(f"Unsupported fixture encoding: {condition.encoding}")


def _fixture_extension(condition: RasterCondition) -> str:
    return ".jpg" if condition.encoding == "jpeg" else f".{condition.encoding}"


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _finite_round(value: float, digits: int = 6) -> float | None:
    numeric = float(value)
    return round(numeric, digits) if isfinite(numeric) else None


def score_extraction(result: ExtractionResult, truth: SvgVectorPath) -> dict[str, Any]:
    if result.mask.shape != truth.mask.shape:
        raise ValueError(
            f"Prediction shape {result.mask.shape} does not match truth {truth.mask.shape}"
        )
    one_pixel = boundary_f1(result.mask, truth.mask, tolerance_px=1.0)
    distances = boundary_distance_summary_px(result.mask, truth.mask)
    corners = geometry_corner_f1(
        result.pixel_geometry,
        truth.geometry,
        tolerance_px=2.0,
    )
    predicted_topology = geometry_topology_signature(result.pixel_geometry)
    reference_topology = geometry_topology_signature(truth.geometry)
    predicted_raster_topology = topology_signature(result.mask)
    reference_raster_topology = topology_signature(truth.mask)
    return {
        "iou": _finite_round(iou(result.mask, truth.mask)),
        "boundary_f1_1px": _finite_round(one_pixel["f1"]),
        "boundary_precision_1px": _finite_round(one_pixel["precision"]),
        "boundary_recall_1px": _finite_round(one_pixel["recall"]),
        "boundary_distance_mean_px": _finite_round(distances["mean_px"]),
        "boundary_distance_p95_px": _finite_round(distances["p95_px"]),
        "boundary_distance_p99_px": _finite_round(distances["p99_px"]),
        "boundary_distance_max_px": _finite_round(distances["max_px"]),
        "corner_f1": _finite_round(corners["f1"]),
        "corner_precision": _finite_round(corners["precision"]),
        "corner_recall": _finite_round(corners["recall"]),
        "corner_angular_error_p95_degrees": _finite_round(
            corners["p95_angular_error_degrees"]
        ),
        "topology_matches": predicted_topology == reference_topology,
        "predicted_topology": predicted_topology,
        "reference_topology": reference_topology,
        "raster_topology_matches": predicted_raster_topology == reference_raster_topology,
        "predicted_raster_topology": predicted_raster_topology,
        "reference_raster_topology": reference_raster_topology,
        "geometry_valid": bool(result.pixel_geometry.is_valid),
        "geometry_empty": bool(result.pixel_geometry.is_empty),
        "reference_geometry_valid": bool(truth.geometry.is_valid),
    }


def _metric_values(scores: Iterable[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for score in scores:
        value = score.get("metrics", {}).get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
            values.append(float(value))
    return values


def summarize_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    active = [score for score in scores if score.get("status") == "active"]
    ious = _metric_values(active, "iou")
    boundary_f1s = _metric_values(active, "boundary_f1_1px")
    p95_distances = _metric_values(active, "boundary_distance_p95_px")
    p99_distances = _metric_values(active, "boundary_distance_p99_px")
    max_distances = _metric_values(active, "boundary_distance_max_px")
    corner_f1s = _metric_values(active, "corner_f1")
    durations = [
        float(score["duration_s"])
        for score in active
        if isinstance(score.get("duration_s"), (int, float))
        and not isinstance(score.get("duration_s"), bool)
        and isfinite(float(score["duration_s"]))
    ]

    def average(values: list[float]) -> float | None:
        return _finite_round(mean(values)) if values else None

    def percentile(values: list[float], quantile: float) -> float | None:
        return _finite_round(np.quantile(values, quantile)) if values else None

    complete_metric_rows = all(
        len(values) == len(active)
        for values in (ious, boundary_f1s, p95_distances, p99_distances, max_distances, corner_f1s)
    )
    return {
        "fixture_count": len(scores),
        "scored_fixtures": len(active),
        "failed_fixtures": len(scores) - len(active),
        "complete_metric_rows": complete_metric_rows,
        "average_iou": average(ious),
        "min_iou": _finite_round(min(ious)) if ious else None,
        "p05_iou": percentile(ious, 0.05),
        "mean_boundary_f1_1px": average(boundary_f1s),
        "p05_boundary_f1_1px": percentile(boundary_f1s, 0.05),
        # These are conservative cross-condition percentiles of each
        # fixture's already-symmetric native-pixel p95/p99 distance.
        "p95_boundary_distance_px": percentile(p95_distances, 0.95),
        "p99_boundary_distance_px": percentile(p99_distances, 0.95),
        "max_boundary_distance_px": _finite_round(max(max_distances))
        if max_distances
        else None,
        "mean_corner_f1": average(corner_f1s),
        "min_corner_f1": _finite_round(min(corner_f1s)) if corner_f1s else None,
        "topology_error_count": sum(
            score.get("metrics", {}).get("topology_matches") is not True for score in active
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
        "p95_duration_s": percentile(durations, 0.95),
        "max_duration_s": _finite_round(max(durations)) if durations else None,
        "native_pixel_references": True,
    }


def _diagnostic_summary(result: ExtractionResult) -> dict[str, Any]:
    diagnostics = result.diagnostics or {}
    return {
        "selected_model_variant": diagnostics.get("model_variant"),
        "extractor": diagnostics.get("extractor"),
        "automatic_guidance": diagnostics.get("automatic_guidance"),
        "model_fallback": diagnostics.get("model_fallback"),
        "confidence": _finite_round(result.confidence),
        "coverage_ratio": _finite_round(result.coverage_ratio),
        "contour_count": result.contour_count,
    }


def benchmark_model(
    fixtures: list[tuple[RasterCondition, Path, SvgVectorPath]],
    *,
    model_variant: str,
    warmup: bool,
) -> list[dict[str, Any]]:
    if warmup and fixtures:
        condition, fixture_path, _truth = fixtures[0]
        extract_service_area(
            fixture_path,
            rgb=_load_rgb(fixture_path),
            cache=False,
            hints=None,
            use_model=model_variant,
        )

    scores: list[dict[str, Any]] = []
    for condition, fixture_path, truth in fixtures:
        started = time.perf_counter()
        try:
            result = extract_service_area(
                fixture_path,
                rgb=_load_rgb(fixture_path),
                cache=False,
                hints=None,
                use_model=model_variant,
            )
            duration = time.perf_counter() - started
            scores.append(
                {
                    "slug": condition.slug,
                    "status": "active",
                    "duration_s": _finite_round(duration),
                    "condition": asdict(condition),
                    "raster_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                    "reference": {
                        "space": "native_pixels",
                        "source": "exact_svg_vector_path",
                        "width": truth.width,
                        "height": truth.height,
                        "flatten_tolerance_px": truth.flatten_tolerance_px,
                    },
                    "metrics": score_extraction(result, truth),
                    "result": _diagnostic_summary(result),
                }
            )
        except Exception as exc:
            scores.append(
                {
                    "slug": condition.slug,
                    "status": "failed",
                    "duration_s": _finite_round(time.perf_counter() - started),
                    "condition": asdict(condition),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return scores


def _report(
    *,
    source_svg: Path,
    source_sha256: str,
    model_variant: str,
    rasterizer: str,
    timing_profile: str,
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "reference_space": "native_pixels",
        "model_variant": model_variant,
        "metadata": {
            "source_svg": str(source_svg.resolve()),
            "source_sha256": source_sha256,
            "reference_space": "native_pixels",
            "truth_source": "exact_svg_vector_path",
            "automatic_production_extraction": True,
            "oracle_hints": False,
            "cache_enabled": False,
            "timing_profile": timing_profile,
            "rasterizer": rasterizer,
        },
        "summary": summarize_scores(scores),
        "scores": scores,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.svg.is_file():
        raise SystemExit(f"SVG fixture does not exist: {args.svg}")
    svg_bytes = read_svg_bytes(args.svg)
    intrinsic = svg_intrinsic_size(svg_bytes)
    if intrinsic is None:
        raise SystemExit("SVG fixture has no finite intrinsic dimensions or viewBox")
    native_width = max(1, round(intrinsic[0]))
    native_height = max(1, round(intrinsic[1]))
    conditions = raster_conditions(native_width, native_height)
    source_sha256 = hashlib.sha256(svg_bytes).hexdigest()

    with TemporaryDirectory(prefix="edgegraph-v20-native-") as directory:
        fixture_root = Path(directory)
        fixtures: list[tuple[RasterCondition, Path, SvgVectorPath]] = []
        rasterizer_names: set[str] = set()
        render_paths: dict[tuple[int, int], Path] = {}
        for condition in conditions:
            size = (condition.width, condition.height)
            render_path = render_paths.get(size)
            if render_path is None:
                render_path = fixture_root / f"render-{condition.width}x{condition.height}.png"
                rasterizer_names.add(
                    _render_svg_png(
                        svg_bytes,
                        render_path,
                        width=condition.width,
                        height=condition.height,
                        source_path=args.svg,
                    )
                )
                render_paths[size] = render_path
            fixture_path = fixture_root / f"{condition.slug}{_fixture_extension(condition)}"
            _encode_fixture(render_path, fixture_path, condition)
            truth = extract_svg_service_path(
                svg_bytes,
                width=condition.width,
                height=condition.height,
            )
            fixtures.append((condition, fixture_path, truth))

        warmup = not args.skip_warmup
        candidate_scores = benchmark_model(
            fixtures,
            model_variant=args.candidate_model,
            warmup=warmup,
        )
        baseline_scores = benchmark_model(
            fixtures,
            model_variant=args.baseline_model,
            warmup=warmup,
        )
        rasterizer = "+".join(sorted(rasterizer_names))

    candidate_report = _report(
        source_svg=args.svg,
        source_sha256=source_sha256,
        model_variant=args.candidate_model,
        rasterizer=rasterizer,
        timing_profile="cold" if args.skip_warmup else "warm",
        scores=candidate_scores,
    )
    baseline_report = _report(
        source_svg=args.svg,
        source_sha256=source_sha256,
        model_variant=args.baseline_model,
        rasterizer=rasterizer,
        timing_profile="cold" if args.skip_warmup else "warm",
        scores=baseline_scores,
    )
    baseline_path = args.baseline_out or default_baseline_path(args.out)
    _write_report(args.out, candidate_report)
    _write_report(baseline_path, baseline_report)

    candidate_average = candidate_report["summary"].get("average_iou")
    baseline_average = baseline_report["summary"].get("average_iou")
    average_improvement = (
        _finite_round(float(candidate_average) - float(baseline_average))
        if isinstance(candidate_average, (int, float))
        and isinstance(baseline_average, (int, float))
        else None
    )
    comparison = {
        "candidate_report": str(args.out),
        "baseline_report": str(baseline_path),
        "candidate": candidate_report["summary"],
        "baseline": baseline_report["summary"],
        "average_iou_improvement": average_improvement,
    }
    print(json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False))
    return 0 if (
        candidate_report["summary"]["failed_fixtures"] == 0
        and baseline_report["summary"]["failed_fixtures"] == 0
        and candidate_report["summary"]["scored_fixtures"] >= MINIMUM_FIXTURE_COUNT
        and baseline_report["summary"]["scored_fixtures"] >= MINIMUM_FIXTURE_COUNT
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
