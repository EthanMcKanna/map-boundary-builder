from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from shapely.geometry import shape

from .catalog_match import catalog_feature_collection, match_service_area_catalog
from .extract import DEFAULT_SIMPLIFY_PX, extract_service_area, load_rgb, write_mask_png, write_overlay_png
from .georeference import (
    CityContext,
    georeference_from_city_context,
    georeference_from_label_context,
    georeference_from_labels,
    infer_city_contexts,
)
from .georef_transform import lonlat_to_mercator
from .geojson import feature_collection, write_geojson
from .image_io import is_svg_image, normalize_image_for_processing
from .ocr import extract_ocr_labels

ProgressCallback = Callable[[dict[str, Any]], None]
MAX_ROAD_CONTEXT_CANDIDATES = 1


@dataclass(frozen=True)
class BoundaryBuildOptions:
    simplify_px: float = DEFAULT_SIMPLIFY_PX
    min_confidence: float = 0.55
    min_control_points: int = 3
    preview_max_dimension: int | None = None
    write_mask_artifact: bool = True


@dataclass(frozen=True)
class BoundaryBuildResult:
    geojson: dict[str, Any]
    summary: dict[str, Any]
    output_path: Path
    mask_path: Path | None = None
    overlay_path: Path | None = None


def emit_progress(
    progress: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    percent: int,
    status: str = "running",
    details: dict[str, Any] | None = None,
) -> None:
    if progress is None:
        return
    event: dict[str, Any] = {
        "stage": stage,
        "message": message,
        "percent": percent,
        "status": status,
    }
    if details:
        event["details"] = details
    progress(event)


def build_boundary(
    image_path: str | Path,
    city: str | None,
    output_path: str | Path,
    *,
    debug_dir: str | Path | None = None,
    options: BoundaryBuildOptions | None = None,
    progress: ProgressCallback | None = None,
) -> BoundaryBuildResult:
    opts = options or BoundaryBuildOptions()
    image_path = Path(image_path)
    output_path = Path(output_path)
    debug_path = Path(debug_dir) if debug_dir else None
    city_input = city.strip() if isinstance(city, str) and city.strip() else None

    emit_progress(
        progress,
        stage="inspect",
        message="Rasterizing SVG upload" if is_svg_image(image_path) else "Reading image metadata",
        percent=5,
    )
    image_path = normalize_image_for_processing(
        image_path,
        output_dir=debug_path or output_path.parent,
        composite_transparent_rasters=False,
    )
    with Image.open(image_path) as img:
        width, height = img.size

    emit_progress(
        progress,
        stage="extract",
        message="Extracting service-area pixels",
        percent=18,
        details={"width": width, "height": height},
    )
    rgb = load_rgb(image_path)
    extraction = extract_service_area(image_path, simplify_px=opts.simplify_px, rgb=rgb)
    emit_progress(
        progress,
        stage="extract",
        message="Pixel polygon extracted",
        percent=36,
        details={
            "style": extraction.style,
            "coverage_ratio": round(extraction.coverage_ratio, 6),
            "contour_count": extraction.contour_count,
            "confidence": extraction.confidence,
        },
    )

    if city_input is None:
        catalog_match = match_service_area_catalog(extraction.pixel_geometry, style=extraction.style)
        if catalog_match is not None:
            data = catalog_feature_collection(
                extraction,
                catalog_match,
                width=width,
                height=height,
                image_path=image_path,
                city_input="Auto",
            )
            combined_confidence = data["features"][0]["properties"]["combined_confidence"]
            emit_progress(
                progress,
                stage="georeference",
                message="Matched known service-area shape",
                percent=78,
                details={
                    "source": "catalog-shape-match",
                    "catalog_slug": catalog_match.entry.slug,
                    "shape_iou": round(catalog_match.iou, 3),
                    "combined_confidence": combined_confidence,
                    "control_points": 0,
                    "median_residual_m": 0.0,
                    "p90_residual_m": 0.0,
                },
            )
            if combined_confidence < opts.min_confidence:
                raise ValueError(
                    f"Combined confidence {combined_confidence:.2f} is below --min-confidence "
                    f"{opts.min_confidence:.2f}. Provide a clearer map crop or lower the threshold."
                )
            return finish_boundary_result(
                data,
                extraction,
                image_path,
                output_path,
                debug_path,
                opts,
                width,
                height,
                city_input="Auto",
                rgb=rgb,
                progress=progress,
            )

    label_y_max = (
        extraction.pixel_geometry.bounds[3] + max(24.0, height * 0.04)
        if extraction.style == "dark-teal"
        else None
    )
    label_y_min = (
        extraction.pixel_geometry.bounds[1] - max(18.0, height * 0.06)
        if extraction.style == "gray-fill"
        else None
    )
    with ThreadPoolExecutor(max_workers=1) as ocr_executor:
        labels_future = ocr_executor.submit(extract_ocr_labels, str(image_path))
        emit_progress(
            progress,
            stage="ocr",
            message="Reading map labels on server",
            percent=44,
        )
        labels = labels_future.result()
        emit_progress(
            progress,
            stage="ocr",
            message="Map labels read",
            percent=47,
            details={
                "label_count": len(labels),
                "top_labels": [label.text for label in labels[:8]],
            },
        )
        georef = fit_georeference(
            labels,
            image_path,
            extraction.pixel_geometry,
            rgb=rgb,
            city_input=city_input,
            width=width,
            height=height,
            coverage_ratio=extraction.coverage_ratio,
            min_control_points=opts.min_control_points,
            label_y_min=label_y_min,
            label_y_max=label_y_max,
            progress=progress,
        )
    if georef is None:
        raise ValueError(
            "Could not infer a reliable map location and georeference from OCR/geocoded map labels. "
            "Provide a higher-resolution map crop with readable city labels or visible roads."
        )

    geo_transform = georef.transform
    data = feature_collection(extraction, width, height, geo_transform, str(image_path), city_input or "Auto")
    geom = shape(data["features"][0]["geometry"])
    combined_confidence = min(extraction.confidence, geo_transform.confidence)
    properties = data["features"][0]["properties"]
    properties["combined_confidence"] = combined_confidence
    properties["geodesic_bbox_lonlat"] = list(geom.bounds)
    properties["georeference_control_points"] = len(georef.control_points)
    properties["georeference_residual_median_m"] = georef.residual_median_m
    properties["georeference_residual_p90_m"] = georef.residual_p90_m
    if city_input is None and should_label_as_regional_area(geom.bounds, properties["city"], georef):
        properties["city"] = "Inferred map area"
    if georef.road_match is not None:
        properties["road_match_score"] = georef.road_match.score
        properties["road_match_base_score"] = georef.road_match.base_score
        properties["road_match_sampled_points"] = georef.road_match.sampled_points
        if georef.road_match.anchor_label is not None:
            properties["road_match_anchor_label"] = georef.road_match.anchor_label.text

    emit_progress(
        progress,
        stage="georeference",
        message="Map transform fitted",
        percent=78,
        details={
            "source": geo_transform.source,
            "combined_confidence": combined_confidence,
            "control_points": len(georef.control_points),
            "median_residual_m": round(georef.residual_median_m, 1),
            "p90_residual_m": round(georef.residual_p90_m, 1),
        },
    )

    if combined_confidence < opts.min_confidence:
        raise ValueError(
            f"Combined confidence {combined_confidence:.2f} is below --min-confidence "
            f"{opts.min_confidence:.2f}. Provide a clearer map crop or lower the threshold."
        )

    return finish_boundary_result(
        data,
        extraction,
        image_path,
        output_path,
        debug_path,
        opts,
        width,
        height,
        city_input=city_input or "Auto",
        rgb=rgb,
        progress=progress,
    )


def finish_boundary_result(
    data: dict[str, Any],
    extraction,
    image_path: Path,
    output_path: Path,
    debug_path: Path | None,
    opts: BoundaryBuildOptions,
    width: int,
    height: int,
    *,
    city_input: str,
    rgb,
    progress: ProgressCallback | None,
) -> BoundaryBuildResult:
    emit_progress(
        progress,
        stage="export",
        message="Writing GeoJSON and previews",
        percent=90,
    )
    write_geojson(data, output_path)
    mask_path: Path | None = None
    overlay_path: Path | None = None
    if debug_path:
        stem = output_path.stem
        overlay_path = debug_path / f"{stem}.overlay.png"
        if opts.write_mask_artifact:
            mask_path = debug_path / f"{stem}.mask.png"
            write_mask_png(extraction.mask, mask_path)
        write_overlay_png(
            image_path,
            extraction.mask,
            overlay_path,
            rgb=rgb,
            max_dimension=opts.preview_max_dimension,
        )

    summary = build_summary(
        data,
        output_path=output_path,
        city=city_input,
        width=width,
        height=height,
        mask_path=mask_path,
        overlay_path=overlay_path,
    )
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)
        (debug_path / f"{output_path.stem}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit_progress(
        progress,
        stage="complete",
        message="Boundary export ready",
        percent=100,
        status="complete",
        details=summary,
    )
    return BoundaryBuildResult(
        geojson=data,
        summary=summary,
        output_path=output_path,
        mask_path=mask_path,
        overlay_path=overlay_path,
    )


def georeference_from_road_contexts(
    image_path: Path,
    pixel_geometry,
    road_context_candidates: list[str],
    *,
    rgb,
    progress: ProgressCallback | None,
):
    if road_context_candidates:
        emit_progress(
            progress,
            stage="georeference",
            message="Trying road-network context",
            percent=62,
            details={"candidates": road_context_candidates},
        )
    for candidate in road_context_candidates:
        georef = georeference_from_city_context(rgb, candidate, pixel_geometry)
        if georef is not None:
            return georef
    return None


def fit_georeference(
    labels: list[Any],
    image_path: Path,
    pixel_geometry,
    *,
    rgb,
    city_input: str | None,
    width: int,
    height: int,
    coverage_ratio: float,
    min_control_points: int,
    label_y_min: float | None,
    label_y_max: float | None,
    progress: ProgressCallback | None,
):
    emit_progress(
        progress,
        stage="georeference",
        message="Inferring map location from labels" if city_input is None else "Matching readable map labels",
        percent=48,
    )
    road_contexts = road_contexts_from_labels(city_input, labels)
    road_context_candidates = [city_input] if city_input is not None else road_context_queries(road_contexts)
    try_ranked_context_first = label_y_min is None and should_try_ranked_context_first(
        city_input,
        coverage_ratio,
        road_contexts,
    )

    georef = None
    if try_ranked_context_first:
        emit_progress(
            progress,
            stage="georeference",
            message="Trying regional label context",
            percent=62,
            details={"candidates": road_context_candidates},
        )
        georef = georeference_from_ranked_label_contexts(
            labels,
            str(image_path),
            road_contexts,
            width,
            height,
            rgb=rgb,
            min_control_points=min_control_points,
        )

    if georef is None:
        georef = georeference_from_labels(
            labels,
            str(image_path),
            city_input,
            width,
            height,
            rgb=rgb,
            min_control_points=min_control_points,
            label_y_min=label_y_min,
            label_y_max=label_y_max,
        )

    if georef is None and road_context_candidates:
        georef = georeference_from_road_contexts(
            image_path,
            pixel_geometry,
            road_context_candidates,
            rgb=rgb,
            progress=progress,
        )
    return georef


def georeference_from_ranked_label_contexts(
    labels: list[Any],
    image_path: str,
    contexts: list[CityContext],
    width: int,
    height: int,
    *,
    rgb: Any,
    min_control_points: int,
):
    best = None
    best_score = -1.0
    for context in contexts:
        result = georeference_from_label_context(
            labels,
            image_path,
            context,
            width,
            height,
            rgb=rgb,
            min_control_points=min_control_points,
        )
        if result is None:
            continue
        score = result.transform.confidence + min(0.12, len(result.control_points) * 0.015)
        score -= min(0.2, result.residual_p90_m / 30000.0)
        if score > best_score:
            best = result
            best_score = score
        if (
            result.transform.confidence >= 0.80
            and len(result.control_points) >= 6
            and result.residual_p90_m <= 3500.0
        ):
            return result
    return best


def road_contexts_from_labels(city: str | None, labels: list[Any]) -> list[CityContext]:
    if city is not None:
        return []

    contexts = infer_city_contexts(labels)
    return rank_road_contexts(contexts)[:MAX_ROAD_CONTEXT_CANDIDATES]


def road_context_queries(contexts: list[CityContext]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        query = context.query.strip()
        if not query or query == "Inferred map area" or query in seen:
            continue
        seen.add(query)
        queries.append(query)
    return queries


def should_try_ranked_context_first(
    city: str | None,
    coverage_ratio: float,
    contexts: list[CityContext],
) -> bool:
    if city is not None or not contexts:
        return False
    return coverage_ratio >= 0.08 and context_bbox_span_m(contexts[0]) >= 30000.0


def rank_road_context_queries(contexts: list[CityContext]) -> list[str]:
    return road_context_queries(rank_road_contexts(contexts))


def rank_road_contexts(contexts: list[CityContext]) -> list[CityContext]:
    scored: list[tuple[float, int, CityContext]] = []
    for index, context in enumerate(contexts):
        query = context.query.strip()
        if not query:
            continue
        scored.append((road_context_score(context), -index, context))

    ranked: list[CityContext] = []
    seen: set[tuple[str, int, int]] = set()
    for _, _, context in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True):
        key = (
            context.center.display_name.lower(),
            round(context.center.lon, 3),
            round(context.center.lat, 3),
        )
        if key in seen:
            continue
        seen.add(key)
        ranked.append(context)
    return ranked


def road_context_score(context: CityContext) -> float:
    span_m = context_bbox_span_m(context)
    place_type = context.center.place_type.lower()
    display_name = context.center.display_name.lower()

    score = context.center.importance
    score += min(4.0, span_m / 18000.0)
    score += min(1.5, len(context.evidence) * 0.25)
    if place_type in {"region", "municipality", "borough"}:
        score += 1.35
    elif place_type in {"city", "town", "village"}:
        score += 0.55
    if span_m >= 30000.0:
        score += 1.25
    elif span_m < 9000.0:
        score -= 1.0
    if any(token in display_name for token in ("school", "district", "campus", "hospital")):
        score -= 1.75
    return score


def context_bbox_span_m(context: CityContext) -> float:
    bbox = context.center.bbox
    if bbox is None:
        return 0.0
    west, south, east, north = bbox
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
    return float(max(abs(east_m - west_m), abs(north_m - south_m)))


def should_label_as_regional_area(bounds: tuple[float, float, float, float], city: str, georef) -> bool:
    if city == "Inferred map area":
        return False
    west, south, east, north = bounds
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
    diagonal_m = hypot(east_m - west_m, north_m - south_m)
    if diagonal_m < 55000:
        return False
    distinct_places = {
        control.geocode.display_name.split(",", 1)[0].strip().lower()
        for control in georef.control_points
        if control.geocode.display_name
    }
    distinct_labels = {control.label.text.strip().lower() for control in georef.control_points if control.label.text}
    return len(distinct_places) >= 4 or len(distinct_labels) >= 4


def build_summary(
    data: dict[str, Any],
    *,
    output_path: Path,
    city: str,
    width: int,
    height: int,
    mask_path: Path | None,
    overlay_path: Path | None,
) -> dict[str, Any]:
    feature = data["features"][0]
    properties = feature["properties"]
    return {
        "output": str(output_path),
        "city_input": city,
        "city": properties["city"],
        "style": properties["style"],
        "image_width": width,
        "image_height": height,
        "coverage_ratio": properties["coverage_ratio"],
        "geometry_type": feature["geometry"]["type"],
        "bbox": properties["geodesic_bbox_lonlat"],
        "combined_confidence": properties["combined_confidence"],
        "extraction_confidence": properties["extraction_confidence"],
        "georeference_confidence": properties["georeference_confidence"],
        "georeference_source": properties["georeference_source"],
        "control_points": properties["georeference_control_points"],
        "rotation_degrees": properties["rotation_degrees"],
        "meters_per_pixel": properties["meters_per_pixel"],
        "median_residual_m": round(properties["georeference_residual_median_m"], 1),
        "p90_residual_m": round(properties["georeference_residual_p90_m"], 1),
        "road_match_score": properties.get("road_match_score"),
        "mask": str(mask_path) if mask_path else None,
        "overlay": str(overlay_path) if overlay_path else None,
    }
