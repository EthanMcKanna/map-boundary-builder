from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from shapely.geometry import shape

from .extract import DEFAULT_SIMPLIFY_PX, extract_service_area, load_rgb, write_mask_png, write_overlay_png
from .georeference import georeference_from_city_context, georeference_from_labels, georeference_from_ocr
from .geojson import feature_collection, write_geojson

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class BoundaryBuildOptions:
    simplify_px: float = DEFAULT_SIMPLIFY_PX
    min_confidence: float = 0.55
    min_control_points: int = 3


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
    ocr_labels: list[Any] | None = None,
) -> BoundaryBuildResult:
    opts = options or BoundaryBuildOptions()
    image_path = Path(image_path)
    output_path = Path(output_path)
    debug_path = Path(debug_dir) if debug_dir else None
    city_input = city.strip() if isinstance(city, str) and city.strip() else None

    emit_progress(
        progress,
        stage="inspect",
        message="Reading image metadata",
        percent=5,
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
    extraction = extract_service_area(image_path, simplify_px=opts.simplify_px)
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

    label_y_max = (
        extraction.pixel_geometry.bounds[3] + max(24.0, height * 0.04)
        if extraction.style == "dark-teal"
        else None
    )
    emit_progress(
        progress,
        stage="georeference",
        message="Inferring map location from labels" if city_input is None else "Matching readable map labels",
        percent=48,
    )
    if ocr_labels is not None:
        georef = georeference_from_labels(
            ocr_labels,
            str(image_path),
            city_input,
            width,
            height,
            min_control_points=opts.min_control_points,
            label_y_max=label_y_max,
        )
    else:
        georef = georeference_from_ocr(
            str(image_path),
            city_input,
            width,
            height,
            min_control_points=opts.min_control_points,
            label_y_max=label_y_max,
        )
    if georef is None and city_input is not None:
        emit_progress(
            progress,
            stage="georeference",
            message="Trying road-network context",
            percent=62,
        )
        georef = georeference_from_city_context(load_rgb(image_path), city_input, extraction.pixel_geometry)
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
        mask_path = debug_path / f"{stem}.mask.png"
        overlay_path = debug_path / f"{stem}.overlay.png"
        write_mask_png(extraction.mask, mask_path)
        write_overlay_png(image_path, extraction.mask, overlay_path)

    summary = build_summary(
        data,
        output_path=output_path,
        city=city_input or "Auto",
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
