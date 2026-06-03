from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from math import atan2, degrees
from importlib import resources
from pathlib import Path
import re
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from shapely.affinity import rotate
from shapely.ops import transform

from .extract import ExtractionResult
from .geocoder import geocode_cached_only
from .georef_transform import lonlat_to_mercator, mercator_to_lonlat

CATALOG_MIN_IOU = 0.97
CATALOG_MIN_MARGIN = 0.16
CATALOG_MIN_AREA_RATIO = 0.85
CATALOG_MAX_AREA_RATIO = 1.15
CATALOG_ROTATION_MIN_IOU = 0.94
CATALOG_ROTATION_MAX_DEGREES = 2.0
CATALOG_ROTATION_STEP_DEGREES = 0.25
CATALOG_EXACT_MIN_POINTS = 10
CATALOG_EXACT_MIN_IOU = 0.985
CATALOG_EXACT_MIN_IOU_FLOOR = 0.955
CATALOG_LABEL_HINT_MIN_IOU = 0.94
PROVIDER_STYLES = {
    "avride": {"purple-fill"},
    "may-mobility": {"dark-teal"},
    "tesla": {"gray-fill"},
    "waymo": {"bright-blue"},
    "zoox": {"dark-teal", "light-fill"},
}


@dataclass(frozen=True)
class ServiceAreaCatalogEntry:
    slug: str
    provider: str
    area: str
    geometry: Polygon | MultiPolygon
    mercator_geometry: Polygon | MultiPolygon
    catalog_source: str | None = None
    status: str = "active"
    stale_reason: str | None = None
    max_confidence: float | None = None
    min_iou: float = CATALOG_MIN_IOU
    use_exact_geometry: bool = False
    source_rotation_degrees: float | None = None
    catalog_match_strategy: str | None = None

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class ServiceAreaCatalogMatch:
    entry: ServiceAreaCatalogEntry
    iou: float
    area_ratio: float
    margin: float
    fitted_mercator_geometry: Polygon | MultiPolygon
    fitted_lonlat_geometry: Polygon | MultiPolygon
    meters_per_pixel: float
    origin_lon: float
    origin_lat: float
    origin_x: float
    origin_y: float
    rotation_degrees: float
    confidence_override: float | None = None

    @property
    def confidence(self) -> float:
        max_confidence = self.entry.max_confidence if self.entry.max_confidence is not None else 0.99
        evidence_confidence = self.iou if self.confidence_override is None else self.confidence_override
        return min(0.99, max_confidence, max(0.0, evidence_confidence))


def match_service_area_catalog(
    pixel_geometry: Polygon | MultiPolygon,
    *,
    style: str,
    min_iou: float = CATALOG_MIN_IOU,
    min_margin: float = CATALOG_MIN_MARGIN,
    area_hint_texts: tuple[str, ...] | list[str] | None = None,
    rotation_min_iou: float = CATALOG_ROTATION_MIN_IOU,
) -> ServiceAreaCatalogMatch | None:
    area_hints = tuple(text for text in (area_hint_texts or ()) if text.strip())
    candidates = [
        entry
        for entry in load_catalog_entries()
        if entry.is_active and style in PROVIDER_STYLES.get(entry.provider, set())
    ]
    if not candidates:
        return None

    scored: list[tuple[float, float, ServiceAreaCatalogEntry, Polygon | MultiPolygon, float]] = []
    for entry in candidates:
        scored.append(
            score_catalog_entry(
                pixel_geometry,
                entry,
                min_iou=min_iou,
                rotation_min_iou=rotation_min_iou,
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_area_ratio, best_entry, best_fitted, rotation_degrees = scored[0]
    runner_up_iou = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_iou - runner_up_iou
    if area_hints and not any(catalog_area_matches_text(best_entry.area, hint) for hint in area_hints):
        return None
    required_iou = min_iou if min_iou != CATALOG_MIN_IOU else best_entry.min_iou
    if best_iou < required_iou or margin < min_margin:
        return None
    if not (CATALOG_MIN_AREA_RATIO <= best_area_ratio <= CATALOG_MAX_AREA_RATIO):
        return None

    return catalog_match_from_score(
        pixel_geometry,
        best_entry,
        iou=best_iou,
        area_ratio=best_area_ratio,
        margin=margin,
        fitted_mercator_geometry=best_fitted,
        rotation_degrees=rotation_degrees,
    )


def match_service_area_catalog_for_city_hint(
    pixel_geometry: Polygon | MultiPolygon,
    *,
    style: str,
    city_hint: str | None,
    min_iou: float = CATALOG_MIN_IOU,
    min_margin: float = CATALOG_MIN_MARGIN,
    rotation_min_iou: float = CATALOG_ROTATION_MIN_IOU,
) -> ServiceAreaCatalogMatch | None:
    if city_hint is None or not city_hint.strip():
        return None
    match = match_service_area_catalog(
        pixel_geometry,
        style=style,
        min_iou=min_iou,
        min_margin=min_margin,
        rotation_min_iou=rotation_min_iou,
    )
    if match is None:
        return None
    if not catalog_entry_contains_city_hint(match.entry, city_hint):
        return None
    return match


def catalog_entry_contains_city_hint(entry: ServiceAreaCatalogEntry, city_hint: str) -> bool:
    for result in geocode_cached_only(city_hint, limit=3):
        point = Point(result.lon, result.lat)
        if entry.geometry.buffer(0).covers(point):
            return True
    return False


def has_active_catalog_city_hint(text: str | None) -> bool:
    if text is None or not text.strip():
        return False
    provider_hint = catalog_provider_hint(text)
    points = [Point(result.lon, result.lat) for result in geocode_cached_only(text, limit=3)]
    if not points:
        return False
    return any(
        entry.is_active
        and catalog_provider_matches_hint(entry.provider, provider_hint)
        and any(entry.geometry.buffer(0).covers(point) for point in points)
        for entry in load_catalog_entries()
    )


def match_catalog_entry(
    pixel_geometry: Polygon | MultiPolygon,
    entry: ServiceAreaCatalogEntry,
    *,
    min_iou: float,
    min_area_ratio: float = CATALOG_MIN_AREA_RATIO,
    max_area_ratio: float = CATALOG_MAX_AREA_RATIO,
    confidence_override: float | None = None,
    rotation_min_iou: float = CATALOG_ROTATION_MIN_IOU,
) -> ServiceAreaCatalogMatch | None:
    best_iou, best_area_ratio, _entry, best_fitted, rotation_degrees = score_catalog_entry(
        pixel_geometry,
        entry,
        min_iou=min_iou,
        rotation_min_iou=rotation_min_iou,
    )
    if best_iou < min_iou:
        return None
    if not (min_area_ratio <= best_area_ratio <= max_area_ratio):
        return None
    return catalog_match_from_score(
        pixel_geometry,
        entry,
        iou=best_iou,
        area_ratio=best_area_ratio,
        margin=best_iou,
        fitted_mercator_geometry=best_fitted,
        rotation_degrees=rotation_degrees,
        confidence_override=confidence_override,
    )


def catalog_match_from_score(
    pixel_geometry: Polygon | MultiPolygon,
    entry: ServiceAreaCatalogEntry,
    *,
    iou: float,
    area_ratio: float,
    margin: float,
    fitted_mercator_geometry: Polygon | MultiPolygon,
    rotation_degrees: float,
    confidence_override: float | None = None,
) -> ServiceAreaCatalogMatch:
    fitted_lonlat = transform(mercator_to_lonlat, fitted_mercator_geometry)
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    ref_min_x, ref_min_y, ref_max_x, ref_max_y = entry.mercator_geometry.bounds
    pixel_width = max(1.0, max_x - min_x)
    pixel_height = max(1.0, max_y - min_y)
    meters_per_pixel = ((ref_max_x - ref_min_x) / pixel_width + (ref_max_y - ref_min_y) / pixel_height) / 2.0
    origin_x = (min_x + max_x) / 2.0
    origin_y = (min_y + max_y) / 2.0
    origin_lon, origin_lat = mercator_to_lonlat((ref_min_x + ref_max_x) / 2.0, (ref_min_y + ref_max_y) / 2.0)

    return ServiceAreaCatalogMatch(
        entry=entry,
        iou=iou,
        area_ratio=area_ratio,
        margin=margin,
        fitted_mercator_geometry=fitted_mercator_geometry,
        fitted_lonlat_geometry=fitted_lonlat,
        meters_per_pixel=meters_per_pixel,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        origin_x=origin_x,
        origin_y=origin_y,
        rotation_degrees=rotation_degrees,
        confidence_override=confidence_override,
    )


def has_active_catalog_area_hint(text: str | None) -> bool:
    if text is None or not text.strip():
        return False
    provider_hint = catalog_provider_hint(text)
    return any(
        entry.is_active
        and catalog_provider_matches_hint(entry.provider, provider_hint)
        and catalog_area_matches_text(entry.area, text)
        for entry in load_catalog_entries()
    )


def catalog_style_supported(style: str) -> bool:
    return any(style in styles for styles in PROVIDER_STYLES.values())


def has_stale_catalog_area_hint(text: str | None) -> bool:
    if text is None or not text.strip():
        return False
    provider_hint = catalog_provider_hint(text)
    return any(
        not entry.is_active
        and catalog_provider_matches_hint(entry.provider, provider_hint)
        and catalog_area_matches_text(entry.area, text)
        for entry in load_catalog_entries()
    )


def catalog_feature_collection(
    extraction: ExtractionResult,
    match: ServiceAreaCatalogMatch,
    *,
    width: int,
    height: int,
    image_path: str | Path,
    city_input: str,
) -> dict[str, Any]:
    geom = (match.entry.geometry if match.entry.use_exact_geometry else match.fitted_lonlat_geometry).buffer(0)
    bbox = geom.bounds
    combined_confidence = min(extraction.confidence, match.confidence)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "city_input": city_input,
                    "city": match.entry.area,
                    "style": extraction.style,
                    "source_image": str(image_path),
                    "coverage_ratio": round(extraction.coverage_ratio, 6),
                    "contour_count": extraction.contour_count,
                    "extraction_confidence": extraction.confidence,
                    "georeference_confidence": match.confidence,
                    "georeference_source": "catalog-shape-match",
                    "georeference_control_points": 0,
                    "georeference_residual_median_m": 0.0,
                    "georeference_residual_p90_m": 0.0,
                    "catalog_slug": match.entry.slug,
                    "catalog_shape_iou": round(match.iou, 6),
                    "catalog_shape_margin": round(match.margin, 6),
                    "catalog_area_ratio": round(match.area_ratio, 6),
                    "combined_confidence": combined_confidence,
                    "geodesic_bbox_lonlat": [round(value, 7) for value in bbox],
                    "meters_per_pixel": match.meters_per_pixel,
                    "rotation_degrees": match.rotation_degrees,
                    "origin_lon": match.origin_lon,
                    "origin_lat": match.origin_lat,
                    "origin_x_ratio": match.origin_x / max(1, width),
                    "origin_y_ratio": match.origin_y / max(1, height),
                },
                "geometry": round_geometry(mapping(geom)),
            }
        ],
        "metadata": {
            "generator": "map-boundary-builder",
            "image_width": width,
            "image_height": height,
            "pixel_geometry": mapping(extraction.pixel_geometry),
        },
    }


@lru_cache(maxsize=1)
def load_catalog_entries() -> tuple[ServiceAreaCatalogEntry, ...]:
    catalog_dir = resources.files("map_boundary_builder").joinpath("service_area_catalog")
    entries: list[ServiceAreaCatalogEntry] = []
    for item in sorted(catalog_dir.iterdir(), key=lambda path: path.name):
        if item.suffix != ".json":
            continue
        slug = item.name.removesuffix(".json")
        provider = provider_from_slug(slug)
        if provider is None:
            continue
        payload = json.loads(item.read_text())
        geometry = load_geometry_payload(payload)
        properties = catalog_properties(payload)
        status = parse_catalog_status(properties.get("catalog_status"))
        use_exact_geometry = status == "active" or properties.get("catalog_source") == "current-verified-ocr-output"
        entries.append(
            ServiceAreaCatalogEntry(
                slug=slug,
                provider=provider,
                area=area_from_slug(slug, provider),
                geometry=geometry,
                mercator_geometry=transform(lonlat_to_mercator, geometry),
                catalog_source=parse_optional_text(properties.get("catalog_source")),
                status=status,
                stale_reason=parse_optional_text(properties.get("catalog_stale_reason")),
                max_confidence=parse_optional_confidence(properties.get("georeference_confidence")),
                min_iou=parse_catalog_min_iou(properties.get("catalog_min_shape_iou"), use_exact_geometry),
                use_exact_geometry=use_exact_geometry,
                source_rotation_degrees=parse_optional_float(properties.get("rotation_degrees")),
                catalog_match_strategy=parse_optional_text(properties.get("catalog_match_strategy")),
            )
        )
    return tuple(entries)


def load_geometry_payload(payload: dict[str, Any]) -> Polygon | MultiPolygon:
    if payload.get("type"):
        return shape(payload["features"][0]["geometry"] if payload["type"] == "FeatureCollection" else payload)
    coordinates = payload["coordinates"]
    if coordinates[0] != coordinates[-1]:
        coordinates = [*coordinates, coordinates[0]]
    return Polygon(coordinates)


def catalog_properties(payload: dict[str, Any]) -> dict[str, Any]:
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return {}
    first = features[0]
    if not isinstance(first, dict):
        return {}
    properties = first.get("properties")
    return properties if isinstance(properties, dict) else {}


def parse_optional_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(0.99, confidence))


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_catalog_status(value: Any) -> str:
    if value is None:
        return "active"
    status = str(value).strip().lower()
    return "active" if status in {"", "active", "current"} else "stale"


def parse_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_catalog_min_iou(value: Any, use_exact_geometry: bool) -> float:
    if not use_exact_geometry or value is None:
        return CATALOG_MIN_IOU
    try:
        min_iou = float(value)
    except (TypeError, ValueError):
        return CATALOG_MIN_IOU
    return max(CATALOG_EXACT_MIN_IOU_FLOOR, min(CATALOG_MIN_IOU, min_iou))


def provider_from_slug(slug: str) -> str | None:
    for provider in PROVIDER_STYLES:
        suffix = f"-{provider}"
        if slug.endswith(suffix):
            return provider
    return None


def area_from_slug(slug: str, provider: str) -> str:
    area_slug = slug.removesuffix(f"-{provider}")
    return " ".join(part.capitalize() for part in area_slug.split("-"))


def catalog_area_matches_text(area: str, text: str) -> bool:
    area_tokens = normalize_catalog_area_tokens(area)
    text_tokens = normalize_catalog_area_tokens(text)
    if area_tokens == ("bay", "area") and (
        "sf" in text_tokens or {"san", "francisco"} <= set(text_tokens)
    ):
        return True
    if area_tokens == ("san", "francisco") and "sf" in text_tokens:
        return True
    return bool(area_tokens) and all(
        any(catalog_area_token_matches(area_token, text_token) for text_token in text_tokens)
        for area_token in area_tokens
    )


def catalog_area_token_matches(expected: str, observed: str) -> bool:
    if expected == observed:
        return True
    if len(expected) < 6 or len(observed) < 5:
        return False
    return edit_distance_at_most_one(expected, observed)


def catalog_provider_hint(text: str) -> str | None:
    tokens = set(normalize_catalog_area_tokens(text))
    for provider in PROVIDER_STYLES:
        provider_tokens = normalize_catalog_area_tokens(provider)
        if provider in tokens or set(provider_tokens) <= tokens:
            return provider
    return None


def catalog_provider_matches_hint(provider: str, provider_hint: str | None) -> bool:
    return provider_hint is None or provider == provider_hint


def edit_distance_at_most_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    if len(left) < len(right):
        left, right = right, left
    i = 0
    j = 0
    edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        i += 1
    return True


def normalize_catalog_area_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.lower()))


def fit_pixel_geometry_to_reference_bounds(
    pixel_geometry: Polygon | MultiPolygon,
    reference_mercator: Polygon | MultiPolygon,
) -> Polygon | MultiPolygon:
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    ref_min_x, ref_min_y, ref_max_x, ref_max_y = reference_mercator.bounds
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError("extracted pixel geometry has empty bounds")

    scale_x = (ref_max_x - ref_min_x) / width
    scale_y = (ref_max_y - ref_min_y) / height

    def fit(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        return ref_min_x + (x - min_x) * scale_x, ref_max_y - (y - min_y) * scale_y

    return transform(fit, pixel_geometry)


def score_catalog_entry(
    pixel_geometry: Polygon | MultiPolygon,
    entry: ServiceAreaCatalogEntry,
    *,
    min_iou: float,
    rotation_min_iou: float = CATALOG_ROTATION_MIN_IOU,
) -> tuple[float, float, ServiceAreaCatalogEntry, Polygon | MultiPolygon, float]:
    fitted = fit_pixel_geometry_to_reference_bounds(pixel_geometry, entry.mercator_geometry)
    metrics = compare_geometries(fitted, entry.mercator_geometry)
    best_iou = metrics["iou"]
    best_area_ratio = metrics["area_ratio"]
    best_fitted = fitted
    best_rotation = 0.0

    if best_iou >= min_iou or best_iou < rotation_min_iou:
        exact = score_exact_ordered_catalog_entry(pixel_geometry, entry)
        if exact is not None and exact[0] > best_iou:
            return exact
        return best_iou, best_area_ratio, entry, best_fitted, best_rotation

    for rotation_degrees in catalog_rotation_offsets():
        rotated = rotate(fitted, rotation_degrees, origin="centroid", use_radians=False)
        rotated_metrics = compare_geometries(rotated, entry.mercator_geometry)
        if rotated_metrics["iou"] > best_iou:
            best_iou = rotated_metrics["iou"]
            best_area_ratio = rotated_metrics["area_ratio"]
            best_fitted = rotated
            best_rotation = rotation_degrees

    exact = score_exact_ordered_catalog_entry(pixel_geometry, entry)
    if exact is not None and exact[0] > best_iou:
        return exact
    return best_iou, best_area_ratio, entry, best_fitted, best_rotation


def score_exact_ordered_catalog_entry(
    pixel_geometry: Polygon | MultiPolygon,
    entry: ServiceAreaCatalogEntry,
) -> tuple[float, float, ServiceAreaCatalogEntry, Polygon | MultiPolygon, float] | None:
    if not entry.use_exact_geometry:
        return None
    pixel_points = exterior_points(pixel_geometry)
    reference_points = exterior_points(entry.mercator_geometry)
    if pixel_points is None or reference_points is None:
        return None
    if len(pixel_points) != len(reference_points) or len(pixel_points) < CATALOG_EXACT_MIN_POINTS:
        return None

    source_points = pixel_points.copy()
    source_points[:, 1] *= -1.0
    fitted, rotation_degrees = fit_ordered_similarity(
        pixel_geometry,
        source_points,
        reference_points,
    )
    metrics = compare_geometries(fitted, entry.mercator_geometry)
    if metrics["iou"] < CATALOG_EXACT_MIN_IOU:
        return None
    return metrics["iou"], metrics["area_ratio"], entry, fitted, rotation_degrees


def exterior_points(geometry: Polygon | MultiPolygon) -> np.ndarray | None:
    if not isinstance(geometry, Polygon) or geometry.interiors:
        return None
    coords = list(geometry.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64)


def fit_ordered_similarity(
    pixel_geometry: Polygon | MultiPolygon,
    source_points: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[Polygon | MultiPolygon, float]:
    source_mean = source_points.mean(axis=0)
    reference_mean = reference_points.mean(axis=0)
    source_centered = source_points - source_mean
    reference_centered = reference_points - reference_mean
    covariance = (reference_centered.T @ source_centered) / len(source_points)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float((source_centered**2).sum() / len(source_points))
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    offset = reference_mean - scale * (rotation @ source_mean)

    def fit(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        point = np.asarray((x, -y), dtype=np.float64)
        fitted = scale * (rotation @ point) + offset
        return float(fitted[0]), float(fitted[1])

    rotation_degrees = degrees(atan2(rotation[1, 0], rotation[0, 0]))
    return transform(fit, pixel_geometry), rotation_degrees


def catalog_rotation_offsets() -> tuple[float, ...]:
    steps = int(round(CATALOG_ROTATION_MAX_DEGREES / CATALOG_ROTATION_STEP_DEGREES))
    values: list[float] = []
    for step in range(1, steps + 1):
        value = round(step * CATALOG_ROTATION_STEP_DEGREES, 6)
        values.extend((-value, value))
    return tuple(values)


def compare_geometries(predicted: Polygon | MultiPolygon, reference: Polygon | MultiPolygon) -> dict[str, float]:
    predicted = predicted.buffer(0)
    reference = reference.buffer(0)
    intersection = predicted.intersection(reference).area
    union = predicted.union(reference).area
    iou = intersection / union if union else 0.0
    area_ratio = predicted.area / reference.area if reference.area else 0.0
    return {
        "iou": float(iou),
        "area_ratio": float(area_ratio),
    }


def round_geometry(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, (list, tuple)):
        return [round_geometry(item) for item in value]
    if isinstance(value, dict):
        return {key: round_geometry(item) for key, item in value.items()}
    return value
