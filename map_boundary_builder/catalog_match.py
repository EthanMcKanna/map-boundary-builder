from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from importlib import resources
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform

from .extract import ExtractionResult
from .georef_transform import lonlat_to_mercator, mercator_to_lonlat

CATALOG_MIN_IOU = 0.97
CATALOG_MIN_MARGIN = 0.16
CATALOG_MIN_AREA_RATIO = 0.85
CATALOG_MAX_AREA_RATIO = 1.15
PROVIDER_STYLES = {
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
    max_confidence: float | None = None


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

    @property
    def confidence(self) -> float:
        max_confidence = self.entry.max_confidence if self.entry.max_confidence is not None else 0.99
        return min(0.99, max_confidence, max(0.0, self.iou))


def match_service_area_catalog(
    pixel_geometry: Polygon | MultiPolygon,
    *,
    style: str,
    min_iou: float = CATALOG_MIN_IOU,
    min_margin: float = CATALOG_MIN_MARGIN,
) -> ServiceAreaCatalogMatch | None:
    candidates = [
        entry
        for entry in load_catalog_entries()
        if style in PROVIDER_STYLES.get(entry.provider, set())
    ]
    if not candidates:
        return None

    scored: list[tuple[float, float, ServiceAreaCatalogEntry, Polygon | MultiPolygon]] = []
    for entry in candidates:
        fitted = fit_pixel_geometry_to_reference_bounds(pixel_geometry, entry.mercator_geometry)
        metrics = compare_geometries(fitted, entry.mercator_geometry)
        scored.append((metrics["iou"], metrics["area_ratio"], entry, fitted))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_area_ratio, best_entry, best_fitted = scored[0]
    runner_up_iou = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_iou - runner_up_iou
    if best_iou < min_iou or margin < min_margin:
        return None
    if not (CATALOG_MIN_AREA_RATIO <= best_area_ratio <= CATALOG_MAX_AREA_RATIO):
        return None

    fitted_lonlat = transform(mercator_to_lonlat, best_fitted)
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    ref_min_x, ref_min_y, ref_max_x, ref_max_y = best_entry.mercator_geometry.bounds
    pixel_width = max(1.0, max_x - min_x)
    pixel_height = max(1.0, max_y - min_y)
    meters_per_pixel = ((ref_max_x - ref_min_x) / pixel_width + (ref_max_y - ref_min_y) / pixel_height) / 2.0
    origin_x = (min_x + max_x) / 2.0
    origin_y = (min_y + max_y) / 2.0
    origin_lon, origin_lat = mercator_to_lonlat((ref_min_x + ref_max_x) / 2.0, (ref_min_y + ref_max_y) / 2.0)

    return ServiceAreaCatalogMatch(
        entry=best_entry,
        iou=best_iou,
        area_ratio=best_area_ratio,
        margin=margin,
        fitted_mercator_geometry=best_fitted,
        fitted_lonlat_geometry=fitted_lonlat,
        meters_per_pixel=meters_per_pixel,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        origin_x=origin_x,
        origin_y=origin_y,
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
    geom = match.fitted_lonlat_geometry.buffer(0)
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
                    "rotation_degrees": 0.0,
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
        entries.append(
            ServiceAreaCatalogEntry(
                slug=slug,
                provider=provider,
                area=area_from_slug(slug, provider),
                geometry=geometry,
                mercator_geometry=transform(lonlat_to_mercator, geometry),
                max_confidence=parse_optional_confidence(properties.get("georeference_confidence")),
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


def provider_from_slug(slug: str) -> str | None:
    for provider in PROVIDER_STYLES:
        suffix = f"-{provider}"
        if slug.endswith(suffix):
            return provider
    return None


def area_from_slug(slug: str, provider: str) -> str:
    area_slug = slug.removesuffix(f"-{provider}")
    return " ".join(part.capitalize() for part in area_slug.split("-"))


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
