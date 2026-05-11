from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon, mapping

from .extract import ExtractionResult
from .georef_transform import GeoreferenceTransform, ring_to_lonlat


def geometry_to_geojson(
    pixel_geometry: Polygon | MultiPolygon,
    width: int,
    height: int,
    geo_transform: GeoreferenceTransform,
) -> dict[str, Any]:
    if isinstance(pixel_geometry, Polygon):
        return {
            "type": "Polygon",
            "coordinates": polygon_coordinates(pixel_geometry, width, height, geo_transform),
        }
    return {
        "type": "MultiPolygon",
        "coordinates": [
            polygon_coordinates(poly, width, height, geo_transform) for poly in pixel_geometry.geoms
        ],
    }


def polygon_coordinates(
    poly: Polygon,
    width: int,
    height: int,
    geo_transform: GeoreferenceTransform,
) -> list[list[list[float]]]:
    exterior = ring_to_lonlat(poly.exterior.coords, width, height, geo_transform)
    interiors = [ring_to_lonlat(ring.coords, width, height, geo_transform) for ring in poly.interiors]
    return [exterior, *interiors]


def feature_collection(
    result: ExtractionResult,
    width: int,
    height: int,
    geo_transform: GeoreferenceTransform,
    image_path: str,
    city_input: str,
) -> dict[str, Any]:
    geom = geometry_to_geojson(result.pixel_geometry, width, height, geo_transform)
    feature = {
        "type": "Feature",
        "properties": {
            "city_input": city_input,
            "city": geo_transform.city,
            "style": result.style,
            "source_image": str(image_path),
            "coverage_ratio": round(result.coverage_ratio, 6),
            "contour_count": result.contour_count,
            "extraction_confidence": result.confidence,
            "georeference_confidence": geo_transform.confidence,
            "georeference_source": geo_transform.source,
            "meters_per_pixel": geo_transform.meters_per_pixel,
            "rotation_degrees": round(geo_transform.rotation_radians * 180.0 / 3.141592653589793, 6),
            "origin_lon": geo_transform.lon,
            "origin_lat": geo_transform.lat,
            "origin_x_ratio": geo_transform.origin_x_ratio,
            "origin_y_ratio": geo_transform.origin_y_ratio,
        },
        "geometry": geom,
    }
    return {
        "type": "FeatureCollection",
        "features": [feature],
        "metadata": {
            "generator": "map-boundary-builder",
            "image_width": width,
            "image_height": height,
            "pixel_geometry": mapping(result.pixel_geometry),
        },
    }


def write_geojson(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
