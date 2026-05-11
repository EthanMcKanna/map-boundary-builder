from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, degrees, exp, log, pi, radians, sin, tan
from typing import Iterable

EARTH_RADIUS_M = 6378137.0
MAX_MERCATOR_LAT = 85.0511287798066


@dataclass(frozen=True)
class GeoreferenceTransform:
    city: str
    lon: float
    lat: float
    origin_x_ratio: float
    origin_y_ratio: float
    meters_per_pixel: float
    rotation_radians: float
    confidence: float
    source: str


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
    x = EARTH_RADIUS_M * radians(lon)
    y = EARTH_RADIUS_M * log(tan(pi / 4.0 + radians(lat) / 2.0))
    return x, y


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = degrees(x / EARTH_RADIUS_M)
    lat = degrees(2.0 * atan(exp(y / EARTH_RADIUS_M)) - pi / 2.0)
    return lon, lat


def pixel_to_lonlat(
    x: float,
    y: float,
    width: int,
    height: int,
    geo_transform: GeoreferenceTransform,
) -> tuple[float, float]:
    origin_x = geo_transform.origin_x_ratio * width
    origin_y = geo_transform.origin_y_ratio * height
    origin_merc_x, origin_merc_y = lonlat_to_mercator(geo_transform.lon, geo_transform.lat)
    pixel_x = x - origin_x
    pixel_y = origin_y - y
    cos_r = cos(geo_transform.rotation_radians)
    sin_r = sin(geo_transform.rotation_radians)
    rotated_x = pixel_x * cos_r - pixel_y * sin_r
    rotated_y = pixel_x * sin_r + pixel_y * cos_r
    merc_x = origin_merc_x + rotated_x * geo_transform.meters_per_pixel
    merc_y = origin_merc_y + rotated_y * geo_transform.meters_per_pixel
    return mercator_to_lonlat(merc_x, merc_y)


def ring_to_lonlat(
    coords: Iterable[tuple[float, float]],
    width: int,
    height: int,
    geo_transform: GeoreferenceTransform,
) -> list[list[float]]:
    return [
        [round(lon, 7), round(lat, 7)]
        for lon, lat in (pixel_to_lonlat(x, y, width, height, geo_transform) for x, y in coords)
    ]
