from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np
from pyproj import Transformer

from .geocoder import GeocodeResult
from .georef_transform import GeoreferenceTransform, lonlat_to_mercator, mercator_to_lonlat
from .ocr import OcrLabel

CACHE_DIR = Path(".cache/map-boundary-builder/overpass")
_TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


@dataclass(frozen=True)
class RoadMatchResult:
    transform: GeoreferenceTransform
    score: float
    sampled_points: int
    anchor_label: OcrLabel | None = None
    base_score: float | None = None


def refine_transform_with_osm_roads(
    rgb: np.ndarray,
    city_center: GeocodeResult,
    initial: GeoreferenceTransform,
) -> RoadMatchResult | None:
    if city_center.bbox is None:
        return None
    road_points = load_road_points(city_center.bbox)
    if road_points.size == 0:
        return None
    feature_distance = image_feature_distance(rgb)
    base_score, base_count = score_georeference_transform(road_points, feature_distance, initial)
    if base_count < 1000:
        return None

    base_tx, base_ty = lonlat_to_mercator(initial.lon, initial.lat)
    best_score = base_score
    best_count = base_count
    best_transform = initial

    coarse = search_near_transform(
        road_points,
        feature_distance,
        initial,
        base_tx,
        base_ty,
        scale_multipliers=np.linspace(0.82, 1.12, 13),
        rotation_offsets=np.deg2rad(np.linspace(-4.0, 4.0, 9)),
        offset_meters=np.linspace(-1600.0, 1600.0, 9),
        min_count=1000,
    )
    if coarse is not None:
        best_score, best_count, best_transform = coarse
        best_tx, best_ty = lonlat_to_mercator(best_transform.lon, best_transform.lat)
        fine = search_near_transform(
            road_points,
            feature_distance,
            best_transform,
            best_tx,
            best_ty,
            scale_multipliers=np.linspace(0.97, 1.03, 7),
            rotation_offsets=np.deg2rad(np.linspace(-1.0, 1.0, 7)),
            offset_meters=np.linspace(-400.0, 400.0, 5),
            min_count=1000,
        )
        if fine is not None and fine[0] > best_score:
            best_score, best_count, best_transform = fine

    if best_score < max(base_score + 0.04, 0.32):
        return None

    refined = GeoreferenceTransform(
        city=initial.city,
        lon=best_transform.lon,
        lat=best_transform.lat,
        origin_x_ratio=initial.origin_x_ratio,
        origin_y_ratio=initial.origin_y_ratio,
        meters_per_pixel=best_transform.meters_per_pixel,
        rotation_radians=best_transform.rotation_radians,
        confidence=min(0.93, max(initial.confidence, initial.confidence + min(0.08, best_score - base_score))),
        source=f"{initial.source}+osm-road-refine",
    )
    return RoadMatchResult(
        transform=refined,
        score=round(best_score, 6),
        sampled_points=best_count,
        base_score=round(base_score, 6),
    )


def search_near_transform(
    road_points: np.ndarray,
    feature_distance: np.ndarray,
    base_transform: GeoreferenceTransform,
    base_tx: float,
    base_ty: float,
    *,
    scale_multipliers: np.ndarray,
    rotation_offsets: np.ndarray,
    offset_meters: np.ndarray,
    min_count: int,
) -> tuple[float, int, GeoreferenceTransform] | None:
    best: tuple[float, int, GeoreferenceTransform] | None = None
    for scale_multiplier in scale_multipliers:
        scale = float(base_transform.meters_per_pixel * scale_multiplier)
        for rotation_offset in rotation_offsets:
            rotation = float(base_transform.rotation_radians + rotation_offset)
            for offset_x in offset_meters:
                for offset_y in offset_meters:
                    lon, lat = mercator_to_lonlat(base_tx + float(offset_x), base_ty + float(offset_y))
                    candidate = GeoreferenceTransform(
                        city=base_transform.city,
                        lon=lon,
                        lat=lat,
                        origin_x_ratio=base_transform.origin_x_ratio,
                        origin_y_ratio=base_transform.origin_y_ratio,
                        meters_per_pixel=scale,
                        rotation_radians=rotation,
                        confidence=base_transform.confidence,
                        source=base_transform.source,
                    )
                    score, count = score_georeference_transform(road_points, feature_distance, candidate)
                    if count < min_count:
                        continue
                    if best is None or score > best[0]:
                        best = (score, count, candidate)
    return best


def georeference_from_osm_roads(
    rgb: np.ndarray,
    city: str,
    city_center: GeocodeResult,
    anchor_labels: list[OcrLabel],
) -> RoadMatchResult | None:
    if not anchor_labels:
        return None
    if city_center.bbox is None:
        return None
    city_anchors = [label for label in anchor_labels[:8] if city.lower() in label.text.lower()]
    if not city_anchors:
        return None

    h, w = rgb.shape[:2]
    road_points = load_road_points(city_center.bbox)
    if road_points.size == 0:
        return None
    feature_distance = image_feature_distance(rgb)

    west, south, east, north = city_center.bbox
    west_m, south_m = _TO_MERCATOR.transform(west, south)
    east_m, north_m = _TO_MERCATOR.transform(east, north)
    base_scale = max(abs(east_m - west_m) / max(w, 1), abs(north_m - south_m) / max(h, 1))
    if base_scale <= 0:
        return None

    best: tuple[float, float, float, float, OcrLabel, int] | None = None
    city_x, city_y = city_center.mercator
    scale_candidates = base_scale * np.geomspace(0.18, 1.25, 44)
    offset_candidates = (-180, -90, 0, 90, 180)

    for anchor in city_anchors:
        anchor_x, anchor_y = city_x, city_y
        for scale in scale_candidates:
            for dx in offset_candidates:
                for dy in offset_candidates:
                    tx = anchor_x - scale * (anchor.x + dx)
                    ty = anchor_y - scale * (-anchor.y + dy)
                    score, count = score_transform(road_points, feature_distance, scale, tx, ty)
                    if count < 1500:
                        continue
                    if best is None or score > best[0]:
                        best = (score, float(scale), float(tx), float(ty), anchor, count)

    if best is None:
        return None
    score, scale, tx, ty, anchor, count = best
    if score < 0.75:
        return None

    lon, lat = mercator_to_lonlat(tx, ty)
    geo_transform = GeoreferenceTransform(
        city=city_center.display_name.split(",")[0],
        lon=lon,
        lat=lat,
        origin_x_ratio=0.0,
        origin_y_ratio=0.0,
        meters_per_pixel=scale,
        rotation_radians=0.0,
        confidence=min(0.82, max(0.56, score + 0.14)),
        source="osm-road-match:nominatim-overpass",
    )
    return RoadMatchResult(transform=geo_transform, score=round(score, 6), sampled_points=count, anchor_label=anchor)


def image_feature_distance(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    bright_linear = (((b > 165) & (g > 150) & (r < 190)) | ((val > 180) & (sat < 80))).astype(np.uint8)
    if bright_linear.mean() < 0.75:
        features = (edges > 0) | (bright_linear > 0)
    else:
        features = edges > 0
    return cv2.distanceTransform((features == 0).astype(np.uint8), cv2.DIST_L2, 5)


def score_transform(
    road_points: np.ndarray,
    feature_distance: np.ndarray,
    scale: float,
    tx: float,
    ty: float,
) -> tuple[float, int]:
    h, w = feature_distance.shape
    px = ((road_points[:, 0] - tx) / scale).round().astype(np.int32)
    py = (-(road_points[:, 1] - ty) / scale).round().astype(np.int32)
    keep = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if keep.sum() == 0:
        return 0.0, 0
    distances = feature_distance[py[keep], px[keep]]
    scores = np.exp(-((distances / 6.0) ** 2))
    return float(scores.mean()), int(keep.sum())


def score_georeference_transform(
    road_points: np.ndarray,
    feature_distance: np.ndarray,
    geo_transform: GeoreferenceTransform,
) -> tuple[float, int]:
    h, w = feature_distance.shape
    tx, ty = lonlat_to_mercator(geo_transform.lon, geo_transform.lat)
    dx = (road_points[:, 0] - tx) / geo_transform.meters_per_pixel
    dy = (road_points[:, 1] - ty) / geo_transform.meters_per_pixel
    cos_r = np.cos(geo_transform.rotation_radians)
    sin_r = np.sin(geo_transform.rotation_radians)
    px = dx * cos_r + dy * sin_r
    py = -(-dx * sin_r + dy * cos_r)
    ix = np.round(px).astype(np.int32)
    iy = np.round(py).astype(np.int32)
    keep = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    if keep.sum() == 0:
        return 0.0, 0
    distances = feature_distance[iy[keep], ix[keep]]
    scores = np.exp(-((distances / 6.0) ** 2))
    return float(scores.mean()), int(keep.sum())


def load_road_points(bbox: tuple[float, float, float, float]) -> np.ndarray:
    payload = load_overpass_roads(bbox)
    points: list[tuple[float, float]] = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry", [])
        projected = [_TO_MERCATOR.transform(point["lon"], point["lat"]) for point in geometry]
        points.extend(sample_line(projected, spacing_m=140.0))
    if not points:
        return np.empty((0, 2), dtype=float)
    arr = np.array(points, dtype=float)
    if len(arr) > 45000:
        step = int(np.ceil(len(arr) / 45000))
        arr = arr[::step]
    return arr


def load_road_segments(bbox: tuple[float, float, float, float]) -> np.ndarray:
    payload = load_overpass_roads(bbox)
    segments: list[tuple[float, float, float, float, float]] = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry", [])
        projected = [_TO_MERCATOR.transform(point["lon"], point["lat"]) for point in geometry]
        for start, end in zip(projected, projected[1:]):
            sx, sy = start
            ex, ey = end
            length = float(np.hypot(ex - sx, ey - sy))
            if length >= 30.0:
                segments.append((sx, sy, ex, ey, length))
    if not segments:
        return np.empty((0, 5), dtype=float)
    return np.array(segments, dtype=float)


def sample_line(points: list[tuple[float, float]], spacing_m: float) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        sx, sy = start
        ex, ey = end
        length = float(np.hypot(ex - sx, ey - sy))
        if length <= 0:
            continue
        count = max(1, int(length / spacing_m))
        for idx in range(count + 1):
            t = idx / count
            samples.append((sx + (ex - sx) * t, sy + (ey - sy) * t))
    return samples


def load_overpass_roads(bbox: tuple[float, float, float, float]) -> dict[str, object]:
    west, south, east, north = bbox
    cache_path = overpass_cache_file(bbox)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    query = f"""
[out:json][timeout:25];
(
  way[highway~"motorway|trunk|primary|secondary|tertiary"]({south},{west},{north},{east});
);
out geom;
"""
    request = Request(
        "https://overpass-api.de/api/interpreter",
        data=urlencode({"data": query}).encode("utf-8"),
        headers={"User-Agent": "map-boundary-builder/0.1 local road matcher"},
    )
    try:
        with urlopen(request, timeout=35) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {"elements": []}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload) + "\n")
    return payload


def overpass_cache_file(bbox: tuple[float, float, float, float]) -> Path:
    rounded = ",".join(f"{value:.4f}" for value in bbox)
    key = hashlib.sha256(rounded.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"
