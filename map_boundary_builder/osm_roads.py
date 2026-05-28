from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
import os
from importlib import resources
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

from .geocoder import GeocodeResult
from .georef_transform import GeoreferenceTransform, lonlat_to_mercator, mercator_to_lonlat
from .ocr import OcrLabel

_CACHE_ROOT = Path(os.environ.get("MAP_BOUNDARY_CACHE_DIR", ".cache/map-boundary-builder"))
CACHE_DIR = _CACHE_ROOT / "overpass"
ROAD_REFINE_CACHE_DIR = _CACHE_ROOT / "road-refine"
ROAD_SEARCH_BATCH_SIZE = max(1, int(os.environ.get("MAP_BOUNDARY_ROAD_SEARCH_BATCH_SIZE", "256")))
ROAD_REFINE_CACHE_VERSION = "road-refine-v1"
OSM_ROAD_POINTS_SEED_FILE = "osm_road_points_seed.npz"
_ROAD_REFINE_MEMORY_CACHE: dict[str, RoadMatchResult | None] = {}
_ROAD_POINTS_SEED: dict[str, np.ndarray] | None = None


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
    *,
    lock_scale: bool = False,
) -> RoadMatchResult | None:
    if city_center.bbox is None:
        return None
    cache_key = road_refine_cache_key(rgb, city_center, initial, lock_scale=lock_scale)
    cached = read_road_refine_cache(cache_key)
    if cached is not None:
        return cached

    road_points = load_road_points(city_center.bbox)
    if road_points.size == 0:
        return None
    road_points = sample_road_points(road_points, max_points=12000)
    feature_distance = image_feature_distance(rgb)
    base_score, base_count = score_georeference_transform(road_points, feature_distance, initial)
    if base_count < 1000:
        return None

    base_tx, base_ty = lonlat_to_mercator(initial.lon, initial.lat)
    best_score = base_score
    best_count = base_count
    best_transform = initial

    if lock_scale:
        coarse_scale_multipliers = np.array([1.0])
        coarse_rotation_offsets = np.deg2rad(np.linspace(-1.0, 1.0, 5))
        coarse_offset_meters = np.linspace(-2400.0, 2400.0, 13)
        fine_scale_multipliers = np.array([1.0])
        fine_rotation_offsets = np.deg2rad(np.linspace(-0.5, 0.5, 5))
        fine_offset_meters = np.linspace(-500.0, 500.0, 5)
    else:
        coarse_scale_multipliers = np.linspace(0.82, 1.12, 13)
        coarse_rotation_offsets = np.deg2rad(np.linspace(-4.0, 4.0, 9))
        coarse_offset_meters = np.linspace(-1600.0, 1600.0, 9)
        fine_scale_multipliers = np.linspace(0.97, 1.03, 7)
        fine_rotation_offsets = np.deg2rad(np.linspace(-1.0, 1.0, 7))
        fine_offset_meters = np.linspace(-400.0, 400.0, 5)

    coarse = search_near_transform(
        road_points,
        feature_distance,
        initial,
        base_tx,
        base_ty,
        scale_multipliers=coarse_scale_multipliers,
        rotation_offsets=coarse_rotation_offsets,
        offset_meters=coarse_offset_meters,
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
            scale_multipliers=fine_scale_multipliers,
            rotation_offsets=fine_rotation_offsets,
            offset_meters=fine_offset_meters,
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
    result = RoadMatchResult(
        transform=refined,
        score=round(best_score, 6),
        sampled_points=best_count,
        base_score=round(base_score, 6),
    )
    write_road_refine_cache(cache_key, result)
    return result


def road_refine_cache_key(
    rgb: np.ndarray,
    city_center: GeocodeResult,
    initial: GeoreferenceTransform,
    *,
    lock_scale: bool,
) -> str:
    image_digest = hashlib.sha256(np.ascontiguousarray(rgb).data).hexdigest()
    bbox = tuple(round(value, 5) for value in (city_center.bbox or ()))
    parts = [
        ROAD_REFINE_CACHE_VERSION,
        image_digest,
        json.dumps(bbox, separators=(",", ":")),
        initial.source,
        f"{initial.lon:.8f}",
        f"{initial.lat:.8f}",
        f"{initial.meters_per_pixel:.8f}",
        f"{initial.rotation_radians:.10f}",
        f"{initial.confidence:.6f}",
        "lock" if lock_scale else "free",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def read_road_refine_cache(cache_key: str) -> RoadMatchResult | None:
    if cache_key in _ROAD_REFINE_MEMORY_CACHE:
        return _ROAD_REFINE_MEMORY_CACHE[cache_key]
    cache_path = ROAD_REFINE_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        transform = GeoreferenceTransform(**data["transform"])
        result = RoadMatchResult(
            transform=transform,
            score=float(data["score"]),
            sampled_points=int(data["sampled_points"]),
            base_score=float(data["base_score"]) if data.get("base_score") is not None else None,
        )
    except Exception:
        return None
    _ROAD_REFINE_MEMORY_CACHE[cache_key] = result
    return result


def write_road_refine_cache(cache_key: str, result: RoadMatchResult) -> None:
    _ROAD_REFINE_MEMORY_CACHE[cache_key] = result
    cache_path = ROAD_REFINE_CACHE_DIR / f"{cache_key}.json"
    payload = {
        "transform": {
            "city": result.transform.city,
            "lon": result.transform.lon,
            "lat": result.transform.lat,
            "origin_x_ratio": result.transform.origin_x_ratio,
            "origin_y_ratio": result.transform.origin_y_ratio,
            "meters_per_pixel": result.transform.meters_per_pixel,
            "rotation_radians": result.transform.rotation_radians,
            "confidence": result.transform.confidence,
            "source": result.transform.source,
        },
        "score": result.score,
        "sampled_points": result.sampled_points,
        "base_score": result.base_score,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")))
        tmp_path.replace(cache_path)
    except OSError:
        return


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
    best: tuple[float, int, float, float, float, float] | None = None
    batch: list[tuple[float, float, float, float]] = []

    def score_batch() -> None:
        nonlocal best, batch
        if not batch:
            return
        scores = score_transform_batch(road_points, feature_distance, batch)
        for (scale, rotation, tx, ty), (score, count) in zip(batch, scores):
            if count < min_count:
                continue
            if best is None or score > best[0]:
                best = (score, count, scale, rotation, tx, ty)
        batch = []

    for scale_multiplier in scale_multipliers:
        scale = float(base_transform.meters_per_pixel * scale_multiplier)
        for rotation_offset in rotation_offsets:
            rotation = float(base_transform.rotation_radians + rotation_offset)
            for offset_x in offset_meters:
                for offset_y in offset_meters:
                    batch.append((scale, rotation, base_tx + float(offset_x), base_ty + float(offset_y)))
                    if len(batch) >= ROAD_SEARCH_BATCH_SIZE:
                        score_batch()
    score_batch()
    if best is None:
        return None
    score, count, scale, rotation, tx, ty = best
    lon, lat = mercator_to_lonlat(tx, ty)
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
    return score, count, candidate


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
    road_points = sample_road_points(road_points, max_points=12000)
    feature_distance = image_feature_distance(rgb)

    west, south, east, north = city_center.bbox
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
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


def score_transform_batch(
    road_points: np.ndarray,
    feature_distance: np.ndarray,
    transforms: list[tuple[float, float, float, float]],
) -> list[tuple[float, int]]:
    if not transforms:
        return []
    h, w = feature_distance.shape
    params = np.asarray(transforms, dtype=float)
    scales = params[:, 0][:, np.newaxis]
    rotations = params[:, 1][:, np.newaxis]
    txs = params[:, 2][:, np.newaxis]
    tys = params[:, 3][:, np.newaxis]
    road_x = road_points[:, 0][np.newaxis, :]
    road_y = road_points[:, 1][np.newaxis, :]

    dx = (road_x - txs) / scales
    dy = (road_y - tys) / scales
    cos_r = np.cos(rotations)
    sin_r = np.sin(rotations)
    px = dx * cos_r + dy * sin_r
    py = -(-dx * sin_r + dy * cos_r)
    ix = np.rint(px).astype(np.int32)
    iy = np.rint(py).astype(np.int32)
    keep = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    counts = keep.sum(axis=1).astype(np.int32)
    if not np.any(counts):
        return [(0.0, 0) for _ in transforms]

    clipped_x = np.clip(ix, 0, max(w - 1, 0))
    clipped_y = np.clip(iy, 0, max(h - 1, 0))
    distances = feature_distance[clipped_y, clipped_x]
    scores = np.exp(-((distances / 6.0) ** 2))
    sums = np.where(keep, scores, 0.0).sum(axis=1)
    means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
    return [(float(score), int(count)) for score, count in zip(means, counts)]


def sample_road_points(road_points: np.ndarray, *, max_points: int) -> np.ndarray:
    if len(road_points) <= max_points:
        return road_points
    step = int(np.ceil(len(road_points) / max_points))
    return road_points[::step]


@lru_cache(maxsize=256)
def load_road_points(bbox: tuple[float, float, float, float]) -> np.ndarray:
    seeded = seed_road_points(overpass_cache_file(bbox).stem)
    if seeded is not None:
        return seeded
    payload = load_overpass_roads(bbox)
    points: list[tuple[float, float]] = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry", [])
        projected = [lonlat_to_mercator(point["lon"], point["lat"]) for point in geometry]
        points.extend(sample_line(projected, spacing_m=140.0))
    if not points:
        return np.empty((0, 2), dtype=float)
    arr = np.array(points, dtype=float)
    if len(arr) > 45000:
        step = int(np.ceil(len(arr) / 45000))
        arr = arr[::step]
    return arr


def seed_road_points(key: str) -> np.ndarray | None:
    seed = load_road_points_seed()
    return seed.get(key)


def load_road_points_seed() -> dict[str, np.ndarray]:
    global _ROAD_POINTS_SEED
    if _ROAD_POINTS_SEED is not None:
        return _ROAD_POINTS_SEED
    try:
        seed_file = resources.files("map_boundary_builder").joinpath(OSM_ROAD_POINTS_SEED_FILE)
        with np.load(io.BytesIO(seed_file.read_bytes())) as archive:
            payload = {key: np.asarray(archive[key], dtype=float) for key in archive.files}
    except Exception:
        payload = {}
    _ROAD_POINTS_SEED = payload
    return _ROAD_POINTS_SEED


@lru_cache(maxsize=256)
def load_road_segments(bbox: tuple[float, float, float, float]) -> np.ndarray:
    payload = load_overpass_roads(bbox)
    segments: list[tuple[float, float, float, float, float]] = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry", [])
        projected = [lonlat_to_mercator(point["lon"], point["lat"]) for point in geometry]
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


@lru_cache(maxsize=256)
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
