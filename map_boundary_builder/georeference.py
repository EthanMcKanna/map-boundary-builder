from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, cos, exp, log, sin, sqrt
import re

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geocoder import GeocodeResult, geocode
from .georef_transform import GeoreferenceTransform, lonlat_to_mercator, mercator_to_lonlat
from .ocr import OcrLabel, extract_ocr_labels
from .osm_places import PlacePoint, load_place_points
from .osm_roads import (
    RoadMatchResult,
    load_road_points,
    load_road_segments,
    refine_transform_with_osm_roads,
    score_georeference_transform,
)

MAX_GEOCODED_LABELS = 16
MAX_PLACE_LABELS = 120
GENERIC_SINGLE_TOKENS = {
    "area",
    "bay",
    "beach",
    "center",
    "city",
    "district",
    "east",
    "heights",
    "hill",
    "lake",
    "los",
    "north",
    "park",
    "san",
    "south",
    "view",
    "west",
}


@dataclass(frozen=True)
class ControlPoint:
    label: OcrLabel
    geocode: GeocodeResult

    @property
    def pixel(self) -> tuple[float, float]:
        return self.label.x, -self.label.y

    @property
    def mercator(self) -> tuple[float, float]:
        return self.geocode.mercator


@dataclass(frozen=True)
class GeoreferenceResult:
    transform: GeoreferenceTransform
    control_points: list[ControlPoint]
    residual_median_m: float
    residual_p90_m: float
    road_match: RoadMatchResult | None = None


@dataclass(frozen=True)
class CityContextCandidate:
    score: float
    road_to_image: float
    image_to_road: float
    density: float
    scale_prior: float
    center_prior: float
    projected_count: int
    transform: GeoreferenceTransform


@dataclass(frozen=True)
class LineFeatureSet:
    midpoints: np.ndarray
    angles: np.ndarray
    weights: np.ndarray
    tree: cKDTree


def georeference_from_ocr(
    image_path: str,
    city: str,
    width: int,
    height: int,
    *,
    min_control_points: int = 3,
    label_y_max: float | None = None,
) -> GeoreferenceResult | None:
    labels = extract_ocr_labels(image_path)
    return georeference_from_labels(
        labels,
        image_path,
        city,
        width,
        height,
        min_control_points=min_control_points,
        label_y_max=label_y_max,
    )


def georeference_from_labels(
    labels: list[OcrLabel],
    image_path: str,
    city: str,
    width: int,
    height: int,
    *,
    min_control_points: int = 3,
    label_y_max: float | None = None,
) -> GeoreferenceResult | None:
    if label_y_max is not None:
        labels = [label for label in labels if label.y <= label_y_max]
    city_results = geocode(city, limit=1)
    if not city_results:
        return None
    city_center = city_results[0]
    controls = build_control_points(labels, city, city_center)
    if len(controls) >= min_control_points:
        fit = robust_similarity_fit(controls)
        if fit is not None:
            scale, rotation, tx, ty, inliers, residuals = fit
            if len(inliers) >= min_control_points:
                residual_values = np.array([residuals[i] for i in inliers], dtype=float)
                residual_median = float(np.median(residual_values))
                residual_p90 = float(np.percentile(residual_values, 90))
                if residual_median <= 2500 and residual_p90 <= 6500 and abs(rotation) <= 0.35:
                    lon, lat = mercator_to_lonlat(tx, ty)
                    confidence = confidence_from_fit(len(inliers), len(controls), residual_median, residual_p90)
                    geo_transform = GeoreferenceTransform(
                        city=city_center.display_name.split(",")[0],
                        lon=lon,
                        lat=lat,
                        origin_x_ratio=0.0,
                        origin_y_ratio=0.0,
                        meters_per_pixel=scale,
                        rotation_radians=rotation,
                        confidence=confidence,
                        source="ocr-georeference:nominatim-label-fit",
                    )
                    from .extract import load_rgb

                    road_refinement = refine_transform_with_osm_roads(load_rgb(image_path), city_center, geo_transform)
                    if road_refinement is not None:
                        geo_transform = road_refinement.transform
                    return GeoreferenceResult(
                        transform=geo_transform,
                        control_points=[controls[i] for i in inliers],
                        residual_median_m=residual_median,
                        residual_p90_m=residual_p90,
                        road_match=road_refinement,
                    )

    return None


def georeference_from_city_context(
    rgb: np.ndarray,
    city: str,
    pixel_geometry,
) -> GeoreferenceResult | None:
    city_results = geocode(city, limit=1)
    if not city_results:
        return None
    city_center = city_results[0]
    if city_center.bbox is None:
        return None

    search_bbox, city_radius_m, base_scale = city_search_context(city_center, rgb.shape[1], rgb.shape[0])
    road_points = load_road_points(search_bbox)
    if len(road_points) < 1200:
        return None
    if len(road_points) > 12000:
        step = int(np.ceil(len(road_points) / 12000))
        road_points = road_points[::step]

    feature_distance = city_context_feature_distance(rgb, pixel_geometry)
    image_features = city_context_feature_points(rgb, pixel_geometry)
    if len(image_features) < 80:
        return None

    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    center_pixel_x = float((min_x + max_x) / 2.0)
    center_pixel_y = float((min_y + max_y) / 2.0)
    city_x, city_y = city_center.mercator

    max_image_dim = max(rgb.shape[:2])
    line_features = city_context_line_features(rgb, pixel_geometry) if max_image_dim <= 900 else None
    if line_features is not None:
        line_best = search_city_context_line_candidates(
            rgb.shape[:2],
            road_points,
            load_road_segments(search_bbox),
            feature_distance,
            image_features,
            line_features,
            city_center.display_name.split(",")[0],
            city_x,
            city_y,
            city_radius_m,
            base_scale,
            center_pixel_x,
            center_pixel_y,
        )
        if line_best is not None:
            score, transform = line_best
            confidence = round(min(0.76, max(0.56, 0.54 + (score - 0.52) * 1.8)), 3)
            transform = GeoreferenceTransform(
                city=transform.city,
                lon=transform.lon,
                lat=transform.lat,
                origin_x_ratio=transform.origin_x_ratio,
                origin_y_ratio=transform.origin_y_ratio,
                meters_per_pixel=transform.meters_per_pixel,
                rotation_radians=transform.rotation_radians,
                confidence=confidence,
                source="city-context:osm-road-line-search",
            )
            return GeoreferenceResult(
                transform=transform,
                control_points=[],
                residual_median_m=0.0,
                residual_p90_m=0.0,
            )

    if min(rgb.shape[:2]) < 260 or max_image_dim > 1600:
        return None

    best = search_city_context_candidates(
        road_points,
        feature_distance,
        image_features,
        city_center.display_name.split(",")[0],
        city_x,
        city_y,
        city_radius_m,
        base_scale,
        center_pixel_x,
        center_pixel_y,
        coarse=True,
    )
    if best is None:
        return None

    fine = search_city_context_candidates(
        road_points,
        feature_distance,
        image_features,
        city_center.display_name.split(",")[0],
        city_x,
        city_y,
        city_radius_m,
        base_scale,
        center_pixel_x,
        center_pixel_y,
        coarse=False,
        seed=best,
    )
    if fine is not None and fine[0] > best[0]:
        best = fine

    score, transform = best
    if score < 0.56:
        return None
    confidence = round(min(0.68, max(0.56, score)), 3)
    transform = GeoreferenceTransform(
        city=transform.city,
        lon=transform.lon,
        lat=transform.lat,
        origin_x_ratio=transform.origin_x_ratio,
        origin_y_ratio=transform.origin_y_ratio,
        meters_per_pixel=transform.meters_per_pixel,
        rotation_radians=transform.rotation_radians,
        confidence=confidence,
        source="city-context:osm-road-search",
    )
    return GeoreferenceResult(
        transform=transform,
        control_points=[],
        residual_median_m=0.0,
        residual_p90_m=0.0,
    )


def city_search_context(
    city_center: GeocodeResult,
    width: int,
    height: int,
) -> tuple[tuple[float, float, float, float], float, float]:
    west, south, east, north = city_center.bbox or (city_center.lon, city_center.lat, city_center.lon, city_center.lat)
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
    city_width_m = max(abs(east_m - west_m), 8000.0)
    city_height_m = max(abs(north_m - south_m), 8000.0)
    radius_m = max(city_width_m, city_height_m) / 2.0
    center_x, center_y = city_center.mercator
    search_radius_m = max(12000.0, min(45000.0, radius_m * 1.35))
    search_west, search_south = mercator_to_lonlat(center_x - search_radius_m, center_y - search_radius_m)
    search_east, search_north = mercator_to_lonlat(center_x + search_radius_m, center_y + search_radius_m)
    base_scale = max(city_width_m / max(width, 1), city_height_m / max(height * 0.55, 1.0))
    return (search_west, search_south, search_east, search_north), radius_m, base_scale


def city_context_feature_distance(rgb: np.ndarray, pixel_geometry) -> np.ndarray:
    feature_mask = city_context_feature_mask(rgb, pixel_geometry)
    return cv2.distanceTransform((feature_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)


def city_context_feature_points(rgb: np.ndarray, pixel_geometry) -> np.ndarray:
    feature_mask = city_context_feature_mask(rgb, pixel_geometry)
    points = np.column_stack(np.where(feature_mask))
    if len(points) > 1200:
        step = int(np.ceil(len(points) / 1200))
        points = points[::step]
    return points


def city_context_feature_mask(rgb: np.ndarray, pixel_geometry) -> np.ndarray:
    h, w = rgb.shape[:2]
    _, min_y, _, max_y = pixel_geometry.bounds
    top = max(0, int(min_y) - 10)
    bottom = min(h, int(max_y) + 10)
    envelope = np.zeros((h, w), dtype=bool)
    envelope[top:bottom, :] = True
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 35, 100) > 0
    return edges & envelope


def search_city_context_candidates(
    road_points: np.ndarray,
    feature_distance: np.ndarray,
    image_features: np.ndarray,
    city_name: str,
    city_x: float,
    city_y: float,
    city_radius_m: float,
    base_scale: float,
    center_pixel_x: float,
    center_pixel_y: float,
    *,
    coarse: bool,
    seed: tuple[float, GeoreferenceTransform] | None = None,
) -> tuple[float, GeoreferenceTransform] | None:
    if coarse:
        offset_values = np.linspace(-0.65 * city_radius_m, 0.65 * city_radius_m, 9)
        scale_values = base_scale * np.geomspace(0.18, 0.46, 9)
        rotation_values = np.deg2rad(np.linspace(-8.0, 8.0, 5))
        centers = [(city_x + dx, city_y + dy) for dx in offset_values for dy in offset_values]
    else:
        assert seed is not None
        _, seed_transform = seed
        seed_tx, seed_ty = projected_center_for_transform(seed_transform, center_pixel_x, center_pixel_y)
        offset_values = np.linspace(-0.18 * city_radius_m, 0.18 * city_radius_m, 5)
        scale_values = seed_transform.meters_per_pixel * np.linspace(0.92, 1.08, 5)
        rotation_values = seed_transform.rotation_radians + np.deg2rad(np.linspace(-3.0, 3.0, 5))
        centers = [(seed_tx + dx, seed_ty + dy) for dx in offset_values for dy in offset_values]

    best: tuple[float, GeoreferenceTransform] | None = None
    for scale in scale_values:
        if scale <= 0:
            continue
        scale_prior = exp(-((log(scale / max(base_scale * 0.35, 1e-6)) / log(1.75)) ** 2))
        for center_x, center_y in centers:
            center_distance = sqrt((center_x - city_x) ** 2 + (center_y - city_y) ** 2)
            center_prior = exp(-((center_distance / max(city_radius_m * 0.45, 1.0)) ** 2))
            for rotation in rotation_values:
                transform = transform_from_projected_center(
                    city_name,
                    center_x,
                    center_y,
                    center_pixel_x,
                    center_pixel_y,
                    float(scale),
                    float(rotation),
                )
                road_to_image, projected_count = score_georeference_transform(road_points, feature_distance, transform)
                if projected_count < 600:
                    continue
                image_to_road, density = image_to_projected_roads_score(
                    road_points,
                    feature_distance.shape,
                    image_features,
                    transform,
                )
                density_prior = exp(-(((density - 0.30) / 0.28) ** 2))
                count_prior = exp(-(((projected_count - 7000.0) / 9000.0) ** 2))
                score = (
                    0.24 * road_to_image
                    + 0.28 * image_to_road
                    + 0.24 * center_prior
                    + 0.16 * scale_prior
                    + 0.05 * density_prior
                    + 0.03 * count_prior
                )
                if best is None or score > best[0]:
                    best = (float(score), transform)
    return best


def transform_from_projected_center(
    city_name: str,
    center_x: float,
    center_y: float,
    center_pixel_x: float,
    center_pixel_y: float,
    scale: float,
    rotation: float,
) -> GeoreferenceTransform:
    cos_r = cos(rotation)
    sin_r = sin(rotation)
    pixel_x = center_pixel_x
    pixel_y = -center_pixel_y
    offset_x = (pixel_x * cos_r - pixel_y * sin_r) * scale
    offset_y = (pixel_x * sin_r + pixel_y * cos_r) * scale
    lon, lat = mercator_to_lonlat(center_x - offset_x, center_y - offset_y)
    return GeoreferenceTransform(
        city=city_name,
        lon=lon,
        lat=lat,
        origin_x_ratio=0.0,
        origin_y_ratio=0.0,
        meters_per_pixel=scale,
        rotation_radians=rotation,
        confidence=0.0,
        source="city-context:osm-road-search",
    )


def projected_center_for_transform(
    transform: GeoreferenceTransform,
    center_pixel_x: float,
    center_pixel_y: float,
) -> tuple[float, float]:
    origin_x, origin_y = lonlat_to_mercator(transform.lon, transform.lat)
    pixel_x = center_pixel_x
    pixel_y = -center_pixel_y
    cos_r = cos(transform.rotation_radians)
    sin_r = sin(transform.rotation_radians)
    return (
        origin_x + (pixel_x * cos_r - pixel_y * sin_r) * transform.meters_per_pixel,
        origin_y + (pixel_x * sin_r + pixel_y * cos_r) * transform.meters_per_pixel,
    )


def image_to_projected_roads_score(
    road_points: np.ndarray,
    shape: tuple[int, int],
    image_features: np.ndarray,
    transform: GeoreferenceTransform,
) -> tuple[float, float]:
    h, w = shape
    origin_x, origin_y = lonlat_to_mercator(transform.lon, transform.lat)
    dx = (road_points[:, 0] - origin_x) / transform.meters_per_pixel
    dy = (road_points[:, 1] - origin_y) / transform.meters_per_pixel
    cos_r = np.cos(transform.rotation_radians)
    sin_r = np.sin(transform.rotation_radians)
    px = dx * cos_r + dy * sin_r
    py = -(-dx * sin_r + dy * cos_r)
    ix = np.round(px).astype(np.int32)
    iy = np.round(py).astype(np.int32)
    keep = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    if keep.sum() == 0:
        return 0.0, 0.0
    raster = np.zeros((h, w), dtype=np.uint8)
    raster[iy[keep], ix[keep]] = 255
    raster = cv2.dilate(raster, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    road_distance = cv2.distanceTransform((raster == 0).astype(np.uint8), cv2.DIST_L2, 5)
    distances = road_distance[image_features[:, 0], image_features[:, 1]]
    score = float(np.exp(-((distances / 5.0) ** 2)).mean())
    return score, float(raster.mean())


def city_context_line_features(rgb: np.ndarray, pixel_geometry) -> LineFeatureSet | None:
    h, w = rgb.shape[:2]
    raster = np.zeros((h, w), dtype=np.uint8)
    for poly in getattr(pixel_geometry, "geoms", [pixel_geometry]):
        exterior = np.array(poly.exterior.coords, dtype=np.int32)
        if len(exterior) >= 3:
            cv2.fillPoly(raster, [exterior], 255)

    scale = 5 if min(h, w) < 360 else 2
    up = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    mask_up = cv2.resize(raster, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(gray, 50, 120)
    edges[mask_up == 0] = 0

    min_line_length = max(28, int(round(min(h, w) * scale * 0.045)))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(24, int(round(min_line_length * 0.7))),
        minLineLength=min_line_length,
        maxLineGap=max(5, int(round(scale * 1.2))),
    )
    if lines is None:
        return None

    midpoints: list[tuple[float, float]] = []
    angles: list[float] = []
    weights: list[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(value) / scale for value in line)
        dx = x2 - x1
        dy = y2 - y1
        length = sqrt(dx * dx + dy * dy)
        if length < max(7.0, min(h, w) * 0.035):
            continue
        midpoints.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        angles.append(float(atan2(dy, dx) % np.pi))
        weights.append(float(min(length, 25.0)))

    if len(midpoints) < 25:
        return None
    midpoint_array = np.array(midpoints, dtype=float)
    return LineFeatureSet(
        midpoints=midpoint_array,
        angles=np.array(angles, dtype=float),
        weights=np.array(weights, dtype=float),
        tree=cKDTree(midpoint_array),
    )


def search_city_context_line_candidates(
    shape: tuple[int, int],
    road_points: np.ndarray,
    road_segments: np.ndarray,
    feature_distance: np.ndarray,
    image_features: np.ndarray,
    line_features: LineFeatureSet,
    city_name: str,
    city_x: float,
    city_y: float,
    city_radius_m: float,
    base_scale: float,
    center_pixel_x: float,
    center_pixel_y: float,
) -> tuple[float, GeoreferenceTransform] | None:
    if road_segments.size == 0:
        return None

    broad_candidates = collect_city_context_candidates(
        road_points,
        feature_distance,
        image_features,
        city_name,
        city_x,
        city_y,
        city_radius_m,
        base_scale,
        center_pixel_x,
        center_pixel_y,
    )
    if not broad_candidates:
        return None

    broad_scored = score_line_candidate_subset(
        select_line_candidate_subset(broad_candidates, per_metric=180, cap=520, per_bucket=4),
        road_segments,
        line_features,
        shape,
    )
    if not broad_scored:
        return None

    seeds = [candidate for _, _, candidate in broad_scored[:4]]
    fine_candidates = collect_fine_line_candidates(
        seeds,
        road_points,
        feature_distance,
        image_features,
        base_scale,
    )
    fine_scored = score_line_candidate_subset(
        select_line_candidate_subset(fine_candidates, per_metric=260, cap=560, per_bucket=6),
        road_segments,
        line_features,
        shape,
    )
    scored = sorted([*broad_scored, *fine_scored], key=lambda item: item[0], reverse=True)
    if not scored:
        return None
    final_score, line_score, candidate = scored[0]
    if final_score < 0.52 or line_score < 0.40:
        return None
    return final_score, candidate.transform


def collect_city_context_candidates(
    road_points: np.ndarray,
    feature_distance: np.ndarray,
    image_features: np.ndarray,
    city_name: str,
    city_x: float,
    city_y: float,
    city_radius_m: float,
    base_scale: float,
    center_pixel_x: float,
    center_pixel_y: float,
) -> list[CityContextCandidate]:
    candidates: list[CityContextCandidate] = []
    offset_values = np.linspace(-0.65 * city_radius_m, 0.65 * city_radius_m, 9)
    scale_values = base_scale * np.geomspace(0.18, 0.46, 9)
    rotation_values = np.deg2rad(np.linspace(-8.0, 8.0, 5))
    for scale in scale_values:
        if scale <= 0:
            continue
        scale_prior = exp(-((log(scale / max(base_scale * 0.35, 1e-6)) / log(1.75)) ** 2))
        for offset_x in offset_values:
            for offset_y in offset_values:
                center_x = city_x + float(offset_x)
                center_y = city_y + float(offset_y)
                center_distance = sqrt(offset_x * offset_x + offset_y * offset_y)
                center_prior = exp(-((center_distance / max(city_radius_m * 0.45, 1.0)) ** 2))
                for rotation in rotation_values:
                    transform = transform_from_projected_center(
                        city_name,
                        center_x,
                        center_y,
                        center_pixel_x,
                        center_pixel_y,
                        float(scale),
                        float(rotation),
                    )
                    candidate = evaluate_line_context_candidate(
                        road_points,
                        feature_distance,
                        image_features,
                        base_scale,
                        transform,
                        center_prior,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return candidates


def collect_fine_line_candidates(
    seeds: list[CityContextCandidate],
    road_points: np.ndarray,
    feature_distance: np.ndarray,
    image_features: np.ndarray,
    base_scale: float,
) -> list[CityContextCandidate]:
    candidates: list[CityContextCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for seed in seeds:
        seed_transform = seed.transform
        origin_x, origin_y = lonlat_to_mercator(seed_transform.lon, seed_transform.lat)
        for scale_multiplier in np.linspace(0.86, 1.12, 7):
            scale = seed_transform.meters_per_pixel * float(scale_multiplier)
            for rotation in seed_transform.rotation_radians + np.deg2rad(np.linspace(-3.0, 3.0, 7)):
                for offset_x in np.linspace(-2400.0, 2400.0, 9):
                    for offset_y in np.linspace(-2400.0, 2400.0, 9):
                        lon, lat = mercator_to_lonlat(origin_x + float(offset_x), origin_y + float(offset_y))
                        key = (
                            round((origin_x + float(offset_x)) / 200.0),
                            round((origin_y + float(offset_y)) / 200.0),
                            round(scale * 10.0),
                            round(float(rotation) * 1000.0),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        transform = GeoreferenceTransform(
                            city=seed_transform.city,
                            lon=lon,
                            lat=lat,
                            origin_x_ratio=seed_transform.origin_x_ratio,
                            origin_y_ratio=seed_transform.origin_y_ratio,
                            meters_per_pixel=scale,
                            rotation_radians=float(rotation),
                            confidence=0.0,
                            source=seed_transform.source,
                        )
                        candidate = evaluate_line_context_candidate(
                            road_points,
                            feature_distance,
                            image_features,
                            base_scale,
                            transform,
                            seed.center_prior,
                        )
                        if candidate is not None:
                            candidates.append(candidate)
    return candidates


def evaluate_line_context_candidate(
    road_points: np.ndarray,
    feature_distance: np.ndarray,
    image_features: np.ndarray,
    base_scale: float,
    transform: GeoreferenceTransform,
    center_prior: float,
) -> CityContextCandidate | None:
    road_to_image, projected_count = score_georeference_transform(road_points, feature_distance, transform)
    if projected_count < 400:
        return None
    image_to_road, density = image_to_projected_roads_score(
        road_points,
        feature_distance.shape,
        image_features,
        transform,
    )
    scale_prior = exp(-((log(transform.meters_per_pixel / max(base_scale * 0.35, 1e-6)) / log(1.75)) ** 2))
    density_prior = exp(-(((density - 0.12) / 0.10) ** 2))
    count_prior = exp(-(((projected_count - 1800.0) / 1600.0) ** 2))
    score = (
        0.42 * road_to_image
        + 0.30 * image_to_road
        + 0.13 * scale_prior
        + 0.08 * density_prior
        + 0.04 * count_prior
        + 0.03 * center_prior
    )
    return CityContextCandidate(
        score=float(score),
        road_to_image=float(road_to_image),
        image_to_road=float(image_to_road),
        density=float(density),
        scale_prior=float(scale_prior),
        center_prior=float(center_prior),
        projected_count=int(projected_count),
        transform=transform,
    )


def select_line_candidate_subset(
    candidates: list[CityContextCandidate],
    *,
    per_metric: int,
    cap: int,
    per_bucket: int = 2,
) -> list[CityContextCandidate]:
    selected: list[CityContextCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    coarse_counts: dict[tuple[int, int, int, int], int] = {}

    def add(candidate: CityContextCandidate) -> None:
        origin_x, origin_y = lonlat_to_mercator(candidate.transform.lon, candidate.transform.lat)
        key = (
            round(origin_x / 200.0),
            round(origin_y / 200.0),
            round(candidate.transform.meters_per_pixel * 10.0),
            round(candidate.transform.rotation_radians * 1000.0),
        )
        coarse_key = (
            round(origin_x / 1400.0),
            round(origin_y / 1400.0),
            round(candidate.transform.meters_per_pixel / 8.0),
            round(candidate.transform.rotation_radians * 12.0),
        )
        if key not in seen:
            if coarse_counts.get(coarse_key, 0) >= per_bucket:
                return
            seen.add(key)
            coarse_counts[coarse_key] = coarse_counts.get(coarse_key, 0) + 1
            selected.append(candidate)

    metrics = [
        lambda candidate: candidate.score,
        lambda candidate: candidate.road_to_image,
        lambda candidate: candidate.image_to_road,
        lambda candidate: candidate.scale_prior,
    ]
    for metric in metrics:
        for candidate in sorted(candidates, key=metric, reverse=True)[:per_metric]:
            add(candidate)
            if len(selected) >= cap:
                return selected
    return selected


def score_line_candidate_subset(
    candidates: list[CityContextCandidate],
    road_segments: np.ndarray,
    line_features: LineFeatureSet,
    shape: tuple[int, int],
) -> list[tuple[float, float, CityContextCandidate]]:
    scored: list[tuple[float, float, CityContextCandidate]] = []
    for candidate in candidates:
        line_score = directional_line_score(road_segments, line_features, candidate.transform, shape)
        final_score = (
            0.64 * line_score
            + 0.17 * candidate.score
            + 0.11 * candidate.road_to_image
            + 0.08 * candidate.scale_prior
        )
        scored.append((float(final_score), float(line_score), candidate))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def directional_line_score(
    road_segments: np.ndarray,
    line_features: LineFeatureSet,
    transform: GeoreferenceTransform,
    shape: tuple[int, int],
) -> float:
    h, w = shape
    origin_x, origin_y = lonlat_to_mercator(transform.lon, transform.lat)
    scale = transform.meters_per_pixel
    cos_r = cos(transform.rotation_radians)
    sin_r = sin(transform.rotation_radians)

    dx1 = (road_segments[:, 0] - origin_x) / scale
    dy1 = (road_segments[:, 1] - origin_y) / scale
    dx2 = (road_segments[:, 2] - origin_x) / scale
    dy2 = (road_segments[:, 3] - origin_y) / scale
    x1 = dx1 * cos_r + dy1 * sin_r
    y1 = -(-dx1 * sin_r + dy1 * cos_r)
    x2 = dx2 * cos_r + dy2 * sin_r
    y2 = -(-dx2 * sin_r + dy2 * cos_r)
    keep = (
        (np.maximum(x1, x2) >= 0)
        & (np.minimum(x1, x2) < w)
        & (np.maximum(y1, y2) >= 0)
        & (np.minimum(y1, y2) < h)
    )
    if int(keep.sum()) < 20:
        return 0.0

    x1 = x1[keep]
    y1 = y1[keep]
    x2 = x2[keep]
    y2 = y2[keep]
    midpoints = np.column_stack(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    road_angles = np.arctan2(y2 - y1, x2 - x1) % np.pi
    road_weights = np.minimum(np.hypot(x2 - x1, y2 - y1), 25.0)

    road_to_image = directional_nearest_score(
        midpoints,
        road_angles,
        road_weights,
        line_features.tree,
        line_features.angles,
        max_distance=12.0,
    )
    road_tree = cKDTree(midpoints)
    image_to_road = directional_nearest_score(
        line_features.midpoints,
        line_features.angles,
        line_features.weights,
        road_tree,
        road_angles,
        max_distance=10.0,
    )
    return float(0.45 * road_to_image + 0.55 * image_to_road)


def directional_nearest_score(
    query_midpoints: np.ndarray,
    query_angles: np.ndarray,
    query_weights: np.ndarray,
    target_tree: cKDTree,
    target_angles: np.ndarray,
    *,
    max_distance: float,
) -> float:
    k = min(8, len(target_angles))
    if k == 0:
        return 0.0
    if len(query_midpoints) == 0:
        return 0.0
    distances, indexes = target_tree.query(query_midpoints, k=k, distance_upper_bound=max_distance)
    distances = np.atleast_2d(distances)
    indexes = np.atleast_2d(indexes)
    if distances.shape[0] != len(query_midpoints):
        distances = distances.T
        indexes = indexes.T
    valid = indexes < len(target_angles)
    safe_indexes = np.where(valid, indexes, 0).astype(int)
    angle_delta = np.abs((query_angles[:, None] - target_angles[safe_indexes] + np.pi / 2.0) % np.pi - np.pi / 2.0)
    values = np.exp(-((distances / 5.0) ** 2)) * np.exp(-((angle_delta / np.deg2rad(12.0)) ** 2))
    values = np.where(valid, values, 0.0)
    best = values.max(axis=1)
    return float(np.average(best, weights=query_weights.astype(float)))


def build_control_points(labels: list[OcrLabel], city: str, city_center: GeocodeResult) -> list[ControlPoint]:
    place_controls = build_osm_place_control_points(labels, city_center)
    if len(place_controls) >= 3:
        return place_controls

    city_x, city_y = city_center.mercator
    max_distance_m = 70000.0
    controls: list[ControlPoint] = []
    used_text: set[tuple[str, ...]] = set()
    city_tokens = place_tokens(city)
    for label in rank_geocode_labels(labels)[:MAX_GEOCODED_LABELS]:
        text_tokens = place_tokens(label.text)
        text_key = tuple(sorted(text_tokens))
        if text_key in used_text:
            continue
        used_text.add(text_key)
        if text_tokens == city_tokens:
            controls.append(ControlPoint(label=label, geocode=city_center))
            continue
        if len(text_tokens) == 1 and next(iter(text_tokens)) in GENERIC_SINGLE_TOKENS:
            continue
        queries = [f"{label.text}, {context}" for context in geocode_contexts(city, city_center)]
        queries.append(label.text)
        if city.lower() in label.text.lower():
            queries.append(label.text)
        best: GeocodeResult | None = None
        best_score = -1.0
        for query in [q for q in queries if q]:
            for candidate in geocode(query, limit=3):
                if not primary_name_matches_label(candidate.display_name, label.text):
                    continue
                cand_x, cand_y = candidate.mercator
                distance = sqrt((cand_x - city_x) ** 2 + (cand_y - city_y) ** 2)
                if distance > max_distance_m:
                    continue
                score = candidate.importance - distance / max_distance_m
                if score > best_score:
                    best = candidate
                    best_score = score
        if best is not None:
            add_or_replace_control(controls, ControlPoint(label=label, geocode=best))
    return controls


def build_osm_place_control_points(labels: list[OcrLabel], city_center: GeocodeResult) -> list[ControlPoint]:
    search_bbox = city_center.bbox
    if search_bbox is None:
        return []
    places = load_place_points(search_bbox)
    if not places:
        return []

    city_x, city_y = city_center.mercator
    max_distance_m = max_place_distance_m(search_bbox, city_center)
    best_by_place: dict[tuple[str, float, float], tuple[float, ControlPoint]] = {}
    used_label_keys: set[tuple[str, int, int]] = set()
    for label in rank_place_labels(labels)[:MAX_PLACE_LABELS]:
        label_tokens = place_tokens(label.text)
        if not label_tokens or len(label_tokens) > 3:
            continue
        if len(label_tokens) == 1 and next(iter(label_tokens)) in GENERIC_SINGLE_TOKENS:
            continue
        label_key = (label.text.lower(), round(label.x / 12), round(label.y / 12))
        if label_key in used_label_keys:
            continue
        used_label_keys.add(label_key)

        for place in places:
            place_tokens_set = place_tokens(place.name)
            if not place_tokens_set:
                continue
            match_score = place_match_score(label_tokens, place_tokens_set)
            if match_score <= 0:
                continue
            place_x, place_y = place.mercator
            distance_m = sqrt((place_x - city_x) ** 2 + (place_y - city_y) ** 2)
            if distance_m > max_distance_m:
                continue
            score = (
                match_score
                + place_type_score(place.place_type)
                + label.confidence / 300.0
                - 0.12 * distance_m / max_distance_m
                - 0.04 * abs(len(place_tokens_set) - len(label_tokens))
            )
            key = (place.name.lower(), round(place.lon, 5), round(place.lat, 5))
            old = best_by_place.get(key)
            if old is not None and old[0] >= score:
                continue
            best_by_place[key] = (
                score,
                ControlPoint(
                    label=label,
                    geocode=GeocodeResult(
                        label=label.text,
                        lon=place.lon,
                        lat=place.lat,
                        display_name=f"{place.name}, {place.place_type}",
                        bbox=None,
                        importance=score,
                    ),
                ),
            )

    controls = [control for _, control in sorted(best_by_place.values(), key=lambda item: item[0], reverse=True)]
    return dedupe_place_controls(controls)


def max_place_distance_m(bbox: tuple[float, float, float, float], city_center: GeocodeResult) -> float:
    center_x, center_y = city_center.mercator
    corners = [
        (bbox[0], bbox[1]),
        (bbox[0], bbox[3]),
        (bbox[2], bbox[1]),
        (bbox[2], bbox[3]),
    ]
    corner_distances = []
    from .georef_transform import lonlat_to_mercator

    for lon, lat in corners:
        x, y = lonlat_to_mercator(lon, lat)
        corner_distances.append(sqrt((x - center_x) ** 2 + (y - center_y) ** 2))
    return max(25000.0, min(95000.0, max(corner_distances, default=70000.0)))


def place_match_score(label_tokens: set[str], place_tokens_set: set[str]) -> float:
    overlap = label_tokens & place_tokens_set
    if not overlap:
        return 0.0
    exact = label_tokens == place_tokens_set
    if exact:
        return 1.7
    if label_tokens <= place_tokens_set:
        return 1.2 * len(overlap) / max(len(place_tokens_set), 1)
    if place_tokens_set <= label_tokens:
        return 0.72 * len(overlap) / max(len(label_tokens), 1)
    overlap_ratio = len(overlap) / max(len(label_tokens), len(place_tokens_set))
    if overlap_ratio >= 0.75:
        return 0.55 * overlap_ratio
    return 0.0


def place_type_score(place_type: str) -> float:
    return {
        "city": 0.45,
        "town": 0.38,
        "village": 0.30,
        "suburb": 0.22,
        "quarter": 0.18,
        "neighbourhood": 0.16,
        "locality": 0.07,
        "hamlet": 0.05,
    }.get(place_type, 0.0)


def rank_place_labels(labels: list[OcrLabel]) -> list[OcrLabel]:
    return sorted(
        labels,
        key=lambda label: (
            token_quality(place_tokens(label.text)),
            label.confidence,
            -len(label.text),
        ),
        reverse=True,
    )


def rank_geocode_labels(labels: list[OcrLabel]) -> list[OcrLabel]:
    return sorted(
        labels,
        key=lambda label: (
            min(2, len(place_tokens(label.text))),
            label.confidence,
            -len(label.text),
        ),
        reverse=True,
    )


def token_quality(tokens: set[str]) -> int:
    if not tokens:
        return 0
    if len(tokens) == 1 and next(iter(tokens)) in GENERIC_SINGLE_TOKENS:
        return 0
    return min(3, len(tokens))


def dedupe_place_controls(controls: list[ControlPoint]) -> list[ControlPoint]:
    deduped: list[ControlPoint] = []
    used_label_positions: set[tuple[int, int]] = set()
    used_place_names: set[str] = set()
    for control in controls:
        position_key = (round(control.label.x / 8), round(control.label.y / 8))
        name_key = control.geocode.display_name.split(",", 1)[0].lower()
        if position_key in used_label_positions or name_key in used_place_names:
            continue
        used_label_positions.add(position_key)
        used_place_names.add(name_key)
        deduped.append(control)
    return deduped


def add_or_replace_control(controls: list[ControlPoint], candidate: ControlPoint) -> None:
    cand_x, cand_y = candidate.geocode.mercator
    for idx, existing in enumerate(controls):
        existing_x, existing_y = existing.geocode.mercator
        if sqrt((cand_x - existing_x) ** 2 + (cand_y - existing_y) ** 2) > 120.0:
            continue
        if control_quality(candidate) > control_quality(existing):
            controls[idx] = candidate
        return
    controls.append(candidate)


def control_quality(control: ControlPoint) -> tuple[int, float]:
    return len(place_tokens(control.label.text)), control.label.confidence


def geocode_contexts(city: str, city_center: GeocodeResult) -> list[str]:
    contexts: list[str] = []
    for value in [city, *city_center.display_name.split(",")[:2]]:
        value = value.strip()
        lowered = value.lower()
        if not value or value.isdigit() or lowered == "united states" or "county" in lowered:
            continue
        if value.lower() not in {context.lower() for context in contexts}:
            contexts.append(value)
    return contexts


def primary_name_matches_label(display_name: str, label_text: str) -> bool:
    label_tokens = place_tokens(label_text)
    if not label_tokens:
        return False
    primary_tokens = place_tokens(display_name.split(",", 1)[0])
    return label_tokens <= primary_tokens


def place_tokens(value: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", value.lower()) if len(part) >= 3}


def robust_similarity_fit(
    controls: list[ControlPoint],
) -> tuple[float, float, float, float, list[int], list[float]] | None:
    if len(controls) < 2:
        return None
    pixel = np.array([control.pixel for control in controls], dtype=float)
    merc = np.array([control.mercator for control in controls], dtype=float)

    best: tuple[float, float, float, float, list[int], list[float]] | None = None
    best_score: float | None = None
    total_spread = control_spread(pixel)
    for i, j in combinations(range(len(controls)), 2):
        p1, p2 = pixel[i], pixel[j]
        m1, m2 = merc[i], merc[j]
        p_vec = p2 - p1
        m_vec = m2 - m1
        p_norm = np.linalg.norm(p_vec)
        m_norm = np.linalg.norm(m_vec)
        if p_norm < 30 or m_norm < 300:
            continue
        scale = float(m_norm / p_norm)
        if scale <= 0 or scale > 500:
            continue
        rotation = float(atan2(m_vec[1], m_vec[0]) - atan2(p_vec[1], p_vec[0]))
        transformed = apply_similarity(pixel, scale, rotation, 0.0, 0.0)
        tx, ty = (m1 - transformed[i]).tolist()
        residuals = np.linalg.norm(apply_similarity(pixel, scale, rotation, tx, ty) - merc, axis=1).tolist()
        threshold = max(1200.0, scale * 90.0)
        inliers = [idx for idx, residual in enumerate(residuals) if residual <= threshold]
        if len(inliers) < 2:
            continue
        refined = fit_similarity(pixel[inliers], merc[inliers])
        if refined is None:
            continue
        r_scale, r_rotation, r_tx, r_ty = refined
        if abs(r_rotation) > 0.35:
            continue
        r_residuals = np.linalg.norm(apply_similarity(pixel, r_scale, r_rotation, r_tx, r_ty) - merc, axis=1).tolist()
        r_threshold = max(1200.0, r_scale * 90.0)
        r_inliers = [idx for idx, residual in enumerate(r_residuals) if residual <= r_threshold]
        if len(r_inliers) < 3:
            continue
        spread = control_spread(pixel[r_inliers])
        residual_values = [r_residuals[idx] for idx in r_inliers]
        median = float(np.median(residual_values)) if residual_values else float("inf")
        p90 = float(np.percentile(residual_values, 90)) if residual_values else float("inf")
        score = fit_candidate_score(len(r_inliers), len(controls), spread, total_spread, median, p90, r_rotation)
        if best_score is None or score > best_score:
            best_score = score
            best = (r_scale, r_rotation, r_tx, r_ty, r_inliers, r_residuals)

    for seed in combinations(range(len(controls)), 3):
        refined = fit_similarity(pixel[list(seed)], merc[list(seed)])
        if refined is None:
            continue
        r_scale, r_rotation, r_tx, r_ty = refined
        if abs(r_rotation) > 0.35:
            continue
        r_residuals = np.linalg.norm(apply_similarity(pixel, r_scale, r_rotation, r_tx, r_ty) - merc, axis=1).tolist()
        r_threshold = max(1200.0, r_scale * 90.0)
        r_inliers = [idx for idx, residual in enumerate(r_residuals) if residual <= r_threshold]
        if len(r_inliers) < 3:
            continue
        spread = control_spread(pixel[r_inliers])
        residual_values = [r_residuals[idx] for idx in r_inliers]
        median = float(np.median(residual_values))
        p90 = float(np.percentile(residual_values, 90))
        score = fit_candidate_score(len(r_inliers), len(controls), spread, total_spread, median, p90, r_rotation)
        if best_score is None or score > best_score:
            best_score = score
            best = (r_scale, r_rotation, r_tx, r_ty, r_inliers, r_residuals)
    return best


def control_spread(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    width, height = np.ptp(points, axis=0)
    return float(width * height)


def fit_candidate_score(
    inlier_count: int,
    total_count: int,
    spread: float,
    total_spread: float,
    median_residual: float,
    p90_residual: float,
    rotation: float,
) -> float:
    residual_score = max(0.0, 1.0 - median_residual / 2000.0) * 0.7
    residual_score += max(0.0, 1.0 - p90_residual / 4500.0) * 0.3
    rotation_score = max(0.0, 1.0 - abs(rotation) / 0.35)
    spread_score = min(1.0, spread / max(total_spread, 1.0))
    inlier_score = min(1.0, inlier_count / max(total_count, 3))
    return 0.35 * residual_score + 0.35 * rotation_score + 0.2 * spread_score + 0.1 * inlier_score


def fit_similarity(pixel: np.ndarray, merc: np.ndarray) -> tuple[float, float, float, float] | None:
    if len(pixel) < 2:
        return None
    p_mean = pixel.mean(axis=0)
    m_mean = merc.mean(axis=0)
    p_centered = pixel - p_mean
    m_centered = merc - m_mean
    variance = float((p_centered**2).sum())
    if variance <= 0:
        return None
    covariance = p_centered.T @ m_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation_matrix = vt.T @ u.T
    if np.linalg.det(rotation_matrix) < 0:
        vt[-1, :] *= -1
        rotation_matrix = vt.T @ u.T
    scale = float(singular_values.sum() / variance)
    translation = m_mean - scale * (rotation_matrix @ p_mean)
    rotation = float(atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    return scale, rotation, float(translation[0]), float(translation[1])


def apply_similarity(points: np.ndarray, scale: float, rotation: float, tx: float, ty: float) -> np.ndarray:
    rot = np.array([[cos(rotation), -sin(rotation)], [sin(rotation), cos(rotation)]])
    return scale * (points @ rot.T) + np.array([tx, ty])


def confidence_from_fit(control_count: int, total_count: int, median_residual: float, p90_residual: float) -> float:
    count_score = min(1.0, control_count / 6.0)
    inlier_score = control_count / max(total_count, 1)
    residual_score = max(0.0, 1.0 - median_residual / 2500.0) * 0.7 + max(0.0, 1.0 - p90_residual / 6500.0) * 0.3
    return round(0.25 + 0.75 * (0.45 * residual_score + 0.35 * count_score + 0.2 * inlier_score), 3)
