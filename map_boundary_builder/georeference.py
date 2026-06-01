from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
from math import atan2, cos, exp, log, sin, sqrt
import os
import re
import time

import cv2
import numpy as np

from .geocoder import GeocodeResult, geocode, geocode_cached_only
from .georef_transform import GeoreferenceTransform, lonlat_to_mercator, mercator_to_lonlat
from .ocr import OcrLabel, extract_ocr_labels
from .osm_places import PlacePoint, load_place_points
from .osm_roads import (
    RoadMatchResult,
    has_local_road_points,
    image_feature_distance,
    load_road_points,
    load_road_segments,
    refine_transform_with_osm_roads,
    sample_road_points,
    score_georeference_transform,
)

MAX_GEOCODED_LABELS = 16
MAX_SPARSE_GEOCODED_LABELS = 32
MAX_PLACE_LABELS = 120
MAX_CITY_INFERENCE_LABELS = 48
MAX_CITY_CONTEXTS = 6
MAX_ROBUST_SIMILARITY_METERS_PER_PIXEL = 500.0
MAX_SPARSE_ROBUST_SIMILARITY_METERS_PER_PIXEL = 600.0
MAX_SPARSE_ROBUST_SIMILARITY_INLIER_RESIDUAL_M = 6500.0
MAX_TWO_CONTROL_SIMILARITY_METERS_PER_PIXEL = 500.0
DIRECT_CONTEXT_QUERY_LIMIT = 6
DIRECT_CONTEXT_MAX_QUERIES = 10
DIRECT_CONTEXT_LIVE_QUERY_LIMIT = 3
DIRECT_CONTEXT_MAX_LIVE_QUERIES = 5
MARKER_DOT_BACKGROUND_SAMPLE_STRIDE = 8
PROMOTED_DIRECT_CONTEXT_LABEL_SCORE = 95.0
STRONG_DIRECT_CONTEXT_MIN_SCORE = 115.0
GEOCODE_BATCH_SIZE = max(1, int(os.environ.get("MAP_BOUNDARY_GEOCODE_BATCH_SIZE", "12")))
GEOCODE_WORKERS = max(1, int(os.environ.get("MAP_BOUNDARY_GEOCODE_WORKERS", "6")))
GEOCODE_LABEL_LOOKAHEAD = max(1, int(os.environ.get("MAP_BOUNDARY_GEOCODE_LABEL_LOOKAHEAD", "3")))
PLACE_FAST_PATH_TIMEOUT_SECONDS = max(0.0, float(os.environ.get("MAP_BOUNDARY_PLACE_FAST_PATH_TIMEOUT_SECONDS", "0.08")))
PLACE_BEFORE_LIVE_TIMEOUT_SECONDS = max(
    0.0,
    float(os.environ.get("MAP_BOUNDARY_PLACE_BEFORE_LIVE_TIMEOUT_SECONDS", "1.0")),
)
EARLY_CONTEXT_MIN_REGIONAL_SPREAD_M = 45000.0
EARLY_CONTEXT_MIN_REGIONAL_NAMES = 6
EARLY_CONTEXT_MIN_CANDIDATES = 24
EARLY_CONTEXT_MIN_NAMES = 8
GENERIC_SINGLE_TOKENS = {
    "area",
    "bay",
    "beach",
    "center",
    "city",
    "district",
    "downtown",
    "east",
    "heights",
    "hill",
    "lake",
    "los",
    "north",
    "park",
    "san",
    "south",
    "vista",
    "view",
    "west",
}
POI_DESCRIPTOR_TOKENS = {
    "airport",
    "arts",
    "asu",
    "botanical",
    "campus",
    "center",
    "college",
    "community",
    "course",
    "desert",
    "fashion",
    "garden",
    "golf",
    "harbor",
    "indian",
    "international",
    "kiwanis",
    "library",
    "mountain",
    "museum",
    "preserve",
    "public",
    "quarter",
    "ranch",
    "recreation",
    "resort",
    "ridge",
    "school",
    "shops",
    "sky",
    "snedigar",
    "steele",
    "stores",
    "sunset",
    "tpc",
}
ROAD_CONTEXT_CUE_RE = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:[nesw]|rd|road|st|street|ave|av|avenue|blvd|bivd|boulevard|"
    r"ln|lane|dr|drive|pkwy|parkway|hwy|highway|ct|court|cir|circle|wy|way)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
OCR_PLACE_TOKEN_ALIASES = {
    "anor": "manor",
    "arverdale": "carverdale",
    "bayarea": "bay area",
    "carverdail": "carverdale",
    "carverdaile": "carverdale",
    "carverdaille": "carverdale",
    "dakland": "oakland",
    "daklanda": "oakland",
    "daklang": "oakland",
    "dall": "dallas",
    "deepellum": "deep ellum",
    "edwood": "redwood",
    "ersey": "jersey",
    "fran": "francisco",
    "frankisco": "francisco",
    "fransisco": "francisco",
    "fredrick": "frederick",
    "huntridg": "huntridge",
    "illowb": "willowbrook",
    "illowbnook": "willowbrook",
    "illowbrook": "willowbrook",
    "isco": "francisco",
    "jakland": "oakland",
    "jaklang": "oakland",
    "jos": "jose",
    "lakewo": "lakewood",
    "lasvegas": "las vegas",
    "lanor": "manor",
    "losangeles": "los angeles",
    "aklawn": "oak lawn",
    "oaklawn": "oak lawn",
    "ook": "willowbrook",
    "rsey": "jersey",
    "sco": "francisco",
    "scyener": "scyene",
    "sey": "jersey",
    "sanantonio": "san antonio",
    "sanfrancisco": "san francisco",
    "uakland": "oakland",
    "vakland": "oakland",
    "villowbrook": "willowbrook",
    "wiishire": "wilshire",
}
ADMIN_CONTEXT_TYPES = {
    "borough",
    "city",
    "municipality",
    "region",
    "town",
    "village",
}
CITY_INFERENCE_STOP_TOKENS = {
    "acy",
    "are",
    "bearing",
    "bam",
    "bee",
    "bnl",
    "bmw",
    "boss",
    "briefly",
    "busy",
    "carve",
    "complete",
    "del",
    "dnt",
    "edit",
    "fan",
    "fas",
    "fen",
    "for",
    "gfa",
    "hla",
    "kel",
    "later",
    "lay",
    "lix",
    "liv",
    "live",
    "mms",
    "min",
    "nis",
    "noun",
    "ont",
    "oss",
    "other",
    "pea",
    "pickup",
    "plate",
    "pst",
    "ral",
    "request",
    "requests",
    "res",
    "ride",
    "rider",
    "riders",
    "rae",
    "sayy",
    "thanks",
    "thr",
    "tap",
    "tile",
    "tiles",
    "try",
    "tns",
    "unavailable",
    "vta",
    "walk",
    "with",
}
FILENAME_CONTEXT_STOP_TOKENS = CITY_INFERENCE_STOP_TOKENS | {
    "app",
    "area",
    "avride",
    "baseline",
    "boundary",
    "boundaries",
    "bust",
    "cache",
    "capture",
    "candidate",
    "cold",
    "control",
    "coverage",
    "copy",
    "current",
    "currentref",
    "debug",
    "default",
    "det",
    "final",
    "frame",
    "gate",
    "geojson",
    "gif",
    "health",
    "hint",
    "image",
    "img",
    "jpeg",
    "jpg",
    "map",
    "maps",
    "ocr",
    "operating",
    "pipeline",
    "polygon",
    "png",
    "probe",
    "prod",
    "production",
    "profile",
    "proof",
    "prune",
    "run",
    "screenshot",
    "service",
    "small",
    "snap",
    "smoke",
    "strict",
    "tail",
    "tesla",
    "tif",
    "tiff",
    "ui",
    "uber",
    "upload",
    "variant",
    "version",
    "warm",
    "waymo",
    "web",
    "webp",
    "zoox",
}
FILENAME_CONTEXT_ALLOWED_PHRASES = {
    ("bay", "area"),
    ("los", "angeles"),
    ("san", "francisco"),
}
FILENAME_CONTEXT_FALLBACK_PHRASES = {
    ("bay", "area"): (("san", "francisco"),),
}
FILENAME_CONTEXT_FALLBACK_PROVIDER_TOKENS = {
    ("bay", "area"): {"tesla"},
}
NON_CONTEXT_COMPONENTS = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "united states",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
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
    road_match_elapsed_s: float | None = None


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
    tree: NearestPointIndex


@dataclass(frozen=True)
class NearestPointIndex:
    points: np.ndarray

    def query(
        self,
        query_points: np.ndarray,
        *,
        k: int = 1,
        distance_upper_bound: float = np.inf,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_array = np.asarray(query_points, dtype=float)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)
        target = np.asarray(self.points, dtype=float)
        k = max(1, min(int(k), len(target))) if len(target) else 1
        if len(query_array) == 0:
            shape = (0,) if k == 1 else (0, k)
            return np.full(shape, np.inf), np.full(shape, len(target), dtype=int)
        if len(target) == 0:
            shape = (len(query_array),) if k == 1 else (len(query_array), k)
            return np.full(shape, np.inf), np.full(shape, 0, dtype=int)

        distance_rows: list[np.ndarray] = []
        index_rows: list[np.ndarray] = []
        for start in range(0, len(query_array), 512):
            chunk = query_array[start : start + 512]
            squared = ((chunk[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
            if k == 1:
                indexes = np.argmin(squared, axis=1)
                distances = np.sqrt(squared[np.arange(len(chunk)), indexes])
            else:
                nearest = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
                nearest_distances = np.take_along_axis(squared, nearest, axis=1)
                order = np.argsort(nearest_distances, axis=1)
                indexes = np.take_along_axis(nearest, order, axis=1)
                distances = np.sqrt(np.take_along_axis(nearest_distances, order, axis=1))
            invalid = distances > distance_upper_bound
            distances = np.where(invalid, np.inf, distances)
            indexes = np.where(invalid, len(target), indexes)
            distance_rows.append(distances)
            index_rows.append(indexes)
        return np.concatenate(distance_rows, axis=0), np.concatenate(index_rows, axis=0)


@dataclass(frozen=True)
class LabelGeocodeCandidate:
    label: OcrLabel
    geocode: GeocodeResult

    @property
    def mercator(self) -> tuple[float, float]:
        return self.geocode.mercator

    @property
    def primary_name(self) -> str:
        return self.geocode.display_name.split(",", 1)[0].strip()


@dataclass(frozen=True)
class CityContext:
    query: str
    center: GeocodeResult
    inferred: bool
    evidence: tuple[str, ...] = ()


def geocode_many(
    requests: list[tuple[str, int]] | list[tuple[str, int, str]],
    *,
    allow_network: bool = True,
) -> list[list[GeocodeResult]]:
    if not requests:
        return []

    normalized: list[tuple[str, int, str]] = []
    for request in requests:
        if len(request) == 2:
            query, limit = request
            country_codes = "us"
        else:
            query, limit, country_codes = request
        normalized.append((query, limit, country_codes))

    unique_requests = list(dict.fromkeys(normalized))
    geocode_backend = geocode if allow_network else geocode_cached_only
    if len(unique_requests) == 1 or GEOCODE_WORKERS == 1:
        results_by_request = {
            request: geocode_backend(request[0], limit=request[1], country_codes=request[2])
            for request in unique_requests
        }
    else:
        max_workers = min(GEOCODE_WORKERS, len(unique_requests))

        def run(request: tuple[str, int, str]) -> tuple[tuple[str, int, str], list[GeocodeResult]]:
            return request, geocode_backend(request[0], limit=request[1], country_codes=request[2])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results_by_request = dict(executor.map(run, unique_requests))

    return [results_by_request[request] for request in normalized]


def georeference_from_ocr(
    image_path: str,
    city: str | None,
    width: int,
    height: int,
    *,
    min_control_points: int = 3,
    label_y_min: float | None = None,
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
        label_y_min=label_y_min,
        label_y_max=label_y_max,
    )


def georeference_from_labels(
    labels: list[OcrLabel],
    image_path: str,
    city: str | None,
    width: int,
    height: int,
    *,
    rgb: np.ndarray | None = None,
    min_control_points: int = 3,
    label_y_min: float | None = None,
    label_y_max: float | None = None,
    road_feature_distance: np.ndarray | None = None,
    anchor_marker_dots: bool = True,
    allow_road_refinement: bool = True,
    allow_sparse_regional_fit: bool = False,
) -> GeoreferenceResult | None:
    control_labels = labels
    if label_y_min is not None:
        control_labels = [label for label in control_labels if label.y >= label_y_min]
    if label_y_max is not None:
        control_labels = [label for label in control_labels if label.y <= label_y_max]
    allow_two_control_fit = label_y_min is not None
    if allow_two_control_fit:
        control_labels = [label for label in control_labels if not is_top_left_title_label(label, width, height)]
    if anchor_marker_dots:
        control_labels = anchor_labels_to_marker_dots(control_labels, image_path, rgb=rgb)
    city_contexts = resolve_city_contexts(labels, city)
    best: tuple[float, GeoreferenceResult, CityContext] | None = None
    for city_context in city_contexts:
        result = georeference_from_label_context(
            control_labels,
            image_path,
            city_context,
            width,
            height,
            rgb=rgb,
            min_control_points=min_control_points,
            allow_two_control_fit=allow_two_control_fit,
            road_feature_distance=road_feature_distance,
            allow_road_refinement=allow_road_refinement,
            allow_sparse_regional_fit=allow_sparse_regional_fit,
        )
        if result is not None:
            if city is None and city_context.query == "Inferred map area":
                result = georeference_result_with_city(result, city_context.center.display_name)
            score = georeference_result_score(result)
            if (
                best is None
                or score > best[0]
                or should_prefer_named_context_result(city_context, result, score, best)
            ):
                best = (score, result, city_context)
            if is_decisive_georeference_result(result):
                break
    return best[1] if best is not None else None


def should_prefer_named_context_result(
    context: CityContext,
    result: GeoreferenceResult,
    score: float,
    best: tuple[float, GeoreferenceResult, CityContext],
) -> bool:
    best_score, best_result, best_context = best
    if context.query == "Inferred map area" or best_context.query != "Inferred map area":
        return False
    if score < best_score - 0.04:
        return False
    if result.transform.confidence < best_result.transform.confidence - 0.025:
        return False
    scale_delta = abs(result.transform.meters_per_pixel - best_result.transform.meters_per_pixel)
    if scale_delta / max(best_result.transform.meters_per_pixel, 1.0) > 0.035:
        return False
    if abs(result.transform.rotation_radians - best_result.transform.rotation_radians) > 0.03:
        return False
    if result.residual_p90_m > max(best_result.residual_p90_m * 1.35, best_result.residual_p90_m + 500.0):
        return False
    return len(result.control_points) >= len(best_result.control_points)


def anchor_labels_to_marker_dots(
    labels: list[OcrLabel],
    image_path: str,
    *,
    rgb: np.ndarray | None = None,
) -> list[OcrLabel]:
    markers = detect_label_marker_dots(image_path, rgb=rgb)
    if not markers:
        return labels

    anchored: list[OcrLabel] = []
    for label in labels:
        marker = nearest_label_marker(label, markers)
        if marker is None:
            anchored.append(label)
            continue
        anchored.append(
            OcrLabel(
                text=label.text,
                x=marker[0],
                y=marker[1],
                width=label.width,
                height=label.height,
                confidence=label.confidence,
            )
        )
    return anchored


def detect_label_marker_dots(image_path: str, *, rgb: np.ndarray | None = None) -> list[tuple[float, float]]:
    try:
        if rgb is None:
            from .extract import load_rgb

            rgb = load_rgb(image_path)
    except Exception:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    background_sample = gray[
        ::MARKER_DOT_BACKGROUND_SAMPLE_STRIDE,
        ::MARKER_DOT_BACKGROUND_SAMPLE_STRIDE,
    ]
    if float(np.median(background_sample)) > 115.0:
        return []
    mask = (gray >= 105).astype(np.uint8)
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    markers: list[tuple[float, float]] = []
    for idx in range(1, component_count):
        _, _, width, height, area = stats[idx]
        if width < 4 or width > 9 or height < 4 or height > 9:
            continue
        aspect = width / max(float(height), 1.0)
        fill = area / max(float(width * height), 1.0)
        if 0.65 <= aspect <= 1.55 and 12 <= area <= 55 and fill >= 0.45:
            x, y = centroids[idx]
            markers.append((float(x), float(y)))
    return markers


def nearest_label_marker(label: OcrLabel, markers: list[tuple[float, float]]) -> tuple[float, float] | None:
    tokens = place_tokens(label.text)
    if token_quality(tokens) == 0 or len(tokens) > 3:
        return None
    if label.height > 36.0 or label.width > 180.0:
        return None

    left = label.x - label.width / 2.0
    right = label.x + label.width / 2.0
    top = label.y - label.height / 2.0
    bottom = label.y + label.height / 2.0
    min_x = left - max(20.0, label.height)
    max_x = right + max(48.0, min(125.0, label.width * 1.35 + 35.0))
    min_y = label.y + max(2.0, label.height * 0.10)
    max_y = label.y + max(38.0, label.height * 1.85)
    max_distance = max(42.0, min(130.0, label.width * 1.35 + 38.0))

    best: tuple[float, tuple[float, float]] | None = None
    for marker_x, marker_y in markers:
        if marker_x < min_x or marker_x > max_x or marker_y < min_y or marker_y > max_y:
            continue
        if left - 2.0 <= marker_x <= right + 2.0 and top - 2.0 <= marker_y <= bottom + 2.0:
            continue
        distance = sqrt((marker_x - label.x) ** 2 + (marker_y - label.y) ** 2)
        if distance > max_distance:
            continue
        score = distance + (12.0 if marker_x < left else 0.0)
        if best is None or score < best[0]:
            best = (score, (marker_x, marker_y))
    return best[1] if best is not None else None


def is_top_left_title_label(label: OcrLabel, width: int, height: int) -> bool:
    if label.x > width * 0.38 or label.y > height * 0.18:
        return False
    if label.width < 44.0:
        return False
    tokens = place_tokens(place_query_text(label.text))
    return bool(tokens)


def georeference_result_with_city(result: GeoreferenceResult, city: str) -> GeoreferenceResult:
    if result.transform.city == city:
        return result
    return GeoreferenceResult(
        transform=replace(result.transform, city=city),
        control_points=result.control_points,
        residual_median_m=result.residual_median_m,
        residual_p90_m=result.residual_p90_m,
        road_match=result.road_match,
    )


def georeference_result_score(result: GeoreferenceResult) -> float:
    road_score = result.road_match.score if result.road_match is not None else 0.0
    road_base_score = result.road_match.base_score if result.road_match is not None else 0.0
    control_count = len(result.control_points)
    road_weight = min(1.0, max(0.15, (control_count - 2) / 8.0))
    control_score = min(0.42, control_count * 0.035)
    spread_score = min(0.28, georeference_control_spread_m(result.control_points) / 220000.0)
    residual_penalty = min(0.35, result.residual_p90_m / 12000.0) + min(0.25, result.residual_median_m / 8000.0)
    return (
        result.transform.confidence
        + control_score
        + spread_score
        + road_weight * road_score
        + 0.2 * road_weight * road_base_score
        - residual_penalty
    )


def is_decisive_georeference_result(result: GeoreferenceResult) -> bool:
    if (
        result.transform.meters_per_pixel >= 15.0
        and result.transform.confidence >= 0.84
        and len(result.control_points) >= 8
        and result.residual_median_m <= 1600
        and result.residual_p90_m <= 3200
    ):
        return True
    if (
        result.road_match is not None
        and result.road_match.score >= 0.60
        and len(result.control_points) >= 5
        and result.residual_p90_m <= 3500
    ):
        return True
    return (
        result.road_match is None
        and result.transform.confidence >= 0.9
        and len(result.control_points) >= 6
        and result.residual_p90_m <= 500
    )


def is_credible_context_hint_georeference(result: GeoreferenceResult | None) -> bool:
    if result is None:
        return False
    if is_decisive_georeference_result(result):
        return True
    if result.transform.confidence < 0.68:
        return False
    if len(result.control_points) >= 4:
        return result.residual_median_m <= 2200.0 and result.residual_p90_m <= 3500.0
    return (
        len(result.control_points) >= 3
        and result.transform.confidence >= 0.70
        and result.residual_median_m <= 1300.0
        and result.residual_p90_m <= 1600.0
    )


def sparse_rotated_fit_without_road_evidence(
    inlier_count: int,
    rotation_radians: float,
    residual_median_m: float,
    residual_p90_m: float,
    road_match: object | None,
) -> bool:
    if road_match is not None:
        return False
    if inlier_count > 3:
        return False
    if abs(rotation_radians) < 0.24:
        return False
    return residual_median_m >= 800.0 and residual_p90_m >= 1200.0


def sparse_high_residual_fit_without_road_evidence(
    inlier_count: int,
    residual_p90_m: float,
    road_match: object | None,
) -> bool:
    if road_match is not None:
        return False
    if inlier_count > 4:
        return False
    return residual_p90_m > 3500.0


def low_res_two_control_regional_fit_without_road_evidence(
    inlier_count: int,
    meters_per_pixel: float,
    width: int,
    height: int,
    road_match: object | None,
) -> bool:
    if road_match is not None:
        return False
    if inlier_count != 2:
        return False
    return min(width, height) < 320 and meters_per_pixel >= 250.0


def georeference_control_spread_m(controls: list[ControlPoint]) -> float:
    if len(controls) < 2:
        return 0.0
    points = np.array([control.mercator for control in controls], dtype=float)
    width, height = np.ptp(points, axis=0)
    return float(max(width, height))


def georeference_from_label_context(
    labels: list[OcrLabel],
    image_path: str,
    city_context: CityContext,
    width: int,
    height: int,
    *,
    rgb: np.ndarray | None = None,
    min_control_points: int,
    allow_two_control_fit: bool = False,
    road_feature_distance: np.ndarray | None = None,
    allow_road_refinement: bool = True,
    allow_sparse_regional_fit: bool = False,
) -> GeoreferenceResult | None:
    city_center = city_context.center
    controls = build_control_points(
        labels,
        city_context.query,
        city_center,
        max_geocoded_labels=MAX_SPARSE_GEOCODED_LABELS if allow_two_control_fit else MAX_GEOCODED_LABELS,
        merge_control_sources=allow_two_control_fit,
    )
    required_available_controls = 2 if allow_two_control_fit else min_control_points
    if len(controls) >= required_available_controls:
        fit = choose_similarity_fit(
            controls,
            image_path,
            city_center,
            allow_two_control_fit=allow_two_control_fit,
            allow_sparse_regional_fit=allow_sparse_regional_fit,
        )
        if fit is not None:
            scale, rotation, tx, ty, inliers, residuals = fit
            min_required_controls = 2 if allow_two_control_fit and len(inliers) == 2 else min_control_points
            if len(inliers) >= min_required_controls:
                residual_median, residual_p90 = residual_median_p90([residuals[i] for i in inliers])
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
                    spread = control_spread(pixel_positions(controls, inliers))
                    road_refinement = None
                    road_refinement_elapsed_s = None
                    if allow_road_refinement and should_try_road_refinement(
                        city_context,
                        scale,
                        len(inliers),
                        residual_median,
                        residual_p90,
                        spread,
                        width,
                        height,
                    ) and not allow_two_control_fit:
                        if rgb is None:
                            from .extract import load_rgb

                            road_rgb = load_rgb(image_path)
                        else:
                            road_rgb = rgb
                        road_started = time.perf_counter()
                        road_refinement = refine_transform_with_osm_roads(
                            road_rgb,
                            city_center,
                            geo_transform,
                            lock_scale=should_lock_road_refinement_scale(
                                scale,
                                len(inliers),
                                residual_median,
                                residual_p90,
                                spread,
                                width,
                                height,
                            ),
                            feature_distance=road_feature_distance,
                        )
                        road_refinement_elapsed_s = max(0.0, time.perf_counter() - road_started)
                    if road_refinement is not None:
                        refined_residuals = control_residuals_for_transform(
                            road_refinement.transform,
                            [controls[i] for i in inliers],
                            width,
                            height,
                        )
                        if refinement_preserves_label_fit(
                            refined_residuals,
                            residual_median,
                            residual_p90,
                            len(inliers),
                        ):
                            geo_transform = road_refinement.transform
                        else:
                            road_refinement = None
                            road_refinement_elapsed_s = None
                    if sparse_rotated_fit_without_road_evidence(
                        len(inliers),
                        rotation,
                        residual_median,
                        residual_p90,
                        road_refinement,
                    ):
                        return None
                    return GeoreferenceResult(
                        transform=geo_transform,
                        control_points=[controls[i] for i in inliers],
                        residual_median_m=residual_median,
                        residual_p90_m=residual_p90,
                        road_match=road_refinement,
                        road_match_elapsed_s=road_refinement_elapsed_s,
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
    if len(road_points) > 7000:
        step = int(np.ceil(len(road_points) / 7000))
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
    line_features = city_context_line_features(rgb, pixel_geometry) if max_image_dim <= 520 else None
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


def resolve_city_contexts(labels: list[OcrLabel], city: str | None) -> list[CityContext]:
    if city:
        city_results = geocode(city, limit=1)
        if city_results:
            return [CityContext(query=city, center=city_results[0], inferred=False)]
        return []
    return infer_city_contexts(labels)


def filename_city_contexts(filename_hint: str | None) -> list[CityContext]:
    queries = filename_context_queries(filename_hint)
    if not queries:
        return []

    contexts: list[CityContext] = []
    for query, results in zip(queries, geocode_many([(query, 2) for query in queries], allow_network=False)):
        query_tokens = place_tokens(query)
        for result in results:
            if not result.bbox:
                continue
            if not primary_name_matches_label(result.display_name, query):
                continue
            if len(query_tokens) == 1 and not is_reliable_single_token_context(result):
                continue
            if result.place_type.lower() not in ADMIN_CONTEXT_TYPES and not is_broad_context_result(result):
                continue
            contexts.append(
                CityContext(
                    query=result.display_name.split(",", 1)[0].strip(),
                    center=result,
                    inferred=True,
                    evidence=(query,),
                )
            )
            break
    return rank_city_contexts_for_georeferencing(dedupe_city_contexts(contexts))[:2]


def filename_context_queries(filename_hint: str | None) -> list[str]:
    if not filename_hint:
        return []
    tokens: list[str] = []
    seen_tokens: set[str] = set()
    for part in re.split(r"[^a-z0-9]+", filename_hint.lower()):
        if len(part) < 3 or any(char.isdigit() for char in part):
            continue
        token = OCR_PLACE_TOKEN_ALIASES.get(part, part)
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        tokens.append(token)

    queries: list[str] = []
    seen_queries: set[str] = set()

    def add_query(parts: tuple[str, ...]) -> None:
        query = " ".join(part.title() for part in parts)
        key = query.lower()
        if key in seen_queries:
            return
        seen_queries.add(key)
        queries.append(query)

    for size in (3, 2):
        for index in range(0, max(0, len(tokens) - size + 1)):
            parts = tuple(tokens[index : index + size])
            if parts in FILENAME_CONTEXT_ALLOWED_PHRASES:
                add_query(parts)
                if seen_tokens & FILENAME_CONTEXT_FALLBACK_PROVIDER_TOKENS.get(parts, set()):
                    for fallback_parts in FILENAME_CONTEXT_FALLBACK_PHRASES.get(parts, ()):
                        add_query(fallback_parts)
                continue
            if any(part in FILENAME_CONTEXT_STOP_TOKENS for part in parts):
                continue
            if is_noisy_poi_query(set(parts)):
                continue
            add_query(parts)

    for token in tokens:
        if token in FILENAME_CONTEXT_STOP_TOKENS or token in GENERIC_SINGLE_TOKENS:
            continue
        if len(token) < 5:
            continue
        add_query((token,))
    if (
        "bay area" in seen_queries
        and seen_tokens & FILENAME_CONTEXT_FALLBACK_PROVIDER_TOKENS[("bay", "area")]
    ):
        for fallback_parts in FILENAME_CONTEXT_FALLBACK_PHRASES[("bay", "area")]:
            add_query(fallback_parts)
    return queries[:8]


def infer_city_context(labels: list[OcrLabel]) -> CityContext | None:
    contexts = infer_city_contexts(labels)
    return contexts[0] if contexts else None


def infer_city_contexts(labels: list[OcrLabel]) -> list[CityContext]:
    inference_labels = city_inference_labels(labels)
    direct_contexts = direct_city_contexts_from_labels(inference_labels, allow_network=False)
    if len(inference_labels) <= 4 and should_use_direct_city_contexts(direct_contexts):
        return direct_contexts
    if should_use_direct_city_contexts(direct_contexts) and is_decisive_direct_context(direct_contexts[0]):
        return direct_contexts

    candidates = geocoded_label_candidates(inference_labels)
    if should_use_direct_city_contexts(direct_contexts):
        expanded_contexts = expanded_contexts_from_label_cluster(candidates, direct_contexts)
        if expanded_contexts:
            return rank_city_contexts_for_georeferencing(
                dedupe_city_contexts([*expanded_contexts, *direct_contexts])
            )[:MAX_CITY_CONTEXTS]
        return direct_contexts

    if not candidates:
        live_direct_contexts = direct_city_contexts_from_labels(inference_labels, allow_network=True)
        if should_use_direct_city_contexts(live_direct_contexts):
            return live_direct_contexts
        return prominent_contexts_from_labels(inference_labels)

    members = best_candidate_cluster(candidates)
    if not members or not has_enough_context_members(members):
        live_direct_contexts = direct_city_contexts_from_labels(inference_labels, allow_network=True)
        if should_use_direct_city_contexts(live_direct_contexts):
            return live_direct_contexts
        return prominent_contexts_from_labels(inference_labels)

    contexts: list[CityContext] = []
    prominent_contexts = prominent_contexts_from_labels(inference_labels)
    covering_prominent_contexts = [
        context
        for context in prominent_contexts
        if context.center.place_type.lower() in ADMIN_CONTEXT_TYPES
        and cluster_coverage(context.center, members) >= 0.55
    ]
    parent_name = choose_parent_component(members)
    if parent_name:
        contexts.extend(contexts_from_parent_name(parent_name, members))

    synthetic = synthetic_context_from_members(members, parent_name)
    if synthetic is not None:
        if synthetic.query == "Inferred map area":
            contexts.insert(0, synthetic)
        else:
            contexts.append(synthetic)

    if contexts and contexts[0].query == "Inferred map area" and covering_prominent_contexts:
        contexts = [
            *covering_prominent_contexts,
            *contexts,
            *[context for context in prominent_contexts if context not in covering_prominent_contexts],
        ]
    elif contexts and contexts[0].query == "Inferred map area":
        contexts.extend(prominent_contexts)
    else:
        contexts = [*prominent_contexts, *contexts]

    anchor = choose_city_anchor(members)
    contexts.append(
        CityContext(
            query=anchor.primary_name,
            center=anchor.geocode,
            inferred=True,
            evidence=tuple(sorted({member.primary_name for member in members})[:8]),
        )
    )
    return rank_city_contexts_for_georeferencing(dedupe_city_contexts(contexts))[:MAX_CITY_CONTEXTS]


def expanded_contexts_from_label_cluster(
    candidates: list[LabelGeocodeCandidate],
    direct_contexts: list[CityContext],
) -> list[CityContext]:
    if not candidates or not direct_contexts:
        return []
    direct_context = direct_contexts[0]
    members = best_candidate_cluster(candidates)
    if not members or not has_enough_context_members(members):
        return []
    if not label_cluster_extends_beyond_context(direct_context, members):
        return []

    contexts: list[CityContext] = []
    parent_name = choose_parent_component(members)
    if parent_name:
        contexts.extend(contexts_from_parent_name(parent_name, members))
    synthetic = synthetic_context_from_members(members, parent_name)
    if synthetic is not None:
        contexts.insert(0, synthetic)
    return contexts


def label_cluster_extends_beyond_context(
    context: CityContext,
    members: list[LabelGeocodeCandidate],
) -> bool:
    bbox = context.center.bbox
    if bbox is None or len(members) < 4:
        return False
    unique_names = {member.primary_name.lower() for member in members}
    if len(unique_names) < 3:
        return False

    west, south, east, north = bbox
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
    min_x, max_x = sorted((west_m, east_m))
    min_y, max_y = sorted((south_m, north_m))
    context_span = max(max_x - min_x, max_y - min_y)
    padding = max(1500.0, context_span * 0.06)

    outside = 0
    for member in members:
        x, y = member.mercator
        if x < min_x - padding or x > max_x + padding or y < min_y - padding or y > max_y + padding:
            outside += 1
    if outside == 0:
        return False

    member_span = cluster_spread_m(members)
    return outside >= 2 or member_span >= max(14000.0, context_span * 1.22)


def city_inference_labels(labels: list[OcrLabel]) -> list[OcrLabel]:
    return sorted(
        [label for label in labels if is_plausible_context_label(label)],
        key=context_label_score,
        reverse=True,
    )


def direct_city_contexts_from_labels(labels: list[OcrLabel], *, allow_network: bool = True) -> list[CityContext]:
    decisive_broad_context = broad_direct_context_from_labels(labels, allow_network=allow_network)
    if decisive_broad_context is not None:
        return [decisive_broad_context]

    query_scores: dict[str, float] = {}
    query_evidence: dict[str, set[str]] = {}
    used_positions: set[tuple[str, int, int]] = set()
    single_token_fragments = single_tokens_supported_by_fuller_labels(labels)
    standalone_queries: set[str] = set()
    for label in labels[:MAX_CITY_INFERENCE_LABELS]:
        query = place_query_text(label.text)
        tokens = place_tokens(query)
        if len(tokens) != 1:
            continue
        token = next(iter(tokens))
        if token in GENERIC_SINGLE_TOKENS or token in CITY_INFERENCE_STOP_TOKENS:
            continue
        if tokens <= single_token_fragments and not is_strong_standalone_context_label(label):
            continue
        standalone_queries.add(token.title())

    for label in labels[:MAX_CITY_INFERENCE_LABELS]:
        query = place_query_text(label.text)
        tokens = place_tokens(query)
        if not tokens or tokens & CITY_INFERENCE_STOP_TOKENS:
            continue
        if len(tokens) > 4:
            continue
        if len(tokens) == 1 and next(iter(tokens)) in GENERIC_SINGLE_TOKENS:
            continue
        if len(tokens) == 1 and tokens <= single_token_fragments and not is_strong_standalone_context_label(label):
            continue
        position_key = (" ".join(sorted(tokens)), round(label.x / 16), round(label.y / 16))
        if position_key in used_positions:
            continue
        used_positions.add(position_key)
        score = direct_context_label_score(label)
        noisy_poi_query = is_noisy_poi_query(tokens)
        if len(tokens) <= 2 and not noisy_poi_query:
            query_scores[query] = query_scores.get(query, 0.0) + score
            query_evidence.setdefault(query, set()).add(label.text)
        token_bonus = 1.0 if len(tokens) == 1 else 0.64
        for token in tokens:
            if token in GENERIC_SINGLE_TOKENS or token in CITY_INFERENCE_STOP_TOKENS or token in POI_DESCRIPTOR_TOKENS:
                continue
            if len(token) < 5:
                continue
            token_query = token.title()
            allow_context_cue_token = "downtown" in tokens
            if noisy_poi_query and not allow_context_cue_token:
                continue
            if len(tokens) > 1 and token_query not in standalone_queries and not allow_context_cue_token:
                continue
            query_scores[token_query] = query_scores.get(token_query, 0.0) + score * token_bonus
            query_evidence.setdefault(token_query, set()).add(label.text)

    promoted_queries = promoted_direct_context_queries(labels, query_scores)
    ranked_queries = append_ranked_context_queries(
        sorted(query_scores.items(), key=lambda item: item[1], reverse=True)[:DIRECT_CONTEXT_QUERY_LIMIT],
        promoted_queries,
        cap=DIRECT_CONTEXT_MAX_QUERIES,
    )
    scored_contexts = score_direct_city_context_queries(
        ranked_queries,
        geocode_many([(query, 2) for query, _score in ranked_queries], allow_network=False),
        query_evidence,
    )
    if not scored_contexts and allow_network:
        live_ranked_queries = append_ranked_context_queries(
            ranked_queries[:DIRECT_CONTEXT_LIVE_QUERY_LIMIT],
            promoted_queries,
            cap=DIRECT_CONTEXT_MAX_LIVE_QUERIES,
        )
        scored_contexts = score_direct_city_context_queries(
            live_ranked_queries,
            geocode_many([(query, 2) for query, _score in live_ranked_queries]),
            query_evidence,
        )

    if not scored_contexts:
        return []
    scored_contexts.sort(key=lambda item: item[0], reverse=True)
    top_score, top_context = scored_contexts[0]
    if top_score < 160.0 and not is_strong_direct_admin_context(top_score, top_context):
        return []
    close_competitors = [context for score, context in scored_contexts[1:] if score >= top_score * 0.62]
    if close_competitors and not all(context_covers_context(top_context, competitor) for competitor in close_competitors):
        return []
    return [top_context]


def direct_context_label_score(label: OcrLabel) -> float:
    return label.confidence + min(35.0, (label.width * label.height) / 650.0)


def is_strong_standalone_context_label(label: OcrLabel) -> bool:
    if label.confidence < 90.0:
        return False
    if label.width < 40.0 or label.height < 14.0:
        return False
    return direct_context_label_score(label) >= PROMOTED_DIRECT_CONTEXT_LABEL_SCORE


def promoted_direct_context_queries(
    labels: list[OcrLabel],
    query_scores: dict[str, float],
) -> list[tuple[str, float]]:
    promoted: dict[str, float] = {}
    for label in labels[:MAX_CITY_INFERENCE_LABELS]:
        query = place_query_text(label.text)
        tokens = place_tokens(query)
        if not is_promotable_direct_context_label(label, query, tokens):
            continue
        score = max(query_scores.get(query, 0.0), direct_context_label_score(label))
        promoted[query] = max(promoted.get(query, 0.0), score)
    return sorted(promoted.items(), key=lambda item: item[1], reverse=True)


def is_promotable_direct_context_label(label: OcrLabel, query: str, tokens: set[str]) -> bool:
    if not query or tokens & CITY_INFERENCE_STOP_TOKENS:
        return False
    if len(tokens) != 2 and not (len(tokens) == 3 and "city" in tokens):
        return False
    if tokens <= GENERIC_SINGLE_TOKENS:
        return False
    if is_noisy_poi_query(tokens):
        return False
    if ROAD_CONTEXT_CUE_RE.search(label.text):
        return False
    return direct_context_label_score(label) >= PROMOTED_DIRECT_CONTEXT_LABEL_SCORE


def append_ranked_context_queries(
    ranked_queries: list[tuple[str, float]],
    promoted_queries: list[tuple[str, float]],
    *,
    cap: int,
) -> list[tuple[str, float]]:
    merged: list[tuple[str, float]] = []
    seen: set[str] = set()
    for query, score in [*ranked_queries, *promoted_queries]:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append((query, score))
        if len(merged) >= cap:
            break
    return merged


def is_strong_direct_admin_context(score: float, context: CityContext) -> bool:
    if score < STRONG_DIRECT_CONTEXT_MIN_SCORE:
        return False
    tokens = place_tokens(context.query)
    if len(tokens) < 2:
        return False
    if context.center.place_type.lower() not in ADMIN_CONTEXT_TYPES:
        return False
    if context.center.importance < 0.58:
        return False
    if geocode_bbox_span_m(context.center) < 24000.0:
        return False
    evidence_tokens = [place_tokens(place_query_text(evidence)) for evidence in context.evidence]
    return any(tokens == tokens_from_evidence for tokens_from_evidence in evidence_tokens)


def score_direct_city_context_queries(
    ranked_queries: list[tuple[str, float]],
    results_by_query: list[list[GeocodeResult]],
    query_evidence: dict[str, set[str]],
) -> list[tuple[float, CityContext]]:
    scored_contexts: list[tuple[float, CityContext]] = []
    for (query, score), results in zip(ranked_queries, results_by_query):
        for result in results:
            if not result.bbox:
                continue
            if not primary_name_matches_label(result.display_name, query):
                continue
            if len(place_tokens(query)) == 1 and not is_reliable_single_token_context(result):
                continue
            if result.place_type.lower() not in ADMIN_CONTEXT_TYPES and not is_broad_context_result(result):
                continue
            regional_bonus = min(65.0, geocode_bbox_span_m(result) / 3500.0) if len(place_tokens(query)) > 1 else 0.0
            scored_contexts.append(
                (
                    score + result.importance * 10.0 + regional_bonus,
                    CityContext(
                        query=result.display_name.split(",", 1)[0],
                        center=result,
                        inferred=True,
                        evidence=tuple(sorted(query_evidence.get(query, {query}))[:4]),
                    ),
                )
            )
            break
    return scored_contexts


def broad_direct_context_from_labels(labels: list[OcrLabel], *, allow_network: bool = True) -> CityContext | None:
    for label in labels[:MAX_CITY_INFERENCE_LABELS]:
        query = place_query_text(label.text)
        tokens = place_tokens(query)
        if len(tokens) != 2 or tokens & CITY_INFERENCE_STOP_TOKENS:
            continue
        if is_noisy_poi_query(tokens):
            continue
        if tokens <= GENERIC_SINGLE_TOKENS and not ({"bay", "area"} <= tokens):
            continue
        score = label.confidence + min(35.0, (label.width * label.height) / 650.0)
        if score < 100.0:
            continue
        results = geocode_cached_only(query, limit=2)
        if not results and allow_network and should_live_geocode_broad_context(tokens, score):
            results = geocode(query, limit=2)
        for result in results:
            if not result.bbox or not primary_name_matches_label(result.display_name, query):
                continue
            if not is_broad_context_result(result):
                continue
            regional_bonus = min(65.0, geocode_bbox_span_m(result) / 3500.0)
            if score + result.importance * 10.0 + regional_bonus < 170.0:
                continue
            return CityContext(
                query=result.display_name.split(",", 1)[0],
                center=result,
                inferred=True,
                evidence=(label.text,),
            )
    return None


def should_live_geocode_broad_context(tokens: set[str], score: float) -> bool:
    return score >= 130.0 and bool(tokens & {"area", "region"})


def context_covers_context(parent: CityContext, child: CityContext) -> bool:
    parent_span = geocode_bbox_span_m(parent.center)
    child_span = geocode_bbox_span_m(child.center)
    if parent.center.bbox is None or parent_span < max(30000.0, child_span * 2.0):
        return False
    west, south, east, north = parent.center.bbox
    padding = max(0.01, (east - west) * 0.03, (north - south) * 0.03)
    return (
        west - padding <= child.center.lon <= east + padding
        and south - padding <= child.center.lat <= north + padding
    )


def should_use_direct_city_contexts(contexts: list[CityContext]) -> bool:
    if not contexts:
        return False
    top = contexts[0]
    tokens = place_tokens(top.query)
    place_type = top.center.place_type.lower()
    if len(tokens) == 1 and place_type not in ADMIN_CONTEXT_TYPES and geocode_bbox_span_m(top.center) < 30000.0:
        return False
    return True


def is_decisive_direct_context(context: CityContext) -> bool:
    place_type = context.center.place_type.lower()
    span_m = geocode_bbox_span_m(context.center)
    if place_type == "region" and span_m >= 85000.0:
        return True
    return place_type in ADMIN_CONTEXT_TYPES and span_m >= 30000.0 and len(context.evidence) >= 1


def is_plausible_context_label(label: OcrLabel) -> bool:
    tokens = place_tokens(label.text)
    if not tokens or tokens & CITY_INFERENCE_STOP_TOKENS:
        return False
    if len(tokens) > 4:
        return False
    if len(tokens) == 1 and next(iter(tokens)) in GENERIC_SINGLE_TOKENS:
        return False
    compact = re.sub(r"[^A-Za-z]", "", label.text)
    if len(compact) < 4:
        return False
    return label.confidence >= 45 or label.width * label.height >= 5000


def context_label_score(label: OcrLabel) -> tuple[float, float, int]:
    tokens = place_tokens(label.text)
    area_score = min(4.0, (label.width * label.height) / 12000.0)
    return label.confidence + area_score * 10.0, area_score, -len(tokens)


def prominent_contexts_from_labels(labels: list[OcrLabel]) -> list[CityContext]:
    scored_contexts: list[tuple[float, CityContext]] = []
    queries = prominent_context_queries(labels)
    for query, results in zip(queries, geocode_many([(query, 3) for query in queries], allow_network=False)):
        for result in results:
            if not result.bbox:
                continue
            if not primary_name_matches_label(result.display_name, query):
                continue
            if not is_broad_context_result(result):
                continue
            scored_contexts.append(
                (
                    prominent_context_score(result, query, labels),
                    CityContext(
                        query=result.display_name.split(",", 1)[0],
                        center=result,
                        inferred=True,
                        evidence=(query,),
                    ),
                )
            )
            break
    contexts = [context for _, context in sorted(scored_contexts, key=lambda item: item[0], reverse=True)]
    return dedupe_city_contexts(contexts)[:5]


def prominent_context_score(result: GeocodeResult, query: str, labels: list[OcrLabel]) -> float:
    type_score = {
        "city": 5.0,
        "town": 3.0,
        "municipality": 3.0,
        "village": 2.0,
        "region": 1.0,
    }.get(result.place_type.lower(), 0.0)
    query_tokens = place_tokens(query)
    support = 0.0
    for label in labels:
        tokens = place_tokens(label.text)
        if query_tokens and query_tokens <= tokens:
            support += 1.0 + min(2.0, label.width * label.height / 25000.0)
    return type_score + result.importance + min(6.0, support)


def prominent_context_queries(labels: list[OcrLabel]) -> list[str]:
    scores: dict[str, float] = {}
    for label in labels[:MAX_CITY_INFERENCE_LABELS]:
        tokens = place_tokens(label.text)
        if not tokens:
            continue
        label_score = context_label_score(label)[0]
        if 1 <= len(tokens) <= 3 and not is_noisy_poi_query(tokens):
            scores[clean_query_text(label.text)] = max(scores.get(clean_query_text(label.text), 0.0), label_score)
        for token in tokens:
            if token in GENERIC_SINGLE_TOKENS or token in CITY_INFERENCE_STOP_TOKENS or token in POI_DESCRIPTOR_TOKENS:
                continue
            token_score = label_score + (label.width * label.height) / 20000.0
            scores[token.title()] = max(scores.get(token.title(), 0.0), token_score)
    return [query for query, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]]


def clean_query_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def place_query_text(value: str) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[^a-z0-9]+", value.lower()):
        if len(part) < 3:
            continue
        token = OCR_PLACE_TOKEN_ALIASES.get(part, part)
        if token in CITY_INFERENCE_STOP_TOKENS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return " ".join(token.title() for token in tokens)


def is_broad_context_result(result: GeocodeResult) -> bool:
    if result.place_type.lower() == "county":
        return False
    if result.place_type.lower() in ADMIN_CONTEXT_TYPES:
        return True
    if result.bbox is None:
        return False
    west, south, east, north = result.bbox
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
    return max(abs(east_m - west_m), abs(north_m - south_m)) >= 12000.0


def is_reliable_single_token_context(result: GeocodeResult) -> bool:
    span_m = geocode_bbox_span_m(result)
    place_type = result.place_type.lower()
    if result.importance >= 0.58:
        return True
    if place_type in {"city", "municipality", "region"} and span_m >= 9000.0:
        return True
    return span_m >= 18000.0


def geocode_bbox_span_m(result: GeocodeResult) -> float:
    if result.bbox is None:
        return 0.0
    west, south, east, north = result.bbox
    west_m, south_m = lonlat_to_mercator(west, south)
    east_m, north_m = lonlat_to_mercator(east, north)
    return float(max(abs(east_m - west_m), abs(north_m - south_m)))


def contexts_from_parent_name(parent_name: str, members: list[LabelGeocodeCandidate]) -> list[CityContext]:
    contexts: list[CityContext] = []
    evidence = tuple(sorted({member.primary_name for member in members})[:8])
    for result in geocode(parent_name, limit=2):
        if not result.bbox:
            continue
        if result.place_type.lower() == "county":
            continue
        if not is_broad_context_result(result):
            continue
        if cluster_coverage(result, members) < 0.55:
            continue
        contexts.append(CityContext(query=parent_name, center=result, inferred=True, evidence=evidence))
    return contexts


def cluster_coverage(result: GeocodeResult, members: list[LabelGeocodeCandidate]) -> float:
    if result.bbox is None or not members:
        return 0.0
    west, south, east, north = result.bbox
    covered = sum(
        1
        for member in members
        if west <= member.geocode.lon <= east and south <= member.geocode.lat <= north
    )
    return covered / len(members)


def synthetic_context_from_members(
    members: list[LabelGeocodeCandidate],
    parent_name: str | None,
) -> CityContext | None:
    if not members:
        return None
    points = np.array([member.mercator for member in members], dtype=float)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    span = max(max_x - min_x, max_y - min_y)
    padding = max(9000.0, min(45000.0, span * 0.45))
    west, south = mercator_to_lonlat(float(min_x - padding), float(min_y - padding))
    east, north = mercator_to_lonlat(float(max_x + padding), float(max_y + padding))
    lon, lat = mercator_to_lonlat(float((min_x + max_x) / 2.0), float((min_y + max_y) / 2.0))
    unique_names = {member.primary_name.lower() for member in members}
    is_multi_city_region = len(unique_names) >= 8 and span >= 30000.0
    name = parent_name if parent_name and not is_multi_city_region else "Inferred map area"
    return CityContext(
        query=name,
        center=GeocodeResult(
            label=name,
            lon=lon,
            lat=lat,
            display_name=name,
            bbox=(west, south, east, north),
            importance=0.5,
            place_type="region",
        ),
        inferred=True,
        evidence=tuple(sorted({member.primary_name for member in members})[:8]),
    )


def choose_parent_component(members: list[LabelGeocodeCandidate]) -> str | None:
    votes: dict[str, float] = {}
    for member in members:
        for component in member.geocode.display_name.split(",")[1:5]:
            component = clean_query_text(component)
            if not is_context_component(component):
                continue
            votes[component] = votes.get(component, 0.0) + 1.0 + member.geocode.importance
    if not votes:
        return None
    name, score = max(votes.items(), key=lambda item: item[1])
    return name if score >= 2.0 else None


def is_context_component(component: str) -> bool:
    lowered = component.lower()
    if not lowered or lowered in NON_CONTEXT_COMPONENTS:
        return False
    if any(char.isdigit() for char in lowered):
        return False
    if "county" in lowered or "parish" in lowered:
        return False
    tokens = place_tokens(component)
    if not tokens or tokens <= GENERIC_SINGLE_TOKENS:
        return False
    return True


def dedupe_city_contexts(contexts: list[CityContext]) -> list[CityContext]:
    deduped: list[CityContext] = []
    seen: set[tuple[str, int, int]] = set()
    for context in contexts:
        key = (context.center.display_name.lower(), round(context.center.lon, 3), round(context.center.lat, 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(context)
    return deduped


def rank_city_contexts_for_georeferencing(contexts: list[CityContext]) -> list[CityContext]:
    return sorted(contexts, key=city_context_georef_score, reverse=True)


def city_context_georef_score(context: CityContext) -> float:
    span_m = geocode_bbox_span_m(context.center)
    place_type = context.center.place_type.lower()
    display_name = context.center.display_name.lower()
    score = context.center.importance
    score += min(4.0, len(context.evidence) * 0.55)
    score += min(3.5, span_m / 22000.0)
    if place_type in {"region", "municipality", "borough"}:
        score += 1.35
    elif place_type in {"city", "town", "village"}:
        score += 0.35
    if span_m < 9000.0:
        score -= 1.4
    if any(token in display_name for token in ("school", "district", "campus", "college", "hospital")):
        score -= 2.2
    if context.query == "Inferred map area":
        score += 0.45
    return score


def geocoded_label_candidates(labels: list[OcrLabel]) -> list[LabelGeocodeCandidate]:
    candidates: list[LabelGeocodeCandidate] = []
    used_text: set[str] = set()
    query_labels: list[tuple[OcrLabel, str]] = []
    single_token_fragments = single_tokens_supported_by_fuller_labels(labels)
    for label in rank_geocode_labels(labels)[:MAX_CITY_INFERENCE_LABELS]:
        query = place_query_text(label.text)
        tokens = place_tokens(query)
        if not tokens or tokens & CITY_INFERENCE_STOP_TOKENS:
            continue
        if len(tokens) > 4:
            continue
        if is_noisy_poi_query(tokens):
            continue
        if len(tokens) == 1 and tokens <= single_token_fragments:
            continue
        if len(tokens) == 1 and next(iter(tokens)) in GENERIC_SINGLE_TOKENS:
            continue
        text_key = " ".join(sorted(tokens))
        if text_key in used_text:
            continue
        used_text.add(text_key)
        query_labels.append((label, query))

    for start in range(0, len(query_labels), GEOCODE_BATCH_SIZE):
        batch = query_labels[start : start + GEOCODE_BATCH_SIZE]
        for (label, query), results in zip(
            batch,
            geocode_many([(query, 2) for _label, query in batch], allow_network=False),
        ):
            for result in results:
                if primary_name_matches_label(result.display_name, query):
                    candidates.append(LabelGeocodeCandidate(label=label, geocode=result))
            if has_reliable_candidate_cluster(candidates):
                return candidates
    return candidates


def has_reliable_candidate_cluster(candidates: list[LabelGeocodeCandidate]) -> bool:
    if len(candidates) < 8:
        return False
    members = best_candidate_cluster(candidates)
    unique_names = {member.primary_name.lower() for member in members}
    if len(unique_names) < 4:
        return False
    spread_m = cluster_spread_m(members)
    if spread_m >= EARLY_CONTEXT_MIN_REGIONAL_SPREAD_M and len(unique_names) >= EARLY_CONTEXT_MIN_REGIONAL_NAMES:
        return True
    return len(candidates) >= EARLY_CONTEXT_MIN_CANDIDATES and len(unique_names) >= EARLY_CONTEXT_MIN_NAMES


def has_enough_context_members(members: list[LabelGeocodeCandidate]) -> bool:
    unique_names = {member.primary_name.lower() for member in members}
    unique_labels = {member.label.text.lower() for member in members}
    return len(members) >= 3 and len(unique_names) >= 2 and len(unique_labels) >= 2


def best_candidate_cluster(candidates: list[LabelGeocodeCandidate]) -> list[LabelGeocodeCandidate]:
    best: tuple[float, list[LabelGeocodeCandidate]] | None = None
    for anchor in candidates:
        anchor_x, anchor_y = anchor.mercator
        members: list[LabelGeocodeCandidate] = []
        for candidate in candidates:
            cand_x, cand_y = candidate.mercator
            if sqrt((cand_x - anchor_x) ** 2 + (cand_y - anchor_y) ** 2) <= 95000.0:
                members.append(candidate)
        unique_names = {member.primary_name.lower() for member in members}
        unique_labels = {member.label.text.lower() for member in members}
        if not unique_names:
            continue
        spread = cluster_spread_m(members)
        confidence = sum(member.label.confidence for member in members) / max(len(members), 1)
        spread_bonus = min(2.5, spread / 25000.0)
        score = len(unique_names) * 2.0 + len(unique_labels) * 0.4 + spread_bonus + confidence / 120.0
        if best is None or score > best[0]:
            best = (score, members)
    return best[1] if best is not None else []


def cluster_spread_m(members: list[LabelGeocodeCandidate]) -> float:
    if len(members) < 2:
        return 0.0
    points = np.array([member.mercator for member in members], dtype=float)
    width, height = np.ptp(points, axis=0)
    return float(max(width, height))


def choose_city_anchor(members: list[LabelGeocodeCandidate]) -> LabelGeocodeCandidate:
    return max(
        members,
        key=lambda member: (
            len(place_tokens(member.primary_name)),
            member.geocode.importance,
            member.label.confidence,
        ),
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
    if len(points) > 800:
        step = int(np.ceil(len(points) / 800))
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
        offset_values = np.linspace(-0.65 * city_radius_m, 0.65 * city_radius_m, 5)
        scale_values = base_scale * np.geomspace(0.18, 0.46, 5)
        rotation_values = np.deg2rad(np.linspace(-8.0, 8.0, 3))
        centers = [(city_x + dx, city_y + dy) for dx in offset_values for dy in offset_values]
    else:
        assert seed is not None
        _, seed_transform = seed
        seed_tx, seed_ty = projected_center_for_transform(seed_transform, center_pixel_x, center_pixel_y)
        offset_values = np.linspace(-0.18 * city_radius_m, 0.18 * city_radius_m, 3)
        scale_values = seed_transform.meters_per_pixel * np.linspace(0.92, 1.08, 3)
        rotation_values = seed_transform.rotation_radians + np.deg2rad(np.linspace(-3.0, 3.0, 3))
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
        tree=NearestPointIndex(midpoint_array),
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
        select_line_candidate_subset(broad_candidates, per_metric=120, cap=320, per_bucket=3),
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
        select_line_candidate_subset(fine_candidates, per_metric=140, cap=360, per_bucket=4),
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
    offset_values = np.linspace(-0.65 * city_radius_m, 0.65 * city_radius_m, 5)
    scale_values = base_scale * np.geomspace(0.18, 0.46, 5)
    rotation_values = np.deg2rad(np.linspace(-8.0, 8.0, 3))
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
        for scale_multiplier in np.linspace(0.86, 1.12, 3):
            scale = seed_transform.meters_per_pixel * float(scale_multiplier)
            for rotation in seed_transform.rotation_radians + np.deg2rad(np.linspace(-3.0, 3.0, 3)):
                for offset_x in np.linspace(-2400.0, 2400.0, 3):
                    for offset_y in np.linspace(-2400.0, 2400.0, 3):
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
    road_tree = NearestPointIndex(midpoints)
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
    target_tree: NearestPointIndex,
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


def build_control_points(
    labels: list[OcrLabel],
    city: str,
    city_center: GeocodeResult,
    *,
    max_geocoded_labels: int = MAX_GEOCODED_LABELS,
    merge_control_sources: bool = False,
) -> list[ControlPoint]:
    # The geocoded and OSM-place paths are independent lookups; overlap them, and
    # accept a fast OSM-place fit only when it is already decisive.
    place_executor = ThreadPoolExecutor(max_workers=1)
    place_future = place_executor.submit(
        build_osm_place_control_points,
        labels,
        city_center,
        prefer_large_text=merge_control_sources,
    )
    try:
        place_controls: list[ControlPoint] | None = None
        if not merge_control_sources and PLACE_FAST_PATH_TIMEOUT_SECONDS > 0:
            try:
                place_controls = place_future.result(timeout=PLACE_FAST_PATH_TIMEOUT_SECONDS)
            except TimeoutError:
                place_controls = None
            else:
                if has_decisive_control_fit(place_controls):
                    return place_controls

        geocoded_controls = build_geocoded_control_points(
            labels,
            city,
            city_center,
            stop_after_controls=6 if merge_control_sources else 4,
            max_labels=max_geocoded_labels,
            prefer_large_text=merge_control_sources,
            allow_network=False,
        )
        if has_decisive_control_fit(geocoded_controls) and not merge_control_sources:
            place_future.cancel()
            return geocoded_controls

        if place_controls is None and PLACE_BEFORE_LIVE_TIMEOUT_SECONDS > 0:
            try:
                place_controls = place_future.result(timeout=PLACE_BEFORE_LIVE_TIMEOUT_SECONDS)
            except TimeoutError:
                place_controls = None

        if place_controls is not None and not merge_control_sources and len(place_controls) >= 3:
            return place_controls

        if place_controls is not None and merge_control_sources:
            merged_controls = dedupe_control_points([*geocoded_controls, *place_controls])
            if has_decisive_control_fit(merged_controls):
                return merged_controls
            if len(merged_controls) >= 2:
                return merged_controls

        geocoded_controls = build_geocoded_control_points(
            labels,
            city,
            city_center,
            stop_after_controls=6 if merge_control_sources else 4,
            max_labels=max_geocoded_labels,
            prefer_large_text=merge_control_sources,
            allow_network=True,
        )
        if has_decisive_control_fit(geocoded_controls) and not merge_control_sources:
            place_future.cancel()
            return geocoded_controls

        if place_controls is None:
            place_controls = place_future.result()
        if not merge_control_sources:
            if len(place_controls) >= 3:
                return place_controls
            return geocoded_controls

        merged_controls = dedupe_control_points([*geocoded_controls, *place_controls])
        if has_decisive_control_fit(merged_controls):
            return merged_controls
        if len(merged_controls) >= 2:
            return merged_controls
        if len(place_controls) >= 3:
            return place_controls
        return geocoded_controls
    finally:
        place_executor.shutdown(wait=False, cancel_futures=True)


def build_geocoded_control_points(
    labels: list[OcrLabel],
    city: str,
    city_center: GeocodeResult,
    *,
    stop_after_controls: int | None = None,
    max_labels: int = MAX_GEOCODED_LABELS,
    prefer_large_text: bool = False,
    allow_network: bool = True,
) -> list[ControlPoint]:
    city_x, city_y = city_center.mercator
    max_distance_m = 70000.0
    controls: list[ControlPoint] = []
    used_text: set[tuple[str, ...]] = set()
    city_tokens = place_tokens(city)
    single_token_fragments = single_tokens_supported_by_fuller_labels(labels)
    label_specs: list[tuple[OcrLabel, str, list[str]]] = []
    for label in rank_geocode_labels(labels, prefer_large_text=prefer_large_text)[:max_labels]:
        raw_text_tokens = place_tokens(label.text)
        if raw_text_tokens & CITY_INFERENCE_STOP_TOKENS:
            continue
        query_text = place_query_text(label.text)
        text_tokens = place_tokens(query_text)
        if not text_tokens or text_tokens & CITY_INFERENCE_STOP_TOKENS:
            continue
        if len(text_tokens) > 4:
            continue
        if is_noisy_poi_query(text_tokens):
            continue
        text_key = tuple(sorted(text_tokens))
        if text_key in used_text:
            continue
        used_text.add(text_key)
        if text_tokens == city_tokens:
            controls.append(ControlPoint(label=label, geocode=city_center))
            continue
        if len(text_tokens) == 1 and text_tokens <= single_token_fragments:
            continue
        if len(text_tokens) == 1 and next(iter(text_tokens)) in GENERIC_SINGLE_TOKENS:
            continue
        if is_noisy_regional_control_query(text_tokens, city_tokens, city_center):
            continue
        if not is_geocodeable_control_query(text_tokens, city_tokens):
            continue
        queries = [f"{query_text}, {context}" for context in geocode_contexts(city, city_center)]
        queries.append(query_text)
        if city.lower() in query_text.lower():
            queries.append(query_text)
        label_specs.append((label, query_text, [query for query in queries if query]))

    for start in range(0, len(label_specs), GEOCODE_LABEL_LOOKAHEAD):
        chunk = label_specs[start : start + GEOCODE_LABEL_LOOKAHEAD]
        requests: list[tuple[str, int]] = []
        lengths: list[int] = []
        for _label, _query_text, query_batch in chunk:
            lengths.append(len(query_batch))
            requests.extend((query, 3) for query in query_batch)
        batch_results = geocode_many(requests, allow_network=allow_network)
        offset = 0
        for (label, query_text, _query_batch), length in zip(chunk, lengths):
            query_results = batch_results[offset : offset + length]
            offset += length

            best: GeocodeResult | None = None
            best_score = -1.0
            for results in query_results:
                for candidate in results:
                    if not primary_name_matches_label(candidate.display_name, query_text):
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
                if stop_after_controls is not None and len(controls) >= stop_after_controls:
                    return controls
    return controls


def single_tokens_supported_by_fuller_labels(labels: list[OcrLabel]) -> set[str]:
    supported: set[str] = set()
    for label in rank_geocode_labels(labels):
        tokens = place_tokens(place_query_text(label.text))
        if len(tokens) >= 2 and not tokens & CITY_INFERENCE_STOP_TOKENS:
            supported.update(tokens)
    return supported


def has_decisive_control_fit(controls: list[ControlPoint]) -> bool:
    if len(controls) < 5:
        return False
    fit = robust_similarity_fit(controls)
    if fit is None:
        return False
    _scale, rotation, _tx, _ty, inliers, residuals = fit
    if len(inliers) < 5 or abs(rotation) > 0.35:
        return False
    residual_values = [residuals[i] for i in inliers]
    if not residual_values:
        return False
    median, p90 = residual_median_p90(residual_values)
    return median <= 1200.0 and p90 <= 3000.0


def is_geocodeable_control_query(tokens: set[str], city_tokens: set[str]) -> bool:
    if tokens == city_tokens or bool(tokens & city_tokens):
        return True
    informative_tokens = tokens - GENERIC_SINGLE_TOKENS
    return any(len(token) >= 5 for token in informative_tokens)


def is_noisy_poi_query(tokens: set[str]) -> bool:
    if not tokens:
        return False
    if tokens == {"bay", "area"}:
        return False
    if tokens <= POI_DESCRIPTOR_TOKENS | GENERIC_SINGLE_TOKENS:
        return True
    if len(tokens) >= 2 and tokens & POI_DESCRIPTOR_TOKENS:
        return True
    return False


def is_noisy_regional_control_query(
    tokens: set[str],
    city_tokens: set[str],
    city_center: GeocodeResult,
) -> bool:
    if geocode_bbox_span_m(city_center) < 85000.0 or not tokens:
        return False
    if tokens == city_tokens:
        return False
    informative_city_tokens = city_tokens - GENERIC_SINGLE_TOKENS
    if informative_city_tokens and tokens & informative_city_tokens and not tokens <= city_tokens:
        return True
    if {"bay", "area"} <= city_tokens and {"bay", "area"} <= tokens and not tokens <= city_tokens:
        return True
    informative_tokens = tokens - GENERIC_SINGLE_TOKENS
    if "city" in tokens and len(informative_tokens) >= 2:
        return True
    return False


def build_osm_place_control_points(
    labels: list[OcrLabel],
    city_center: GeocodeResult,
    *,
    prefer_large_text: bool = False,
) -> list[ControlPoint]:
    search_bbox = city_center.bbox
    if search_bbox is None:
        return []
    places = load_place_points(search_bbox)
    if not places:
        return []

    city_x, city_y = city_center.mercator
    max_distance_m = max_place_distance_m(search_bbox, city_center)
    indexed_places = [
        (place, place_tokens(place.name), place.mercator)
        for place in places
        if place_tokens(place.name)
    ]
    best_by_place: dict[tuple[str, float, float], tuple[float, ControlPoint]] = {}
    used_label_keys: set[tuple[str, int, int]] = set()
    for label in candidate_place_labels(labels):
        label_tokens = place_tokens(place_query_text(label.text))
        if not label_tokens or len(label_tokens) > 3:
            continue
        if len(label_tokens) == 1 and next(iter(label_tokens)) in GENERIC_SINGLE_TOKENS:
            continue
        label_key = (label.text.lower(), round(label.x / 12), round(label.y / 12))
        if label_key in used_label_keys:
            continue
        used_label_keys.add(label_key)

        for place, place_tokens_set, (place_x, place_y) in indexed_places:
            if (
                place_tokens_set < label_tokens
                and (label_tokens - place_tokens_set) & POI_DESCRIPTOR_TOKENS
                and place.place_type in {"city", "town", "village", "suburb", "quarter", "neighbourhood"}
            ):
                continue
            match_score = place_match_score(label_tokens, place_tokens_set)
            if match_score <= 0:
                continue
            distance_m = sqrt((place_x - city_x) ** 2 + (place_y - city_y) ** 2)
            if distance_m > max_distance_m:
                continue
            score = (
                match_score
                + place_type_score(place.place_type)
                + label.confidence / 300.0
                + (min(0.24, label.width / 520.0) if prefer_large_text else 0.0)
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


def candidate_place_labels(labels: list[OcrLabel]) -> list[OcrLabel]:
    candidates: list[OcrLabel] = []
    seen: set[tuple[str, int, int]] = set()

    def add(label: OcrLabel) -> None:
        raw_tokens = place_tokens(label.text)
        if raw_tokens & CITY_INFERENCE_STOP_TOKENS:
            return
        tokens = place_tokens(place_query_text(label.text))
        if not tokens or tokens & CITY_INFERENCE_STOP_TOKENS:
            return
        if len(tokens) == 1 and next(iter(tokens)) in POI_DESCRIPTOR_TOKENS:
            return
        key = (" ".join(sorted(tokens)), round(label.x / 12), round(label.y / 12))
        if key in seen:
            return
        seen.add(key)
        candidates.append(label)

    for label in rank_place_labels(labels)[:MAX_PLACE_LABELS]:
        add(label)

    concise_labels = sorted(
        labels,
        key=lambda label: (
            len(place_tokens(label.text)) in {1, 2},
            label.confidence,
            -label.width * label.height,
        ),
        reverse=True,
    )
    for label in concise_labels:
        raw_tokens = place_tokens(label.text)
        if raw_tokens & CITY_INFERENCE_STOP_TOKENS:
            continue
        tokens = place_tokens(place_query_text(label.text))
        if not tokens or tokens & CITY_INFERENCE_STOP_TOKENS or len(tokens) > 2:
            continue
        if len(tokens) == 1 and next(iter(tokens)) in POI_DESCRIPTOR_TOKENS:
            continue
        if len(tokens) == 1 and next(iter(tokens)) in GENERIC_SINGLE_TOKENS:
            continue
        if label.confidence < 55.0:
            continue
        add(label)
        if len(candidates) >= MAX_PLACE_LABELS + 64:
            break
    return candidates


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
            token_quality(place_tokens(place_query_text(label.text))),
            label.confidence,
            -len(label.text),
        ),
        reverse=True,
    )


def rank_geocode_labels(labels: list[OcrLabel], *, prefer_large_text: bool = False) -> list[OcrLabel]:
    if prefer_large_text:
        return sorted(
            labels,
            key=lambda label: (
                min(2, len(place_tokens(place_query_text(label.text)))),
                min(35.0, (label.width * label.height) / 90.0),
                label.confidence,
                len(label.text),
            ),
            reverse=True,
        )
    return sorted(
        labels,
        key=lambda label: (
            min(2, len(place_tokens(place_query_text(label.text)))),
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


def dedupe_control_points(controls: list[ControlPoint]) -> list[ControlPoint]:
    selected: list[ControlPoint] = []
    for control in sorted(controls, key=control_quality, reverse=True):
        if any(is_duplicate_control(control, existing) for existing in selected):
            continue
        selected.append(control)
    return sorted(selected, key=control_quality, reverse=True)


def is_duplicate_control(candidate: ControlPoint, existing: ControlPoint) -> bool:
    cand_name = candidate.geocode.display_name.split(",", 1)[0].lower()
    existing_name = existing.geocode.display_name.split(",", 1)[0].lower()
    cand_x, cand_y = candidate.geocode.mercator
    existing_x, existing_y = existing.geocode.mercator
    same_place = cand_name == existing_name or sqrt((cand_x - existing_x) ** 2 + (cand_y - existing_y) ** 2) <= 180.0
    if same_place:
        return True
    cand_tokens = place_tokens(place_query_text(candidate.label.text))
    existing_tokens = place_tokens(place_query_text(existing.label.text))
    same_label = (
        cand_tokens == existing_tokens
        and abs(candidate.label.x - existing.label.x) <= 4.0
        and abs(candidate.label.y - existing.label.y) <= 4.0
    )
    if same_label:
        return True
    same_line = abs(candidate.label.y - existing.label.y) <= max(candidate.label.height, existing.label.height, 12.0)
    close_x = abs(candidate.label.x - existing.label.x) <= max(candidate.label.width, existing.label.width, 80.0)
    return bool(cand_tokens and existing_tokens and (cand_tokens < existing_tokens or existing_tokens < cand_tokens) and same_line and close_x)


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


def control_quality(control: ControlPoint) -> tuple[int, float, float]:
    return (
        len(place_tokens(place_query_text(control.label.text))),
        min(35.0, (control.label.width * control.label.height) / 90.0),
        control.label.confidence,
    )


def geocode_contexts(city: str, city_center: GeocodeResult) -> list[str]:
    contexts: list[str] = []
    for value in [city, *city_center.display_name.split(",")[:5]]:
        value = value.strip()
        lowered = value.lower()
        if (
            not value
            or value.isdigit()
            or lowered in {"inferred map area", "united states"}
            or "county" in lowered
        ):
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
    tokens: set[str] = set()
    for part in re.split(r"[^a-z0-9]+", value.lower()):
        if len(part) < 3:
            continue
        tokens.add(OCR_PLACE_TOKEN_ALIASES.get(part, part))
    return tokens


def choose_similarity_fit(
    controls: list[ControlPoint],
    image_path: str,
    city_center: GeocodeResult,
    *,
    allow_two_control_fit: bool,
    allow_sparse_regional_fit: bool = False,
) -> tuple[float, float, float, float, list[int], list[float]] | None:
    use_sparse_robust_limits = allow_two_control_fit and allow_sparse_regional_fit
    robust_fit = robust_similarity_fit(
        controls,
        max_meters_per_pixel=(
            MAX_SPARSE_ROBUST_SIMILARITY_METERS_PER_PIXEL
            if use_sparse_robust_limits
            else MAX_ROBUST_SIMILARITY_METERS_PER_PIXEL
        ),
        max_inlier_residual_m=(
            MAX_SPARSE_ROBUST_SIMILARITY_INLIER_RESIDUAL_M if use_sparse_robust_limits else None
        ),
    )
    candidates: list[tuple[str, tuple[float, float, float, float, list[int], list[float]]]] = []
    if robust_fit is not None:
        candidates.append(("multi", robust_fit))

    if allow_two_control_fit:
        for i, j in combinations(range(len(controls)), 2):
            pair_fit = two_control_similarity_fit(controls, i, j)
            if pair_fit is not None:
                candidates.append(("pair", pair_fit))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][1]

    road_scorer = None if allow_two_control_fit else build_similarity_road_scorer(image_path, city_center)
    best: tuple[float, tuple[float, float, float, float, list[int], list[float]]] | None = None
    total_spread = control_spread(np.array([control.pixel for control in controls], dtype=float))
    for kind, fit in candidates:
        scale, rotation, tx, ty, inliers, residuals = fit
        median, p90 = residual_median_p90([residuals[idx] for idx in inliers])
        spread = control_spread(pixel_positions(controls, inliers))
        fit_score = fit_candidate_score(len(inliers), len(controls), spread, total_spread, median, p90, rotation)
        road_score, road_count = road_scorer(scale, rotation, tx, ty) if road_scorer is not None else (0.0, 0)
        pair_penalty = 0.08 if kind == "pair" and road_score < 0.48 else 0.0
        pair_road_bonus = 0.24 if kind == "pair" and road_score >= 0.78 else 0.0
        score = (
            fit_score
            + 1.25 * road_score
            + min(0.16, road_count / 8000.0)
            + (0.04 if len(inliers) >= 3 else 0.0)
            + pair_road_bonus
            - pair_penalty
        )
        if best is None or score > best[0]:
            best = (score, fit)
    return best[1] if best is not None else candidates[0][1]


def two_control_similarity_fit(
    controls: list[ControlPoint],
    first_index: int,
    second_index: int,
) -> tuple[float, float, float, float, list[int], list[float]] | None:
    first = controls[first_index]
    second = controls[second_index]
    p1 = np.array(first.pixel, dtype=float)
    p2 = np.array(second.pixel, dtype=float)
    m1 = np.array(first.mercator, dtype=float)
    m2 = np.array(second.mercator, dtype=float)
    pixel_vector = p2 - p1
    mercator_vector = m2 - m1
    pixel_distance = float(np.linalg.norm(pixel_vector))
    mercator_distance = float(np.linalg.norm(mercator_vector))
    if pixel_distance < 80.0 or mercator_distance < 1200.0:
        return None
    scale = mercator_distance / pixel_distance
    if scale <= 0.0 or scale > MAX_TWO_CONTROL_SIMILARITY_METERS_PER_PIXEL:
        return None
    rotation = float(atan2(mercator_vector[1], mercator_vector[0]) - atan2(pixel_vector[1], pixel_vector[0]))
    if abs(rotation) > 0.35:
        return None
    transformed = apply_similarity(np.array([first.pixel, second.pixel], dtype=float), scale, rotation, 0.0, 0.0)
    tx, ty = (m1 - transformed[0]).tolist()
    all_pixels = np.array([control.pixel for control in controls], dtype=float)
    all_mercator = np.array([control.mercator for control in controls], dtype=float)
    residuals = np.linalg.norm(apply_similarity(all_pixels, scale, rotation, tx, ty) - all_mercator, axis=1).tolist()
    return scale, rotation, float(tx), float(ty), [first_index, second_index], residuals


def robust_similarity_inlier_threshold(scale: float, max_inlier_residual_m: float | None) -> float:
    threshold = max(1200.0, scale * 90.0)
    if max_inlier_residual_m is not None:
        threshold = min(max(0.0, float(max_inlier_residual_m)), threshold)
    return threshold


def build_similarity_road_scorer(
    image_path: str,
    city_center: GeocodeResult,
):
    if city_center.bbox is None:
        return None
    if geocode_bbox_span_m(city_center) >= 85000.0:
        return None
    road_points = sample_road_points(load_road_points(city_center.bbox), max_points=12000)
    if road_points.size == 0:
        return None
    from .extract import load_rgb

    feature_distance = image_feature_distance(load_rgb(image_path))

    def score(scale: float, rotation: float, tx: float, ty: float) -> tuple[float, int]:
        lon, lat = mercator_to_lonlat(tx, ty)
        transform = GeoreferenceTransform(
            city=city_center.display_name.split(",", 1)[0],
            lon=lon,
            lat=lat,
            origin_x_ratio=0.0,
            origin_y_ratio=0.0,
            meters_per_pixel=scale,
            rotation_radians=rotation,
            confidence=0.0,
            source="control-fit-candidate",
        )
        return score_georeference_transform(road_points, feature_distance, transform)

    return score


def robust_similarity_fit(
    controls: list[ControlPoint],
    *,
    max_meters_per_pixel: float = MAX_ROBUST_SIMILARITY_METERS_PER_PIXEL,
    max_inlier_residual_m: float | None = None,
) -> tuple[float, float, float, float, list[int], list[float]] | None:
    if len(controls) < 2:
        return None
    key = (
        round(max(0.0, float(max_meters_per_pixel)), 6),
        None if max_inlier_residual_m is None else round(max(0.0, float(max_inlier_residual_m)), 6),
        tuple(
            (
                float(control.pixel[0]),
                float(control.pixel[1]),
                float(control.mercator[0]),
                float(control.mercator[1]),
            )
            for control in controls
        ),
    )
    return _robust_similarity_fit_cached(key)


@lru_cache(maxsize=256)
def _robust_similarity_fit_cached(
    key: tuple[float, float | None, tuple[tuple[float, float, float, float], ...]],
) -> tuple[float, float, float, float, list[int], list[float]] | None:
    max_meters_per_pixel, max_inlier_residual_m, control_key = key
    if len(control_key) < 2:
        return None
    pixel = np.array([(item[0], item[1]) for item in control_key], dtype=float)
    merc = np.array([(item[2], item[3]) for item in control_key], dtype=float)

    best: tuple[float, float, float, float, list[int], list[float]] | None = None
    best_score: float | None = None
    total_spread = control_spread(pixel)
    for i, j in combinations(range(len(control_key)), 2):
        p1, p2 = pixel[i], pixel[j]
        m1, m2 = merc[i], merc[j]
        p_vec = p2 - p1
        m_vec = m2 - m1
        p_norm = np.linalg.norm(p_vec)
        m_norm = np.linalg.norm(m_vec)
        if p_norm < 30 or m_norm < 300:
            continue
        scale = float(m_norm / p_norm)
        if scale <= 0 or scale > max_meters_per_pixel:
            continue
        rotation = float(atan2(m_vec[1], m_vec[0]) - atan2(p_vec[1], p_vec[0]))
        transformed = apply_similarity(pixel, scale, rotation, 0.0, 0.0)
        tx, ty = (m1 - transformed[i]).tolist()
        residuals = np.linalg.norm(apply_similarity(pixel, scale, rotation, tx, ty) - merc, axis=1).tolist()
        threshold = robust_similarity_inlier_threshold(scale, max_inlier_residual_m)
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
        r_threshold = robust_similarity_inlier_threshold(r_scale, max_inlier_residual_m)
        r_inliers = [idx for idx, residual in enumerate(r_residuals) if residual <= r_threshold]
        if len(r_inliers) < 3:
            continue
        spread = control_spread(pixel[r_inliers])
        median, p90 = residual_median_p90([r_residuals[idx] for idx in r_inliers], empty=float("inf"))
        if median > 2500.0 or p90 > 6500.0:
            continue
        score = fit_candidate_score(
            len(r_inliers),
            len(control_key),
            spread,
            total_spread,
            median,
            p90,
            r_rotation,
        )
        if best_score is None or score > best_score:
            best_score = score
            best = (r_scale, r_rotation, r_tx, r_ty, r_inliers, r_residuals)

    for seed in combinations(range(len(control_key)), 3):
        refined = fit_similarity(pixel[list(seed)], merc[list(seed)])
        if refined is None:
            continue
        r_scale, r_rotation, r_tx, r_ty = refined
        if abs(r_rotation) > 0.35:
            continue
        r_residuals = np.linalg.norm(apply_similarity(pixel, r_scale, r_rotation, r_tx, r_ty) - merc, axis=1).tolist()
        r_threshold = robust_similarity_inlier_threshold(r_scale, max_inlier_residual_m)
        r_inliers = [idx for idx, residual in enumerate(r_residuals) if residual <= r_threshold]
        if len(r_inliers) < 3:
            continue
        spread = control_spread(pixel[r_inliers])
        median, p90 = residual_median_p90([r_residuals[idx] for idx in r_inliers])
        if median > 2500.0 or p90 > 6500.0:
            continue
        score = fit_candidate_score(
            len(r_inliers),
            len(control_key),
            spread,
            total_spread,
            median,
            p90,
            r_rotation,
        )
        if best_score is None or score > best_score:
            best_score = score
            best = (r_scale, r_rotation, r_tx, r_ty, r_inliers, r_residuals)
    if best is None:
        return None
    return prune_single_noisy_similarity_control(best, pixel, merc, total_spread)


def prune_single_noisy_similarity_control(
    fit: tuple[float, float, float, float, list[int], list[float]],
    pixel: np.ndarray,
    merc: np.ndarray,
    total_spread: float,
) -> tuple[float, float, float, float, list[int], list[float]]:
    scale, rotation, tx, ty, inliers, residuals = fit
    if len(inliers) < 6:
        return fit

    base_median, base_p90 = residual_median_p90([residuals[idx] for idx in inliers])
    base_score = fit_candidate_score(
        len(inliers),
        len(pixel),
        control_spread(pixel[inliers]),
        total_spread,
        base_median,
        base_p90,
        rotation,
    )
    best = fit
    best_improvement = 0.0
    for drop_index in inliers:
        candidate_inliers = [idx for idx in inliers if idx != drop_index]
        if len(candidate_inliers) < 5:
            continue
        refined = fit_similarity(pixel[candidate_inliers], merc[candidate_inliers])
        if refined is None:
            continue
        r_scale, r_rotation, r_tx, r_ty = refined
        if abs(r_rotation) > 0.35:
            continue
        r_residuals = np.linalg.norm(
            apply_similarity(pixel, r_scale, r_rotation, r_tx, r_ty) - merc,
            axis=1,
        ).tolist()
        median, p90 = residual_median_p90([r_residuals[idx] for idx in candidate_inliers])
        if median > 2500.0 or p90 > 6500.0:
            continue
        spread = control_spread(pixel[candidate_inliers])
        if spread < total_spread * 0.55:
            continue
        median_improvement = base_median - median
        p90_improvement = base_p90 - p90
        enough_median_improvement = median_improvement >= max(150.0, base_median * 0.25)
        enough_tail_improvement = (
            median_improvement >= max(50.0, base_median * 0.10)
            and p90_improvement >= max(800.0, base_p90 * 0.35)
        )
        if not (enough_median_improvement or enough_tail_improvement):
            continue
        if p90_improvement < max(250.0, base_p90 * 0.30):
            continue
        score = fit_candidate_score(
            len(candidate_inliers),
            len(pixel),
            spread,
            total_spread,
            median,
            p90,
            r_rotation,
        )
        if score < base_score - 0.08:
            continue
        improvement = median_improvement + p90_improvement
        if improvement > best_improvement:
            best_improvement = improvement
            best = (r_scale, r_rotation, r_tx, r_ty, candidate_inliers, r_residuals)
    return best


def should_try_road_refinement(
    city_context: CityContext,
    meters_per_pixel: float,
    inlier_count: int,
    residual_median_m: float,
    residual_p90_m: float,
    spread: float,
    width: int,
    height: int,
) -> bool:
    if inlier_count <= 2:
        return False
    if city_context.query == "Inferred map area" or geocode_bbox_span_m(city_context.center) >= 85000.0:
        return False
    if (
        inlier_count <= 4
        and residual_median_m <= 1300.0
        and residual_p90_m <= 1800.0
        and not has_local_road_points(city_context.center.bbox)
    ):
        return False
    image_area = max(float(width * height), 1.0)
    if inlier_count >= 6 and residual_median_m <= 900.0 and residual_p90_m <= 1200.0:
        return False
    if (
        meters_per_pixel >= 20.0
        and inlier_count >= 8
        and residual_median_m <= 1600.0
        and residual_p90_m <= 3200.0
        and spread >= image_area * 0.08
    ):
        return False
    if inlier_count >= 5 and residual_median_m <= 500.0 and residual_p90_m <= 1000.0:
        return False
    return True


def should_lock_road_refinement_scale(
    meters_per_pixel: float,
    inlier_count: int,
    residual_median_m: float,
    residual_p90_m: float,
    spread: float,
    width: int,
    height: int,
) -> bool:
    image_area = max(float(width * height), 1.0)
    # Wide regional screenshots have enough label spread to establish scale;
    # road matching should improve placement without stretching that fit.
    return (
        meters_per_pixel >= 20.0
        and inlier_count >= 8
        and residual_median_m <= 1600.0
        and residual_p90_m <= 3200.0
        and spread >= image_area * 0.08
    )


def control_residuals_for_transform(
    transform: GeoreferenceTransform,
    controls: list[ControlPoint],
    width: int,
    height: int,
) -> np.ndarray:
    origin_x = transform.origin_x_ratio * width
    origin_y = transform.origin_y_ratio * height
    pixel = np.array([(control.label.x - origin_x, origin_y - control.label.y) for control in controls], dtype=float)
    merc = np.array([control.mercator for control in controls], dtype=float)
    tx, ty = lonlat_to_mercator(transform.lon, transform.lat)
    transformed = apply_similarity(pixel, transform.meters_per_pixel, transform.rotation_radians, tx, ty)
    return np.linalg.norm(transformed - merc, axis=1)


def refinement_preserves_label_fit(
    residuals: np.ndarray,
    base_median_m: float,
    base_p90_m: float,
    control_count: int,
) -> bool:
    if residuals.size == 0:
        return False
    median, p90 = residual_median_p90(residuals)
    if control_count <= 4:
        median_limit = max(650.0, base_median_m + 650.0)
        p90_limit = max(1200.0, base_p90_m + 900.0)
    else:
        median_limit = max(1200.0, base_median_m * 2.5)
        p90_limit = max(2500.0, base_p90_m * 2.2)
    return median <= median_limit and p90 <= p90_limit


def pixel_positions(controls: list[ControlPoint], indexes: list[int]) -> np.ndarray:
    return np.array([controls[idx].pixel for idx in indexes], dtype=float)


def residual_median_p90(values: list[float] | np.ndarray, *, empty: float = 0.0) -> tuple[float, float]:
    if isinstance(values, np.ndarray):
        if values.size == 0:
            return empty, empty
        sorted_values = sorted(float(value) for value in values.ravel())
    else:
        if not values:
            return empty, empty
        sorted_values = sorted(float(value) for value in values)
    return linear_percentile(sorted_values, 50.0), linear_percentile(sorted_values, 90.0)


def linear_percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


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
    score = 0.25 * residual_score + 0.2 * rotation_score + 0.3 * spread_score + 0.25 * inlier_score
    if abs(rotation) > 0.05:
        score -= min(0.25, (abs(rotation) - 0.05) / 0.2)
    return score


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
    rotation = float(atan2(covariance[0, 1] - covariance[1, 0], covariance[0, 0] + covariance[1, 1]))
    cos_r = cos(rotation)
    sin_r = sin(rotation)
    scale = float(
        (
            cos_r * (covariance[0, 0] + covariance[1, 1])
            + sin_r * (covariance[0, 1] - covariance[1, 0])
        )
        / variance
    )
    translation = m_mean - scale * np.array(
        [
            cos_r * p_mean[0] - sin_r * p_mean[1],
            sin_r * p_mean[0] + cos_r * p_mean[1],
        ],
        dtype=float,
    )
    return scale, rotation, float(translation[0]), float(translation[1])


def apply_similarity(points: np.ndarray, scale: float, rotation: float, tx: float, ty: float) -> np.ndarray:
    rot = np.array([[cos(rotation), -sin(rotation)], [sin(rotation), cos(rotation)]])
    return scale * (points @ rot.T) + np.array([tx, ty])


def confidence_from_fit(control_count: int, total_count: int, median_residual: float, p90_residual: float) -> float:
    count_score = min(1.0, control_count / 6.0)
    inlier_score = control_count / max(total_count, 1)
    residual_score = max(0.0, 1.0 - median_residual / 2500.0) * 0.7 + max(0.0, 1.0 - p90_residual / 6500.0) * 0.3
    return round(0.25 + 0.75 * (0.45 * residual_score + 0.35 * count_score + 0.2 * inlier_score), 3)
