"""Coarse-to-fine, boundary-aware service-area extraction.

The global selector identifies the intended region at low resolution. A compact
refiner then evaluates only native-resolution tiles intersecting the uncertain
boundary band and predicts an inside logit, signed distance, and corner map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .extract import AUTO_FILL_STYLE, ExtractionResult, extraction_confidence, orient_pixel_polygon
from .model_extract import InferenceSessionLike, ModelExtractionConfig, guidance_channels, predict_mask_probabilities


@dataclass(frozen=True)
class BoundaryFieldConfig:
    coarse_input_size: int = 256
    coarse_threshold: float = 0.45
    refiner_patch_size: int = 192
    refiner_core_size: int = 128
    refine_band_px: int = 24
    geometry_tolerance_px: float = 0.75

    def __post_init__(self) -> None:
        if self.refiner_patch_size <= self.refiner_core_size:
            raise ValueError("refiner patch must be larger than its core")
        if (self.refiner_patch_size - self.refiner_core_size) % 2:
            raise ValueError("refiner context must split evenly around the core")
        if not 0.0 < self.coarse_threshold < 1.0:
            raise ValueError("coarse threshold must be between 0 and 1")


def extract_service_area_with_boundaryfield(
    rgb: np.ndarray,
    coarse_session: InferenceSessionLike,
    refiner_session: InferenceSessionLike | None = None,
    *,
    hints: Any = None,
    config: BoundaryFieldConfig | None = None,
) -> ExtractionResult:
    cfg = config or BoundaryFieldConfig()
    coarse_config = ModelExtractionConfig(
        input_width=cfg.coarse_input_size,
        input_height=cfg.coarse_input_size,
        threshold=cfg.coarse_threshold,
        simplify_px=0.0,
        style=AUTO_FILL_STYLE,
        output_activation="logits",
        input_channels=5,
    )
    coarse_probabilities = predict_mask_probabilities(rgb, coarse_session, config=coarse_config, hints=hints)
    coarse_mask = coarse_probabilities >= cfg.coarse_threshold
    if not coarse_mask.any():
        raise ValueError("BoundaryField global selector did not identify a service area.")

    if refiner_session is None:
        refined_mask = clean_refined_mask(coarse_mask, coarse_mask)
        signed_distance = None
        corner_map = np.zeros(coarse_mask.shape, dtype=np.float32)
        tile_count = 0
    else:
        refined_mask, signed_distance, corner_map, tile_count = refine_boundary_band(
            rgb,
            coarse_probabilities,
            coarse_mask,
            refiner_session,
            hints=hints,
            config=cfg,
        )
    refined_mask = select_guided_component(refined_mask, hints)
    geometry, contour_count = boundaryfield_mask_to_geometry(
        refined_mask,
        corner_map=corner_map,
        tolerance_px=cfg.geometry_tolerance_px,
    )
    boundary = _boundary_mask(refined_mask)
    confidence = extraction_confidence(refined_mask, AUTO_FILL_STYLE, contour_count)
    band_uncertainty: float | None = None
    if signed_distance is not None:
        band_uncertainty = float((np.abs(signed_distance[boundary]) < 0.08).mean()) if boundary.any() else 1.0
        confidence = min(confidence, max(0.0, 1.0 - band_uncertainty))
    return ExtractionResult(
        mask=refined_mask,
        style=AUTO_FILL_STYLE,
        pixel_geometry=geometry,
        coverage_ratio=float(refined_mask.mean()),
        contour_count=contour_count,
        confidence=confidence,
        diagnostics={
            "boundaryfield": True,
            "coarse_input_shape": [cfg.coarse_input_size, cfg.coarse_input_size],
            "coarse_threshold": cfg.coarse_threshold,
            "refiner_patch_size": cfg.refiner_patch_size,
            "refiner_core_size": cfg.refiner_core_size,
            "refiner_tile_count": tile_count,
            "refiner_enabled": refiner_session is not None,
            "refine_band_px": cfg.refine_band_px,
            "geometry_tolerance_px": cfg.geometry_tolerance_px,
            "corner_peak": float(corner_map.max()) if corner_map.size else 0.0,
            "boundary_uncertainty": band_uncertainty,
        },
    )


def refine_boundary_band(
    rgb: np.ndarray,
    coarse_probabilities: np.ndarray,
    coarse_mask: np.ndarray,
    refiner_session: InferenceSessionLike,
    *,
    hints: Any,
    config: BoundaryFieldConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    height, width = coarse_mask.shape
    patch_size = config.refiner_patch_size
    core_size = config.refiner_core_size
    margin = (patch_size - core_size) // 2
    boundary = _boundary_mask(coarse_mask)
    if not boundary.any():
        raise ValueError("BoundaryField coarse mask has no usable boundary.")

    guidance = guidance_channels(rgb, width, height, hints=hints)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge /= max(1e-6, float(np.quantile(edge, 0.99)))
    full_input = np.concatenate(
        [
            np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)),
            coarse_probabilities[np.newaxis].astype(np.float32),
            guidance.astype(np.float32),
            np.clip(edge, 0.0, 1.0)[np.newaxis].astype(np.float32),
        ],
        axis=0,
    )

    ys, xs = np.where(boundary)
    cells = sorted({(int(y // core_size), int(x // core_size)) for y, x in zip(ys, xs, strict=True)})
    padded = np.pad(full_input, ((0, 0), (margin, margin), (margin, margin)), mode="reflect")
    patches: list[np.ndarray] = []
    origins: list[tuple[int, int, int, int]] = []
    for cell_y, cell_x in cells:
        y0 = cell_y * core_size
        x0 = cell_x * core_size
        core_h = min(core_size, height - y0)
        core_w = min(core_size, width - x0)
        patch = padded[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        if patch.shape[-2:] != (patch_size, patch_size):
            patch = np.pad(
                patch,
                ((0, 0), (0, patch_size - patch.shape[-2]), (0, patch_size - patch.shape[-1])),
                mode="edge",
            )
        patches.append(patch)
        origins.append((y0, x0, core_h, core_w))

    batch = np.ascontiguousarray(np.stack(patches, axis=0), dtype=np.float32)
    input_name = refiner_session.get_inputs()[0].name
    outputs = refiner_session.run(None, {input_name: batch})
    if not outputs:
        raise ValueError("BoundaryField refiner returned no output.")
    fields = np.asarray(outputs[0], dtype=np.float32)
    if fields.ndim != 4 or fields.shape[0] != len(patches) or fields.shape[1] < 3:
        raise ValueError(f"BoundaryField refiner returned unexpected shape {fields.shape}.")

    base_sdf = signed_distance_from_mask(coarse_mask, clip_px=float(config.refine_band_px))
    refined_logits = np.where(coarse_mask, 8.0, -8.0).astype(np.float32)
    refined_sdf = base_sdf.copy()
    corner_map = np.zeros((height, width), dtype=np.float32)
    for index, (y0, x0, core_h, core_w) in enumerate(origins):
        core = fields[index, :, margin : margin + core_h, margin : margin + core_w]
        refined_logits[y0 : y0 + core_h, x0 : x0 + core_w] = core[0]
        refined_sdf[y0 : y0 + core_h, x0 : x0 + core_w] = np.clip(core[1], -1.0, 1.0)
        corner_map[y0 : y0 + core_h, x0 : x0 + core_w] = _sigmoid(core[2])

    kernel_size = (config.refine_band_px * 2) + 1
    band = cv2.dilate(boundary.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8), iterations=1) > 0
    refined_probability = _sigmoid(refined_logits)
    candidate = refined_probability >= 0.5
    refined_mask = np.where(band, candidate, coarse_mask)
    refined_mask = clean_refined_mask(refined_mask.astype(bool), coarse_mask)
    return refined_mask, refined_sdf, corner_map, len(patches)


def clean_refined_mask(mask: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
    """Remove refiner speckle without applying corner-rounding morphology."""
    height, width = mask.shape
    minimum_component_area = max(48, int(round(mask.size * 0.00015)))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        overlap = int(np.logical_and(component, coarse_mask).sum())
        if area >= minimum_component_area and overlap >= max(8, int(area * 0.08)):
            keep |= component

    contours, hierarchy = cv2.findContours(keep.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return keep
    maximum_noise_hole_area = max(32.0, height * width * 0.00045)
    filled = keep.astype(np.uint8) * 255
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] >= 0 and cv2.contourArea(contour) < maximum_noise_hole_area:
            cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    return filled > 0


def select_guided_component(mask: np.ndarray, hints: Any) -> np.ndarray:
    seed = hints.get("seed_point") if isinstance(hints, dict) else getattr(hints, "seed_point", None)
    if seed is None:
        return mask
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    x = int(round(float(np.clip(seed[0], 0, mask.shape[1] - 1))))
    y = int(round(float(np.clip(seed[1], 0, mask.shape[0] - 1))))
    label = int(labels[y, x])
    if label == 0:
        ys, xs = np.where(mask)
        if not len(xs):
            return mask
        nearest = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        label = int(labels[ys[nearest], xs[nearest]])
    return labels == label


def signed_distance_from_mask(mask: np.ndarray, *, clip_px: float = 24.0) -> np.ndarray:
    binary = mask.astype(np.uint8)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    return np.clip((inside - outside) / max(1.0, clip_px), -1.0, 1.0).astype(np.float32)


def boundaryfield_mask_to_geometry(
    mask: np.ndarray,
    *,
    corner_map: np.ndarray | None = None,
    tolerance_px: float = 0.75,
) -> tuple[Polygon | MultiPolygon, int]:
    """Polygonize while retaining holes and protecting high-confidence corners."""
    height, width = mask.shape
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8) * 255,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    if hierarchy is None:
        raise ValueError("No service-area polygon could be extracted from the BoundaryField mask.")
    hierarchy = hierarchy[0]
    min_area = max(32.0, height * width * 0.00002)
    polygons: list[Polygon] = []
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1 or cv2.contourArea(contour) < min_area:
            continue
        exterior = _approximate_contour(contour, corner_map, tolerance_px)
        holes: list[list[tuple[float, float]]] = []
        child = int(hierarchy[index][2])
        while child != -1:
            if cv2.contourArea(contours[child]) >= 12.0:
                holes.append(_approximate_contour(contours[child], corner_map, tolerance_px))
            child = int(hierarchy[child][0])
        if len(exterior) < 4:
            continue
        polygon = Polygon(exterior, holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area < min_area:
            continue
        if isinstance(polygon, Polygon):
            polygons.append(orient_pixel_polygon(polygon))
        elif isinstance(polygon, MultiPolygon):
            polygons.extend(orient_pixel_polygon(part) for part in polygon.geoms if part.area >= min_area)
    if not polygons:
        raise ValueError("No valid service-area polygon could be extracted from the BoundaryField mask.")
    merged = unary_union(polygons)
    if isinstance(merged, Polygon):
        return orient_pixel_polygon(merged), len(polygons)
    if isinstance(merged, MultiPolygon):
        return MultiPolygon([orient_pixel_polygon(part) for part in merged.geoms]), len(polygons)
    raise ValueError("BoundaryField extraction did not form a polygon.")


def _approximate_contour(
    contour: np.ndarray,
    corner_map: np.ndarray | None,
    tolerance_px: float,
) -> list[tuple[float, float]]:
    epsilon = max(0.25, float(tolerance_px))
    if corner_map is not None and len(contour):
        points = contour[:, 0, :]
        xs = np.clip(points[:, 0], 0, corner_map.shape[1] - 1)
        ys = np.clip(points[:, 1], 0, corner_map.shape[0] - 1)
        if float(corner_map[ys, xs].max(initial=0.0)) >= 0.45:
            epsilon = min(epsilon, 0.5)
    approximated = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
    coords = [(float(point[0][0]), float(point[0][1])) for point in approximated]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return (binary > 0) & (eroded == 0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))
