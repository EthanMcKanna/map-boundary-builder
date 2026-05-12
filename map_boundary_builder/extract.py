from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage import morphology

DEFAULT_SIMPLIFY_PX = 6.0


@dataclass(frozen=True)
class ExtractionResult:
    mask: np.ndarray
    style: str
    pixel_geometry: Polygon | MultiPolygon
    coverage_ratio: float
    contour_count: int
    confidence: float


def load_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    rgba = np.asarray(image)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    # Composite transparent screenshots over white. Most provided app-map PNGs
    # are fully opaque, but this keeps alpha edges predictable.
    composited = rgb * alpha + 255.0 * (1.0 - alpha)
    return np.clip(composited, 0, 255).astype(np.uint8)


def extract_service_area(image_path: str | Path, simplify_px: float = DEFAULT_SIMPLIFY_PX) -> ExtractionResult:
    rgb = load_rgb(image_path)
    style = classify_style(rgb)
    if style == "bright-blue":
        raw_mask = blue_service_mask(rgb)
    elif style == "gray-fill":
        raw_mask = gray_fill_service_mask(rgb)
    else:
        raw_mask = dark_teal_service_mask(rgb)
    mask = repair_mask(raw_mask, style)
    if style == "gray-fill":
        mask = keep_main_components(mask, max_components=1)
    mask = remove_dark_teal_chrome(mask, style)
    geometry, contour_count = mask_to_geometry(mask, simplify_px=simplify_px)
    coverage_ratio = float(mask.mean())
    confidence = extraction_confidence(mask, style, contour_count)
    return ExtractionResult(
        mask=mask,
        style=style,
        pixel_geometry=geometry,
        coverage_ratio=coverage_ratio,
        contour_count=contour_count,
        confidence=confidence,
    )


def classify_style(rgb: np.ndarray) -> str:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    bright_blue = ((hue >= 92) & (hue <= 116) & (sat >= 90) & (val >= 130)).mean()
    dark_pixels = (val < 95).mean()
    low_saturation = (sat < 25).mean()
    teal_pixels = ((hue >= 78) & (hue <= 104) & (sat >= 45) & (val >= 50) & (val <= 190)).mean()
    r, g, _b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    green_pixels = (
        ((hue >= 55) & (hue <= 90) & (sat >= 45) & (val >= 80) & (g.astype(np.int16) > r.astype(np.int16) + 25))
    ).mean()
    if bright_blue > 0.02 and bright_blue > teal_pixels * 1.5:
        return "bright-blue"
    if green_pixels > 0.015:
        return "dark-teal"
    if dark_pixels > 0.80 and teal_pixels < 0.01 and bright_blue < 0.01:
        return "gray-fill"
    if low_saturation > 0.85 and dark_pixels > 0.35:
        return "gray-fill"
    if dark_pixels > 0.35 or teal_pixels > 0.08:
        return "dark-teal"
    return "bright-blue"


def blue_service_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    saturated_blue = (hue >= 92) & (hue <= 116) & (sat >= 75) & (val >= 105)
    app_blue = (b >= 145) & (g >= 80) & (r <= 95) & ((b.astype(np.int16) - r.astype(np.int16)) >= 80)
    return saturated_blue | app_blue


def dark_teal_service_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    zoox_green = green_service_fill_mask(rgb, hue, sat, val)
    if zoox_green is not None:
        return zoox_green

    teal = (hue >= 78) & (hue <= 104) & (sat >= 35) & (val >= 45)
    channel_teal = (g >= 55) & (b >= 55) & (r <= 65) & ((g.astype(np.int16) + b.astype(np.int16)) > 2 * r.astype(np.int16) + 45)
    not_white_text = val < 225
    broad = (teal | channel_teal) & not_white_text

    # Some dark map exports tint the full background teal, so hue alone marks
    # the whole canvas. In that case switch to brighter rendered map ink; the
    # repair pass turns the dense street/building texture into its envelope.
    if broad.mean() > 0.75:
        return (g > 70) & (b > 60) & (r < 80) & (val > 65)
    return broad


def green_service_fill_mask(
    rgb: np.ndarray,
    hue: np.ndarray,
    sat: np.ndarray,
    val: np.ndarray,
) -> np.ndarray | None:
    r, g, _b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    green = (hue >= 55) & (hue <= 90) & (sat >= 45) & (val >= 80) & (g.astype(np.int16) > r.astype(np.int16) + 25)
    labels, count = ndimage.label(green)
    if count == 0:
        return None

    areas = ndimage.sum(green, labels, index=np.arange(1, count + 1))
    largest_label = int(np.argmax(areas) + 1)
    largest_area = float(areas[largest_label - 1])
    h, w = green.shape
    if largest_area < max(4000.0, green.size * 0.015):
        return None

    ys, xs = np.where(labels == largest_label)
    if len(ys) == 0:
        return None
    component_h = int(ys.max() - ys.min() + 1)
    component_w = int(xs.max() - xs.min() + 1)
    if component_h < h * 0.15 or component_w < w * 0.15:
        return None
    if ys.min() <= h * 0.05 or ys.max() >= h * 0.90:
        return None

    return labels == largest_label


def gray_fill_service_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = int(np.clip(otsu - 4, 20, 70))
    upper = int(np.clip(otsu + 175, lower + 8, 220))
    return (gray >= lower) & (gray <= upper)


def repair_mask(raw_mask: np.ndarray, style: str) -> np.ndarray:
    h, w = raw_mask.shape
    min_dim = min(h, w)
    mask = raw_mask.astype(bool)

    min_object = max(64, int(mask.size * (0.00001 if style == "bright-blue" else 0.00002)))
    mask = morphology.remove_small_objects(mask, max_size=max(0, min_object - 1))

    close_px = max(7, int(round(min_dim * (0.010 if style == "bright-blue" else 0.006))))
    open_px = max(3, int(round(min_dim * (0.0025 if style == "bright-blue" else 0.0015))))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px | 1, close_px | 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px | 1, open_px | 1))

    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
    mask = mask_u8 > 0

    mask = keep_main_components(mask, max_components=4 if style == "bright-blue" else 8)
    mask = ndimage.binary_fill_holes(mask)

    fill_kernel_size = max(9, int(round(min_dim * (0.018 if style == "bright-blue" else 0.012))))
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_kernel_size | 1, fill_kernel_size | 1))
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, fill_kernel)
    mask = mask_u8 > 0
    mask = keep_main_components(mask, max_components=4 if style == "bright-blue" else 8)
    return mask


def keep_main_components(mask: np.ndarray, max_components: int) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    areas = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    sorted_labels = np.argsort(areas)[::-1] + 1
    keep: list[int] = []
    max_area = float(areas.max())
    for label in sorted_labels[:max_components]:
        area = float(areas[label - 1])
        if area >= max(150.0, max_area * 0.015):
            keep.append(int(label))
    return np.isin(labels, keep)


def remove_dark_teal_chrome(mask: np.ndarray, style: str) -> np.ndarray:
    if style != "dark-teal":
        return mask
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask

    areas = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    max_area = float(areas.max())
    h, w = mask.shape
    cleaned = mask.copy()

    for label in range(1, count + 1):
        area = float(areas[label - 1])
        ys, xs = np.where(labels == label)
        if len(ys) == 0:
            continue
        component_h = int(ys.max() - ys.min() + 1)
        component_w = int(xs.max() - xs.min() + 1)
        touches_top = int(ys.min()) == 0
        shallow_top_bar = touches_top and component_h <= max(28, int(h * 0.12)) and area <= max_area * 0.35
        tiny_island = area < max(max_area * 0.03, mask.size * 0.005) and (
            component_h <= max(18, int(h * 0.06)) or component_w <= max(18, int(w * 0.10))
        )
        if shallow_top_bar or tiny_island:
            cleaned[labels == label] = False
    return cleaned


def mask_to_geometry(mask: np.ndarray, simplify_px: float) -> tuple[Polygon | MultiPolygon, int]:
    h, w = mask.shape
    tolerance = max(0.0, float(simplify_px))
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[Polygon] = []
    min_area = max(200.0, h * w * 0.00005)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        coords = [(float(point[0][0]), float(point[0][1])) for point in contour]
        if len(coords) < 4:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < min_area:
            continue
        poly = simplify_geometry(poly, tolerance)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)

    if not polygons:
        raise ValueError("No service-area polygon could be extracted from the image.")

    merged = unary_union(polygons)
    merged = simplify_geometry(merged, tolerance)
    if isinstance(merged, Polygon):
        return orient_pixel_polygon(merged), len(polygons)
    if isinstance(merged, MultiPolygon):
        return MultiPolygon([orient_pixel_polygon(poly) for poly in merged.geoms]), len(polygons)
    raise ValueError("Extracted service area did not form a polygon.")


def simplify_geometry(geometry: Polygon | MultiPolygon, tolerance: float) -> Polygon | MultiPolygon:
    if tolerance <= 0:
        return geometry
    simplified = geometry.simplify(tolerance, preserve_topology=True)
    if not simplified.is_valid:
        simplified = simplified.buffer(0)
    return simplified


def orient_pixel_polygon(poly: Polygon) -> Polygon:
    if poly.exterior.is_ccw:
        return poly
    return Polygon(list(poly.exterior.coords)[::-1], [list(ring.coords) for ring in poly.interiors])


def extraction_confidence(mask: np.ndarray, style: str, contour_count: int) -> float:
    coverage = float(mask.mean())
    if coverage <= 0.0:
        return 0.0
    expected_min, expected_max = (0.03, 0.85) if style == "bright-blue" else (0.015, 0.95)
    coverage_score = 1.0 if expected_min <= coverage <= expected_max else 0.55
    component_score = 1.0 if contour_count <= 3 else max(0.55, 1.0 - (contour_count - 3) * 0.08)
    return round(0.75 * coverage_score + 0.25 * component_score, 3)


def write_mask_png(mask: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def write_overlay_png(rgb_path: str | Path, mask: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(rgb_path).astype(np.float32)
    overlay_color = np.array([255, 60, 0], dtype=np.float32)
    alpha = 0.38
    out = rgb.copy()
    out[mask] = out[mask] * (1.0 - alpha) + overlay_color * alpha
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB").save(path)
