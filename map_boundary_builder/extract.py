from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
from shapely.affinity import scale as scale_geometry
from shapely.affinity import translate as translate_geometry
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

from .pipeline_version import runtime_dependency_signature

DEFAULT_SIMPLIFY_PX = 6.0
EXTRACT_MAX_DIMENSION = max(0, int(os.environ.get("MAP_BOUNDARY_EXTRACT_MAX_DIMENSION", "0")))
_CACHE_ROOT = Path(os.environ.get("MAP_BOUNDARY_CACHE_DIR", ".cache/map-boundary-builder"))
EXTRACTION_CACHE_DIR = _CACHE_ROOT / "extractions"
EXTRACTION_CACHE_VERSION = "extraction-v1"
EXTRACTION_BORDER_COLOR_TOLERANCE = 6
EXTRACTION_BORDER_ROW_MATCH_RATIO = 0.995
EXTRACTION_MEMORY_CACHE_MAX = 24
EXTRACTION_DISK_CACHE_ENABLED = os.environ.get("MAP_BOUNDARY_EXTRACTION_DISK_CACHE", "").lower() in {
    "1",
    "true",
    "yes",
}
EXTRACTION_CACHE_DEPENDENCY_PACKAGES = (
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "pillow",
    "shapely",
)
_EXTRACTION_MEMORY_CACHE: OrderedDict[str, ExtractionResult] = OrderedDict()
_EXTRACTION_CACHE_DEPENDENCY_SIGNATURE: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    mask: np.ndarray
    style: str
    pixel_geometry: Polygon | MultiPolygon
    coverage_ratio: float
    contour_count: int
    confidence: float


def load_rgb(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    if image_path.suffix.lower() == ".webp":
        cv2_rgb = load_webp_rgb_with_cv2(image_path)
        if cv2_rgb is not None:
            return cv2_rgb
    with Image.open(path) as image:
        return pil_image_to_rgb_array(image)


def load_webp_rgb_with_cv2(path: str | Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    if image.ndim != 3:
        return None
    if image.shape[2] == 3:
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if image.shape[2] != 4:
        return None
    alpha = image[:, :, 3]
    if not np.all(alpha == 255):
        return None
    return np.ascontiguousarray(cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB))


def load_rgb_at_max_dimension(path: str | Path, max_dimension: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb_image = pil_image_to_rgb(image)
        max_dimension = max(0, int(max_dimension))
        if max_dimension > 0:
            largest = max(rgb_image.size)
            if largest > max_dimension:
                scale = max_dimension / float(largest)
                rgb_image = rgb_image.resize(
                    (
                        max(1, round(rgb_image.width * scale)),
                        max(1, round(rgb_image.height * scale)),
                    ),
                    Image.Resampling.BILINEAR,
                )
        return np.array(rgb_image, dtype=np.uint8, copy=True)


def pil_image_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(pil_image_to_rgb(image), dtype=np.uint8, copy=True)


def pil_image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if "A" not in image.getbands():
        return image.convert("RGB")

    rgba_image = image.convert("RGBA")
    if rgba_image.getchannel("A").getextrema()[0] == 255:
        return rgba_image.convert("RGB")
    background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    background.alpha_composite(rgba_image)
    return background.convert("RGB")


def extract_service_area(
    image_path: str | Path,
    simplify_px: float = DEFAULT_SIMPLIFY_PX,
    *,
    rgb: np.ndarray | None = None,
    max_dimension: int | None = None,
    cache: bool = True,
) -> ExtractionResult:
    if rgb is None:
        rgb = load_rgb(image_path)
    max_dimension = EXTRACT_MAX_DIMENSION if max_dimension is None else max(0, int(max_dimension))
    rgb = np.ascontiguousarray(rgb)
    canonical_key: str | None = None
    canonical_origin = (0.0, 0.0)
    if cache:
        canonical_rgb, canonical_origin = canonical_extract_rgb(rgb)
        canonical_key = extraction_visual_cache_key(
            canonical_rgb,
            simplify_px=simplify_px,
            max_dimension=max_dimension,
        )
        if canonical_key is not None:
            cached = read_extraction_cache(canonical_key, rgb.shape[:2], canonical_origin)
            if cached is not None:
                return cached
    scale = extraction_scale_factor(rgb, max_dimension)
    if scale < 1.0:
        height, width = rgb.shape[:2]
        scaled_rgb = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        scaled = extract_service_area_from_rgb(scaled_rgb, simplify_px=simplify_px * scale)
        result = rescale_extraction_result(scaled, width=width, height=height, scale=scale)
    else:
        result = extract_service_area_from_rgb(rgb, simplify_px=simplify_px)
    if canonical_key is not None:
        write_extraction_cache(canonical_key, result, canonical_rgb.shape[:2], canonical_origin)
    return result


def extract_service_area_from_rgb(rgb: np.ndarray, simplify_px: float = DEFAULT_SIMPLIFY_PX) -> ExtractionResult:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    style = classify_style(rgb, hsv=hsv)
    if style == "bright-blue":
        raw_mask = blue_service_mask(rgb, hsv=hsv)
    elif style == "purple-fill":
        raw_mask = purple_service_mask(rgb, hsv=hsv)
    elif style == "light-fill":
        raw_mask = light_fill_service_mask(rgb, hsv=hsv)
    elif style == "gray-fill":
        raw_mask = gray_fill_service_mask(rgb)
    else:
        raw_mask = dark_teal_service_mask(rgb, hsv=hsv)
    mask = repair_mask(raw_mask, style)
    if style in {"gray-fill", "light-fill"}:
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


def extraction_scale_factor(rgb: np.ndarray, max_dimension: int) -> float:
    if max_dimension <= 0:
        return 1.0
    height, width = rgb.shape[:2]
    largest = max(width, height)
    if largest <= max_dimension:
        return 1.0
    return max_dimension / float(largest)


def rescale_extraction_result(
    result: ExtractionResult,
    *,
    width: int,
    height: int,
    scale: float,
) -> ExtractionResult:
    mask = cv2.resize(
        result.mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    geometry = scale_geometry(result.pixel_geometry, xfact=1.0 / scale, yfact=1.0 / scale, origin=(0, 0))
    coverage_ratio = float(mask.mean())
    confidence = extraction_confidence(mask, result.style, result.contour_count)
    return ExtractionResult(
        mask=mask,
        style=result.style,
        pixel_geometry=geometry,
        coverage_ratio=coverage_ratio,
        contour_count=result.contour_count,
        confidence=confidence,
    )


def extraction_visual_cache_key(
    rgb: np.ndarray | None,
    *,
    simplify_px: float,
    max_dimension: int,
) -> str | None:
    if rgb is None:
        return None
    contiguous = np.ascontiguousarray(rgb)
    digest = hashlib.sha256()
    digest.update(b"rgb-canonical-extraction")
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.data)
    payload = (
        f"{EXTRACTION_CACHE_VERSION}:"
        f"simplify={round(float(simplify_px), 4)}:"
        f"max-dimension={int(max_dimension)}:"
        f"deps={extraction_cache_dependency_signature()}:"
        f"{digest.hexdigest()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_extraction_cache(
    cache_key: str,
    output_shape: tuple[int, int],
    origin: tuple[float, float],
) -> ExtractionResult | None:
    cached = _EXTRACTION_MEMORY_CACHE.get(cache_key)
    if cached is None:
        if not EXTRACTION_DISK_CACHE_ENABLED:
            return None
        cache_path = EXTRACTION_CACHE_DIR / f"{cache_key}.npz"
        if not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                mask = data["mask"].astype(bool)
                style = str(data["style"].item())
                geometry_payload = str(data["geometry"].item())
                geometry = shape(json.loads(geometry_payload))
                contour_count = int(data["contour_count"].item())
        except Exception:
            return None
        cached = ExtractionResult(
            mask=mask,
            style=style,
            pixel_geometry=geometry,
            coverage_ratio=float(mask.mean()),
            contour_count=contour_count,
            confidence=extraction_confidence(mask, style, contour_count),
        )
        remember_extraction_memory_cache(cache_key, cached)
    else:
        _EXTRACTION_MEMORY_CACHE.move_to_end(cache_key)

    return shift_cached_extraction(cached, output_shape=output_shape, origin=origin)


def write_extraction_cache(
    cache_key: str,
    result: ExtractionResult,
    canonical_shape: tuple[int, int],
    origin: tuple[float, float],
) -> None:
    left, top = rounded_origin(origin)
    height, width = canonical_shape
    if height <= 0 or width <= 0:
        return
    if top < 0 or left < 0 or top + height > result.mask.shape[0] or left + width > result.mask.shape[1]:
        return
    canonical_mask = np.ascontiguousarray(result.mask[top : top + height, left : left + width])
    canonical_geometry = translate_geometry(result.pixel_geometry, xoff=-left, yoff=-top)
    cached = ExtractionResult(
        mask=canonical_mask,
        style=result.style,
        pixel_geometry=canonical_geometry,
        coverage_ratio=float(canonical_mask.mean()),
        contour_count=result.contour_count,
        confidence=extraction_confidence(canonical_mask, result.style, result.contour_count),
    )
    remember_extraction_memory_cache(cache_key, cached)
    if not EXTRACTION_DISK_CACHE_ENABLED:
        return
    try:
        EXTRACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=EXTRACTION_CACHE_DIR,
            prefix=f"{cache_key}.",
            suffix=".tmp.npz",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            np.savez(
                tmp,
                mask=canonical_mask.astype(np.uint8),
                style=np.array(result.style),
                geometry=np.array(json.dumps(mapping(canonical_geometry), separators=(",", ":"))),
                contour_count=np.array(result.contour_count, dtype=np.int32),
            )
        tmp_path.replace(EXTRACTION_CACHE_DIR / f"{cache_key}.npz")
    except OSError:
        return


def remember_extraction_memory_cache(cache_key: str, result: ExtractionResult) -> None:
    _EXTRACTION_MEMORY_CACHE[cache_key] = result
    _EXTRACTION_MEMORY_CACHE.move_to_end(cache_key)
    while len(_EXTRACTION_MEMORY_CACHE) > EXTRACTION_MEMORY_CACHE_MAX:
        _EXTRACTION_MEMORY_CACHE.popitem(last=False)


def shift_cached_extraction(
    result: ExtractionResult,
    *,
    output_shape: tuple[int, int],
    origin: tuple[float, float],
) -> ExtractionResult | None:
    left, top = rounded_origin(origin)
    height, width = result.mask.shape
    output_height, output_width = output_shape
    if top < 0 or left < 0 or top + height > output_height or left + width > output_width:
        return None
    mask = np.zeros((output_height, output_width), dtype=bool)
    mask[top : top + height, left : left + width] = result.mask
    geometry = translate_geometry(result.pixel_geometry, xoff=left, yoff=top)
    coverage_ratio = float(mask.mean())
    confidence = extraction_confidence(mask, result.style, result.contour_count)
    return ExtractionResult(
        mask=mask,
        style=result.style,
        pixel_geometry=geometry,
        coverage_ratio=coverage_ratio,
        contour_count=result.contour_count,
        confidence=confidence,
    )


def canonical_extract_rgb(rgb: np.ndarray | None) -> tuple[np.ndarray | None, tuple[float, float]]:
    if rgb is None or rgb.ndim != 3 or rgb.shape[0] < 3 or rgb.shape[1] < 3:
        return rgb, (0.0, 0.0)
    contiguous = np.ascontiguousarray(rgb)
    border_color = canonical_extract_border_color(contiguous)

    height, width = contiguous.shape[:2]
    top = leading_matching_border_rows(contiguous, border_color, reverse=False)
    bottom_trim = leading_matching_border_rows(contiguous, border_color, reverse=True)
    left = leading_matching_border_cols(contiguous, border_color, reverse=False)
    right_trim = leading_matching_border_cols(contiguous, border_color, reverse=True)
    bottom = height - bottom_trim
    right = width - right_trim
    if top >= bottom or left >= right:
        return contiguous, (0.0, 0.0)
    if top == 0 and left == 0 and bottom == height and right == width:
        return contiguous, (0.0, 0.0)
    return np.ascontiguousarray(contiguous[top:bottom, left:right]), (float(left), float(top))


def canonical_extract_border_color(rgb: np.ndarray) -> np.ndarray:
    border_samples = np.concatenate(
        (
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ),
        axis=0,
    )
    return np.median(border_samples.astype(np.int16), axis=0)


def leading_matching_border_rows(rgb: np.ndarray, border_color: np.ndarray, *, reverse: bool) -> int:
    height = rgb.shape[0]
    count = 0
    indexes = range(height - 1, -1, -1) if reverse else range(height)
    for index in indexes:
        if not border_pixels_match(rgb[index, :, :], border_color):
            break
        count += 1
    return count


def leading_matching_border_cols(rgb: np.ndarray, border_color: np.ndarray, *, reverse: bool) -> int:
    width = rgb.shape[1]
    count = 0
    indexes = range(width - 1, -1, -1) if reverse else range(width)
    for index in indexes:
        if not border_pixels_match(rgb[:, index, :], border_color):
            break
        count += 1
    return count


def border_pixels_match(pixels: np.ndarray, border_color: np.ndarray) -> bool:
    delta = np.max(np.abs(pixels.astype(np.int16) - border_color), axis=1)
    return bool(np.mean(delta <= EXTRACTION_BORDER_COLOR_TOLERANCE) >= EXTRACTION_BORDER_ROW_MATCH_RATIO)


def rounded_origin(origin: tuple[float, float]) -> tuple[int, int]:
    return int(round(origin[0])), int(round(origin[1]))


def extraction_cache_dependency_signature() -> str:
    global _EXTRACTION_CACHE_DEPENDENCY_SIGNATURE
    if _EXTRACTION_CACHE_DEPENDENCY_SIGNATURE is not None:
        return _EXTRACTION_CACHE_DEPENDENCY_SIGNATURE
    _EXTRACTION_CACHE_DEPENDENCY_SIGNATURE = runtime_dependency_signature(EXTRACTION_CACHE_DEPENDENCY_PACKAGES)
    return _EXTRACTION_CACHE_DEPENDENCY_SIGNATURE


def classify_style(rgb: np.ndarray, *, hsv: np.ndarray | None = None) -> str:
    if hsv is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    bright_blue = ((hue >= 92) & (hue <= 116) & (sat >= 90) & (val >= 130)).mean()
    teal_pixels = ((hue >= 78) & (hue <= 104) & (sat >= 45) & (val >= 50) & (val <= 190)).mean()
    if bright_blue > 0.02 and bright_blue > teal_pixels * 1.5:
        return "bright-blue"

    dark_pixels = (val < 95).mean()
    low_saturation = (sat < 25).mean()
    r, g, _b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    green_pixels = (
        ((hue >= 55) & (hue <= 90) & (sat >= 45) & (val >= 80) & (g.astype(np.int16) > r.astype(np.int16) + 25))
    ).mean()
    if dark_pixels > 0.35 and teal_pixels > 0.08 and green_pixels > 0.015:
        return "dark-teal"
    saturated_bright_pixels = ((sat >= 45) & (val >= 80)).mean()
    if dark_pixels > 0.80 and teal_pixels < 0.01 and bright_blue < 0.01 and saturated_bright_pixels < 0.01:
        return "gray-fill"
    purple_fill = purple_service_mask(rgb, hsv=hsv).mean()
    if purple_fill > 0.02:
        return "purple-fill"
    light_fill = light_fill_service_mask(rgb, hsv=hsv)
    light_fill_ratio = float(light_fill.mean())
    if 0.025 <= light_fill_ratio <= 0.55:
        return "light-fill"
    if green_pixels > 0.015:
        return "dark-teal"
    if dark_pixels > 0.80 and teal_pixels < 0.01 and bright_blue < 0.01:
        return "gray-fill"
    if low_saturation > 0.85 and dark_pixels > 0.35:
        return "gray-fill"
    if dark_pixels > 0.35 or teal_pixels > 0.08:
        return "dark-teal"
    return "bright-blue"


def blue_service_mask(rgb: np.ndarray, *, hsv: np.ndarray | None = None) -> np.ndarray:
    if hsv is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    saturated_blue = (hue >= 92) & (hue <= 116) & (sat >= 75) & (val >= 105)
    app_blue = (b >= 145) & (g >= 80) & (r <= 95) & ((b.astype(np.int16) - r.astype(np.int16)) >= 80)
    return saturated_blue | app_blue


def purple_service_mask(rgb: np.ndarray, *, hsv: np.ndarray | None = None) -> np.ndarray:
    if hsv is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    r, _g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (
        (hue >= 112)
        & (hue <= 145)
        & (sat >= 55)
        & (val >= 120)
        & ((b.astype(np.int16) - r.astype(np.int16)) >= 45)
    )


def light_fill_service_mask(rgb: np.ndarray, *, hsv: np.ndarray | None = None) -> np.ndarray:
    if hsv is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    light = (val >= 230) & (sat <= 75)
    return remove_edge_connected_components(light)


def remove_edge_connected_components(mask: np.ndarray) -> np.ndarray:
    labels, count, _stats = connected_components(mask)
    if count == 0:
        return mask
    h, w = mask.shape
    edge_labels = set(labels[0, :])
    edge_labels.update(labels[h - 1, :])
    edge_labels.update(labels[:, 0])
    edge_labels.update(labels[:, w - 1])
    edge_labels.discard(0)
    if not edge_labels:
        return mask
    return mask & ~select_component_labels(labels, list(edge_labels))


def dark_teal_service_mask(rgb: np.ndarray, *, hsv: np.ndarray | None = None) -> np.ndarray:
    if hsv is None:
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
    labels, count, stats = connected_components(green)
    if count == 0:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
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
    mask = remove_small_components(mask, min_area=min_object)
    if style == "bright-blue":
        return repair_bright_blue_mask(mask)

    close_px = max(7, int(round(min_dim * 0.006)))
    open_px = max(3, int(round(min_dim * 0.0015)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px | 1, close_px | 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px | 1, open_px | 1))

    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
    mask = mask_u8 > 0

    mask = keep_main_components(mask, max_components=4 if style == "bright-blue" else 8)
    mask = fill_binary_holes(mask)

    fill_kernel_size = max(9, int(round(min_dim * 0.012)))
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_kernel_size | 1, fill_kernel_size | 1))
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, fill_kernel)
    mask = mask_u8 > 0
    mask = keep_main_components(mask, max_components=8)
    return mask


def repair_bright_blue_mask(seed_mask: np.ndarray) -> np.ndarray:
    h, w = seed_mask.shape
    min_dim = min(h, w)
    mask = fill_binary_holes(seed_mask)

    crack_px = max(7, min(11, int(round(min_dim * 0.003))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (crack_px | 1, crack_px | 1))
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    mask = mask_u8 > 0
    mask = remove_exterior_repair_bleeds(mask, seed_mask)
    return keep_main_components(mask, max_components=4)


def keep_main_components(mask: np.ndarray, max_components: int) -> np.ndarray:
    labels, count, stats = connected_components(mask)
    if count == 0:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    sorted_labels = np.argsort(areas)[::-1] + 1
    keep: list[int] = []
    max_area = float(areas.max())
    for label in sorted_labels[:max_components]:
        area = float(areas[label - 1])
        if area >= max(150.0, max_area * 0.015):
            keep.append(int(label))
    if len(keep) == count:
        return mask.astype(bool, copy=False)
    return select_component_labels(labels, keep)


def remove_dark_teal_chrome(mask: np.ndarray, style: str) -> np.ndarray:
    if style != "dark-teal":
        return mask
    labels, count, stats = connected_components(mask)
    if count == 0:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    max_area = float(areas.max())
    h, w = mask.shape
    cleaned = mask.copy()

    for label in range(1, count + 1):
        area = float(areas[label - 1])
        _left, top, component_w, component_h, _component_area = stats[label]
        touches_top = int(top) == 0
        shallow_top_bar = touches_top and component_h <= max(28, int(h * 0.12)) and area <= max_area * 0.35
        tiny_island = area < max(max_area * 0.03, mask.size * 0.005) and (
            component_h <= max(18, int(h * 0.06)) or component_w <= max(18, int(w * 0.10))
        )
        if shallow_top_bar or tiny_island:
            cleaned[labels == label] = False
    return cleaned


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return labels, count - 1, stats


def remove_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    labels, count, stats = connected_components(mask)
    if count == 0:
        return mask
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(1, int(min_area)))
    keep = keep[keep != 0]
    if len(keep) == 0:
        return np.zeros_like(mask, dtype=bool)
    if len(keep) == count:
        return mask.astype(bool, copy=False)
    return select_component_labels(labels, keep)


def select_component_labels(labels: np.ndarray, keep: list[int] | np.ndarray) -> np.ndarray:
    if len(keep) == 0:
        return np.zeros(labels.shape, dtype=bool)
    keep_array = np.asarray(keep, dtype=np.intp)
    label_count = int(labels.max(initial=0)) + 1
    selected = np.zeros(label_count, dtype=bool)
    selected[keep_array[(keep_array >= 0) & (keep_array < label_count)]] = True
    return selected[labels]


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    background = (~padded).astype(np.uint8) * 255
    cv2.floodFill(background, None, (0, 0), 0)
    holes = background[1:-1, 1:-1] > 0
    return mask | holes


def edge_connected_background(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    background = (~padded).astype(np.uint8) * 255
    cv2.floodFill(background, None, (0, 0), 128)
    return background[1:-1, 1:-1] == 128


def remove_exterior_repair_bleeds(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    min_dim = min(h, w)
    outside = edge_connected_background(seed_mask)
    added_outside = mask & outside
    labels, count, stats = connected_components(added_outside)
    if count == 0:
        return mask

    min_area = max(500, int(round(mask.size * 0.00018)))
    min_long_span = max(48, int(round(min_dim * 0.028)))
    min_short_span = max(18, int(round(min_dim * 0.010)))
    remove = np.zeros_like(mask, dtype=bool)
    for label in range(1, count + 1):
        _left, _top, component_w, component_h, area = stats[label]
        long_span = max(int(component_w), int(component_h))
        short_span = min(int(component_w), int(component_h))
        if int(area) >= min_area and long_span >= min_long_span and short_span >= min_short_span:
            remove[labels == label] = True
    return (mask & ~remove) | seed_mask


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


def write_overlay_png(
    rgb_path: str | Path,
    mask: np.ndarray,
    path: str | Path,
    *,
    rgb: np.ndarray | None = None,
    max_dimension: int | None = None,
) -> None:
    write_overlay_image(rgb_path, mask, path, rgb=rgb, max_dimension=max_dimension)


def write_overlay_image(
    rgb_path: str | Path,
    mask: np.ndarray,
    path: str | Path,
    *,
    rgb: np.ndarray | None = None,
    max_dimension: int | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = load_rgb(rgb_path)
    target_width = mask.shape[1]
    target_height = mask.shape[0]
    if max_dimension is not None and max_dimension > 0:
        h, w = mask.shape[:2]
        largest = max(h, w)
        if largest > max_dimension:
            preview_scale = max_dimension / float(largest)
            target_width = max(1, round(w * preview_scale))
            target_height = max(1, round(h * preview_scale))
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
    if rgb.shape[:2] != (target_height, target_width):
        rgb = cv2.resize(
            rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA if max(rgb.shape[:2]) > max(target_height, target_width) else cv2.INTER_CUBIC,
        )
    rgb = rgb.astype(np.float32)
    overlay_color = np.array([255, 60, 0], dtype=np.float32)
    outline_color = (23, 33, 29)
    alpha = 0.38
    out = rgb.copy()
    out[mask] = out[mask] * (1.0 - alpha) + overlay_color * alpha
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        outline_width = max(2, min(6, round(min(mask.shape) / 360)))
        cv2.drawContours(out, contours, -1, outline_color, thickness=outline_width, lineType=cv2.LINE_AA)
    image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
    if Path(path).suffix.lower() == ".webp":
        image.save(path, format="WEBP", quality=82, method=2)
    else:
        image.save(path)
