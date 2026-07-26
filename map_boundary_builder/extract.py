"""Shared extraction primitives for the unified pipeline.

Everything style-specific was removed in the revamp: this module now holds
image loading, the LAB auto-fill fallback used by ``segment``, mask repair,
polygonization, and artifact writers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

DEFAULT_SIMPLIFY_PX = 6.0


DEFAULT_EXTRACTION_PROFILE = "map-overlay"


AUTO_FILL_STYLE = "auto-fill"


AUTO_FILL_ANALYSIS_MAX_DIMENSION = 512


AUTO_FILL_CLUSTER_COUNT = 12


AUTO_FILL_KMEANS_SEED = 7


AUTO_FILL_KMEANS_FIT_MAX_SAMPLES = 60000


AUTO_FILL_CENTER_PREMERGE_DISTANCE = 6.0


AUTO_FILL_TINT_MERGE_DISTANCE = 22.0


AUTO_FILL_MIN_BORDER_DISTINCTNESS = 12.0


# A recolored or dark basemap land cluster sits near the basemap border color
# (the border samples are basemap), clearing the lenient floor only barely
# (chroma-weighted distance 12-18); a real saturated overlay sits on top of the
# basemap and is far from it (>=32). The group path uses this stricter floor in
# the empty gap between them; the ring path and light-fill gate keep the 12.0.
AUTO_FILL_GROUP_MIN_BORDER_DISTINCTNESS = 24.0


AUTO_FILL_LIGHTNESS_DISTINCTNESS_WEIGHT = 0.25


AUTO_FILL_MAX_BORDER_FRACTION = 0.55


AUTO_FILL_MIN_CLUSTER_COVERAGE = 0.008


AUTO_FILL_MAX_CLUSTER_COVERAGE = 0.65


AUTO_FILL_MIN_COMPONENT_SPAN_RATIO = 0.12


AUTO_FILL_MIN_COMPONENT_DENSITY = 0.45


AUTO_FILL_MIN_COMPONENT_INTERIOR_RATIO = 0.35


AUTO_FILL_TEXTURE_LAPLACIAN_THRESHOLD = 6


AUTO_FILL_MIN_INTERIOR_TEXTURE = 0.07


AUTO_FILL_MIN_SEED_CHROMA = 20.0


AUTO_FILL_CHROMA_SCORE_SCALE = 24.0


AUTO_FILL_RING_CLOSE_RATIO = 0.03


AUTO_FILL_RING_MIN_CHROMA = 20.0


AUTO_FILL_RING_MEMBER_MIN_CHROMA = 15.0


AUTO_FILL_RING_HUE_MERGE_DEGREES = 25.0


AUTO_FILL_RING_MAX_STROKE_COVERAGE = 0.20


AUTO_FILL_RING_MIN_ENCLOSED_RATIO = 1.2


AUTO_FILL_RING_DOMINANT_HOLE_RATIO = 0.80


AUTO_FILL_RING_MAX_BORDER_FRACTION = 0.30


AUTO_FILL_RESULT_MIN_COVERAGE = 0.01


AUTO_FILL_RESULT_MAX_COVERAGE = 0.85


AUTO_FILL_CONFIDENCE_DISCOUNT = 0.9


AUTO_FILL_REPAIR_MAX_DIMENSION = 1024


BRIGHT_BLUE_WATER_MIN_INTERIOR_TEXTURE = 0.20


@dataclass(frozen=True)
class ExtractionProfile:
    name: str
    auto_fill_group_min_border_distinctness: float = AUTO_FILL_GROUP_MIN_BORDER_DISTINCTNESS
    auto_fill_tint_merge_distance: float = AUTO_FILL_TINT_MERGE_DISTANCE
    auto_fill_max_border_fraction: float = AUTO_FILL_MAX_BORDER_FRACTION
    auto_fill_min_component_density: float = AUTO_FILL_MIN_COMPONENT_DENSITY
    auto_fill_min_component_interior_ratio: float = AUTO_FILL_MIN_COMPONENT_INTERIOR_RATIO
    auto_fill_min_interior_texture: float = AUTO_FILL_MIN_INTERIOR_TEXTURE
    auto_fill_min_seed_chroma: float = AUTO_FILL_MIN_SEED_CHROMA
    bright_blue_water_min_interior_texture: float = BRIGHT_BLUE_WATER_MIN_INTERIOR_TEXTURE


@dataclass(frozen=True)
class ExtractionHints:
    seed_point: tuple[float, float] | None = None
    target_rgb: tuple[int, int, int] | None = None


EXTRACTION_PROFILES: dict[str, ExtractionProfile] = {
    DEFAULT_EXTRACTION_PROFILE: ExtractionProfile(name=DEFAULT_EXTRACTION_PROFILE),
    # Satellite/aerial screenshots tend to have textured, high-variance base
    # imagery and subtler semi-transparent annotation fills. Keep this opt-in so
    # the production map-overlay path does not inherit looser false-positive
    # gates intended for annotated imagery.
    "satellite-overlay": ExtractionProfile(
        name="satellite-overlay",
        auto_fill_group_min_border_distinctness=16.0,
        auto_fill_tint_merge_distance=12.0,
        auto_fill_max_border_fraction=0.70,
        auto_fill_min_component_density=0.35,
        auto_fill_min_component_interior_ratio=0.25,
        auto_fill_min_interior_texture=0.035,
        auto_fill_min_seed_chroma=10.0,
        bright_blue_water_min_interior_texture=0.08,
    ),
}


EXTRACTION_PROFILE_ALIASES = {
    "default": DEFAULT_EXTRACTION_PROFILE,
    "map": DEFAULT_EXTRACTION_PROFILE,
    "map-overlay": DEFAULT_EXTRACTION_PROFILE,
    "satellite": "satellite-overlay",
    "aerial": "satellite-overlay",
    "imagery": "satellite-overlay",
}


@dataclass(frozen=True)
class ExtractionResult:
    mask: np.ndarray
    style: str
    pixel_geometry: Polygon | MultiPolygon
    coverage_ratio: float
    contour_count: int
    confidence: float
    scaled_cache_status: str | None = None
    scaled_cache_shape: tuple[int, int] | None = None
    extraction_profile: str = DEFAULT_EXTRACTION_PROFILE
    diagnostics: dict[str, object] | None = None


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
    if "transparency" in image.info and "A" not in image.getbands():
        image = image.convert("RGBA")
    if "A" not in image.getbands():
        return image.convert("RGB")

    rgba_image = image.convert("RGBA")
    if rgba_image.getchannel("A").getextrema()[0] == 255:
        return rgba_image.convert("RGB")
    background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    background.alpha_composite(rgba_image)
    return background.convert("RGB")


def resolve_extraction_profile(profile: str | ExtractionProfile | None = None) -> ExtractionProfile:
    if isinstance(profile, ExtractionProfile):
        return profile
    profile_name = DEFAULT_EXTRACTION_PROFILE if profile is None else str(profile).strip().lower()
    profile_name = EXTRACTION_PROFILE_ALIASES.get(profile_name, profile_name)
    try:
        return EXTRACTION_PROFILES[profile_name]
    except KeyError as exc:
        supported = ", ".join(sorted(EXTRACTION_PROFILES))
        raise ValueError(f"Unsupported extraction profile {profile!r}; supported profiles: {supported}") from exc


def resolve_extraction_hints(hints: ExtractionHints | dict[str, object] | None = None) -> ExtractionHints:
    if hints is None:
        return ExtractionHints()
    if isinstance(hints, ExtractionHints):
        return hints
    seed = hints.get("seed_point")
    target = hints.get("target_rgb")
    return ExtractionHints(
        seed_point=normalize_point_hint(seed),
        target_rgb=normalize_rgb_hint(target),
    )


def normalize_point_hint(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("seed_point hint must be a two-item (x, y) pair")
    return float(value[0]), float(value[1])


def normalize_rgb_hint(value: object) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("target_rgb hint must be a three-item (r, g, b) tuple")
    rgb = tuple(int(round(float(channel))) for channel in value)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError("target_rgb hint channels must be between 0 and 255")
    return rgb


def chroma_weighted_color_distance(color: np.ndarray, reference: np.ndarray) -> float:
    """LAB distance with lightness damped, so a map fill must differ in color
    rather than in brightness alone — gray basemap blocks and white UI chrome
    differ from each other only in lightness."""
    delta = color.astype(np.float32) - reference.astype(np.float32)
    delta[0] *= AUTO_FILL_LIGHTNESS_DISTINCTNESS_WEIGHT
    return float(np.linalg.norm(delta))


def auto_fill_extraction_result(
    rgb: np.ndarray,
    *,
    simplify_px: float,
    profile: ExtractionProfile | None = None,
    hints: ExtractionHints | dict[str, object] | None = None,
) -> ExtractionResult | None:
    extraction_profile = resolve_extraction_profile(profile)
    extraction_hints = resolve_extraction_hints(hints)
    raw = auto_fill_service_mask(rgb, profile=extraction_profile, hints=extraction_hints)
    if raw is None:
        return None
    raw_mask, diagnostics = raw
    if not raw_mask.any():
        return None
    # The mask is derived from a small analysis frame, so it carries no detail
    # finer than its upscale step; repair at a capped resolution (cheap
    # morphology) and upscale the result instead of grinding full-res pixels.
    full_shape = rgb.shape[:2]
    repair_shape = capped_repair_shape(full_shape)
    repair_input = upscale_bool_mask(raw_mask, repair_shape)
    repaired = repair_mask(repair_input, AUTO_FILL_STYLE)
    repaired = keep_main_components(repaired, max_components=3)
    mask = upscale_bool_mask(repaired, full_shape)
    coverage_ratio = float(mask.mean())
    if not AUTO_FILL_RESULT_MIN_COVERAGE <= coverage_ratio <= AUTO_FILL_RESULT_MAX_COVERAGE:
        return None
    try:
        geometry, contour_count = mask_to_geometry(mask, simplify_px=simplify_px)
    except ValueError:
        return None
    return ExtractionResult(
        mask=mask,
        style=AUTO_FILL_STYLE,
        pixel_geometry=geometry,
        coverage_ratio=coverage_ratio,
        contour_count=contour_count,
        confidence=extraction_confidence(mask, AUTO_FILL_STYLE, contour_count),
        extraction_profile=extraction_profile.name,
        diagnostics={"auto_fill": diagnostics},
    )


def auto_fill_service_mask(
    rgb: np.ndarray,
    *,
    profile: ExtractionProfile | None = None,
    hints: ExtractionHints | dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Color-agnostic fill detection for palettes outside the tuned styles.

    Clusters the image in LAB space and scores each color group as a candidate
    service-area fill: it must be a mostly contiguous, reasonably large blob
    that is distinct from the border/basemap color, does not dominate the image
    border, and is solid rather than a road-like lattice once small gaps close.
    """
    extraction_profile = resolve_extraction_profile(profile)
    extraction_hints = resolve_extraction_hints(hints)
    height, width = rgb.shape[:2]
    if min(height, width) < 32:
        return None
    analysis = rgb
    largest = max(height, width)
    if largest > AUTO_FILL_ANALYSIS_MAX_DIMENSION:
        scale = AUTO_FILL_ANALYSIS_MAX_DIMENSION / float(largest)
        analysis = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    lab = cv2.cvtColor(analysis, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] < AUTO_FILL_CLUSTER_COUNT * 4:
        return None
    cv2.setRNGSeed(AUTO_FILL_KMEANS_SEED)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    # Fitting on a strided subsample finds the same palette (clusters are large,
    # contiguous color regions) at a fraction of the iteration cost; every
    # analysis pixel is then assigned to its nearest fitted center.
    fit_pixels = pixels
    if pixels.shape[0] > AUTO_FILL_KMEANS_FIT_MAX_SAMPLES:
        stride = int(np.ceil(pixels.shape[0] / AUTO_FILL_KMEANS_FIT_MAX_SAMPLES))
        fit_pixels = np.ascontiguousarray(pixels[::stride])
    _compactness, _fit_labels, centers = cv2.kmeans(
        fit_pixels,
        AUTO_FILL_CLUSTER_COUNT,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = assign_to_nearest_center(pixels, centers)
    label_image = labels.reshape(lab.shape[:2])
    group_of_label, group_centers = merge_close_cluster_centers(
        centers,
        np.bincount(labels.ravel(), minlength=AUTO_FILL_CLUSTER_COUNT),
    )
    group_image = group_of_label[label_image]
    border_fractions, border_color = auto_fill_border_stats(group_image, group_centers.shape[0], lab)
    texture = analysis_texture_mask(lab)
    analysis_hints = scale_extraction_hints(extraction_hints, rgb.shape[:2], analysis.shape[:2])
    selected_groups = select_auto_fill_groups(
        group_image,
        group_centers,
        border_fractions=border_fractions,
        border_color=border_color,
        texture=texture,
        profile=extraction_profile,
        hints=analysis_hints,
    )
    if selected_groups is not None:
        selected = sorted(selected_groups)
        return np.isin(group_image, selected), {
            "profile": extraction_profile.name,
            "method": "cluster-fill",
            "selected_groups": selected,
            "seed_point": list(extraction_hints.seed_point) if extraction_hints.seed_point is not None else None,
            "target_rgb": list(extraction_hints.target_rgb) if extraction_hints.target_rgb is not None else None,
        }
    ring = select_auto_fill_ring(
        group_image,
        group_centers,
        border_color=border_color,
    )
    if ring is None:
        return None
    return ring, {
        "profile": extraction_profile.name,
        "method": "outline-ring",
    }


def scale_extraction_hints(
    hints: ExtractionHints,
    source_shape: tuple[int, int],
    analysis_shape: tuple[int, int],
) -> ExtractionHints:
    if hints.seed_point is None or source_shape == analysis_shape:
        return hints
    source_h, source_w = source_shape
    analysis_h, analysis_w = analysis_shape
    if source_h <= 0 or source_w <= 0:
        return hints
    x, y = hints.seed_point
    return ExtractionHints(
        seed_point=(x * analysis_w / source_w, y * analysis_h / source_h),
        target_rgb=hints.target_rgb,
    )


def assign_to_nearest_center(pixels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    centers32 = np.ascontiguousarray(centers, dtype=np.float32)
    center_sq = (centers32**2).sum(axis=1)
    scores = pixels @ centers32.T
    scores *= -2.0
    scores += center_sq[np.newaxis, :]
    return np.argmin(scores, axis=1).astype(np.int32)


def capped_repair_shape(shape: tuple[int, int]) -> tuple[int, int]:
    height, width = shape
    largest = max(height, width)
    if largest <= AUTO_FILL_REPAIR_MAX_DIMENSION:
        return shape
    scale = AUTO_FILL_REPAIR_MAX_DIMENSION / float(largest)
    return max(1, round(height * scale)), max(1, round(width * scale))


def auto_fill_border_stats(
    group_image: np.ndarray,
    group_count: int,
    lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    border_groups = np.concatenate(
        (group_image[0, :], group_image[-1, :], group_image[:, 0], group_image[:, -1])
    )
    border_fractions = np.bincount(border_groups, minlength=group_count).astype(np.float64)
    border_fractions /= float(border_groups.size)
    border_lab = np.concatenate(
        (lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]),
        axis=0,
    ).astype(np.float32)
    return border_fractions, np.median(border_lab, axis=0)


def analysis_texture_mask(lab: np.ndarray) -> np.ndarray:
    """Pixels with meaningful local lightness structure. A translucent service
    overlay keeps the basemap's street/label texture visible, while water,
    parkland, and solid UI fills are flat."""
    laplacian = cv2.Laplacian(lab[:, :, 0], cv2.CV_16S, ksize=3)
    return np.abs(laplacian) > AUTO_FILL_TEXTURE_LAPLACIAN_THRESHOLD


def upscale_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == tuple(shape):
        return mask
    return (
        cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    )


def merge_close_cluster_centers(
    centers: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = centers.shape[0]
    parent = np.arange(count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(count):
        for second in range(first + 1, count):
            if float(np.linalg.norm(centers[first] - centers[second])) <= AUTO_FILL_CENTER_PREMERGE_DISTANCE:
                parent[find(second)] = find(first)

    roots = np.array([find(index) for index in range(count)])
    unique_roots, group_of_label = np.unique(roots, return_inverse=True)
    group_centers = np.zeros((len(unique_roots), centers.shape[1]), dtype=np.float64)
    weights = counts.astype(np.float64)
    for group in range(len(unique_roots)):
        members = group_of_label == group
        member_weights = weights[members]
        total = float(member_weights.sum())
        if total > 0.0:
            group_centers[group] = (centers[members] * member_weights[:, np.newaxis]).sum(axis=0) / total
        else:
            group_centers[group] = centers[members].mean(axis=0)
    return group_of_label, group_centers.astype(np.float32)


def select_auto_fill_groups(
    group_image: np.ndarray,
    group_centers: np.ndarray,
    *,
    border_fractions: np.ndarray,
    border_color: np.ndarray,
    texture: np.ndarray,
    profile: ExtractionProfile,
    hints: ExtractionHints,
) -> set[int] | None:
    group_count = group_centers.shape[0]
    target_lab: np.ndarray | None = None
    if hints.target_rgb is not None:
        target_lab = cv2.cvtColor(np.array([[hints.target_rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(
            np.float32
        )
    # A semi-transparent overlay renders as a family of nearby tints (fill over
    # background, fill over roads, fill over parks), so each candidate is the
    # union of one seed group with its color neighbors.
    best_members: list[int] | None = None
    best_score = 0.0
    scored_member_sets: set[tuple[int, ...]] = set()
    for seed in range(group_count):
        distances = np.linalg.norm(group_centers - group_centers[seed], axis=1)
        members = sorted(np.flatnonzero(distances <= profile.auto_fill_tint_merge_distance).tolist())
        member_key = tuple(members)
        if member_key in scored_member_sets:
            continue
        scored_member_sets.add(member_key)
        candidate_mask = np.isin(group_image, members)
        if not candidate_matches_hints(candidate_mask, hints):
            continue
        score = auto_fill_candidate_score(
            candidate_mask,
            group_centers[seed],
            border_color,
            float(border_fractions[members].sum()),
            texture=texture,
            profile=profile,
            target_lab=target_lab,
        )
        if score is not None and score > best_score:
            best_score = score
            best_members = members
    if best_members is None:
        return None
    return set(best_members)


def auto_fill_candidate_score(
    candidate_mask: np.ndarray,
    seed_center: np.ndarray,
    border_color: np.ndarray,
    border_fraction: float,
    *,
    texture: np.ndarray,
    profile: ExtractionProfile,
    target_lab: np.ndarray | None = None,
) -> float | None:
    if border_fraction > profile.auto_fill_max_border_fraction:
        return None
    coverage = float(candidate_mask.mean())
    if not AUTO_FILL_MIN_CLUSTER_COVERAGE <= coverage <= AUTO_FILL_MAX_CLUSTER_COVERAGE:
        return None
    distinctness = chroma_weighted_color_distance(seed_center, border_color)
    if distinctness < profile.auto_fill_group_min_border_distinctness:
        return None

    h, w = candidate_mask.shape
    close_px = max(3, round(min(h, w) * 0.012)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    closed = cv2.morphologyEx(candidate_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0
    labels, count, stats = connected_components(closed)
    if count == 0:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    largest_label = int(np.argmax(areas) + 1)
    _left, _top, component_w, component_h, area = stats[largest_label]
    if component_w < w * AUTO_FILL_MIN_COMPONENT_SPAN_RATIO:
        return None
    if component_h < h * AUTO_FILL_MIN_COMPONENT_SPAN_RATIO:
        return None
    area_ratio = float(area) / float(candidate_mask.size)
    if area_ratio < AUTO_FILL_MIN_CLUSTER_COVERAGE:
        return None
    # A road/grid lattice closes into a sparse web while a real fill stays
    # solid: it keeps most of its hole-filled envelope and survives erosion.
    component = labels == largest_label
    filled_area = float(fill_binary_holes(component).sum())
    if float(area) / max(1.0, filled_area) < profile.auto_fill_min_component_density:
        return None
    eroded = cv2.erode(component.astype(np.uint8), kernel) > 0
    eroded_area = float(eroded.sum())
    if eroded_area / float(area) < profile.auto_fill_min_component_interior_ratio:
        return None
    # A translucent service fill keeps basemap streets/labels visible inside
    # it; flat regions (water, parkland, solid cards) carry no texture.
    interior_probe = eroded if eroded.any() else component
    texture_fraction = float(texture[interior_probe].mean())
    if texture_fraction < profile.auto_fill_min_interior_texture:
        return None
    # Basemap water and land read as pale (low LAB chroma) even when roads or
    # labels give them texture; announcement overlays are saturated.
    seed_chroma = float(np.linalg.norm(seed_center[1:].astype(np.float32) - 128.0))
    if seed_chroma < profile.auto_fill_min_seed_chroma:
        return None
    target_score = 1.0
    if target_lab is not None:
        target_distance = chroma_weighted_color_distance(seed_center, target_lab)
        if target_distance > 80.0:
            return None
        target_score = max(0.25, 1.0 - (target_distance / 80.0))
    chroma_score = min(1.0, seed_chroma / AUTO_FILL_CHROMA_SCORE_SCALE)
    contiguity = min(1.0, float(area) / max(1.0, float(candidate_mask.sum())))
    return area_ratio * contiguity * (1.0 - border_fraction) * min(1.0, distinctness / 40.0) * chroma_score * target_score


def candidate_matches_hints(candidate_mask: np.ndarray, hints: ExtractionHints) -> bool:
    if hints.seed_point is None:
        return True
    x, y = hints.seed_point
    col = int(round(x))
    row = int(round(y))
    if row < 0 or col < 0 or row >= candidate_mask.shape[0] or col >= candidate_mask.shape[1]:
        return False
    return bool(candidate_mask[row, col])


def select_auto_fill_ring(
    group_image: np.ndarray,
    group_centers: np.ndarray,
    *,
    border_color: np.ndarray,
) -> np.ndarray | None:
    """Outline-only service maps draw a colored boundary stroke with no fill.
    Find a chromatic, mostly-closed thin ring whose closure encloses one
    dominant interior region, and return the filled envelope."""
    h, w = group_image.shape
    group_count = group_centers.shape[0]
    ring_close_px = max(5, round(min(h, w) * AUTO_FILL_RING_CLOSE_RATIO)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_close_px, ring_close_px))
    ab_offsets = group_centers[:, 1:].astype(np.float32) - 128.0
    chromas = np.linalg.norm(ab_offsets, axis=1)
    hue_angles = np.degrees(np.arctan2(ab_offsets[:, 1], ab_offsets[:, 0]))
    best_mask: np.ndarray | None = None
    best_score = 0.0
    scored_member_sets: set[tuple[int, ...]] = set()
    for group in range(group_count):
        center = group_centers[group]
        chroma = float(chromas[group])
        if chroma < AUTO_FILL_RING_MIN_CHROMA:
            continue
        if chroma_weighted_color_distance(center, border_color) < AUTO_FILL_MIN_BORDER_DISTINCTNESS:
            continue
        # Antialiasing splits a thin stroke into several tints along the
        # background-to-stroke color line; union groups that share the seed's
        # hue direction so dashed/thin rings stay closed.
        angle_delta = np.abs((hue_angles - hue_angles[group] + 180.0) % 360.0 - 180.0)
        members = sorted(
            np.flatnonzero(
                (chromas >= AUTO_FILL_RING_MEMBER_MIN_CHROMA)
                & (angle_delta <= AUTO_FILL_RING_HUE_MERGE_DEGREES)
            ).tolist()
        )
        member_key = tuple(members)
        if member_key in scored_member_sets:
            continue
        scored_member_sets.add(member_key)
        group_mask = np.isin(group_image, members)
        stroke_coverage = float(group_mask.mean())
        if not 0.002 <= stroke_coverage <= AUTO_FILL_RING_MAX_STROKE_COVERAGE:
            continue
        # Dilate-fill-erode rather than a plain close: closing reconnects
        # dashes only transiently (erosion re-cuts the thin necks), while
        # dilation bridges gaps long enough for hole-filling to capture the
        # interior, and the final erosion undoes the boundary inflation.
        dilated = cv2.dilate(group_mask.astype(np.uint8), kernel) > 0
        labels, count, stats = connected_components(dilated)
        if count == 0:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
        largest_label = int(np.argmax(areas) + 1)
        component = labels == largest_label
        component_area = float(areas[largest_label - 1])
        filled = fill_binary_holes(component)
        enclosed = filled & ~component
        enclosed_area = float(enclosed.sum())
        if enclosed_area < component_area * AUTO_FILL_RING_MIN_ENCLOSED_RATIO:
            continue
        enclosed_coverage = enclosed_area / float(group_mask.size)
        if not 0.01 <= enclosed_coverage <= 0.80:
            continue
        # One dominant enclosed region separates a boundary ring from a road
        # lattice, whose closure traps many small cells instead.
        enclosed_labels, enclosed_count, enclosed_stats = connected_components(enclosed)
        if enclosed_count == 0:
            continue
        enclosed_areas = enclosed_stats[1:, cv2.CC_STAT_AREA].astype(float)
        if float(enclosed_areas.max()) < enclosed_area * AUTO_FILL_RING_DOMINANT_HOLE_RATIO:
            continue
        ys, xs = np.nonzero(filled)
        if (xs.max() - xs.min() + 1) < w * AUTO_FILL_MIN_COMPONENT_SPAN_RATIO:
            continue
        if (ys.max() - ys.min() + 1) < h * AUTO_FILL_MIN_COMPONENT_SPAN_RATIO:
            continue
        border_hits = (
            int(filled[0, :].sum())
            + int(filled[-1, :].sum())
            + int(filled[:, 0].sum())
            + int(filled[:, -1].sum())
        )
        if border_hits > (2 * (h + w)) * AUTO_FILL_RING_MAX_BORDER_FRACTION:
            continue
        score = enclosed_coverage * min(1.0, chroma / 40.0)
        if score > best_score:
            best_score = score
            best_mask = cv2.erode(filled.astype(np.uint8), kernel) > 0
    return best_mask


def repair_mask(raw_mask: np.ndarray, style: str) -> np.ndarray:
    h, w = raw_mask.shape
    min_dim = min(h, w)
    mask = raw_mask.astype(bool)

    # A color threshold that matched nothing has no fill to repair; skip the
    # full-resolution morphology, which otherwise burns time on an empty frame
    # (common when a tuned style is tried on an off-palette map before the
    # color-agnostic fallback runs).
    if not mask.any():
        return mask

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
    confidence = 0.75 * coverage_score + 0.25 * component_score
    if style == AUTO_FILL_STYLE:
        confidence *= AUTO_FILL_CONFIDENCE_DISCOUNT
    return round(confidence, 3)


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
        image.save(path, format="WEBP", quality=82, method=0)
    else:
        image.save(path)

