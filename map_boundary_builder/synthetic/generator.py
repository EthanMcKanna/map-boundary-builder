"""Deterministic synthetic map-boundary sample generation.

This module is intentionally lightweight. It creates a local, reproducible
renderer that exercises the extraction/evaluation stack while leaving the
future MapLibre/Playwright renderer as a drop-in replacement for the same
artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from shapely.affinity import scale
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping
from shapely.ops import unary_union

from ..evaluation import rasterize_geometry_mask
from .manifest import (
    OverlayStyleMetadata,
    SyntheticArtifactPaths,
    SyntheticDatasetManifest,
    SyntheticSampleMetadata,
)

GENERATOR_VERSION = "synthetic-generator-v21-negative-ui-scenes"


@dataclass(frozen=True)
class SyntheticOverlayStyle:
    name: str
    fill_color: str
    fill_opacity: float
    stroke_color: str | None = None
    stroke_width_px: float = 0.0
    dashed: bool = False
    fill_enabled: bool = True
    labels_on_top: bool = False
    circular_viewport: bool = False
    pattern: str | None = None
    stroke_join: str = "miter"

    def metadata(self) -> OverlayStyleMetadata:
        return OverlayStyleMetadata(
            name=self.name,
            fill_color=self.fill_color,
            fill_opacity=self.fill_opacity if self.fill_enabled else 0.0,
            stroke_color=self.stroke_color,
            stroke_width_px=self.stroke_width_px if self.stroke_width_px else None,
        )


@dataclass(frozen=True)
class SyntheticSceneConfig:
    provider: str = "synthetic"
    service_area: str = "test-city"
    variant: str = "default"
    width: int = 960
    height: int = 640
    seed: int = 1
    base_map: str = "procedural-open-map"
    overlay_style: SyntheticOverlayStyle | None = None
    touch_border: bool = False
    include_ui_chrome: bool = False
    include_hole: bool = False
    jpeg_quality: int | None = None
    labels_on_top: bool = False
    circular_viewport: bool = False
    complex_boundary: bool = False
    large_service_area: bool = False
    include_distractor: bool = False
    shape_family: str = "radial"
    # Negative scenes render an app-style screen (route map, cards, buttons)
    # with NO service area; the truth mask is empty. They teach the model that
    # ride/trip UI screenshots contain nothing to extract.
    negative_scene: bool = False
    # Large desaturated district-style polygons beneath the target overlay,
    # excluded from the truth mask — real maps shade school districts, admin
    # areas, and water bodies that must not read as service areas.
    include_admin_shading: bool = False
    # Noise texture over the basemap approximating satellite/aerial imagery.
    textured_basemap: bool = False
    # Named provider look: "waymo-web" (light basemap, near-opaque brand blue,
    # labels on top, pale secondary zones) or "tesla-dark" (near-black
    # basemap, translucent salmon fill, corner city chip). These bias sample
    # density toward the real maps users most often upload.
    provider_style: str | None = None


@dataclass(frozen=True)
class SyntheticRenderResult:
    sample: SyntheticSampleMetadata
    polygon: Polygon
    mask_area_px: int


DEFAULT_OVERLAY_STYLES: tuple[SyntheticOverlayStyle, ...] = (
    SyntheticOverlayStyle("bright-blue-fill", "#2f7df6", 0.38, "#175fe0", 3.0),
    SyntheticOverlayStyle("waymo-solid-blue", "#0087ff", 0.94, "#0070d8", 1.0, labels_on_top=True, circular_viewport=True),
    SyntheticOverlayStyle("waymo-solid-blue-no-stroke", "#0087ff", 0.96, None, 0.0, labels_on_top=True),
    SyntheticOverlayStyle("waymo-cyan-blue", "#0797ff", 0.88, "#0072dd", 2.0, labels_on_top=True, circular_viewport=True),
    SyntheticOverlayStyle("muted-green-fill", "#78b77b", 0.34, "#43864e", 3.0),
    SyntheticOverlayStyle("purple-fill", "#9b65dc", 0.36, "#7347b8", 3.0),
    SyntheticOverlayStyle("orange-fill", "#f28c38", 0.32, "#cf641b", 3.0),
    SyntheticOverlayStyle("low-contrast-gray", "#b9c0c8", 0.28, "#8f98a3", 2.0),
    SyntheticOverlayStyle("solid-outline-only", "#ffffff", 0.0, "#2367dc", 5.0, fill_enabled=False),
    SyntheticOverlayStyle("dashed-outline-only", "#ffffff", 0.0, "#2367dc", 5.0, dashed=True, fill_enabled=False),
)


def generate_synthetic_dataset(
    output_dir: str | Path,
    *,
    count: int,
    seed: int = 1,
    width: int = 960,
    height: int = 640,
    negative_every: int = 0,
    negative_start_index: int = 0,
) -> SyntheticDatasetManifest:
    if count < 1:
        raise ValueError("count must be positive")

    root = Path(output_dir)
    samples: list[SyntheticSampleMetadata] = []
    for index in range(count):
        negative = (
            negative_every > 0
            and index >= negative_start_index
            and (index - negative_start_index) % negative_every == 0
        )
        provider = None
        if not negative:
            if index % 8 == 6:
                provider = "waymo-web"
            elif index % 8 == 2:
                provider = "tesla-dark"
        if provider == "waymo-web":
            style = waymo_web_overlay_style(seed + index)
        elif provider == "tesla-dark":
            style = tesla_dark_overlay_style(seed + index)
        else:
            style = randomized_overlay_style(seed + index, index=index)
        config = SyntheticSceneConfig(
            provider="synthetic",
            service_area=f"sample-city-{index % 5}",
            variant=f"{style.name}-{index}",
            width=width,
            height=height,
            seed=seed + index,
            overlay_style=style,
            touch_border=index % 7 == 3,
            include_ui_chrome=index % 5 == 2,
            include_hole=index % 6 == 4,
            labels_on_top=style.labels_on_top or index % 9 == 5,
            circular_viewport=style.circular_viewport or index % 11 == 6,
            complex_boundary=style.labels_on_top or index % 4 == 0,
            large_service_area=style.labels_on_top or index % 10 == 8,
            include_distractor=index % 3 == 1,
            shape_family=("rectilinear", "road-following", "angular", "radial")[index % 4],
            jpeg_quality=82 if index % 4 == 1 else None,
            negative_scene=negative,
            include_admin_shading=index % 5 == 0 or provider == "waymo-web",
            textured_basemap=index % 7 == 5 and provider is None,
            provider_style=provider,
        )
        samples.append(generate_synthetic_sample(root, config).sample)

    manifest = SyntheticDatasetManifest(
        name="synthetic-boundary-dataset",
        version=GENERATOR_VERSION,
        samples=samples,
        properties={
            "generator": GENERATOR_VERSION,
            "seed": seed,
            "count": count,
            "width": width,
            "height": height,
        },
    )
    manifest.write_json(root / "manifest.json")
    return manifest


def waymo_web_overlay_style(seed: int) -> SyntheticOverlayStyle:
    """Waymo coverage-map look: near-opaque brand blue, labels drawn on top."""
    rng = random.Random(seed)
    hue = rng.uniform(0.55, 0.60)
    saturation = rng.uniform(0.85, 1.0)
    lightness = rng.uniform(0.45, 0.55)
    fill = colorsys.hls_to_rgb(hue, lightness, saturation)
    stroke = colorsys.hls_to_rgb(hue, max(0.15, lightness - 0.15), saturation)
    return SyntheticOverlayStyle(
        name=f"waymo-web-{seed % 97}",
        fill_color=rgb_hex(fill),
        fill_opacity=rng.uniform(0.88, 0.98),
        stroke_color=rgb_hex(stroke) if rng.random() < 0.5 else None,
        stroke_width_px=rng.uniform(0.0, 2.0),
        labels_on_top=True,
        circular_viewport=rng.random() < 0.2,
    )


def tesla_dark_overlay_style(seed: int) -> SyntheticOverlayStyle:
    """Tesla app look: translucent salmon/orange fill over a dark basemap."""
    rng = random.Random(seed)
    hue = rng.uniform(0.02, 0.07)
    saturation = rng.uniform(0.55, 0.85)
    lightness = rng.uniform(0.55, 0.68)
    fill = colorsys.hls_to_rgb(hue, lightness, saturation)
    return SyntheticOverlayStyle(
        name=f"tesla-dark-{seed % 97}",
        fill_color=rgb_hex(fill),
        fill_opacity=rng.uniform(0.45, 0.75),
        stroke_color=None,
        stroke_width_px=0.0,
        labels_on_top=True,
    )


def randomized_overlay_style(seed: int, *, index: int = 0) -> SyntheticOverlayStyle:
    """Sample the full visual space instead of teaching the model a short palette."""
    rng = random.Random(seed * 104729 + index)
    hue = rng.random()
    saturation = rng.uniform(0.08, 1.0)
    lightness = rng.uniform(0.18, 0.84)
    fill_color = rgb_hex(colorsys.hls_to_rgb(hue, lightness, saturation))
    stroke_hue = (hue + rng.uniform(-0.12, 0.12)) % 1.0
    stroke_color = rgb_hex(
        colorsys.hls_to_rgb(stroke_hue, max(0.08, min(0.92, lightness + rng.uniform(-0.28, 0.20))), saturation)
    )
    outline_only = rng.random() < 0.14
    return SyntheticOverlayStyle(
        name=f"domain-random-{index:04d}",
        fill_color=fill_color,
        fill_opacity=0.0 if outline_only else rng.uniform(0.16, 0.98),
        stroke_color=stroke_color if rng.random() < 0.82 or outline_only else None,
        stroke_width_px=rng.uniform(0.0, 8.0) if not outline_only else rng.uniform(2.0, 9.0),
        dashed=rng.random() < 0.16,
        fill_enabled=not outline_only,
        labels_on_top=rng.random() < 0.55,
        circular_viewport=rng.random() < 0.16,
        pattern=rng.choice((None, None, None, "hatch", "dots")),
        stroke_join=rng.choice(("miter", "miter", "miter", "round", "bevel")),
    )


def rgb_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(channel * 255))):02x}" for channel in rgb)


def generate_synthetic_sample(
    output_dir: str | Path,
    config: SyntheticSceneConfig,
) -> SyntheticRenderResult:
    if config.width <= 0 or config.height <= 0:
        raise ValueError("width and height must be positive")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sample_dir = root / _sample_slug(config)
    sample_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    if config.negative_scene:
        return _generate_negative_sample(root, sample_dir, config, rng)
    style = config.overlay_style or DEFAULT_OVERLAY_STYLES[config.seed % len(DEFAULT_OVERLAY_STYLES)]
    labels_on_top = config.labels_on_top or style.labels_on_top
    # Border-pressure fixtures must expose the actual rectangular capture edge;
    # a circular crop can clip that contact back out of the saved truth mask.
    circular_viewport = (config.circular_viewport or style.circular_viewport) and not config.touch_border
    polygon, hole = _sample_polygon(
        config.width,
        config.height,
        rng,
        config.touch_border,
        config.include_hole,
        config.complex_boundary or style.labels_on_top,
        config.large_service_area or style.labels_on_top,
        config.shape_family,
    )

    if hole is not None:
        polygon = Polygon(polygon.exterior.coords, [hole.exterior.coords])
        hole = None
    if circular_viewport:
        polygon = _largest_polygon(polygon.intersection(_circular_viewport_geometry(config)).buffer(0))

    base = _render_basemap(config, rng)
    if config.textured_basemap:
        base = _apply_basemap_texture(base, random.Random(config.seed + 700_003))
    if config.include_admin_shading:
        base = _render_admin_shading(base, config, random.Random(config.seed + 500_017))
    mask = _render_mask(config.width, config.height, polygon, hole)
    if config.touch_border and not _mask_touches_border(mask):
        raise RuntimeError("touch_border sample did not reach an outer mask pixel")
    overlay = _render_overlay(base, polygon, hole, style)
    if config.include_distractor:
        distractor = _sample_distractor_polygon(config.width, config.height, random.Random(config.seed + 400_009))
        distractor_style = randomized_overlay_style(config.seed + 800_011, index=config.seed)
        overlay = _render_overlay(overlay, distractor, None, distractor_style)
    if labels_on_top:
        overlay = _render_top_map_details(overlay, config, random.Random(config.seed + 900_001))
    if circular_viewport:
        overlay = _apply_circular_viewport(overlay, config)
    image = _apply_capture_effects(overlay, config)

    screenshot_path = sample_dir / "image.jpg" if config.jpeg_quality else sample_dir / "image.png"
    overlay_path = sample_dir / "overlay.png"
    mask_path = sample_dir / "mask.png"
    geojson_path = sample_dir / "boundary.geojson"
    metadata_path = sample_dir / "metadata.json"

    if config.jpeg_quality:
        image.save(screenshot_path, quality=config.jpeg_quality)
    else:
        image.save(screenshot_path)
    overlay.save(overlay_path)
    mask.save(mask_path)
    _write_geojson(geojson_path, polygon, hole, config)

    artifacts = SyntheticArtifactPaths(
        screenshot=str(screenshot_path.relative_to(root)),
        overlay=str(overlay_path.relative_to(root)),
        mask=str(mask_path.relative_to(root)),
        geojson=str(geojson_path.relative_to(root)),
        metadata=str(metadata_path.relative_to(root)),
    )
    sample = SyntheticSampleMetadata.create(
        provider=config.provider,
        service_area=config.service_area,
        variant=config.variant,
        image_size=(config.width, config.height),
        overlay_style=style.metadata(),
        artifacts=artifacts,
        base_map=config.base_map,
        seed=config.seed,
        generator_version=GENERATOR_VERSION,
        properties={
            "touch_border": config.touch_border,
            "include_ui_chrome": config.include_ui_chrome,
            "include_hole": config.include_hole,
            "labels_on_top": labels_on_top,
            "circular_viewport": circular_viewport,
            "complex_boundary": config.complex_boundary,
            "large_service_area": config.large_service_area,
            "include_distractor": config.include_distractor,
            "shape_family": config.shape_family,
            "jpeg_quality": config.jpeg_quality,
            "overlay_pattern": style.pattern,
            "overlay_dashed": style.dashed,
            "stroke_join": style.stroke_join,
            "provider_style": config.provider_style,
            "renderer": "procedural-pillow",
        },
    )
    metadata_path.write_text(sample.to_json(), encoding="utf-8")
    return SyntheticRenderResult(sample=sample, polygon=polygon, mask_area_px=_count_mask_pixels(mask))


def _generate_negative_sample(
    root: Path,
    sample_dir: Path,
    config: SyntheticSceneConfig,
    rng: random.Random,
) -> SyntheticRenderResult:
    base = _render_basemap(config, rng)
    image = _render_app_ui(base, config, rng)
    image = _apply_capture_effects(image, config)
    mask = Image.new("L", (config.width, config.height), 0)

    screenshot_path = sample_dir / "image.jpg" if config.jpeg_quality else sample_dir / "image.png"
    if config.jpeg_quality:
        image.save(screenshot_path, quality=config.jpeg_quality)
    else:
        image.save(screenshot_path)
    overlay_path = sample_dir / "overlay.png"
    mask_path = sample_dir / "mask.png"
    image.save(overlay_path)
    mask.save(mask_path)
    geojson_path = sample_dir / "boundary.geojson"
    _write_geojson(geojson_path, None, None, config)

    style = config.overlay_style or DEFAULT_OVERLAY_STYLES[0]
    artifacts = SyntheticArtifactPaths(
        screenshot=str(screenshot_path.relative_to(root)),
        overlay=str(overlay_path.relative_to(root)),
        mask=str(mask_path.relative_to(root)),
        geojson=str(geojson_path.relative_to(root)),
        metadata=str((sample_dir / "metadata.json").relative_to(root)),
    )
    sample = SyntheticSampleMetadata.create(
        provider=config.provider,
        service_area=config.service_area,
        variant=config.variant,
        image_size=(config.width, config.height),
        overlay_style=style.metadata(),
        artifacts=artifacts,
        base_map=config.base_map,
        seed=config.seed,
        generator_version=GENERATOR_VERSION,
        properties={
            "negative_scene": True,
            "touch_border": False,
            "include_ui_chrome": True,
            "include_hole": False,
            "labels_on_top": False,
            "include_distractor": False,
            "shape_family": "none",
            "circular_viewport": False,
        },
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps(sample.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SyntheticRenderResult(sample=sample, polygon=Polygon(), mask_area_px=0)


def _apply_basemap_texture(base: Image.Image, rng: random.Random) -> Image.Image:
    """Blend blurred noise over the basemap to mimic satellite imagery grain."""
    import numpy as np

    width, height = base.size
    noise_rng = np.random.default_rng(rng.randint(0, 2**31))
    coarse = noise_rng.integers(0, 255, size=(height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
    noise = Image.fromarray(coarse, "RGB").resize((width, height), Image.Resampling.BILINEAR)
    strength = rng.uniform(0.08, 0.22)
    return Image.blend(base, noise, strength)


def _render_admin_shading(base: Image.Image, config: SyntheticSceneConfig, rng: random.Random) -> Image.Image:
    """Draw 1-2 large muted district polygons that are NOT the service area."""
    image = base.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(rng.randint(1, 2)):
        polygon = _sample_distractor_polygon(config.width, config.height, random.Random(rng.randint(0, 2**31)))
        if config.provider_style == "waymo-web":
            # Waymo's coverage pages shade secondary zones in pale salmon.
            red = rng.randint(225, 250)
            tint = (red, rng.randint(150, 185), rng.randint(125, 160))
        else:
            # Muted, desaturated palettes: grays, tans, pale washes.
            gray = rng.randint(150, 235)
            tint = rng.choice(
                [
                    (gray, gray, gray),
                    (gray, gray - rng.randint(0, 18), gray - rng.randint(10, 30)),
                    (gray - rng.randint(10, 25), gray, gray - rng.randint(0, 15)),
                ]
            )
        opacity = rng.randint(90, 210)
        points = _int_points(polygon.exterior.coords)
        draw.polygon(points, fill=(*tint, opacity))
        if rng.random() < 0.7:
            draw.line([*points, points[0]], fill=(60, 62, 66, 200), width=rng.randint(2, 4))
    return image


def _render_app_ui(base: Image.Image, config: SyntheticSceneConfig, rng: random.Random) -> Image.Image:
    """Compose a ride/trip-app style screen over the basemap: an optional
    route polyline with pin markers, a stack of UI cards, buttons, and a
    status bar. Nothing in the result is a service-area overlay."""
    image = base.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    font = ImageFont.load_default()

    # Optional warm/cool full-map tint, like themed app basemaps.
    if rng.random() < 0.5:
        tint = rng.choice([(244, 170, 150), (150, 190, 244), (240, 210, 150), (170, 220, 180)])
        draw.rectangle((0, 0, width, height), fill=(*tint, rng.randint(28, 80)))

    # Route polyline with jittered waypoints and endpoint pins.
    if rng.random() < 0.8:
        points = [(rng.randint(int(width * 0.15), int(width * 0.85)), rng.randint(int(height * 0.08), int(height * 0.5)))]
        for _ in range(rng.randint(3, 7)):
            prev_x, prev_y = points[-1]
            points.append(
                (
                    max(8, min(width - 8, prev_x + rng.randint(-width // 5, width // 5))),
                    max(8, min(height - 8, prev_y + rng.randint(0, height // 6))),
                )
            )
        route_color = rng.choice([(84, 92, 214), (30, 120, 240), (30, 30, 34), (219, 68, 55)])
        draw.line(points, fill=(*route_color, 255), width=rng.randint(4, 9), joint="curve")
        for px, py in (points[0], points[-1]):
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=(40, 40, 44, 255), width=4, fill=(255, 255, 255, 255))

    # Card stack rising from the bottom, covering 25-60% of the screen.
    card_color = rng.choice([(255, 255, 255), (247, 247, 249), (28, 29, 33)])
    text_color = (120, 122, 128, 255) if card_color[0] > 128 else (170, 172, 180, 255)
    top = int(height * rng.uniform(0.4, 0.75))
    y = top
    while y < height - 20:
        card_height = rng.randint(60, 180)
        margin = rng.randint(8, 24)
        draw.rounded_rectangle(
            (margin, y, width - margin, min(height - 12, y + card_height)),
            radius=rng.randint(8, 18),
            fill=(*card_color, rng.randint(235, 255)),
        )
        # Text-like bars and an occasional pill button inside the card.
        bar_y = y + 18
        while bar_y < min(height - 24, y + card_height - 16):
            bar_width = rng.randint(width // 5, int(width * 0.7))
            draw.rounded_rectangle(
                (margin + 18, bar_y, margin + 18 + bar_width, bar_y + rng.randint(8, 16)),
                radius=5,
                fill=text_color,
            )
            bar_y += rng.randint(22, 40)
        if rng.random() < 0.5:
            button_width = rng.randint(width // 4, width // 2)
            button_x = rng.randint(margin + 12, max(margin + 13, width - margin - button_width - 12))
            draw.rounded_rectangle(
                (button_x, y + card_height - 52, button_x + button_width, y + card_height - 16),
                radius=16,
                fill=(rng.randint(20, 240), rng.randint(20, 120), rng.randint(60, 240), 255),
            )
        y += card_height + rng.randint(8, 20)

    # Status bar.
    bar_color = (250, 250, 250, 240) if card_color[0] > 128 else (18, 18, 20, 240)
    draw.rectangle((0, 0, width, rng.randint(24, 44)), fill=bar_color)
    draw.text((width // 12, 8), "12:17", fill=text_color, font=font)
    return image


def _sample_slug(config: SyntheticSceneConfig) -> str:
    parts = [config.provider, config.service_area, config.variant, str(config.seed)]
    return "-".join(_slug_part(part) for part in parts if _slug_part(part))[:96]


def _slug_part(value: object) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    return "-".join(part for part in text.split("-") if part)


def _render_basemap(config: SyntheticSceneConfig, rng: random.Random) -> Image.Image:
    if config.provider_style == "tesla-dark":
        dark = True
    elif config.provider_style == "waymo-web":
        dark = False
    else:
        dark = config.seed % 4 == 0
    background = (37, 43, 48) if dark else (242, 240, 234)
    image = Image.new("RGB", (config.width, config.height), background)
    draw = ImageDraw.Draw(image, "RGBA")

    water = (42, 83, 110, 210) if dark else (177, 213, 226, 230)
    park = (51, 91, 65, 190) if dark else (194, 224, 181, 210)
    road = (196, 199, 199, 210) if dark else (255, 255, 255, 235)
    arterial = (220, 160, 86, 210) if dark else (245, 190, 105, 230)
    label = (224, 228, 230, 220) if dark else (72, 78, 84, 220)

    draw.polygon(
        [
            (0, int(config.height * 0.72)),
            (int(config.width * 0.2), int(config.height * 0.65)),
            (int(config.width * 0.55), config.height),
            (0, config.height),
        ],
        fill=water,
    )
    draw.polygon(
        [
            (int(config.width * 0.68), int(config.height * 0.08)),
            (config.width, int(config.height * 0.02)),
            (config.width, int(config.height * 0.28)),
            (int(config.width * 0.74), int(config.height * 0.32)),
        ],
        fill=park,
    )

    for _ in range(16):
        y = rng.randint(30, max(31, config.height - 30))
        x_offset = rng.randint(-120, 120)
        draw.line([(x_offset, y), (config.width + x_offset, y + rng.randint(-80, 80))], fill=road, width=3)
    for _ in range(10):
        x = rng.randint(30, max(31, config.width - 30))
        y_offset = rng.randint(-80, 80)
        draw.line([(x, y_offset), (x + rng.randint(-80, 80), config.height + y_offset)], fill=road, width=3)
    for _ in range(3):
        y = rng.randint(90, max(91, config.height - 90))
        draw.line([(-20, y), (config.width + 20, y + rng.randint(-50, 50))], fill=arterial, width=7)

    font = ImageFont.load_default()
    for index, text in enumerate(("Downtown", "Central", "River Park", "Station", "Market", "Heights")):
        x = int((index + 1) * config.width / 7) + rng.randint(-28, 28)
        y = rng.randint(55, max(56, config.height - 70))
        draw.text((x, y), text, fill=label, font=font)

    if config.include_ui_chrome or config.provider_style == "tesla-dark":
        chip_fill = (28, 29, 33, 240) if dark else (255, 255, 255, 235)
        chip_text = (240, 241, 244, 245) if dark else (42, 44, 48, 235)
        city_names = ("Tampa, FL", "Austin, TX", "Miami, FL", "Phoenix, AZ", "Dallas, TX", "Service area")
        chip_label = city_names[config.seed % len(city_names)]
        draw.rounded_rectangle((18, 18, min(config.width - 18, 240), 68), radius=14, fill=chip_fill)
        draw.text((36, 36), chip_label, fill=chip_text, font=font)
        draw.rounded_rectangle(
            (config.width - 168, config.height - 72, config.width - 24, config.height - 24),
            radius=10,
            fill=chip_fill,
        )

    return image


def _sample_polygon(
    width: int,
    height: int,
    rng: random.Random,
    touch_border: bool,
    include_hole: bool,
    complex_boundary: bool = False,
    large_service_area: bool = False,
    shape_family: str = "radial",
) -> tuple[Polygon, Polygon | None]:
    if shape_family == "rectilinear":
        polygon = _sample_rectilinear_polygon(width, height, rng, large_service_area)
    elif shape_family == "angular":
        polygon = _sample_angular_polygon(
            width,
            height,
            rng,
            complex_boundary,
            large_service_area,
        )
    elif shape_family == "road-following":
        polygon = _sample_road_following_polygon(width, height, rng, large_service_area)
    else:
        polygon = _sample_radial_polygon(width, height, rng, complex_boundary, large_service_area)

    if touch_border:
        anchor = polygon.representative_point()
        half_height = max(2.0, min(width, height) * 0.015)
        polygon = _largest_polygon(
            unary_union(
                [
                    polygon,
                    box(
                        -1.0,
                        max(-1.0, anchor.y - half_height),
                        anchor.x + 1.0,
                        min(float(height), anchor.y + half_height),
                    ),
                ]
            ).buffer(0)
        )

    hole = None
    if include_hole:
        center = polygon.representative_point()
        hole_w = width * 0.035
        hole_h = height * 0.045
        candidate = box(center.x - hole_w, center.y - hole_h, center.x + hole_w, center.y + hole_h)
        if polygon.buffer(-3).contains(candidate):
            hole = candidate
    return polygon, hole


def _sample_radial_polygon(
    width: int,
    height: int,
    rng: random.Random,
    complex_boundary: bool,
    large_service_area: bool,
) -> Polygon:
    cx = width * rng.uniform(0.42, 0.58)
    cy = height * rng.uniform(0.42, 0.58)
    if large_service_area:
        radius_x = width * rng.uniform(0.28, 0.42)
        radius_y = height * rng.uniform(0.30, 0.45)
    else:
        radius_x = width * rng.uniform(0.20, 0.34)
        radius_y = height * rng.uniform(0.20, 0.34)
    vertices = rng.randint(14, 24) if complex_boundary else rng.randint(7, 12)
    points: list[tuple[float, float]] = []
    for index in range(vertices):
        angle = (2.0 * math.pi * index / vertices) + rng.uniform(-0.20, 0.20)
        scale = rng.uniform(0.64, 1.20) if complex_boundary else rng.uniform(0.74, 1.12)
        if complex_boundary and index % 5 == 2:
            scale *= rng.uniform(0.45, 0.70)
        if complex_boundary and index % 7 == 3:
            scale *= rng.uniform(1.08, 1.32)
        x = cx + math.cos(angle) * radius_x * scale
        y = cy + math.sin(angle) * radius_y * scale
        points.append((min(width - 2, max(2, x)), min(height - 2, max(2, y))))
    return _largest_polygon(Polygon(points).buffer(0))


def _sample_angular_polygon(
    width: int,
    height: int,
    rng: random.Random,
    complex_boundary: bool,
    large_service_area: bool,
) -> Polygon:
    """Build a deliberate straight-edge polygon with acute, obtuse, and reflex corners.

    Angular fixtures used to fall through to the radial generator. This family
    alternates outer and inset radii around a mildly perturbed angular lattice,
    producing stable concave corners and long oblique line segments rather than
    a many-sided approximation of a curve.
    """

    cx = width * rng.uniform(0.43, 0.57)
    cy = height * rng.uniform(0.42, 0.58)
    if large_service_area:
        radius_x = width * rng.uniform(0.31, 0.42)
        radius_y = height * rng.uniform(0.32, 0.44)
    else:
        radius_x = width * rng.uniform(0.23, 0.34)
        radius_y = height * rng.uniform(0.24, 0.36)

    vertex_count = rng.choice((8, 10, 12, 14)) if complex_boundary else rng.choice((8, 10, 12))
    rotation = rng.uniform(-math.pi, math.pi)
    points: list[tuple[float, float]] = []
    for index in range(vertex_count):
        lattice_angle = rotation + (2.0 * math.pi * index / vertex_count)
        angle = lattice_angle + rng.uniform(-0.045, 0.045)
        if index % 2:
            radial_scale = rng.uniform(0.50, 0.68)
        else:
            radial_scale = rng.uniform(0.92, 1.08)
        # Break perfect star symmetry while keeping every edge genuinely linear.
        if complex_boundary and index % 5 == 0:
            radial_scale *= rng.uniform(1.04, 1.14)
        x = cx + math.cos(angle) * radius_x * radial_scale
        y = cy + math.sin(angle) * radius_y * radial_scale
        points.append((min(width - 2.0, max(2.0, x)), min(height - 2.0, max(2.0, y))))

    polygon = _largest_polygon(Polygon(points).buffer(0))
    return _largest_polygon(polygon.intersection(box(2, 2, width - 2, height - 2)).buffer(0))


def _sample_rectilinear_polygon(
    width: int,
    height: int,
    rng: random.Random,
    large_service_area: bool,
) -> Polygon:
    """Build connected orthogonal regions with real corner and notch pressure."""
    span_x = width * rng.uniform(0.48, 0.72 if large_service_area else 0.60)
    span_y = height * rng.uniform(0.48, 0.76 if large_service_area else 0.62)
    left = width * rng.uniform(0.14, 0.28)
    top = height * rng.uniform(0.12, 0.28)
    right = min(width - 3.0, left + span_x)
    bottom = min(height - 3.0, top + span_y)
    pieces = [box(left, top, right, bottom)]
    for _ in range(rng.randint(2, 5)):
        side = rng.choice(("left", "right", "top", "bottom"))
        if side in {"left", "right"}:
            arm_h = (bottom - top) * rng.uniform(0.12, 0.35)
            arm_y = rng.uniform(top, bottom - arm_h)
            arm_w = width * rng.uniform(0.06, 0.18)
            x0 = left - arm_w if side == "left" else right - 1.0
            pieces.append(box(x0, arm_y, x0 + arm_w + 1.0, arm_y + arm_h))
        else:
            arm_w = (right - left) * rng.uniform(0.12, 0.35)
            arm_x = rng.uniform(left, right - arm_w)
            arm_h = height * rng.uniform(0.06, 0.18)
            y0 = top - arm_h if side == "top" else bottom - 1.0
            pieces.append(box(arm_x, y0, arm_x + arm_w, y0 + arm_h + 1.0))
    polygon = _largest_polygon(unary_union(pieces).buffer(0))
    for _ in range(rng.randint(1, 3)):
        min_x, min_y, max_x, max_y = polygon.bounds
        notch_w = (max_x - min_x) * rng.uniform(0.05, 0.14)
        notch_h = (max_y - min_y) * rng.uniform(0.05, 0.14)
        edge = rng.choice(("top", "bottom", "left", "right"))
        if edge in {"top", "bottom"}:
            x0 = rng.uniform(min_x + 2, max_x - notch_w - 2)
            y0 = min_y - 1 if edge == "top" else max_y - notch_h + 1
        else:
            x0 = min_x - 1 if edge == "left" else max_x - notch_w + 1
            y0 = rng.uniform(min_y + 2, max_y - notch_h - 2)
        carved = polygon.difference(box(x0, y0, x0 + notch_w, y0 + notch_h)).buffer(0)
        if not carved.is_empty:
            polygon = _largest_polygon(carved)
    return _largest_polygon(polygon.intersection(box(2, 2, width - 2, height - 2)).buffer(0))


def _sample_road_following_polygon(
    width: int,
    height: int,
    rng: random.Random,
    large_service_area: bool,
) -> Polygon:
    """Create a sharp, road-aligned envelope with oblique and square turns."""
    cx = width * rng.uniform(0.38, 0.55)
    cy = height * rng.uniform(0.38, 0.58)
    points = [(cx, cy)]
    headings = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi, 5 * math.pi / 4, 3 * math.pi / 2, 7 * math.pi / 4]
    for _ in range(rng.randint(5, 9)):
        heading = rng.choice(headings)
        length = min(width, height) * rng.uniform(0.08, 0.19)
        x = min(width - 12.0, max(12.0, points[-1][0] + math.cos(heading) * length))
        y = min(height - 12.0, max(12.0, points[-1][1] + math.sin(heading) * length))
        points.append((x, y))
    width_px = min(width, height) * rng.uniform(0.075, 0.14 if large_service_area else 0.11)
    line = LineString(points)
    corridor = line.buffer(width_px, cap_style="square", join_style="bevel")
    hub = box(
        cx - width_px * rng.uniform(1.0, 1.8),
        cy - width_px * rng.uniform(1.0, 1.8),
        cx + width_px * rng.uniform(1.0, 1.8),
        cy + width_px * rng.uniform(1.0, 1.8),
    )
    polygon = unary_union([corridor, hub]).intersection(box(2, 2, width - 2, height - 2)).buffer(0)
    return _largest_polygon(polygon)


def _render_mask(width: int, height: int, polygon: Polygon, hole: Polygon | None) -> Image.Image:
    geometry = polygon
    if hole is not None:
        interiors = [interior.coords for interior in polygon.interiors]
        interiors.append(hole.exterior.coords)
        geometry = Polygon(polygon.exterior.coords, interiors)
    mask = rasterize_geometry_mask(geometry, width=width, height=height)
    return Image.fromarray(mask.astype("uint8") * 255)


def _mask_touches_border(mask: Image.Image) -> bool:
    width, height = mask.size
    edges = (
        mask.crop((0, 0, width, 1)),
        mask.crop((0, height - 1, width, height)),
        mask.crop((0, 0, 1, height)),
        mask.crop((width - 1, 0, width, height)),
    )
    return any(edge.getbbox() is not None for edge in edges)


def _sample_distractor_polygon(width: int, height: int, rng: random.Random) -> Polygon:
    left = rng.choice((width * 0.03, width * 0.70))
    top = rng.choice((height * 0.05, height * 0.68))
    box_w = rng.uniform(width * 0.12, width * 0.25)
    box_h = rng.uniform(height * 0.10, height * 0.24)
    return Polygon(
        [
            (left, top + box_h * 0.25),
            (left + box_w * 0.45, top),
            (left + box_w, top + box_h * 0.35),
            (left + box_w * 0.80, top + box_h),
            (left + box_w * 0.18, top + box_h * 0.88),
        ]
    )


def _render_overlay(
    base: Image.Image,
    polygon: Polygon,
    hole: Polygon | None,
    style: SyntheticOverlayStyle,
) -> Image.Image:
    image = base.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill = _hex_rgba(style.fill_color, style.fill_opacity)
    stroke = _hex_rgba(style.stroke_color or style.fill_color, 1.0)

    if style.fill_enabled and style.fill_opacity > 0:
        draw.polygon(_int_points(polygon.exterior.coords), fill=fill)
        for interior in polygon.interiors:
            draw.polygon(_int_points(interior.coords), fill=(0, 0, 0, 0))
        if hole is not None:
            draw.polygon(_int_points(hole.exterior.coords), fill=(0, 0, 0, 0))
        if style.pattern is not None:
            pattern = Image.new("RGBA", image.size, (0, 0, 0, 0))
            pattern_draw = ImageDraw.Draw(pattern)
            pattern_color = _hex_rgba(style.stroke_color or style.fill_color, min(0.75, style.fill_opacity + 0.25))
            if style.pattern == "hatch":
                for offset in range(-image.height, image.width, 18):
                    pattern_draw.line((offset, 0, offset + image.height, image.height), fill=pattern_color, width=1)
            else:
                for y in range(7, image.height, 16):
                    for x in range(7, image.width, 16):
                        pattern_draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=pattern_color)
            clip = _render_mask(image.width, image.height, polygon, hole)
            layer.alpha_composite(Image.composite(pattern, Image.new("RGBA", image.size), clip))
    if style.stroke_width_px > 0:
        points = _int_points(polygon.exterior.coords)
        if style.dashed:
            _draw_dashed_ring(draw, points, fill=stroke, width=max(1, round(style.stroke_width_px)))
        elif style.stroke_join == "round":
            draw.line(points, fill=stroke, width=max(1, round(style.stroke_width_px)), joint="curve")
        elif style.stroke_join == "bevel":
            draw.line(points, fill=stroke, width=max(1, round(style.stroke_width_px)))
        else:
            # Pillow grows a polygon outline entirely inward. That makes the
            # rendered edge disagree with its vector label and rewards a
            # contracted prediction. Browser SVG, Canvas, and map renderers
            # center the stroke on the vector path, so construct that miter
            # band explicitly.
            _draw_centered_miter_ring(
                layer,
                polygon.exterior.coords,
                fill=stroke,
                width=max(1, round(style.stroke_width_px)),
            )
    return Image.alpha_composite(image, layer).convert("RGB")


def _draw_centered_miter_ring(
    layer: Image.Image,
    coordinates,
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    """Paint a closed, centered miter stroke without erasing the fill."""

    band = LineString(list(coordinates)).buffer(
        float(width) / 2.0,
        cap_style="flat",
        join_style="mitre",
    )
    if band.is_empty:
        return
    stroke_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    stroke_draw = ImageDraw.Draw(stroke_layer)
    polygons = (
        [band]
        if isinstance(band, Polygon)
        else list(band.geoms)
        if isinstance(band, MultiPolygon)
        else []
    )
    for part in polygons:
        stroke_draw.polygon(_int_points(part.exterior.coords), fill=fill)
        for interior in part.interiors:
            stroke_draw.polygon(_int_points(interior.coords), fill=(0, 0, 0, 0))
    layer.alpha_composite(stroke_layer)


def _render_top_map_details(image: Image.Image, config: SyntheticSceneConfig, rng: random.Random) -> Image.Image:
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    road_light = (210, 234, 255, 135)
    road_strong = (184, 222, 255, 185)
    label = (222, 242, 255, 220)
    shield = (54, 76, 96, 230)

    for _ in range(12):
        y = rng.randint(35, max(36, config.height - 35))
        x_offset = rng.randint(-140, 140)
        draw.line(
            [(x_offset, y), (config.width + x_offset, y + rng.randint(-90, 90))],
            fill=road_light,
            width=rng.choice((1, 2, 3)),
        )
    for _ in range(8):
        x = rng.randint(35, max(36, config.width - 35))
        y_offset = rng.randint(-90, 90)
        draw.line(
            [(x, y_offset), (x + rng.randint(-90, 90), config.height + y_offset)],
            fill=road_light,
            width=rng.choice((1, 2, 3)),
        )
    for _ in range(4):
        x = rng.randint(int(config.width * 0.24), int(config.width * 0.76))
        draw.line(
            [(x, -20), (x + rng.randint(-90, 90), config.height + 20)],
            fill=road_strong,
            width=rng.choice((3, 4, 5)),
        )
    for index, text in enumerate(("Houston", "Downtown", "Midtown", "Heights", "Museum District", "First Ward")):
        x = int((index + 1) * config.width / 7) + rng.randint(-42, 42)
        y = rng.randint(int(config.height * 0.22), int(config.height * 0.78))
        draw.text((x, y), text, fill=label, font=font)
    for _ in range(7):
        x = rng.randint(int(config.width * 0.20), int(config.width * 0.82))
        y = rng.randint(int(config.height * 0.15), int(config.height * 0.85))
        draw.rounded_rectangle((x - 9, y - 7, x + 9, y + 7), radius=5, fill=shield)
        draw.text((x - 5, y - 4), str(rng.choice((10, 45, 69, 90, 288, 610))), fill=(255, 255, 255, 230), font=font)
    return image.convert("RGB")


def _apply_circular_viewport(image: Image.Image, config: SyntheticSceneConfig) -> Image.Image:
    background = Image.new("RGB", image.size, (255, 255, 255))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    margin = -int(min(config.width, config.height) * 0.01)
    draw.ellipse((margin, margin, config.width - margin, config.height - margin), fill=255)
    background.paste(image, (0, 0), mask)
    return background


def _circular_viewport_geometry(config: SyntheticSceneConfig) -> Polygon:
    """Match the visible ellipse so hidden screenshot corners are not labels."""
    margin = -int(min(config.width, config.height) * 0.01)
    radius_x = (config.width - (2 * margin)) / 2.0
    radius_y = (config.height - (2 * margin)) / 2.0
    center = (config.width / 2.0, config.height / 2.0)
    circle = Point(*center).buffer(1.0, quad_segs=128)
    return scale(circle, xfact=radius_x, yfact=radius_y, origin=center)


def _apply_capture_effects(image: Image.Image, config: SyntheticSceneConfig) -> Image.Image:
    rng = random.Random(config.seed + 700_003)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.78, 1.22))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.72, 1.30))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.35, 1.55))
    if config.seed % 3 == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))
    if config.seed % 5 == 0:
        small = image.resize((max(1, config.width // 2), max(1, config.height // 2)), Image.Resampling.BILINEAR)
        image = small.resize((config.width, config.height), Image.Resampling.BILINEAR)
    return image


def _write_geojson(path: Path, polygon: Polygon | None, hole: Polygon | None, config: SyntheticSceneConfig) -> None:
    if polygon is None:
        data = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "generator": GENERATOR_VERSION,
                "negative_scene": True,
                "image_width": config.width,
                "image_height": config.height,
            },
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    geometry = polygon
    if hole is not None:
        geometry = Polygon(polygon.exterior.coords, [hole.exterior.coords])
    feature = {
        "type": "Feature",
        "properties": {
            "provider": config.provider,
            "service_area": config.service_area,
            "variant": config.variant,
            "synthetic": True,
        },
        "geometry": _pixel_geometry_to_lonlat(mapping(geometry), config.width, config.height),
    }
    data = {
        "type": "FeatureCollection",
        "features": [feature],
        "metadata": {
            "generator": GENERATOR_VERSION,
            "pixel_geometry": mapping(geometry),
            "image_width": config.width,
            "image_height": config.height,
        },
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pixel_geometry_to_lonlat(geometry: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    origin_lon = -122.45
    origin_lat = 37.72
    lon_span = 0.16
    lat_span = lon_span * (height / width)

    def convert_ring(ring: Sequence[Sequence[float]]) -> list[list[float]]:
        return [
            [
                round(origin_lon + (float(x) / width) * lon_span, 7),
                round(origin_lat + ((height - float(y)) / height) * lat_span, 7),
            ]
            for x, y in ring
        ]

    if geometry["type"] == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [convert_ring(ring) for ring in geometry["coordinates"]],
        }
    raise ValueError(f"unsupported synthetic geometry type: {geometry['type']}")


def _largest_polygon(geometry) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon) and geometry.geoms:
        return max(geometry.geoms, key=lambda item: item.area)
    raise ValueError("synthetic boundary did not form a polygon")


def _int_points(coords: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    return [(round(float(x)), round(float(y))) for x, y, *_rest in coords]


def _hex_rgba(value: str, opacity: float) -> tuple[int, int, int, int]:
    color = value.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"expected #rrggbb color, got {value!r}")
    return (
        int(color[0:2], 16),
        int(color[2:4], 16),
        int(color[4:6], 16),
        max(0, min(255, round(float(opacity) * 255))),
    )


def _draw_dashed_ring(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash_px: int = 18,
    gap_px: int = 12,
) -> None:
    for start, end in zip(points, points[1:]):
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            continue
        distance = 0.0
        while distance < length:
            segment_end = min(length, distance + dash_px)
            sx = x1 + (x2 - x1) * (distance / length)
            sy = y1 + (y2 - y1) * (distance / length)
            ex = x1 + (x2 - x1) * (segment_end / length)
            ey = y1 + (y2 - y1) * (segment_end / length)
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
            distance += dash_px + gap_px


def _count_mask_pixels(mask: Image.Image) -> int:
    return sum(mask.histogram()[1:])
