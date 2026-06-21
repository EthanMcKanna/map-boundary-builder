"""Deterministic synthetic map-boundary sample generation.

This module is intentionally lightweight. It creates a local, reproducible
renderer that exercises the extraction/evaluation stack while leaving the
future MapLibre/Playwright renderer as a drop-in replacement for the same
artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from shapely.geometry import MultiPolygon, Polygon, mapping

from .manifest import (
    OverlayStyleMetadata,
    SyntheticArtifactPaths,
    SyntheticDatasetManifest,
    SyntheticSampleMetadata,
)

GENERATOR_VERSION = "synthetic-generator-v10-real-style"


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
) -> SyntheticDatasetManifest:
    if count < 1:
        raise ValueError("count must be positive")

    root = Path(output_dir)
    samples: list[SyntheticSampleMetadata] = []
    for index in range(count):
        style = DEFAULT_OVERLAY_STYLES[index % len(DEFAULT_OVERLAY_STYLES)]
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
            jpeg_quality=82 if index % 4 == 1 else None,
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
    style = config.overlay_style or DEFAULT_OVERLAY_STYLES[config.seed % len(DEFAULT_OVERLAY_STYLES)]
    labels_on_top = config.labels_on_top or style.labels_on_top
    circular_viewport = config.circular_viewport or style.circular_viewport
    polygon, hole = _sample_polygon(
        config.width,
        config.height,
        rng,
        config.touch_border,
        config.include_hole,
        config.complex_boundary or style.labels_on_top,
        config.large_service_area or style.labels_on_top,
    )

    base = _render_basemap(config, rng)
    mask = _render_mask(config.width, config.height, polygon, hole)
    overlay = _render_overlay(base, polygon, hole, style)
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
            "jpeg_quality": config.jpeg_quality,
            "renderer": "procedural-pillow",
        },
    )
    metadata_path.write_text(sample.to_json(), encoding="utf-8")
    return SyntheticRenderResult(sample=sample, polygon=polygon, mask_area_px=_count_mask_pixels(mask))


def _sample_slug(config: SyntheticSceneConfig) -> str:
    parts = [config.provider, config.service_area, config.variant, str(config.seed)]
    return "-".join(_slug_part(part) for part in parts if _slug_part(part))[:96]


def _slug_part(value: object) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    return "-".join(part for part in text.split("-") if part)


def _render_basemap(config: SyntheticSceneConfig, rng: random.Random) -> Image.Image:
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

    if config.include_ui_chrome:
        draw.rounded_rectangle((18, 18, config.width - 18, 68), radius=10, fill=(255, 255, 255, 235))
        draw.text((36, 36), "Service area", fill=(42, 44, 48, 235), font=font)
        draw.rounded_rectangle(
            (config.width - 168, config.height - 72, config.width - 24, config.height - 24),
            radius=10,
            fill=(255, 255, 255, 235),
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
) -> tuple[Polygon, Polygon | None]:
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
    if touch_border:
        points[0] = (1.0, points[0][1])
        points[1] = (1.0, points[1][1])
    polygon = _largest_polygon(Polygon(points).buffer(0))

    hole = None
    if include_hole:
        hole_w = width * 0.045
        hole_h = height * 0.055
        hole = Polygon(
            [
                (cx - hole_w, cy - hole_h),
                (cx + hole_w, cy - hole_h),
                (cx + hole_w, cy + hole_h),
                (cx - hole_w, cy + hole_h),
            ]
        )
        if not polygon.contains(hole):
            hole = None
    return polygon, hole


def _render_mask(width: int, height: int, polygon: Polygon, hole: Polygon | None) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(_int_points(polygon.exterior.coords), fill=255)
    if hole is not None:
        draw.polygon(_int_points(hole.exterior.coords), fill=0)
    return mask


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
        if hole is not None:
            draw.polygon(_int_points(hole.exterior.coords), fill=(0, 0, 0, 0))
    if style.stroke_width_px > 0:
        points = _int_points(polygon.exterior.coords)
        if style.dashed:
            _draw_dashed_ring(draw, points, fill=stroke, width=max(1, round(style.stroke_width_px)))
        else:
            draw.line(points, fill=stroke, width=max(1, round(style.stroke_width_px)), joint="curve")
    return Image.alpha_composite(image, layer).convert("RGB")


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


def _apply_capture_effects(image: Image.Image, config: SyntheticSceneConfig) -> Image.Image:
    if config.seed % 3 == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
    if config.seed % 5 == 0:
        small = image.resize((max(1, config.width // 2), max(1, config.height // 2)), Image.Resampling.BILINEAR)
        image = small.resize((config.width, config.height), Image.Resampling.BILINEAR)
    return image


def _write_geojson(path: Path, polygon: Polygon, hole: Polygon | None, config: SyntheticSceneConfig) -> None:
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
