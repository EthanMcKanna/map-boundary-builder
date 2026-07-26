import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
from shapely.geometry import shape

from map_boundary_builder.evaluation import rasterize_geometry_mask
from map_boundary_builder.synthetic import (
    DEFAULT_OVERLAY_STYLES,
    SyntheticSceneConfig,
    generate_synthetic_dataset,
    generate_synthetic_sample,
)
from map_boundary_builder.synthetic.generator import _draw_centered_miter_ring, _sample_angular_polygon


def test_generate_synthetic_sample_writes_required_artifacts(tmp_path: Path) -> None:
    result = generate_synthetic_sample(
        tmp_path,
        SyntheticSceneConfig(
            provider="Waymo",
            service_area="Phoenix",
            variant="outline",
            width=320,
            height=220,
            seed=42,
            overlay_style=DEFAULT_OVERLAY_STYLES[-1],
            touch_border=True,
            include_ui_chrome=True,
            include_hole=True,
        ),
    )

    result.sample.validate_required_artifacts(tmp_path)
    assert result.sample.image_size == (320, 220)
    assert result.sample.properties["renderer"] == "procedural-pillow"
    assert result.mask_area_px > 0

    screenshot = Image.open(tmp_path / result.sample.artifacts.screenshot)
    overlay = Image.open(tmp_path / result.sample.artifacts.overlay)
    mask = Image.open(tmp_path / result.sample.artifacts.mask)

    assert screenshot.size == (320, 220)
    assert overlay.size == (320, 220)
    assert mask.size == (320, 220)
    mask_array = np.asarray(mask)
    assert set(np.unique(mask_array)).issubset({0, 255})
    assert any(
        edge.any()
        for edge in (
            mask_array[0, :],
            mask_array[-1, :],
            mask_array[:, 0],
            mask_array[:, -1],
        )
    )

    geojson = json.loads((tmp_path / result.sample.artifacts.geojson).read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"][0]["geometry"]["type"] == "Polygon"
    assert geojson["metadata"]["pixel_geometry"]["type"] == "Polygon"
    assert geojson["metadata"]["image_width"] == 320
    pixel_geometry = shape(geojson["metadata"]["pixel_geometry"])
    assert pixel_geometry.bounds[0] <= 0.0
    assert np.array_equal(
        mask_array > 0,
        rasterize_geometry_mask(pixel_geometry, width=320, height=220),
    )


def test_generate_synthetic_sample_is_deterministic(tmp_path: Path) -> None:
    config = SyntheticSceneConfig(width=240, height=180, seed=7, variant="deterministic")
    first = generate_synthetic_sample(tmp_path / "first", config)
    second = generate_synthetic_sample(tmp_path / "second", config)

    first_mask = (tmp_path / "first" / first.sample.artifacts.mask).read_bytes()
    second_mask = (tmp_path / "second" / second.sample.artifacts.mask).read_bytes()
    first_geojson = (tmp_path / "first" / first.sample.artifacts.geojson).read_text(encoding="utf-8")
    second_geojson = (tmp_path / "second" / second.sample.artifacts.geojson).read_text(encoding="utf-8")

    assert first.sample.content_hash == second.sample.content_hash
    assert first.sample.sample_id == second.sample.sample_id
    assert first_mask == second_mask
    assert first_geojson == second_geojson


def test_touch_border_takes_precedence_over_circular_viewport(tmp_path: Path) -> None:
    result = generate_synthetic_sample(
        tmp_path,
        SyntheticSceneConfig(
            width=120,
            height=90,
            seed=24,
            variant="border-before-circle",
            touch_border=True,
            circular_viewport=True,
            overlay_style=DEFAULT_OVERLAY_STYLES[1],
        ),
    )
    mask = np.asarray(Image.open(tmp_path / result.sample.artifacts.mask).convert("L")) > 0

    assert result.sample.properties["touch_border"] is True
    assert result.sample.properties["circular_viewport"] is False
    assert mask[:, 0].any()


def test_generate_synthetic_dataset_writes_manifest(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, count=5, seed=100, width=200, height=140)

    manifest.validate_required_artifacts(tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert saved["properties"]["count"] == 5
    assert len(saved["samples"]) == 5
    assert len({sample.sample_id for sample in manifest.samples}) == 5


def test_angular_family_has_oblique_edges_and_reflex_corners() -> None:
    polygon = _sample_angular_polygon(
        320,
        220,
        random.Random(17),
        complex_boundary=True,
        large_service_area=False,
    )
    points = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    edges = np.roll(points, -1, axis=0) - points
    turns = []
    for index in range(len(edges)):
        first = edges[index]
        second = edges[(index + 1) % len(edges)]
        turns.append((first[0] * second[1]) - (first[1] * second[0]))

    assert polygon.is_valid
    assert len(points) >= 8
    assert np.all(np.abs(edges[:, 0]) > 1e-6)
    assert np.all(np.abs(edges[:, 1]) > 1e-6)
    assert min(turns) < 0 < max(turns)


def test_miter_stroke_is_centered_on_vector_path() -> None:
    layer = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    _draw_centered_miter_ring(
        layer,
        [(10, 8), (24, 8), (24, 24), (10, 24), (10, 8)],
        fill=(20, 80, 200, 255),
        width=4,
    )
    alpha = np.asarray(layer.getchannel("A"))

    # The left-hand path is x=10, so a centered 4px band must be visible on
    # both sides. Pillow's old polygon outline failed the exterior assertion.
    assert alpha[16, 8] == 255
    assert alpha[16, 10] == 255
    assert alpha[16, 11] == 255
    assert alpha[16, 7] == 0
    assert alpha[16, 12] == 0
