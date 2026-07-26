import numpy as np
import pytest
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

from map_boundary_builder.extract import (
    AUTO_FILL_STYLE,
    DEFAULT_EXTRACTION_PROFILE,
    ExtractionHints,
    auto_fill_extraction_result,
    extraction_confidence,
    fill_binary_holes,
    keep_main_components,
    load_rgb,
    mask_to_geometry,
    resolve_extraction_hints,
    resolve_extraction_profile,
    simplify_geometry,
    write_mask_png,
    write_overlay_png,
)


def textured_fill_image(width: int = 320, height: int = 240) -> np.ndarray:
    """Light basemap with a road grid and a translucent blue overlay fill.

    The auto-fill gates require interior texture (real overlays keep roads
    visible), so fixtures blend the fill over the grid instead of stamping a
    flat rectangle.
    """
    rgb = np.full((height, width, 3), 245, dtype=np.uint8)
    for x in range(0, width, 24):
        rgb[:, x : x + 2] = 210
    for y in range(0, height, 24):
        rgb[y : y + 2, :] = 210
    region = rgb[60:180, 80:240].astype(np.float64)
    overlay = np.array([60.0, 120.0, 235.0])
    rgb[60:180, 80:240] = (0.35 * region + 0.65 * overlay).astype(np.uint8)
    return rgb


def test_auto_fill_extracts_textured_overlay():
    result = auto_fill_extraction_result(textured_fill_image(), simplify_px=6.0)
    assert result is not None
    assert result.style == AUTO_FILL_STYLE
    assert result.mask[120, 160]
    assert not result.mask[10, 10]
    assert result.contour_count >= 1
    assert 0.0 < result.confidence <= 1.0


def test_auto_fill_rejects_flat_image():
    flat = np.full((120, 160, 3), 250, dtype=np.uint8)
    assert auto_fill_extraction_result(flat, simplify_px=6.0) is None


def test_auto_fill_respects_seed_hint():
    rgb = textured_fill_image()
    hints = ExtractionHints(seed_point=(160.0, 120.0))
    result = auto_fill_extraction_result(rgb, simplify_px=6.0, hints=hints)
    assert result is not None
    assert result.mask[120, 160]


def test_mask_to_geometry_simple_rectangle():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True
    geometry, contour_count = mask_to_geometry(mask, simplify_px=2.0)
    assert contour_count == 1
    assert isinstance(geometry, (Polygon, MultiPolygon))
    assert geometry.area == pytest.approx(60 * 40, rel=0.15)


def test_mask_to_geometry_raises_on_empty():
    with pytest.raises(ValueError):
        mask_to_geometry(np.zeros((50, 50), dtype=bool), simplify_px=2.0)


def test_simplify_geometry_reduces_vertices():
    mask = np.zeros((200, 200), dtype=bool)
    yy, xx = np.mgrid[0:200, 0:200]
    mask[((yy - 100) ** 2 + (xx - 100) ** 2) < 70**2] = True
    raw, _ = mask_to_geometry(mask, simplify_px=0.0)
    simplified = simplify_geometry(raw, tolerance=6.0)
    assert len(simplified.exterior.coords) < len(raw.exterior.coords)


def test_fill_binary_holes():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True
    mask[25:35, 25:35] = False
    filled = fill_binary_holes(mask)
    assert filled[30, 30]


def test_keep_main_components_drops_specks():
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:120, 20:120] = True
    mask[150:152, 150:152] = True
    kept = keep_main_components(mask, max_components=3)
    assert kept[50, 50]
    assert not kept[150, 150]


def test_extraction_confidence_bounds():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True
    confidence = extraction_confidence(mask, "model-mask", 1)
    assert 0.0 < confidence <= 1.0
    assert extraction_confidence(np.zeros((10, 10), dtype=bool), "model-mask", 0) == 0.0


def test_resolve_extraction_profile_aliases():
    assert resolve_extraction_profile(None).name == DEFAULT_EXTRACTION_PROFILE
    assert resolve_extraction_profile("satellite").name == "satellite-overlay"
    with pytest.raises(ValueError):
        resolve_extraction_profile("bogus-profile")


def test_resolve_extraction_hints_accepts_dict():
    hints = resolve_extraction_hints({"seed_point": (5, 6), "target_rgb": (1, 2, 3)})
    assert hints.seed_point == (5.0, 6.0)
    assert hints.target_rgb == (1, 2, 3)


def test_load_rgb_and_artifact_writers(tmp_path):
    rgb = textured_fill_image(64, 48)
    source = tmp_path / "sample.png"
    Image.fromarray(rgb).save(source)
    loaded = load_rgb(source)
    assert loaded.shape == (48, 64, 3)

    mask = np.zeros((48, 64), dtype=bool)
    mask[10:30, 10:40] = True
    mask_path = tmp_path / "artifacts" / "mask.png"
    overlay_path = tmp_path / "artifacts" / "overlay.png"
    write_mask_png(mask, mask_path)
    write_overlay_png(source, mask, overlay_path, rgb=loaded)
    assert mask_path.is_file()
    assert overlay_path.is_file()
