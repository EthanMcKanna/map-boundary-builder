from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

import map_boundary_builder.extract as extract_module
from map_boundary_builder.gray_outline import detect_gray_outline_mask


def dark_gray_service_map(points: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    height = width = 480
    core = np.full((height, width, 3), 15, dtype=np.uint8)
    for coordinate in range(24, width, 48):
        cv2.line(core, (coordinate, 0), (coordinate, height - 1), (38, 38, 38), 2)
    for coordinate in range(30, height, 54):
        cv2.line(core, (0, coordinate), (width - 1, coordinate), (34, 34, 34), 2)
    cv2.line(core, (0, 440), (470, 0), (44, 44, 44), 4)

    polygon = np.asarray(points, dtype=np.int32)
    target = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(target, [polygon], 255)
    inside = target > 0
    # A translucent neutral fill brightens both land and the road texture.
    core[inside] = np.clip(
        core[inside].astype(np.int16) + 45,
        0,
        255,
    ).astype(np.uint8)
    cv2.polylines(core, [polygon], True, (230, 230, 230), 2, cv2.LINE_AA)

    # Representative UI and route-shield chrome: both are bright but remain
    # separate from the large service outline.
    cv2.rectangle(core, (18, 18), (142, 72), (31, 31, 31), -1)
    cv2.rectangle(core, (18, 18), (142, 72), (70, 70, 70), 2)
    cv2.rectangle(core, (265, 205), (298, 235), (225, 225, 225), 2)

    padded = cv2.copyMakeBorder(core, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded_target = cv2.copyMakeBorder(target, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0) > 0
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(padded, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    assert ok
    decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return decoded, padded_target


TAMPA_SHAPE = [
    (118, 75),
    (360, 78),
    (360, 180),
    (404, 180),
    (404, 240),
    (430, 240),
    (430, 315),
    (365, 315),
    (338, 355),
    (280, 365),
    (260, 355),
    (260, 390),
    (205, 390),
    (205, 445),
    (145, 445),
    (145, 290),
    (92, 290),
    (92, 180),
    (118, 180),
]

ORLANDO_SHAPE = [
    (62, 85),
    (135, 85),
    (240, 142),
    (365, 142),
    (418, 385),
    (330, 392),
    (225, 392),
    (185, 382),
    (62, 382),
]


@pytest.mark.parametrize("points", [TAMPA_SHAPE, ORLANDO_SHAPE])
def test_dark_gray_filled_white_outline_recovers_sharp_polygon(
    points: list[tuple[int, int]],
) -> None:
    rgb, target = dark_gray_service_map(points)
    canonical_rgb, _origin = extract_module.canonical_extract_rgb(rgb)
    outlined = detect_gray_outline_mask(canonical_rgb)

    result = extract_module.extract_service_area(
        "unused.jpeg",
        rgb=rgb,
        cache=False,
        simplify_px=1.0,
    )

    intersection = int(np.logical_and(result.mask, target).sum())
    union = int(np.logical_or(result.mask, target).sum())
    assert intersection / union >= 0.96
    assert result.style == "gray-fill"
    assert result.contour_count == 1
    assert result.pixel_geometry.is_valid
    assert result.pixel_geometry.geom_type == "Polygon"
    assert result.confidence == 1.0
    assert outlined is not None
    assert outlined.diagnostics["method"] == "luminance-outline"
    assert outlined.diagnostics["verified_source_native"] is True
    assert outlined.diagnostics["ensemble_minimum_iou"] >= 0.985
    assert outlined.diagnostics["low_frequency_fill_iou"] >= 0.95
    assert len(result.pixel_geometry.exterior.coords) <= 48


def test_gray_outline_requires_brighter_fill_evidence() -> None:
    rgb = np.full((360, 420, 3), 16, dtype=np.uint8)
    for coordinate in range(20, 420, 35):
        cv2.line(rgb, (coordinate, 0), (coordinate, 359), (38, 38, 38), 2)
    # A large white panel/road rectangle alone is not a service area: its
    # interior has the same luminance distribution as its exterior.
    cv2.rectangle(rgb, (80, 70), (340, 290), (225, 225, 225), 3)

    assert detect_gray_outline_mask(rgb) is None


def test_gray_outline_rejects_bright_road_lattice() -> None:
    rgb = np.full((420, 440, 3), 14, dtype=np.uint8)
    for coordinate in range(55, 390, 45):
        cv2.line(rgb, (coordinate, 45), (coordinate, 370), (225, 225, 225), 2)
    for coordinate in range(55, 375, 45):
        cv2.line(rgb, (45, coordinate), (395, coordinate), (225, 225, 225), 2)

    assert detect_gray_outline_mask(rgb) is None


def test_model_path_preflights_verified_source_native_gray_outline() -> None:
    rgb, _target = dark_gray_service_map(TAMPA_SHAPE)

    with patch.object(extract_module, "maybe_extract_with_model") as model_extract:
        result = extract_module.extract_service_area(
            "unused.jpeg",
            rgb=rgb,
            cache=False,
            simplify_px=1.0,
            use_model=extract_module.EDGEGRAPH_MODEL_VARIANT,
        )

    model_extract.assert_not_called()
    assert result.style == "gray-fill"
    assert result.diagnostics["gray_outline"]["method"] == "luminance-outline"
    assert result.diagnostics["automatic_route"] == (
        "verified-source-native-preflight-v1"
    )
