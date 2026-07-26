import numpy as np

from map_boundary_builder.boundaryfield import (
    BoundaryFieldConfig,
    boundaryfield_mask_to_geometry,
    extract_service_area_with_boundaryfield,
    signed_distance_from_mask,
)


class FakeInput:
    def __init__(self, name: str):
        self.name = name


class CoarseSession:
    def get_inputs(self):
        return [FakeInput("image")]

    def run(self, _outputs, feed):
        values = feed["image"]
        height, width = values.shape[-2:]
        logits = np.full((1, 1, height, width), -8.0, dtype=np.float32)
        logits[:, :, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 8.0
        return [logits]


class RefinerSession:
    def get_inputs(self):
        return [FakeInput("boundary_patch")]

    def run(self, _outputs, feed):
        values = feed["boundary_patch"]
        coarse = values[:, 3:4]
        logits = (coarse - 0.5) * 16.0
        sdf = (coarse - 0.5) * 2.0
        corners = np.full_like(coarse, -8.0)
        return [np.concatenate([logits, sdf, corners], axis=1).astype(np.float32)]


def test_boundaryfield_runs_coarse_and_native_refiner_end_to_end() -> None:
    rgb = np.full((160, 224, 3), 220, dtype=np.uint8)
    result = extract_service_area_with_boundaryfield(
        rgb,
        CoarseSession(),
        RefinerSession(),
        config=BoundaryFieldConfig(refiner_patch_size=96, refiner_core_size=64),
    )

    assert result.mask.shape == (160, 224)
    assert result.pixel_geometry.is_valid
    assert result.contour_count == 1
    assert result.diagnostics["boundaryfield"] is True
    assert result.diagnostics["refiner_tile_count"] > 0


def test_selector_only_confidence_uses_mask_quality_without_fake_refiner_uncertainty() -> None:
    rgb = np.full((160, 224, 3), 220, dtype=np.uint8)
    result = extract_service_area_with_boundaryfield(
        rgb,
        CoarseSession(),
        config=BoundaryFieldConfig(coarse_input_size=160),
    )

    assert result.confidence >= 0.55
    assert result.diagnostics["refiner_enabled"] is False
    assert result.diagnostics["boundary_uncertainty"] is None


def test_boundaryfield_polygonizer_preserves_holes_and_sharp_corners() -> None:
    mask = np.zeros((128, 160), dtype=bool)
    mask[16:112, 20:140] = True
    mask[48:80, 64:96] = False

    geometry, contour_count = boundaryfield_mask_to_geometry(mask, tolerance_px=0.5)

    assert geometry.is_valid
    assert geometry.geom_type == "Polygon"
    assert len(geometry.interiors) == 1
    assert contour_count == 1
    assert len(geometry.exterior.coords) <= 6


def test_signed_distance_changes_sign_at_mask_boundary() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    distance = signed_distance_from_mask(mask, clip_px=8)

    assert distance[20, 20] > 0
    assert distance[0, 0] < 0
