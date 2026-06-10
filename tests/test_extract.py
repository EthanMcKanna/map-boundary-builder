import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import cv2

import map_boundary_builder.extract as extract_module
from map_boundary_builder.extract import (
    _EXTRACTION_MEMORY_CACHE,
    _SCALED_EXTRACTION_MEMORY_CACHE,
    EXTRACTION_CACHE_ENV,
    extract_service_area,
    extraction_cache_dependency_signature,
    classify_style,
    green_service_fill_mask,
    keep_main_components,
    repair_mask,
    remove_small_components,
)


class MaskRepairTests(unittest.TestCase):
    def test_bright_blue_repair_preserves_exterior_notches(self) -> None:
        raw = np.zeros((240, 240), dtype=bool)
        raw[40:200, 40:200] = True
        raw[84:108, 40:116] = False
        raw[144:156, 144:156] = False

        repaired = repair_mask(raw, "bright-blue")

        self.assertFalse(repaired[96, 80])
        self.assertTrue(repaired[150, 150])

    def test_bright_blue_repair_still_fills_thin_edge_artifacts(self) -> None:
        raw = np.zeros((240, 240), dtype=bool)
        raw[40:200, 40:200] = True
        raw[96:102, 40:116] = False

        repaired = repair_mask(raw, "bright-blue")

        self.assertTrue(repaired[99, 80])

    def test_remove_small_components_returns_noop_mask_without_reselecting(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[4:14, 4:14] = True
        mask[24:34, 24:34] = True

        with patch.object(
            extract_module,
            "select_component_labels",
            side_effect=AssertionError("all components should be kept without relabel selection"),
        ):
            cleaned = remove_small_components(mask, min_area=20)

        np.testing.assert_array_equal(cleaned, mask)

    def test_keep_main_components_returns_noop_mask_without_reselecting(self) -> None:
        mask = np.zeros((70, 70), dtype=bool)
        mask[4:24, 4:24] = True
        mask[44:64, 44:64] = True

        with patch.object(
            extract_module,
            "select_component_labels",
            side_effect=AssertionError("all main components should be kept without relabel selection"),
        ):
            cleaned = keep_main_components(mask, max_components=2)

        np.testing.assert_array_equal(cleaned, mask)

    def test_classify_style_shortcuts_obvious_dark_teal(self) -> None:
        rgb = np.full((120, 120, 3), 18, dtype=np.uint8)
        teal_hsv = np.zeros((1, 1, 3), dtype=np.uint8)
        teal_hsv[0, 0] = (85, 150, 130)
        teal_rgb = cv2.cvtColor(teal_hsv, cv2.COLOR_HSV2RGB)[0, 0]
        rgb[24:96, 24:96] = teal_rgb

        with (
            patch.object(
                extract_module,
                "purple_service_mask",
                side_effect=AssertionError("obvious dark teal should not need purple mask"),
            ),
            patch.object(
                extract_module,
                "light_fill_service_mask",
                side_effect=AssertionError("obvious dark teal should not need light-fill component pass"),
            ),
        ):
            self.assertEqual(classify_style(rgb), "dark-teal")

    def test_classify_style_shortcuts_obvious_gray_fill(self) -> None:
        rgb = np.full((100, 100, 3), 30, dtype=np.uint8)

        with (
            patch.object(
                extract_module,
                "purple_service_mask",
                side_effect=AssertionError("obvious gray fill should not need purple mask"),
            ),
            patch.object(
                extract_module,
                "light_fill_service_mask",
                side_effect=AssertionError("obvious gray fill should not need light-fill component pass"),
            ),
        ):
            self.assertEqual(classify_style(rgb), "gray-fill")

    def test_green_service_fill_expands_muted_may_style_area(self) -> None:
        rgb = np.full((260, 220, 3), 255, dtype=np.uint8)
        service_area = np.array(
            [[88, 22], [145, 44], [172, 128], [148, 224], [86, 206], [55, 126]],
            dtype=np.int32,
        )
        cv2.fillPoly(rgb, [service_area], (193, 219, 195))

        ink_color = (104, 179, 116)
        cv2.line(rgb, (88, 40), (104, 202), ink_color, 17)
        cv2.line(rgb, (65, 126), (160, 126), ink_color, 15)
        cv2.line(rgb, (116, 58), (153, 178), ink_color, 15)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        saturated_seed = (
            (hue >= 55)
            & (hue <= 90)
            & (sat >= 45)
            & (val >= 80)
            & (rgb[:, :, 1].astype(np.int16) > rgb[:, :, 0].astype(np.int16) + 25)
        )

        mask = green_service_fill_mask(rgb, hue, sat, val)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(mask.sum(), saturated_seed.sum() * 1.9)
        self.assertTrue(mask[44, 118])
        self.assertTrue(mask[204, 92])
        self.assertFalse(mask[18, 88])

    def test_downscaled_extraction_returns_original_coordinate_space(self) -> None:
        rgb = np.full((240, 240, 3), 255, dtype=np.uint8)
        rgb[60:190, 50:180] = (46, 119, 246)

        result = extract_service_area("unused.png", rgb=rgb, max_dimension=120)

        self.assertEqual(result.mask.shape, (240, 240))
        min_x, min_y, max_x, max_y = result.pixel_geometry.bounds
        self.assertLess(abs(min_x - 50), 3)
        self.assertLess(abs(min_y - 60), 3)
        self.assertLess(abs(max_x - 179), 3)
        self.assertLess(abs(max_y - 189), 3)

    def test_canonical_extraction_cache_shifts_uniform_border_hit(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)
        bordered = np.full((90, 112, 3), 255, dtype=np.uint8)
        bordered[4:84, 7:107] = base

        with TemporaryDirectory() as workdir:
            with patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)):
                _EXTRACTION_MEMORY_CACHE.clear()
                try:
                    base_result = extract_service_area("base.png", rgb=base)
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        side_effect=AssertionError("bordered cache hit should avoid extraction"),
                    ):
                        bordered_result = extract_service_area("bordered.png", rgb=bordered)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()

        self.assertEqual(bordered_result.mask.shape, bordered.shape[:2])
        self.assertEqual(bordered_result.style, base_result.style)
        self.assertAlmostEqual(bordered_result.coverage_ratio, float(bordered_result.mask.mean()))
        np.testing.assert_array_equal(bordered_result.mask[4:84, 7:107], base_result.mask)
        self.assertFalse(bordered_result.mask[:4, :].any())
        self.assertFalse(bordered_result.mask[:, :7].any())
        base_min_x, base_min_y, base_max_x, base_max_y = base_result.pixel_geometry.bounds
        min_x, min_y, max_x, max_y = bordered_result.pixel_geometry.bounds
        self.assertAlmostEqual(min_x, base_min_x + 7)
        self.assertAlmostEqual(min_y, base_min_y + 4)
        self.assertAlmostEqual(max_x, base_max_x + 7)
        self.assertAlmostEqual(max_y, base_max_y + 4)

    def test_extraction_cache_does_not_write_disk_by_default(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)

        with TemporaryDirectory() as workdir:
            cache_dir = Path(workdir)
            with (
                patch.object(extract_module, "EXTRACTION_CACHE_DIR", cache_dir),
                patch.object(extract_module, "EXTRACTION_DISK_CACHE_ENABLED", False),
            ):
                _EXTRACTION_MEMORY_CACHE.clear()
                try:
                    extract_service_area("base.png", rgb=base)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()

            self.assertEqual(list(cache_dir.glob("*.npz")), [])

    def test_cache_false_bypasses_memory_cache(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)

        with TemporaryDirectory() as workdir:
            with patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)):
                _EXTRACTION_MEMORY_CACHE.clear()
                try:
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        wraps=extract_module.extract_service_area_from_rgb,
                    ) as wrapped:
                        extract_service_area("base.png", rgb=base)
                        extract_service_area("base.png", rgb=base, cache=False)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()

        self.assertEqual(wrapped.call_count, 2)

    def test_extraction_cache_env_bypasses_memory_cache(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)

        with TemporaryDirectory() as workdir:
            with patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)):
                _EXTRACTION_MEMORY_CACHE.clear()
                try:
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        wraps=extract_module.extract_service_area_from_rgb,
                    ) as wrapped:
                        extract_service_area("base.png", rgb=base)
                        with patch.dict(os.environ, {EXTRACTION_CACHE_ENV: "0"}):
                            extract_service_area("base.png", rgb=base)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()

        self.assertEqual(wrapped.call_count, 2)

    def test_large_untrimmed_extraction_skips_memory_cache(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)
        base[0, :2] = (240, 240, 240)
        base[-1, :2] = (240, 240, 240)
        base[:2, 0] = (240, 240, 240)
        base[:2, -1] = (240, 240, 240)

        with TemporaryDirectory() as workdir:
            with (
                patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)),
                patch.object(extract_module, "EXTRACTION_DISK_CACHE_ENABLED", False),
                patch.object(extract_module, "EXTRACTION_UNTRIMMED_CACHE_MAX_PIXELS", 1),
            ):
                _EXTRACTION_MEMORY_CACHE.clear()
                try:
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        wraps=extract_module.extract_service_area_from_rgb,
                    ) as wrapped:
                        first = extract_service_area("first.png", rgb=base)
                        second = extract_service_area("second.png", rgb=base)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()

        self.assertEqual(wrapped.call_count, 2)
        self.assertTrue(first.pixel_geometry.equals_exact(second.pixel_geometry, 0.0))

    def test_scaled_extraction_cache_reuses_large_downscaled_result(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)
        base[0, :2] = (240, 240, 240)
        base[-1, :2] = (240, 240, 240)
        base[:2, 0] = (240, 240, 240)
        base[:2, -1] = (240, 240, 240)

        with TemporaryDirectory() as workdir:
            with (
                patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)),
                patch.object(extract_module, "EXTRACTION_DISK_CACHE_ENABLED", False),
                patch.object(extract_module, "EXTRACTION_UNTRIMMED_CACHE_MAX_PIXELS", 1),
                patch.object(extract_module, "SCALED_EXTRACTION_MEMORY_CACHE_MAX", 4),
                patch.object(extract_module, "SCALED_EXTRACTION_CACHE_MAX_PIXELS", 10_000),
            ):
                _EXTRACTION_MEMORY_CACHE.clear()
                _SCALED_EXTRACTION_MEMORY_CACHE.clear()
                try:
                    first = extract_service_area("first.png", rgb=base, max_dimension=40)
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        side_effect=AssertionError("scaled cache hit should avoid extraction"),
                    ):
                        second = extract_service_area("second.png", rgb=base, max_dimension=40)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()
                    _SCALED_EXTRACTION_MEMORY_CACHE.clear()

        self.assertEqual(first.scaled_cache_status, "miss-stored")
        self.assertEqual(first.scaled_cache_shape, (32, 40))
        self.assertEqual(second.scaled_cache_status, "hit")
        self.assertEqual(second.scaled_cache_shape, (32, 40))
        self.assertEqual(second.mask.shape, base.shape[:2])
        np.testing.assert_array_equal(first.mask, second.mask)
        self.assertTrue(first.pixel_geometry.equals_exact(second.pixel_geometry, 0.0))

    def test_trimmed_extraction_uses_memory_cache_above_untrimmed_limit(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)
        bordered = np.full((90, 112, 3), 255, dtype=np.uint8)
        bordered[4:84, 7:107] = base

        with TemporaryDirectory() as workdir:
            with (
                patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)),
                patch.object(extract_module, "EXTRACTION_DISK_CACHE_ENABLED", False),
                patch.object(extract_module, "EXTRACTION_UNTRIMMED_CACHE_MAX_PIXELS", 1),
            ):
                _EXTRACTION_MEMORY_CACHE.clear()
                try:
                    first = extract_service_area("bordered-a.png", rgb=bordered)
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        side_effect=AssertionError("trimmed cache hit should avoid extraction"),
                    ):
                        second = extract_service_area("bordered-b.png", rgb=bordered)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()

        np.testing.assert_array_equal(first.mask, second.mask)
        self.assertTrue(first.pixel_geometry.equals_exact(second.pixel_geometry, 0.0))

    def test_large_trimmed_extraction_skips_memory_cache_above_trimmed_limit(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)
        bordered = np.full((90, 112, 3), 255, dtype=np.uint8)
        bordered[4:84, 7:107] = base

        with TemporaryDirectory() as workdir:
            with (
                patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)),
                patch.object(extract_module, "EXTRACTION_DISK_CACHE_ENABLED", False),
                patch.object(extract_module, "EXTRACTION_TRIMMED_CACHE_MAX_PIXELS", 1),
            ):
                _EXTRACTION_MEMORY_CACHE.clear()
                try:
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        wraps=extract_module.extract_service_area_from_rgb,
                    ) as wrapped:
                        first = extract_service_area("bordered-a.png", rgb=bordered)
                        second = extract_service_area("bordered-b.png", rgb=bordered)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()

        self.assertEqual(wrapped.call_count, 2)
        self.assertTrue(first.pixel_geometry.equals_exact(second.pixel_geometry, 0.0))

    def test_large_trimmed_downscaled_extraction_uses_scaled_cache(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)
        bordered = np.full((90, 112, 3), 255, dtype=np.uint8)
        bordered[4:84, 7:107] = base

        with TemporaryDirectory() as workdir:
            with (
                patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)),
                patch.object(extract_module, "EXTRACTION_DISK_CACHE_ENABLED", False),
                patch.object(extract_module, "EXTRACTION_TRIMMED_CACHE_MAX_PIXELS", 1),
                patch.object(extract_module, "SCALED_EXTRACTION_MEMORY_CACHE_MAX", 4),
                patch.object(extract_module, "SCALED_EXTRACTION_CACHE_MAX_PIXELS", 10_000),
            ):
                _EXTRACTION_MEMORY_CACHE.clear()
                _SCALED_EXTRACTION_MEMORY_CACHE.clear()
                try:
                    first = extract_service_area("bordered-a.png", rgb=bordered, max_dimension=80)
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        side_effect=AssertionError("scaled cache hit should avoid extraction"),
                    ):
                        second = extract_service_area("bordered-b.png", rgb=bordered, max_dimension=80)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()
                    _SCALED_EXTRACTION_MEMORY_CACHE.clear()

        self.assertEqual(first.scaled_cache_status, "miss-stored")
        self.assertEqual(second.scaled_cache_status, "hit")
        np.testing.assert_array_equal(first.mask, second.mask)
        self.assertTrue(first.pixel_geometry.equals_exact(second.pixel_geometry, 0.0))

    def test_canonical_extraction_disk_cache_can_be_enabled(self) -> None:
        base = np.full((80, 100, 3), 255, dtype=np.uint8)
        base[24:58, 30:74] = (46, 119, 246)
        bordered = np.full((90, 112, 3), 255, dtype=np.uint8)
        bordered[4:84, 7:107] = base

        with TemporaryDirectory() as workdir:
            with (
                patch.object(extract_module, "EXTRACTION_CACHE_DIR", Path(workdir)),
                patch.object(extract_module, "EXTRACTION_DISK_CACHE_ENABLED", True),
            ):
                _EXTRACTION_MEMORY_CACHE.clear()
                try:
                    base_result = extract_service_area("base.png", rgb=base)
                    self.assertTrue(list(Path(workdir).glob("*.npz")))
                    _EXTRACTION_MEMORY_CACHE.clear()
                    with patch.object(
                        extract_module,
                        "extract_service_area_from_rgb",
                        side_effect=AssertionError("bordered disk cache hit should avoid extraction"),
                    ):
                        bordered_result = extract_service_area("bordered.png", rgb=bordered)
                finally:
                    _EXTRACTION_MEMORY_CACHE.clear()

        np.testing.assert_array_equal(bordered_result.mask[4:84, 7:107], base_result.mask)
        self.assertFalse(bordered_result.mask[:4, :].any())
        self.assertFalse(bordered_result.mask[:, :7].any())

    def test_extraction_cache_signature_tracks_cv2_runtime(self) -> None:
        original = extract_module._EXTRACTION_CACHE_DEPENDENCY_SIGNATURE
        extract_module._EXTRACTION_CACHE_DEPENDENCY_SIGNATURE = None
        try:
            with patch.object(extract_module, "runtime_dependency_signature", return_value="deps=cv2-a"):
                first = extraction_cache_dependency_signature()
            extract_module._EXTRACTION_CACHE_DEPENDENCY_SIGNATURE = None
            with patch.object(extract_module, "runtime_dependency_signature", return_value="deps=cv2-b"):
                second = extraction_cache_dependency_signature()
        finally:
            extract_module._EXTRACTION_CACHE_DEPENDENCY_SIGNATURE = original

        self.assertNotEqual(first, second)

    def test_extraction_cache_key_tracks_pipeline_version(self) -> None:
        rgb = np.full((12, 14, 3), 255, dtype=np.uint8)
        with (
            patch.object(extract_module, "get_pipeline_version", return_value="pipeline-a"),
            patch.object(extract_module, "extraction_cache_dependency_signature", return_value="deps"),
        ):
            first = extract_module.extraction_visual_cache_key(rgb, simplify_px=6.0, max_dimension=0)
        with (
            patch.object(extract_module, "get_pipeline_version", return_value="pipeline-b"),
            patch.object(extract_module, "extraction_cache_dependency_signature", return_value="deps"),
        ):
            second = extract_module.extraction_visual_cache_key(rgb, simplify_px=6.0, max_dimension=0)

        self.assertNotEqual(first, second)


class AutoFillExtractionTests(unittest.TestCase):
    @staticmethod
    def _basemap(
        background: tuple[int, int, int],
        road_color: tuple[int, int, int],
        size: tuple[int, int] = (360, 360),
    ) -> np.ndarray:
        height, width = size
        rgb = np.full((height, width, 3), background, dtype=np.uint8)
        for x in range(24, width, 48):
            cv2.line(rgb, (x, 0), (x, height - 1), road_color, 3)
        for y in range(24, height, 48):
            cv2.line(rgb, (0, y), (width - 1, y), road_color, 3)
        return rgb

    @staticmethod
    def _blend_overlay(
        rgb: np.ndarray,
        polygon: np.ndarray,
        color: tuple[int, int, int],
        alpha: float,
    ) -> np.ndarray:
        region = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(region, [polygon], 255)
        region_mask = region > 0
        blended = rgb.astype(np.float32)
        blended[region_mask] = blended[region_mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
        rgb[:] = np.clip(blended, 0, 255).astype(np.uint8)
        return region_mask

    @staticmethod
    def _intersection_over_union(mask: np.ndarray, expected: np.ndarray) -> float:
        union = float(np.logical_or(mask, expected).sum())
        if union == 0.0:
            return 0.0
        return float(np.logical_and(mask, expected).sum()) / union

    _SERVICE_POLYGON = np.array(
        [[70, 60], [300, 80], [320, 210], [240, 320], [90, 300], [50, 170]],
        dtype=np.int32,
    )

    def test_red_overlay_on_light_basemap_uses_auto_fill(self) -> None:
        rgb = self._basemap((226, 226, 226), (204, 204, 204))
        expected = self._blend_overlay(rgb, self._SERVICE_POLYGON, (220, 36, 32), 0.5)

        result = extract_service_area("unused-red.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "auto-fill")
        self.assertGreaterEqual(self._intersection_over_union(result.mask, expected), 0.85)

    def test_orange_overlay_on_light_basemap_uses_auto_fill(self) -> None:
        rgb = self._basemap((230, 228, 222), (206, 204, 200))
        expected = self._blend_overlay(rgb, self._SERVICE_POLYGON, (245, 140, 20), 0.5)

        result = extract_service_area("unused-orange.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "auto-fill")
        self.assertGreaterEqual(self._intersection_over_union(result.mask, expected), 0.85)

    def test_yellow_overlay_on_light_basemap_uses_auto_fill(self) -> None:
        rgb = self._basemap((228, 226, 220), (202, 200, 196))
        expected = self._blend_overlay(rgb, self._SERVICE_POLYGON, (240, 210, 30), 0.55)

        result = extract_service_area("unused-yellow.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "auto-fill")
        self.assertGreaterEqual(self._intersection_over_union(result.mask, expected), 0.85)

    def test_magenta_overlay_on_dark_basemap_uses_auto_fill(self) -> None:
        rgb = self._basemap((30, 30, 34), (66, 66, 72))
        expected = self._blend_overlay(rgb, self._SERVICE_POLYGON, (208, 44, 164), 0.55)

        result = extract_service_area("unused-magenta.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "auto-fill")
        self.assertGreaterEqual(self._intersection_over_union(result.mask, expected), 0.85)

    def test_app_chrome_frame_does_not_steal_auto_fill_pick(self) -> None:
        # White UI chrome around a gray basemap: the interior gray region is
        # "distinct" from the white border in lightness only, so the chroma
        # gate must keep it from outscoring the actual colored fill.
        rgb = np.full((400, 400, 3), 255, dtype=np.uint8)
        rgb[40:360, 20:380] = self._basemap((221, 221, 221), (200, 200, 200), size=(320, 360))
        polygon = self._SERVICE_POLYGON + np.array([[30, 50]], dtype=np.int32)
        expected = self._blend_overlay(rgb, polygon, (235, 90, 40), 0.5)

        result = extract_service_area("unused-framed.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "auto-fill")
        self.assertGreaterEqual(self._intersection_over_union(result.mask, expected), 0.85)

    def test_solid_outline_ring_extracts_enclosed_region(self) -> None:
        rgb = self._basemap((240, 239, 236), (218, 217, 214), size=(420, 420))
        polygon = self._SERVICE_POLYGON + np.array([[20, 30]], dtype=np.int32)
        expected = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(expected, [polygon], 255)
        cv2.polylines(rgb, [polygon], isClosed=True, color=(217, 48, 37), thickness=5)

        result = extract_service_area("unused-ring-solid.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "auto-fill")
        self.assertGreaterEqual(self._intersection_over_union(result.mask, expected > 0), 0.85)

    def test_dashed_outline_ring_extracts_enclosed_region(self) -> None:
        rgb = self._basemap((240, 239, 236), (218, 217, 214), size=(420, 420))
        polygon = self._SERVICE_POLYGON + np.array([[20, 30]], dtype=np.int32)
        expected = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(expected, [polygon], 255)
        points = polygon.astype(np.float64)
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            length = float(np.hypot(*(end - start)))
            steps = np.arange(0.0, length, 1.0)
            on = (steps % 22.0) < 14.0
            for step, draw in zip(steps, on):
                if not draw:
                    continue
                point = start + (end - start) * (step / length)
                cv2.circle(rgb, (int(round(point[0])), int(round(point[1]))), 2, (26, 115, 232), -1)

        result = extract_service_area("unused-ring-dashed.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "auto-fill")
        self.assertGreaterEqual(self._intersection_over_union(result.mask, expected > 0), 0.85)

    def test_pale_textured_water_corner_fails_closed(self) -> None:
        # Basemap water with white roads/labels rendered over it has texture,
        # but reads pale: the chroma floor must keep auto-fill off it.
        rgb = self._basemap((241, 240, 236), (255, 255, 255), size=(400, 400))
        water = np.array([[0, 240], [180, 400], [0, 400]], dtype=np.int32)
        cv2.fillPoly(rgb, [water], (170, 211, 235))
        for x in range(24, 400, 48):
            cv2.line(rgb, (x, 0), (x, 399), (255, 255, 255), 3)
        cv2.rectangle(rgb, (40, 330), (90, 340), (110, 112, 115), -1)

        with self.assertRaises(ValueError):
            extract_service_area("unused-water-corner.png", rgb=rgb, cache=False)

    def test_overlay_beats_textured_water_corner(self) -> None:
        rgb = self._basemap((241, 240, 236), (220, 219, 216), size=(400, 400))
        water = np.array([[0, 240], [180, 400], [0, 400]], dtype=np.int32)
        cv2.fillPoly(rgb, [water], (170, 211, 235))
        polygon = np.array(
            [[150, 40], [360, 60], [375, 200], [290, 300], [160, 280], [130, 150]],
            dtype=np.int32,
        )
        expected = self._blend_overlay(rgb, polygon, (220, 36, 32), 0.5)

        result = extract_service_area("unused-water-overlay.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "auto-fill")
        self.assertGreaterEqual(self._intersection_over_union(result.mask, expected), 0.85)

    def test_tiny_disagreeing_generic_does_not_replace_styled(self) -> None:
        mask_styled = np.zeros((100, 100), dtype=bool)
        mask_styled[10:70, 10:70] = True
        mask_generic = np.zeros((100, 100), dtype=bool)
        mask_generic[80:90, 80:90] = True
        styled = extract_module.ExtractionResult(
            mask=mask_styled,
            style="light-fill",
            pixel_geometry=extract_module.Polygon([(10, 10), (70, 10), (70, 70), (10, 70)]),
            coverage_ratio=float(mask_styled.mean()),
            contour_count=1,
            confidence=1.0,
        )
        generic = extract_module.ExtractionResult(
            mask=mask_generic,
            style="auto-fill",
            pixel_geometry=extract_module.Polygon([(80, 80), (90, 80), (90, 90), (80, 90)]),
            coverage_ratio=float(mask_generic.mean()),
            contour_count=1,
            confidence=0.9,
        )

        self.assertFalse(extract_module.auto_fill_should_take_over(styled, generic))

    def test_oversized_styled_mask_yields_to_disagreeing_generic(self) -> None:
        mask_styled = np.zeros((100, 100), dtype=bool)
        mask_styled[2:98, 2:98] = True
        mask_generic = np.zeros((100, 100), dtype=bool)
        mask_generic[20:60, 20:60] = True
        styled = extract_module.ExtractionResult(
            mask=mask_styled,
            style="gray-fill",
            pixel_geometry=extract_module.Polygon([(2, 2), (98, 2), (98, 98), (2, 98)]),
            coverage_ratio=float(mask_styled.mean()),
            contour_count=1,
            confidence=1.0,
        )
        generic = extract_module.ExtractionResult(
            mask=mask_generic,
            style="auto-fill",
            pixel_geometry=extract_module.Polygon([(20, 20), (60, 20), (60, 60), (20, 60)]),
            coverage_ratio=float(mask_generic.mean()),
            contour_count=1,
            confidence=0.9,
        )

        self.assertTrue(extract_module.auto_fill_should_take_over(styled, generic))

    def test_plain_basemap_without_overlay_fails_closed(self) -> None:
        rgb = self._basemap((226, 226, 226), (204, 204, 204))

        with self.assertRaises(ValueError):
            extract_service_area("unused-plain.png", rgb=rgb, cache=False)

    def test_bright_blue_extraction_skips_auto_fill(self) -> None:
        rgb = np.full((240, 240, 3), 255, dtype=np.uint8)
        rgb[60:190, 50:180] = (46, 119, 246)

        with patch.object(
            extract_module,
            "auto_fill_extraction_result",
            side_effect=AssertionError("styled extraction should not invoke auto-fill"),
        ):
            result = extract_service_area("unused-blue.png", rgb=rgb, cache=False)

        self.assertEqual(result.style, "bright-blue")

    def test_auto_fill_confidence_is_discounted_below_styled(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        styled = extract_module.extraction_confidence(mask, "dark-teal", 1)
        generic = extract_module.extraction_confidence(mask, "auto-fill", 1)

        self.assertLess(generic, styled)


if __name__ == "__main__":
    unittest.main()
