import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import cv2

import map_boundary_builder.extract as extract_module
from map_boundary_builder.extract import (
    _EXTRACTION_MEMORY_CACHE,
    _SCALED_EXTRACTION_MEMORY_CACHE,
    extract_service_area,
    extraction_cache_dependency_signature,
    classify_style,
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


if __name__ == "__main__":
    unittest.main()
