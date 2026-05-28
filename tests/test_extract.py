import unittest

import numpy as np

from map_boundary_builder.extract import extract_service_area, repair_mask


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


if __name__ == "__main__":
    unittest.main()
