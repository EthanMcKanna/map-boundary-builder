import unittest

import numpy as np

from map_boundary_builder.extract import repair_mask


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


if __name__ == "__main__":
    unittest.main()
