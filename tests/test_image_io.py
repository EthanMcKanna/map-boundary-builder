import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from map_boundary_builder.extract import load_rgb, write_overlay_png
from map_boundary_builder.image_io import normalize_image_for_processing, safe_image_extension


class SvgImageIoTests(unittest.TestCase):
    def test_safe_image_extension_preserves_svg(self) -> None:
        self.assertEqual(safe_image_extension("mi.svg"), ".svg")

    def test_svg_is_rasterized_before_pillow_reads_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            svg_path = workdir / "input.png"
            svg_path.write_text(
                """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 8">
<rect width="12" height="8" fill="#ffffff"/>
<rect x="2" y="1" width="8" height="6" fill="#00a6ff"/>
</svg>
""",
                encoding="utf-8",
            )

            image_path = normalize_image_for_processing(svg_path, output_dir=workdir)
            rgb = load_rgb(image_path)

            self.assertEqual(image_path.suffix, ".png")
            self.assertEqual(rgb.shape, (8, 12, 3))
            self.assertGreater(int(rgb[:, :, 2].max()), 200)

    def test_transparent_png_is_composited_before_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            png_path = workdir / "circle.png"
            image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
            image.putpixel((1, 1), (0, 128, 255, 255))
            image.save(png_path)

            image_path = normalize_image_for_processing(png_path, output_dir=workdir)
            rgb = load_rgb(image_path)

            self.assertEqual(image_path.name, "circle.opaque.png")
            self.assertEqual(tuple(rgb[0, 0]), (255, 255, 255))
            self.assertEqual(tuple(rgb[1, 1]), (0, 128, 255))


class OverlayPreviewTests(unittest.TestCase):
    def test_overlay_preview_draws_dark_mask_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "overlay.png"
            rgb = np.full((20, 20, 3), 255, dtype=np.uint8)
            mask = np.zeros((20, 20), dtype=bool)
            mask[5:15, 5:15] = True

            write_overlay_png("unused.png", mask, out_path, rgb=rgb)

            overlay = np.asarray(Image.open(out_path).convert("RGB"))
            self.assertTrue(np.all(overlay[5, 10] < 60))
            self.assertGreater(int(overlay[10, 10][0]), 240)
            self.assertLess(int(overlay[10, 10][1]), 240)
            self.assertEqual(tuple(overlay[0, 0]), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
