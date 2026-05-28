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

    def test_transparent_png_can_skip_temp_composite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            png_path = workdir / "circle.png"
            image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
            image.putpixel((1, 1), (0, 128, 255, 255))
            image.putpixel((2, 2), (10, 20, 30, 128))
            image.save(png_path)

            opaque_path = normalize_image_for_processing(png_path, output_dir=workdir)
            opaque_rgb = load_rgb(opaque_path)
            image_path = normalize_image_for_processing(
                png_path,
                output_dir=workdir,
                composite_transparent_rasters=False,
            )
            rgb = load_rgb(image_path)

            self.assertEqual(image_path, png_path)
            self.assertEqual(tuple(rgb[0, 0]), (255, 255, 255))
            self.assertEqual(tuple(rgb[1, 1]), (0, 128, 255))
            np.testing.assert_array_equal(rgb, opaque_rgb)

    def test_opaque_rgba_png_loads_as_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            png_path = Path(tmp) / "opaque.png"
            image = Image.new("RGBA", (2, 1), (0, 0, 0, 255))
            image.putpixel((0, 0), (12, 34, 56, 255))
            image.putpixel((1, 0), (98, 76, 54, 255))
            image.save(png_path)

            rgb = load_rgb(png_path)

            self.assertEqual(rgb.shape, (1, 2, 3))
            self.assertEqual(tuple(rgb[0, 0]), (12, 34, 56))
            self.assertEqual(tuple(rgb[0, 1]), (98, 76, 54))


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

    def test_overlay_preview_can_be_bounded_for_inline_api_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "overlay.png"
            rgb = np.full((40, 20, 3), 255, dtype=np.uint8)
            mask = np.zeros((40, 20), dtype=bool)
            mask[10:30, 5:15] = True

            write_overlay_png("unused.png", mask, out_path, rgb=rgb, max_dimension=10)

            with Image.open(out_path) as overlay:
                self.assertEqual(overlay.size, (5, 10))


if __name__ == "__main__":
    unittest.main()
