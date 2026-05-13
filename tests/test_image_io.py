import tempfile
import unittest
from pathlib import Path

from map_boundary_builder.extract import load_rgb
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


if __name__ == "__main__":
    unittest.main()
