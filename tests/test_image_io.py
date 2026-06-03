import gzip
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, features

from map_boundary_builder.extract import load_rgb, write_overlay_image, write_overlay_png
from map_boundary_builder.image_io import normalize_image_for_processing, safe_image_extension, svg_rasterizer_diagnostics


class SvgImageIoTests(unittest.TestCase):
    def test_safe_image_extension_preserves_svg(self) -> None:
        self.assertEqual(safe_image_extension("mi.svg"), ".svg")
        self.assertEqual(safe_image_extension("mi.svgz"), ".svgz")

    def test_safe_image_extension_preserves_avif(self) -> None:
        self.assertEqual(safe_image_extension("map.avif"), ".avif")

    def test_safe_image_extension_preserves_gif(self) -> None:
        self.assertEqual(safe_image_extension("map.gif"), ".gif")

    def test_safe_image_extension_preserves_bmp(self) -> None:
        self.assertEqual(safe_image_extension("map.bmp"), ".bmp")

    def test_safe_image_extension_preserves_tiff(self) -> None:
        self.assertEqual(safe_image_extension("map.tif"), ".tif")
        self.assertEqual(safe_image_extension("map.tiff"), ".tiff")

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

    def test_svgz_is_rasterized_before_pillow_reads_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            svg_path = workdir / "input.svgz"
            svg_text = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">
<rect width="10" height="6" fill="#ffffff"/>
<rect x="2" y="1" width="6" height="4" fill="#00a6ff"/>
</svg>
"""
            svg_path.write_bytes(gzip.compress(svg_text.encode("utf-8")))

            image_path = normalize_image_for_processing(svg_path, output_dir=workdir)
            rgb = load_rgb(image_path)

            self.assertEqual(image_path.name, "input.raster.png")
            self.assertEqual(rgb.shape, (6, 10, 3))
            self.assertGreater(int(rgb[:, :, 2].max()), 200)

    def test_large_svg_can_be_rasterized_at_max_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            svg_path = workdir / "wide-map.png"
            svg_path.write_text(
                """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4000 2000">
<rect width="4000" height="2000" fill="#ffffff"/>
<rect x="1000" y="500" width="2000" height="1000" fill="#00a6ff"/>
</svg>
""",
                encoding="utf-8",
            )

            image_path = normalize_image_for_processing(
                svg_path,
                output_dir=workdir,
                svg_max_dimension=1000,
            )
            rgb = load_rgb(image_path)

            self.assertEqual(rgb.shape, (500, 1000, 3))
            self.assertGreater(int(rgb[:, :, 2].max()), 200)

    def test_svg_max_dimension_does_not_upscale_small_svgs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            svg_path = workdir / "small.svg"
            svg_path.write_text(
                """<svg xmlns="http://www.w3.org/2000/svg" width="12" height="8">
<rect width="12" height="8" fill="#ffffff"/>
<rect x="2" y="1" width="8" height="6" fill="#00a6ff"/>
</svg>
""",
                encoding="utf-8",
            )

            image_path = normalize_image_for_processing(
                svg_path,
                output_dir=workdir,
                svg_max_dimension=1000,
            )
            rgb = load_rgb(image_path)

            self.assertEqual(rgb.shape, (8, 12, 3))

    def test_svg_rasterization_prefers_resvg_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            svg_path = workdir / "input.svg"
            svg_path.write_text(
                """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4000 2000">
<rect width="4000" height="2000" fill="#ffffff"/>
<rect x="1000" y="500" width="2000" height="1000" fill="#00a6ff"/>
</svg>
""",
                encoding="utf-8",
            )
            png_bytes = png_fixture_bytes((1000, 500), rect=(250, 125, 750, 375))
            calls: list[dict[str, object]] = []

            def fake_import_module(name: str):
                if name == "resvg_py":
                    return SimpleNamespace(
                        svg_to_bytes=lambda **kwargs: calls.append(kwargs) or png_bytes,
                    )
                if name == "cairosvg":
                    raise AssertionError("resvg-py should be tried before CairoSVG")
                raise AssertionError(name)

            with patch("map_boundary_builder.image_io.importlib.import_module", side_effect=fake_import_module):
                image_path = normalize_image_for_processing(
                    svg_path,
                    output_dir=workdir,
                    svg_max_dimension=1000,
                )

            rgb = load_rgb(image_path)

            self.assertEqual(image_path.name, "input.raster.png")
            self.assertEqual(rgb.shape, (500, 1000, 3))
            self.assertEqual(calls[0]["width"], 1000)
            self.assertEqual(calls[0]["height"], 500)
            self.assertEqual(calls[0]["resources_dir"], str(workdir))
            self.assertGreater(int(rgb[:, :, 2].max()), 200)

    def test_svg_rasterization_falls_back_to_cairosvg_when_resvg_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            svg_path = workdir / "input.svg"
            svg_path.write_text(
                """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4000 2000">
<rect width="4000" height="2000" fill="#ffffff"/>
<rect x="1000" y="500" width="2000" height="1000" fill="#00a6ff"/>
</svg>
""",
                encoding="utf-8",
            )
            calls: list[dict[str, object]] = []
            png_bytes = png_fixture_bytes((1000, 500), rect=(250, 125, 750, 375))

            def fake_svg2png(**kwargs):
                calls.append(kwargs)
                Path(kwargs["write_to"]).write_bytes(png_bytes)

            def fake_import_module(name: str):
                if name == "resvg_py":
                    raise OSError("resvg unavailable")
                if name == "cairosvg":
                    return SimpleNamespace(svg2png=fake_svg2png)
                raise AssertionError(name)

            with patch("map_boundary_builder.image_io.importlib.import_module", side_effect=fake_import_module):
                image_path = normalize_image_for_processing(
                    svg_path,
                    output_dir=workdir,
                    svg_max_dimension=1000,
                )

            rgb = load_rgb(image_path)

            self.assertEqual(image_path.name, "input.raster.png")
            self.assertEqual(rgb.shape, (500, 1000, 3))
            self.assertEqual(calls[0]["output_width"], 1000)
            self.assertEqual(calls[0]["output_height"], 500)
            self.assertEqual(calls[0]["write_to"], str(image_path))
            self.assertGreater(int(rgb[:, :, 2].max()), 200)

    def test_svg_rasterizer_diagnostics_prefers_resvg_when_cairo_fails(self) -> None:
        png_bytes = png_fixture_bytes((4, 3), rect=(1, 1, 3, 2))
        svg_rasterizer_diagnostics.cache_clear()
        self.addCleanup(svg_rasterizer_diagnostics.cache_clear)

        def fake_import_module(name: str):
            if name == "cairosvg":
                raise OSError("libcairo missing")
            if name == "resvg_py":
                return SimpleNamespace(svg_to_bytes=lambda **_kwargs: png_bytes)
            raise AssertionError(name)

        with patch("map_boundary_builder.image_io.importlib.import_module", side_effect=fake_import_module):
            diagnostics = svg_rasterizer_diagnostics()

        self.assertTrue(diagnostics["ok"])
        self.assertEqual(diagnostics["preferred"], "resvg-py")
        self.assertFalse(diagnostics["cairosvg"]["ok"])
        self.assertIn("libcairo missing", diagnostics["cairosvg"]["error"])
        self.assertTrue(diagnostics["resvg_py"]["ok"])

    def test_svg_rasterizer_diagnostics_caches_defensive_copy(self) -> None:
        png_bytes = png_fixture_bytes((4, 3), rect=(1, 1, 3, 2))
        calls: list[str] = []
        svg_rasterizer_diagnostics.cache_clear()
        self.addCleanup(svg_rasterizer_diagnostics.cache_clear)

        def fake_import_module(name: str):
            calls.append(name)
            if name == "cairosvg":
                return SimpleNamespace(
                    svg2png=lambda **kwargs: kwargs["write_to"].write(png_bytes)
                )
            if name == "resvg_py":
                return SimpleNamespace(svg_to_bytes=lambda **_kwargs: png_bytes)
            raise AssertionError(name)

        with patch("map_boundary_builder.image_io.importlib.import_module", side_effect=fake_import_module):
            first = svg_rasterizer_diagnostics()
            first["preferred"] = "mutated"
            first["resvg_py"]["ok"] = False
            second = svg_rasterizer_diagnostics()

        self.assertEqual(calls, ["cairosvg", "resvg_py"])
        self.assertEqual(second["preferred"], "resvg-py")
        self.assertTrue(second["resvg_py"]["ok"])
        self.assertIsNot(first, second)
        self.assertIsNot(first["resvg_py"], second["resvg_py"])

    def test_svg_rasterizer_diagnostics_retries_failed_probe(self) -> None:
        png_bytes = png_fixture_bytes((4, 3), rect=(1, 1, 3, 2))
        calls: list[str] = []
        fail_resvg_once = True
        svg_rasterizer_diagnostics.cache_clear()
        self.addCleanup(svg_rasterizer_diagnostics.cache_clear)

        def fake_import_module(name: str):
            nonlocal fail_resvg_once
            calls.append(name)
            if name == "cairosvg":
                raise OSError("libcairo missing")
            if name == "resvg_py":
                if fail_resvg_once:
                    fail_resvg_once = False
                    raise RuntimeError("transient resvg failure")
                return SimpleNamespace(svg_to_bytes=lambda **_kwargs: png_bytes)
            raise AssertionError(name)

        with patch("map_boundary_builder.image_io.importlib.import_module", side_effect=fake_import_module):
            first = svg_rasterizer_diagnostics()
            second = svg_rasterizer_diagnostics()

        self.assertEqual(calls, ["cairosvg", "resvg_py", "cairosvg", "resvg_py"])
        self.assertFalse(first["ok"])
        self.assertIsNone(first["preferred"])
        self.assertTrue(second["ok"])
        self.assertEqual(second["preferred"], "resvg-py")

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

    def test_webp_loads_as_rgb_without_pillow_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            webp_path = Path(tmp) / "opaque.webp"
            image = Image.new("RGB", (2, 1), (0, 0, 0))
            image.putpixel((0, 0), (12, 34, 56))
            image.putpixel((1, 0), (98, 76, 54))
            image.save(webp_path, format="WEBP", lossless=True)

            rgb = load_rgb(webp_path)

            self.assertEqual(rgb.shape, (1, 2, 3))
            self.assertEqual(tuple(rgb[0, 0]), (12, 34, 56))
            self.assertEqual(tuple(rgb[0, 1]), (98, 76, 54))

    def test_transparent_gif_falls_back_to_white_composite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gif_path = Path(tmp) / "transparent.gif"
            image = Image.new("P", (2, 1), 0)
            image.putpalette([0, 0, 0, 0, 128, 255] + [0, 0, 0] * 254)
            image.info["transparency"] = 0
            image.putpixel((1, 0), 1)
            image.save(gif_path, format="GIF", transparency=0)

            rgb = load_rgb(gif_path)

            self.assertEqual(rgb.shape, (1, 2, 3))
            self.assertEqual(tuple(rgb[0, 0]), (255, 255, 255))
            self.assertEqual(tuple(rgb[0, 1]), (0, 128, 255))

    def test_bmp_loads_as_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bmp_path = Path(tmp) / "opaque.bmp"
            image = Image.new("RGB", (2, 1), (0, 0, 0))
            image.putpixel((0, 0), (12, 34, 56))
            image.putpixel((1, 0), (98, 76, 54))
            image.save(bmp_path, format="BMP")

            rgb = load_rgb(bmp_path)

            self.assertEqual(rgb.shape, (1, 2, 3))
            self.assertEqual(tuple(rgb[0, 0]), (12, 34, 56))
            self.assertEqual(tuple(rgb[0, 1]), (98, 76, 54))

    def test_tiff_loads_as_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = Path(tmp) / "opaque.tiff"
            image = Image.new("RGB", (2, 1), (0, 0, 0))
            image.putpixel((0, 0), (12, 34, 56))
            image.putpixel((1, 0), (98, 76, 54))
            image.save(tiff_path, format="TIFF")

            rgb = load_rgb(tiff_path)

            self.assertEqual(rgb.shape, (1, 2, 3))
            self.assertEqual(tuple(rgb[0, 0]), (12, 34, 56))
            self.assertEqual(tuple(rgb[0, 1]), (98, 76, 54))

    def test_transparent_webp_falls_back_to_white_composite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            webp_path = Path(tmp) / "transparent.webp"
            image = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
            image.putpixel((0, 0), (0, 128, 255, 255))
            image.putpixel((1, 0), (0, 0, 0, 0))
            image.save(webp_path, format="WEBP", lossless=True)

            rgb = load_rgb(webp_path)

            self.assertEqual(rgb.shape, (1, 2, 3))
            self.assertEqual(tuple(rgb[0, 0]), (0, 128, 255))
            self.assertEqual(tuple(rgb[0, 1]), (255, 255, 255))

    @unittest.skipUnless(features.check("avif"), "Pillow AVIF support required")
    def test_avif_loads_as_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            avif_path = Path(tmp) / "opaque.avif"
            image = Image.new("RGB", (16, 8), (12, 34, 56))
            for x in range(8, 16):
                for y in range(8):
                    image.putpixel((x, y), (98, 76, 54))
            image.save(avif_path, format="AVIF", quality=100)

            rgb = load_rgb(avif_path)

            self.assertEqual(rgb.shape, (8, 16, 3))
            self.assertLess(np.abs(rgb[:, :8].mean(axis=(0, 1)) - np.array([12, 34, 56])).max(), 8)
            self.assertLess(np.abs(rgb[:, 8:].mean(axis=(0, 1)) - np.array([98, 76, 54])).max(), 8)


def png_fixture_bytes(size: tuple[int, int], *, rect: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGB", size, (255, 255, 255))
    for y in range(rect[1], rect[3]):
        for x in range(rect[0], rect[2]):
            image.putpixel((x, y), (0, 166, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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

    def test_overlay_preview_resizes_caller_rgb_to_mask_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "overlay.png"
            rgb = np.full((12, 8, 3), 255, dtype=np.uint8)
            mask = np.zeros((40, 20), dtype=bool)
            mask[10:30, 5:15] = True

            write_overlay_png("unused.png", mask, out_path, rgb=rgb)

            with Image.open(out_path) as overlay:
                self.assertEqual(overlay.size, (20, 40))

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support required")
    def test_overlay_preview_can_write_webp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "overlay.webp"
            rgb = np.full((20, 20, 3), 255, dtype=np.uint8)
            mask = np.zeros((20, 20), dtype=bool)
            mask[5:15, 5:15] = True

            write_overlay_image("unused.png", mask, out_path, rgb=rgb)

            with Image.open(out_path) as overlay:
                self.assertEqual(overlay.format, "WEBP")
                self.assertEqual(overlay.size, (20, 20))


if __name__ == "__main__":
    unittest.main()
