import unittest
import base64
import builtins
import gzip
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, PngImagePlugin, features

import api.index as api_index
from api.index import (
    CRON_WARM_PATH,
    CRON_WARM_PATHS,
    LEGACY_CRON_WARM_PATH,
    INLINE_OVERLAY_OPTIMIZE_BYTES,
    allow_catalog_for_request,
    authorized_cron_request,
    bool_field,
    cached_run_payload,
    cached_run_response_status,
    cron_warm_generation_payload,
    event_stage_elapsed_seconds,
    generation_error_payload,
    generation_error_status,
    health_response_status,
    health_payload,
    include_overlay_for_request,
    inline_overlay,
    jpeg_commentless_run_result_cache_key,
    jpeg_commentless_sha256,
    jpeg_visual_run_result_cache_key,
    jpeg_visual_sha256,
    json_response_body,
    filename_hint_cache_value,
    normalized_image_sha256,
    png_visual_run_result_cache_key,
    png_visual_sha256,
    _RUN_RESULT_MEMORY_CACHE,
    raw_run_result_cache_key,
    remember_run_result_cache,
    read_run_result_cache,
    RUN_RESULT_MEMORY_CACHE_MAX_BYTES,
    RUN_RESULT_MEMORY_CACHE_MAX,
    run_result_cache_tmp_path,
    run_result_cache_key,
    webp_visual_run_result_cache_key,
    webp_visual_sha256,
    write_run_result_cache,
)
from map_boundary_builder.asset_response import web_asset_response, web_asset_version
from map_boundary_builder.runner import (
    BoundaryBuildOptions,
    CatalogProbeMiss,
    catalog_matching_enabled,
    should_overlap_ocr_with_extraction,
    should_try_pre_ocr_catalog,
)


def jpeg_bytes(image: Image.Image, **save_options: object) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", **save_options)
    return buffer.getvalue()


def insert_jpeg_segment(image_bytes: bytes, marker: int, payload: bytes) -> bytes:
    segment_length = len(payload) + 2
    return image_bytes[:2] + bytes((0xFF, marker)) + segment_length.to_bytes(2, "big") + payload + image_bytes[2:]


def insert_webp_chunk(image_bytes: bytes, chunk_type: bytes, payload: bytes) -> bytes:
    assert len(chunk_type) == 4
    chunk = chunk_type + len(payload).to_bytes(4, "little") + payload
    if len(payload) % 2:
        chunk += b"\x00"
    riff_size = int.from_bytes(image_bytes[4:8], "little") + len(chunk)
    return image_bytes[:4] + riff_size.to_bytes(4, "little") + image_bytes[8:] + chunk


class ApiRunCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        _RUN_RESULT_MEMORY_CACHE.clear()

    def test_catalog_matching_defaults_on_for_api_options_namespace(self) -> None:
        self.assertTrue(catalog_matching_enabled(SimpleNamespace()))
        self.assertFalse(catalog_matching_enabled(SimpleNamespace(allow_catalog=False)))

    def test_api_safe_extension_allows_avif(self) -> None:
        self.assertEqual(api_index.safe_extension("upload.avif"), ".avif")

    def test_api_safe_extension_allows_gif(self) -> None:
        self.assertEqual(api_index.safe_extension("upload.gif"), ".gif")

    def test_api_safe_extension_allows_bmp(self) -> None:
        self.assertEqual(api_index.safe_extension("upload.bmp"), ".bmp")

    def test_api_safe_extension_allows_tiff(self) -> None:
        self.assertEqual(api_index.safe_extension("upload.tif"), ".tif")
        self.assertEqual(api_index.safe_extension("upload.tiff"), ".tiff")

    def test_api_safe_extension_allows_svgz(self) -> None:
        self.assertEqual(api_index.safe_extension("upload.svgz"), ".svgz")

    def test_ocr_overlap_only_when_pre_ocr_catalog_cannot_return(self) -> None:
        self.assertFalse(should_overlap_ocr_with_extraction(city_input=None, allow_catalog=True))
        self.assertTrue(should_overlap_ocr_with_extraction(city_input=None, allow_catalog=False))
        self.assertFalse(
            should_overlap_ocr_with_extraction(
                city_input=None,
                allow_catalog=True,
                filename_hint="Waymo Phoenix",
            )
        )
        self.assertFalse(
            should_overlap_ocr_with_extraction(
                city_input=None,
                allow_catalog=True,
                filename_hint="Waymo Miami",
            )
        )
        self.assertFalse(
            should_overlap_ocr_with_extraction(
                city_input=None,
                allow_catalog=True,
                filename_hint="Waymo Bay Area",
            )
        )
        self.assertFalse(
            should_overlap_ocr_with_extraction(
                city_input=None,
                allow_catalog=True,
                filename_hint="Zoox San Francisco",
            )
        )
        self.assertFalse(
            should_overlap_ocr_with_extraction(
                city_input=None,
                allow_catalog=True,
                filename_hint="Tesla Bay Area",
            )
        )
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Phoenix", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Houston", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Bay Area", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Tesla Houston", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Zoox San Francisco", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Tesla Bay Area", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Miami", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Santa Monica", allow_catalog=True))
        self.assertTrue(should_overlap_ocr_with_extraction(city_input="Atlantis", allow_catalog=True))
        self.assertTrue(should_overlap_ocr_with_extraction(city_input="Phoenix", allow_catalog=False))

    def test_pre_ocr_catalog_only_runs_when_it_can_match(self) -> None:
        self.assertTrue(should_try_pre_ocr_catalog(city_input=None, allow_catalog=True))
        self.assertTrue(
            should_try_pre_ocr_catalog(
                city_input=None,
                allow_catalog=True,
                filename_hint="Waymo Phoenix",
            )
        )
        self.assertTrue(
            should_try_pre_ocr_catalog(
                city_input=None,
                allow_catalog=True,
                filename_hint="Waymo Miami",
            )
        )
        self.assertTrue(
            should_try_pre_ocr_catalog(
                city_input=None,
                allow_catalog=True,
                filename_hint="Waymo Bay Area",
            )
        )
        self.assertTrue(
            should_try_pre_ocr_catalog(
                city_input=None,
                allow_catalog=True,
                filename_hint="Zoox San Francisco",
            )
        )
        self.assertTrue(
            should_try_pre_ocr_catalog(
                city_input=None,
                allow_catalog=True,
                filename_hint="Tesla Bay Area",
            )
        )
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Phoenix", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Houston", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Bay Area", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Tesla Houston", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Zoox San Francisco", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Tesla Bay Area", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Miami", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Santa Monica", allow_catalog=True))
        self.assertFalse(should_try_pre_ocr_catalog(city_input="Atlantis", allow_catalog=True))
        self.assertFalse(should_try_pre_ocr_catalog(city_input="Phoenix", allow_catalog=False))

    def test_run_cache_key_depends_on_image_and_options(self) -> None:
        base = run_result_cache_key(b"image-a", None, BoundaryBuildOptions())
        changed_image = run_result_cache_key(b"image-b", None, BoundaryBuildOptions())
        changed_city = run_result_cache_key(b"image-a", "Miami", BoundaryBuildOptions())
        changed_options = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(min_control_points=4),
        )
        changed_filename = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(filename_hint="Phoenix.png"),
        )
        changed_preview_options = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(preview_max_dimension=1200),
        )
        changed_overlay_format = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(overlay_format="webp"),
        )
        changed_mask_options = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(write_mask_artifact=False),
        )
        changed_catalog_probe_options = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(catalog_probe_only=True),
        )
        changed_catalog_probe_missed_options = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(catalog_probe_missed=True),
        )
        changed_catalog_probe_miss_low_iou_options = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(catalog_probe_missed=True, catalog_probe_miss_low_iou=True),
        )
        changed_allow_catalog = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(allow_catalog=False),
        )
        changed_overlay_mode = run_result_cache_key(
            b"image-a",
            None,
            SimpleNamespace(
                simplify_px=BoundaryBuildOptions().simplify_px,
                min_confidence=BoundaryBuildOptions().min_confidence,
                min_control_points=BoundaryBuildOptions().min_control_points,
                include_overlay=False,
                preview_max_dimension=None,
                write_mask_artifact=False,
            ),
        )

        self.assertNotEqual(base, changed_image)
        self.assertNotEqual(base, changed_city)
        self.assertNotEqual(base, changed_options)
        self.assertNotEqual(base, changed_filename)
        self.assertNotEqual(base, changed_preview_options)
        self.assertNotEqual(base, changed_overlay_format)
        self.assertNotEqual(base, changed_mask_options)
        self.assertNotEqual(base, changed_catalog_probe_options)
        self.assertNotEqual(base, changed_catalog_probe_missed_options)
        self.assertNotEqual(changed_catalog_probe_missed_options, changed_catalog_probe_miss_low_iou_options)
        self.assertNotEqual(base, changed_allow_catalog)
        self.assertNotEqual(base, changed_overlay_mode)

    def test_run_cache_filename_hint_uses_semantic_basename(self) -> None:
        options = BoundaryBuildOptions(filename_hint="/tmp/uploads/Dallas.png")
        same_basename = BoundaryBuildOptions(filename_hint="Dallas.png")
        different_basename = BoundaryBuildOptions(filename_hint="Phoenix.png")
        same_context_cache_bust = BoundaryBuildOptions(
            filename_hint="Dallas pipeline-version-1780067151-e527924-ui.png"
        )
        same_generic_probe_words = BoundaryBuildOptions(
            filename_hint="Dallas map repeat after-roadskip-1780067151.png"
        )

        self.assertEqual(filename_hint_cache_value(options.filename_hint), "png:dallas")
        self.assertEqual(
            run_result_cache_key(b"image-a", None, options),
            run_result_cache_key(b"image-a", None, same_basename),
        )
        self.assertEqual(
            run_result_cache_key(b"image-a", None, options),
            run_result_cache_key(b"image-a", None, same_context_cache_bust),
        )
        self.assertEqual(
            run_result_cache_key(b"image-a", None, options),
            run_result_cache_key(b"image-a", None, same_generic_probe_words),
        )
        self.assertEqual(
            raw_run_result_cache_key(b"image-a", None, options),
            raw_run_result_cache_key(b"image-a", None, same_generic_probe_words),
        )
        self.assertNotEqual(
            run_result_cache_key(b"image-a", None, options),
            run_result_cache_key(b"image-a", None, different_basename),
        )

    def test_run_cache_filename_hint_ignores_generic_probe_tokens(self) -> None:
        self.assertEqual(
            filename_hint_cache_value("neutral-map-after-roadskip-1780146244.webp"),
            "webp:",
        )
        self.assertEqual(filename_hint_cache_value("neutral-map-1780145995.webp"), "webp:")
        self.assertEqual(filename_hint_cache_value("uploaded-map-variant.avif"), "avif:")
        self.assertEqual(filename_hint_cache_value("uploaded-map-variant.bmp"), "bmp:")
        self.assertEqual(filename_hint_cache_value("uploaded-map.png"), "png:")
        self.assertEqual(
            filename_hint_cache_value("dallas-map-repeat-1780146013-1.webp"),
            "webp:dallas",
        )

    def test_run_cache_filename_hint_preserves_provider_and_multiword_area(self) -> None:
        waymo = BoundaryBuildOptions(filename_hint="Waymo Bay Area screenshot-1780067151.png")
        tesla = BoundaryBuildOptions(filename_hint="Tesla Bay Area screenshot-1780067151.png")

        self.assertEqual(filename_hint_cache_value(waymo.filename_hint), "png:waymo bay area")
        self.assertEqual(filename_hint_cache_value(tesla.filename_hint), "png:tesla bay area")
        self.assertNotEqual(
            run_result_cache_key(b"image-a", None, waymo),
            run_result_cache_key(b"image-a", None, tesla),
        )

    def test_run_cache_key_depends_on_pipeline_version(self) -> None:
        with patch("api.index.get_pipeline_version", return_value="pipeline-a"):
            first = run_result_cache_key(b"image-a", None, BoundaryBuildOptions())
        with patch("api.index.get_pipeline_version", return_value="pipeline-b"):
            second = run_result_cache_key(b"image-a", None, BoundaryBuildOptions())

        self.assertNotEqual(first, second)

    def test_run_cache_key_depends_on_ocr_runtime_config(self) -> None:
        with patch("api.index.ocr_runtime_config", return_value={"rapidocr_rec_batch_num": 24}):
            first = run_result_cache_key(b"image-a", None, BoundaryBuildOptions())
        with patch("api.index.ocr_runtime_config", return_value={"rapidocr_rec_batch_num": 32}):
            second = run_result_cache_key(b"image-a", None, BoundaryBuildOptions())

        self.assertNotEqual(first, second)

    def test_run_cache_key_depends_on_generation_runtime_env(self) -> None:
        with (
            patch.dict("os.environ", {"MAP_BOUNDARY_GENERAL_EXTRACT_MAX_DIMENSION": "1600"}, clear=True),
            patch("api.index.ocr_runtime_config", return_value={}),
        ):
            first = run_result_cache_key(b"image-a", None, BoundaryBuildOptions())
        with (
            patch.dict("os.environ", {"MAP_BOUNDARY_GENERAL_EXTRACT_MAX_DIMENSION": "1200"}, clear=True),
            patch("api.index.ocr_runtime_config", return_value={}),
        ):
            second = run_result_cache_key(b"image-a", None, BoundaryBuildOptions())

        self.assertNotEqual(first, second)

    def test_generation_runtime_env_config_defaults_and_overrides(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            defaults = api_index.generation_runtime_env_config()
        with patch.dict("os.environ", {"MAP_BOUNDARY_GEOCODE_WORKERS": "1"}, clear=True):
            changed = api_index.generation_runtime_env_config()

        self.assertEqual(defaults["MAP_BOUNDARY_GEOCODE_WORKERS"], "6")
        self.assertEqual(changed["MAP_BOUNDARY_GEOCODE_WORKERS"], "1")

    def test_run_cache_key_uses_decoded_pixels(self) -> None:
        first = BytesIO()
        second = BytesIO()
        image = Image.new("RGBA", (3, 2), (12, 34, 56, 255))
        image.save(first, format="PNG", compress_level=0)
        image.save(second, format="PNG", compress_level=9)

        self.assertNotEqual(first.getvalue(), second.getvalue())
        self.assertEqual(
            normalized_image_sha256(first.getvalue()),
            normalized_image_sha256(second.getvalue()),
        )
        self.assertEqual(
            run_result_cache_key(first.getvalue(), None, BoundaryBuildOptions()),
            run_result_cache_key(second.getvalue(), None, BoundaryBuildOptions()),
        )
        self.assertNotEqual(
            raw_run_result_cache_key(first.getvalue(), None, BoundaryBuildOptions()),
            raw_run_result_cache_key(second.getvalue(), None, BoundaryBuildOptions()),
        )

    def test_png_visual_hash_ignores_text_metadata_only(self) -> None:
        first = BytesIO()
        second = BytesIO()
        changed_pixel = BytesIO()
        first_metadata = PngImagePlugin.PngInfo()
        first_metadata.add_text("probe", "first")
        second_metadata = PngImagePlugin.PngInfo()
        second_metadata.add_text("probe", "second")
        Image.new("RGBA", (3, 2), (12, 34, 56, 255)).save(first, format="PNG", pnginfo=first_metadata)
        Image.new("RGBA", (3, 2), (12, 34, 56, 255)).save(second, format="PNG", pnginfo=second_metadata)
        Image.new("RGBA", (3, 2), (12, 34, 57, 255)).save(changed_pixel, format="PNG", pnginfo=first_metadata)

        self.assertNotEqual(first.getvalue(), second.getvalue())
        self.assertEqual(png_visual_sha256(first.getvalue()), png_visual_sha256(second.getvalue()))
        self.assertNotEqual(png_visual_sha256(first.getvalue()), png_visual_sha256(changed_pixel.getvalue()))
        self.assertIsNone(png_visual_sha256(b"not a png"))

    def test_png_visual_run_cache_key_ignores_text_metadata_but_keeps_options(self) -> None:
        first = BytesIO()
        second = BytesIO()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("cache_bust", "a")
        Image.new("RGBA", (3, 2), (12, 34, 56, 255)).save(first, format="PNG", pnginfo=metadata)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("cache_bust", "b")
        Image.new("RGBA", (3, 2), (12, 34, 56, 255)).save(second, format="PNG", pnginfo=metadata)

        self.assertNotEqual(
            raw_run_result_cache_key(first.getvalue(), None, BoundaryBuildOptions()),
            raw_run_result_cache_key(second.getvalue(), None, BoundaryBuildOptions()),
        )
        self.assertEqual(
            png_visual_run_result_cache_key(first.getvalue(), None, BoundaryBuildOptions()),
            png_visual_run_result_cache_key(second.getvalue(), None, BoundaryBuildOptions()),
        )
        self.assertNotEqual(
            png_visual_run_result_cache_key(first.getvalue(), None, BoundaryBuildOptions()),
            png_visual_run_result_cache_key(first.getvalue(), "Dallas", BoundaryBuildOptions()),
        )
        self.assertIsNone(png_visual_run_result_cache_key(b"not a png", None, BoundaryBuildOptions()))

    def test_jpeg_commentless_hash_ignores_comments_only(self) -> None:
        base = jpeg_bytes(Image.new("RGB", (4, 3), (12, 34, 56)), quality=95)
        first = insert_jpeg_segment(base, 0xFE, b"first comment")
        second = insert_jpeg_segment(base, 0xFE, b"second comment")
        changed_pixel = insert_jpeg_segment(
            jpeg_bytes(Image.new("RGB", (4, 3), (200, 34, 57)), quality=95),
            0xFE,
            b"first comment",
        )
        first_exif = insert_jpeg_segment(base, 0xE1, b"Exif\x00\x00first")
        second_exif = insert_jpeg_segment(base, 0xE1, b"Exif\x00\x00second")

        self.assertNotEqual(first, second)
        self.assertEqual(jpeg_commentless_sha256(first), jpeg_commentless_sha256(second))
        self.assertNotEqual(jpeg_commentless_sha256(first), jpeg_commentless_sha256(changed_pixel))
        self.assertNotEqual(jpeg_commentless_sha256(first_exif), jpeg_commentless_sha256(second_exif))
        self.assertIsNone(jpeg_commentless_sha256(b"not a jpeg"))
        self.assertIsNone(jpeg_commentless_sha256(base[:-2]))

    def test_jpeg_commentless_run_cache_key_ignores_comments_but_keeps_options(self) -> None:
        base = jpeg_bytes(Image.new("RGB", (4, 3), (12, 34, 56)), quality=95)
        first = insert_jpeg_segment(base, 0xFE, b"cache bust a")
        second = insert_jpeg_segment(base, 0xFE, b"cache bust b")

        self.assertNotEqual(
            raw_run_result_cache_key(first, None, BoundaryBuildOptions()),
            raw_run_result_cache_key(second, None, BoundaryBuildOptions()),
        )
        self.assertEqual(
            jpeg_commentless_run_result_cache_key(first, None, BoundaryBuildOptions()),
            jpeg_commentless_run_result_cache_key(second, None, BoundaryBuildOptions()),
        )
        self.assertNotEqual(
            jpeg_commentless_run_result_cache_key(first, None, BoundaryBuildOptions()),
            jpeg_commentless_run_result_cache_key(first, "Dallas", BoundaryBuildOptions()),
        )
        self.assertIsNone(jpeg_commentless_run_result_cache_key(b"not a jpeg", None, BoundaryBuildOptions()))

    def test_jpeg_visual_hash_ignores_metadata_only(self) -> None:
        base = jpeg_bytes(Image.new("RGB", (4, 3), (12, 34, 56)), quality=95)
        changed_pixel = jpeg_bytes(Image.new("RGB", (4, 3), (200, 34, 57)), quality=95)
        first_comment = insert_jpeg_segment(base, 0xFE, b"first comment")
        second_exif = insert_jpeg_segment(base, 0xE1, b"Exif\x00\x00second")
        third_xmp = insert_jpeg_segment(base, 0xE1, b"http://ns.adobe.com/xap/1.0/\x00third")
        color_profile = insert_jpeg_segment(base, 0xE2, b"ICC_PROFILE\x00profile")
        unknown_app1 = insert_jpeg_segment(base, 0xE1, b"unknown app1 payload")

        self.assertNotEqual(first_comment, second_exif)
        self.assertNotEqual(second_exif, third_xmp)
        self.assertEqual(jpeg_visual_sha256(first_comment), jpeg_visual_sha256(second_exif))
        self.assertEqual(jpeg_visual_sha256(first_comment), jpeg_visual_sha256(third_xmp))
        self.assertNotEqual(jpeg_visual_sha256(first_comment), jpeg_visual_sha256(changed_pixel))
        self.assertNotEqual(jpeg_visual_sha256(base), jpeg_visual_sha256(color_profile))
        self.assertNotEqual(jpeg_visual_sha256(base), jpeg_visual_sha256(unknown_app1))
        self.assertIsNone(jpeg_visual_sha256(b"not a jpeg"))
        self.assertIsNone(jpeg_visual_sha256(base[:-2]))

    def test_jpeg_visual_run_cache_key_ignores_metadata_but_keeps_options(self) -> None:
        base = jpeg_bytes(Image.new("RGB", (4, 3), (12, 34, 56)), quality=95)
        first = insert_jpeg_segment(base, 0xE1, b"Exif\x00\x00cache bust a")
        second = insert_jpeg_segment(base, 0xE1, b"http://ns.adobe.com/xap/1.0/\x00cache bust b")

        self.assertNotEqual(
            raw_run_result_cache_key(first, None, BoundaryBuildOptions()),
            raw_run_result_cache_key(second, None, BoundaryBuildOptions()),
        )
        self.assertNotEqual(
            jpeg_commentless_run_result_cache_key(first, None, BoundaryBuildOptions()),
            jpeg_commentless_run_result_cache_key(second, None, BoundaryBuildOptions()),
        )
        self.assertEqual(
            jpeg_visual_run_result_cache_key(first, None, BoundaryBuildOptions()),
            jpeg_visual_run_result_cache_key(second, None, BoundaryBuildOptions()),
        )
        self.assertNotEqual(
            jpeg_visual_run_result_cache_key(first, None, BoundaryBuildOptions()),
            jpeg_visual_run_result_cache_key(first, "Dallas", BoundaryBuildOptions()),
        )
        self.assertIsNone(jpeg_visual_run_result_cache_key(b"not a jpeg", None, BoundaryBuildOptions()))

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support required")
    def test_webp_visual_hash_ignores_metadata_only(self) -> None:
        base = BytesIO()
        changed = BytesIO()
        Image.new("RGB", (4, 3), (12, 34, 56)).save(base, format="WEBP", lossless=True)
        Image.new("RGB", (4, 3), (12, 34, 57)).save(changed, format="WEBP", lossless=True)
        first = insert_webp_chunk(base.getvalue(), b"EXIF", b"first metadata")
        second = insert_webp_chunk(base.getvalue(), b"XMP ", b"second metadata")
        color_profile = insert_webp_chunk(base.getvalue(), b"ICCP", b"profile")

        self.assertNotEqual(first, second)
        self.assertEqual(webp_visual_sha256(first), webp_visual_sha256(second))
        self.assertNotEqual(webp_visual_sha256(first), webp_visual_sha256(changed.getvalue()))
        self.assertNotEqual(webp_visual_sha256(base.getvalue()), webp_visual_sha256(color_profile))
        self.assertIsNone(webp_visual_sha256(b"not a webp"))
        self.assertIsNone(webp_visual_sha256(base.getvalue()[:-1]))

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support required")
    def test_webp_visual_run_cache_key_ignores_metadata_but_keeps_options(self) -> None:
        base = BytesIO()
        Image.new("RGB", (4, 3), (12, 34, 56)).save(base, format="WEBP", lossless=True)
        first = insert_webp_chunk(base.getvalue(), b"EXIF", b"cache bust a")
        second = insert_webp_chunk(base.getvalue(), b"XMP ", b"cache bust b")

        self.assertNotEqual(
            raw_run_result_cache_key(first, None, BoundaryBuildOptions()),
            raw_run_result_cache_key(second, None, BoundaryBuildOptions()),
        )
        self.assertEqual(
            webp_visual_run_result_cache_key(first, None, BoundaryBuildOptions()),
            webp_visual_run_result_cache_key(second, None, BoundaryBuildOptions()),
        )
        self.assertNotEqual(
            webp_visual_run_result_cache_key(first, None, BoundaryBuildOptions()),
            webp_visual_run_result_cache_key(first, "Dallas", BoundaryBuildOptions()),
        )
        self.assertIsNone(webp_visual_run_result_cache_key(b"not a webp", None, BoundaryBuildOptions()))

    def test_run_cache_round_trip_and_payload_rehydration(self) -> None:
        cache_key = run_result_cache_key(b"unit-cache-image", None, BoundaryBuildOptions())
        cached = {
            "city": "Miami",
            "summary": {"city": "Miami", "combined_confidence": 0.93},
            "artifacts": {"geojson_inline": {"type": "FeatureCollection", "features": []}},
        }

        write_run_result_cache(cache_key, cached)
        restored = read_run_result_cache(cache_key)
        payload = cached_run_payload(
            restored or {},
            "1234-abcd",
            "Miami.png",
            [{"stage": "queued", "message": "Run queued", "percent": 1, "status": "queued"}],
        )

        self.assertEqual(restored, cached)
        self.assertEqual(payload["id"], "1234-abcd")
        self.assertTrue(payload["cached"])
        self.assertEqual(payload["filename"], "Miami.png")
        self.assertEqual(payload["events"][-1]["message"], "Boundary export ready from cache")
        self.assertEqual(cached_run_response_status(payload), HTTPStatus.CREATED)

    def test_failed_run_cache_round_trip_and_payload_rehydration(self) -> None:
        cache_key = raw_run_result_cache_key(b"unit-cache-image", None, BoundaryBuildOptions())
        failed_payload = {
            "status": "failed",
            "error": "Could not infer a reliable map location.",
            "profile": {"build_boundary_s": 2.0},
            "events": [{"stage": "ocr"}],
        }

        write_run_result_cache(cache_key, failed_payload)
        restored = read_run_result_cache(cache_key)
        payload = cached_run_payload(
            restored or {},
            "1234-failed",
            "Bad.png",
            [{"stage": "queued", "message": "Run queued", "percent": 1, "status": "queued"}],
        )

        self.assertEqual(
            restored,
            {"status": "failed", "error": "Could not infer a reliable map location."},
        )
        self.assertEqual(payload["id"], "1234-failed")
        self.assertTrue(payload["cached"])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["filename"], "Bad.png")
        self.assertIn("Could not infer", payload["error"])
        self.assertEqual(payload["events"][-1]["message"], "Generation failure ready from cache")
        self.assertEqual(
            payload["events"][-1]["details"],
            {"error": "Could not infer a reliable map location."},
        )
        self.assertEqual(cached_run_response_status(payload), HTTPStatus.UNPROCESSABLE_ENTITY)

    def test_catalog_miss_cache_round_trip_and_payload_rehydration(self) -> None:
        cache_key = raw_run_result_cache_key(
            b"unit-cache-image",
            "Bay Area",
            BoundaryBuildOptions(catalog_probe_only=True),
        )
        miss_payload = {
            "status": "catalog_miss",
            "error": "No known service-area shape matched the catalog probe.",
            "catalog_probe_miss": {
                "active_shape_iou_is_low": False,
                "best_active_catalog_slug": "bay-area-waymo",
                "best_active_catalog_iou": 0.91,
            },
            "profile": {"build_boundary_s": 0.05},
            "events": [{"stage": "extract"}],
        }

        write_run_result_cache(cache_key, miss_payload)
        restored = read_run_result_cache(cache_key)
        payload = cached_run_payload(
            restored or {},
            "1234-miss",
            "Bay Area probe.jpg",
            [{"stage": "queued", "message": "Run queued", "percent": 1, "status": "queued"}],
        )

        self.assertEqual(
            restored,
            {
                "status": "catalog_miss",
                "error": "No known service-area shape matched the catalog probe.",
                "catalog_probe_miss": {
                    "active_shape_iou_is_low": False,
                    "best_active_catalog_slug": "bay-area-waymo",
                    "best_active_catalog_iou": 0.91,
                },
            },
        )
        self.assertEqual(payload["id"], "1234-miss")
        self.assertTrue(payload["cached"])
        self.assertEqual(payload["status"], "catalog_miss")
        self.assertEqual(payload["catalog_probe_miss"]["best_active_catalog_slug"], "bay-area-waymo")
        self.assertEqual(payload["filename"], "Bay Area probe.jpg")
        self.assertIn("No known service-area shape", payload["error"])
        self.assertEqual(payload["events"][-1]["message"], "Catalog miss ready from cache")
        self.assertEqual(payload["events"][-1]["details"]["best_active_catalog_slug"], "bay-area-waymo")
        self.assertEqual(cached_run_response_status(payload), HTTPStatus.OK)

    def test_run_cache_uses_memory_cache_before_disk(self) -> None:
        cached = {
            "city": "Dallas",
            "summary": {"city": "Dallas", "combined_confidence": 0.84},
            "artifacts": {"geojson_inline": {"type": "FeatureCollection", "features": []}},
        }

        remember_run_result_cache("memory-only", cached)
        restored = read_run_result_cache("memory-only")

        self.assertEqual(restored, cached)
        assert restored is not None
        restored["summary"]["city"] = "Changed"
        self.assertEqual(read_run_result_cache("memory-only"), cached)
        self.assertIsInstance(_RUN_RESULT_MEMORY_CACHE["memory-only"], str)

    def test_run_memory_cache_evicts_oldest_entries(self) -> None:
        for index in range(RUN_RESULT_MEMORY_CACHE_MAX + 1):
            remember_run_result_cache(
                f"key-{index}",
                {"city": str(index), "summary": {}, "artifacts": {}},
            )

        self.assertNotIn("key-0", _RUN_RESULT_MEMORY_CACHE)
        self.assertIn(f"key-{RUN_RESULT_MEMORY_CACHE_MAX}", _RUN_RESULT_MEMORY_CACHE)

    def test_run_memory_cache_skips_oversized_entries(self) -> None:
        remember_run_result_cache(
            "large-overlay",
            {
                "city": "Dallas",
                "summary": {},
                "artifacts": {"overlay_svg": "x" * RUN_RESULT_MEMORY_CACHE_MAX_BYTES},
            },
        )

        self.assertNotIn("large-overlay", _RUN_RESULT_MEMORY_CACHE)

    def test_run_memory_cache_drops_corrupt_memory_entries(self) -> None:
        _RUN_RESULT_MEMORY_CACHE["bad"] = "{"

        self.assertIsNone(read_run_result_cache("bad"))
        self.assertNotIn("bad", _RUN_RESULT_MEMORY_CACHE)

    def test_run_result_memory_cache_survives_parallel_access(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        _RUN_RESULT_MEMORY_CACHE.clear()

        def write_and_read(index: int) -> dict[str, object] | None:
            key = f"parallel-key-{index}"
            remember_run_result_cache(key, {"city": str(index), "summary": {}, "artifacts": {}})
            return read_run_result_cache(key)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(write_and_read, range(32)))
            list(executor.map(write_and_read, range(32, RUN_RESULT_MEMORY_CACHE_MAX + 32)))

        self.assertTrue(all(result for result in results))
        self.assertLessEqual(len(_RUN_RESULT_MEMORY_CACHE), RUN_RESULT_MEMORY_CACHE_MAX)

    def test_run_result_cache_tmp_path_is_thread_specific(self) -> None:
        cache_path = Path("/tmp/map-boundary-builder-cache/run-results/key.json")

        with patch.object(api_index.threading, "get_ident", return_value=111):
            first = run_result_cache_tmp_path(cache_path)
        with patch.object(api_index.threading, "get_ident", return_value=222):
            second = run_result_cache_tmp_path(cache_path)

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, cache_path.parent)
        self.assertTrue(first.name.startswith("key.json."))
        self.assertTrue(first.name.endswith(".111.tmp"))

    def test_cached_run_payload_can_include_request_profile(self) -> None:
        cached = {
            "city": "Phoenix",
            "summary": {"city": "Phoenix"},
            "artifacts": {"geojson_inline": {"type": "FeatureCollection", "features": []}},
        }
        profile = {"cache_hit": "raw", "total_before_send_s": 0.012345}

        payload = cached_run_payload(cached, "run-id", "Phoenix.png", [], profile=profile)

        self.assertEqual(payload["profile"], profile)

    def test_create_run_profile_includes_pipeline_version_on_cache_hit(self) -> None:
        request = api_index.handler.__new__(api_index.handler)
        request.parse_upload_request = lambda: (
            {},
            {"image": ("Phoenix.png", b"image-bytes")},
            "multipart",
        )
        captured: dict[str, object] = {}

        def send_json(payload: dict[str, object], *, status: HTTPStatus) -> None:
            captured["payload"] = payload
            captured["status"] = status

        request.send_json = send_json
        cached = {
            "city": "Phoenix",
            "summary": {"city": "Phoenix"},
            "artifacts": {"geojson_inline": {"type": "FeatureCollection", "features": []}},
        }
        with (
            patch("api.index.get_pipeline_version", return_value="pipeline-profile"),
            patch("api.index.raw_run_result_cache_key", return_value="raw-key"),
            patch("api.index.read_run_result_cache", return_value=cached),
        ):
            request.handle_create_run()

        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(captured["status"], HTTPStatus.CREATED)
        self.assertEqual(payload["profile"]["pipeline_version"], "pipeline-profile")
        self.assertEqual(payload["profile"]["cache_hit"], "raw")

    def test_create_run_reports_runner_import_failure_without_secondary_error(self) -> None:
        request = api_index.handler.__new__(api_index.handler)
        request.parse_upload_request = lambda: (
            {},
            {"image": ("upload.png", b"not-an-image")},
            "multipart",
        )
        captured: dict[str, object] = {}

        def send_json(payload: dict[str, object], *, status: HTTPStatus) -> None:
            captured["payload"] = payload
            captured["status"] = status

        real_import = builtins.__import__

        def import_with_runner_failure(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "map_boundary_builder.runner" and "build_boundary" in fromlist:
                raise ImportError("runner import unavailable")
            return real_import(name, globals, locals, fromlist, level)

        request.send_json = send_json
        with (
            patch("api.index.get_pipeline_version", return_value="pipeline-import-failure"),
            patch("api.index.raw_run_result_cache_key", return_value="raw-key"),
            patch("api.index.read_run_result_cache", return_value=None),
            patch("builtins.__import__", side_effect=import_with_runner_failure),
        ):
            request.handle_create_run()

        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(captured["status"], HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("runner import unavailable", payload["error"])
        self.assertEqual(payload["profile"]["pipeline_version"], "pipeline-import-failure")
        self.assertEqual(payload["profile"]["cache_hit"], "miss")
        self.assertIn("build_boundary_s", payload["profile"])
        self.assertEqual(payload["events"][-1]["stage"], "failed")
        self.assertEqual(payload["events"][-1]["status"], "failed")
        self.assertEqual(payload["events"][-1]["details"], {"error": "runner import unavailable"})

    def test_create_run_catalog_miss_includes_terminal_event(self) -> None:
        request = api_index.handler.__new__(api_index.handler)
        request.parse_upload_request = lambda: (
            {"catalog_probe_only": "1"},
            {"image": ("Bay Area.png", b"not-an-image")},
            "multipart",
        )
        captured: dict[str, object] = {}

        def send_json(payload: dict[str, object], *, status: HTTPStatus) -> None:
            captured["payload"] = payload
            captured["status"] = status

        details = {
            "active_shape_iou_is_low": False,
            "best_active_catalog_slug": "bay-area-waymo",
            "best_active_catalog_iou": 0.91,
        }

        def build_boundary_miss(*args, **kwargs):
            raise CatalogProbeMiss("No known service-area shape matched the catalog probe.", details=details)

        request.send_json = send_json
        with (
            patch("api.index.get_pipeline_version", return_value="pipeline-catalog-miss-terminal"),
            patch("api.index.read_run_result_cache", return_value=None),
            patch("api.index.write_run_result_cache"),
            patch("map_boundary_builder.runner.build_boundary", side_effect=build_boundary_miss),
        ):
            request.handle_create_run()

        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(captured["status"], HTTPStatus.OK)
        self.assertEqual(payload["status"], "catalog_miss")
        self.assertEqual(payload["catalog_probe_miss"], details)
        self.assertEqual(payload["events"][-1]["stage"], "catalog_miss")
        self.assertEqual(payload["events"][-1]["status"], "catalog_miss")
        self.assertEqual(payload["events"][-1]["message"], "Catalog probe missed")
        self.assertEqual(payload["events"][-1]["details"], details)

    def test_event_stage_elapsed_seconds_sums_repeated_stages(self) -> None:
        events = [
            {"stage": "queued", "message": "Run queued"},
            {"stage": "inspect", "timestamp": 10.0},
            {"stage": "extract", "timestamp": 10.25},
            {"stage": "extract", "timestamp": 10.75},
            {"stage": "ocr", "timestamp": 11.0},
            {"stage": "georeference", "timestamp": 11.4},
            {"stage": "complete", "timestamp": 11.5},
        ]

        self.assertEqual(
            event_stage_elapsed_seconds(events),
            {
                "inspect": 0.25,
                "extract": 0.75,
                "ocr": 0.4,
                "georeference": 0.1,
            },
        )

    def test_json_response_body_gzips_large_payloads_when_supported(self) -> None:
        payload = {"data": "x" * 4096}

        encoded, headers = json_response_body(payload, accept_encoding="br, gzip")

        self.assertEqual(headers["Content-Encoding"], "gzip")
        self.assertLess(len(encoded), 512)
        self.assertEqual(json_response_body(payload)[0], gzip.decompress(encoded))

    def test_generation_value_errors_are_client_safe_failures(self) -> None:
        events = [
            {"stage": "queued", "timestamp": 10.0},
            {"stage": "ocr", "timestamp": 10.5},
        ]
        profile = {"build_boundary_s": 0.42}

        payload = generation_error_payload(
            ValueError("Could not infer a reliable map location."),
            "run-1",
            "input.png",
            events,
            profile,
        )

        self.assertEqual(generation_error_status(ValueError("bad map")), HTTPStatus.UNPROCESSABLE_ENTITY)
        self.assertEqual(generation_error_status(RuntimeError("boom")), HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertEqual(payload["id"], "run-1")
        self.assertEqual(payload["filename"], "input.png")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["percent"], 100)
        self.assertEqual(payload["profile"], profile)
        self.assertIn("Could not infer", payload["error"])
        self.assertEqual(payload["events"][:-1], events)
        self.assertEqual(payload["events"][-1]["stage"], "failed")
        self.assertEqual(payload["events"][-1]["status"], "failed")
        self.assertEqual(
            payload["events"][-1]["details"],
            {"error": "Could not infer a reliable map location."},
        )

    def test_health_payload_can_prewarm_generation_runtime(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("api.index.prewarm_generation_runtime", return_value={"status": "ok", "total_s": 0.1}) as prewarm,
        ):
            cold = health_payload()
            warm = health_payload(warm="ocr")

        self.assertEqual(cold["ocr"]["rapidocr_max_dimension"], 1600)
        self.assertEqual(cold["ocr"]["rapidocr_detector_limit_side_len"], 608)
        self.assertEqual(cold["ocr"]["rapidocr_large_image_detector_limit_side_len"], 608)
        self.assertEqual(cold["ocr"]["rapidocr_bright_blue_detector_limit_side_len"], 480)
        self.assertEqual(cold["ocr"]["rapidocr_bright_blue_detector_limit_type"], "max")
        self.assertEqual(cold["ocr"]["rapidocr_large_image_detector_limit_min_dimension"], 1000)
        self.assertEqual(cold["ocr"]["current_catalog_label_ocr_max_dimension"], 875)
        self.assertEqual(cold["ocr"]["rapidocr_warm_detector_limit"], 608)
        self.assertEqual(cold["ocr"]["rapidocr_warm_detector_limits"], [608, 480])
        self.assertTrue(cold["ocr"]["onnxruntime_enable_cpu_mem_arena"])
        self.assertTrue(cold["ocr"]["onnxruntime_allow_spinning"])
        self.assertEqual(cold["ocr"]["fast_text_ocr_styles"], ["bright-blue", "gray-fill", "light-fill"])
        self.assertEqual(cold["ocr"]["fast_text_ocr_min_area"], 1300.0)
        self.assertEqual(cold["ocr"]["fast_text_ocr_rescue_min_area"], 900.0)
        self.assertEqual(cold["ocr"]["fast_text_ocr_rescue_min_aspect"], 2.8)
        self.assertEqual(cold["ocr"]["fast_text_ocr_fallback_confidence"], 0.70)
        self.assertEqual(cold["generation_env"]["MAP_BOUNDARY_GENERAL_EXTRACT_MAX_DIMENSION"], "1600")
        self.assertEqual(cold["generation_env"]["MAP_BOUNDARY_GEOCODE_WORKERS"], "6")
        self.assertEqual(cold["runtime_dependencies"]["onnxruntime"], "1.26.0")
        self.assertIn("cv2", cold["runtime_dependencies"])
        self.assertIn("rapidocr-onnxruntime", cold["runtime_dependencies"])
        self.assertNotIn("warm", cold)
        self.assertEqual(warm["warm"], {"status": "ok", "total_s": 0.1})
        prewarm.assert_called_once_with()

    def test_health_payload_marks_runtime_or_warm_failures_unhealthy(self) -> None:
        dependencies = [
            ("numpy", "2.4.6"),
            ("onnxruntime", "1.26.0"),
            ("opencv-python", "missing"),
            ("opencv-python-headless", "4.10.0.84"),
            ("pillow", "12.2.0"),
            ("rapidocr-onnxruntime", "1.4.4"),
            ("shapely", "2.1.2"),
            ("cv2", "missing"),
        ]
        with (
            patch("api.index.pipeline_version_dependency_versions", return_value=dependencies),
            patch("api.index.prewarm_generation_runtime", return_value={"status": "error", "error": "No module named cv2"}),
        ):
            cold = health_payload()
            warm = health_payload(warm="ocr")

        self.assertFalse(cold["ok"])
        self.assertFalse(warm["ok"])
        self.assertEqual(warm["warm"]["status"], "error")
        self.assertEqual(health_response_status(warm), HTTPStatus.SERVICE_UNAVAILABLE)

    def test_cron_warm_generation_requires_bearer_secret(self) -> None:
        self.assertEqual(CRON_WARM_PATH, "/api/cron/warm-generation-v2")
        self.assertIn(LEGACY_CRON_WARM_PATH, CRON_WARM_PATHS)
        self.assertIn(CRON_WARM_PATH, CRON_WARM_PATHS)

        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(authorized_cron_request("Bearer secret"))

        with patch.dict("os.environ", {"CRON_SECRET": "secret"}):
            self.assertTrue(authorized_cron_request("Bearer secret"))
            self.assertFalse(authorized_cron_request(None))
            self.assertFalse(authorized_cron_request("secret"))
            self.assertFalse(authorized_cron_request("Bearer different"))

    def test_cron_warm_generation_payload_runs_authenticated_prewarm(self) -> None:
        with (
            patch.dict("os.environ", {"CRON_SECRET": "secret"}),
            patch("api.index.get_pipeline_version", return_value="pipeline-test"),
            patch("api.index.prewarm_generation_runtime", return_value={"status": "ok"}) as prewarm,
        ):
            unauthorized_payload, unauthorized_status = cron_warm_generation_payload(
                authorization_header="Bearer wrong"
            )
            payload, status = cron_warm_generation_payload(authorization_header="Bearer secret")

        self.assertEqual(unauthorized_status, HTTPStatus.UNAUTHORIZED)
        self.assertEqual(unauthorized_payload["error"], "Unauthorized")
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload["pipeline_version"], "pipeline-test")
        self.assertEqual(payload["warm"], {"status": "ok"})
        prewarm.assert_called_once_with()

    def test_cron_warm_generation_payload_surfaces_prewarm_failure(self) -> None:
        with (
            patch.dict("os.environ", {"CRON_SECRET": "secret"}),
            patch("api.index.get_pipeline_version", return_value="pipeline-test"),
            patch("api.index.prewarm_generation_runtime", return_value={"status": "error", "error": "cv2 missing"}),
        ):
            payload, status = cron_warm_generation_payload(authorization_header="Bearer secret")

        self.assertEqual(status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["pipeline_version"], "pipeline-test")
        self.assertEqual(payload["warm"]["status"], "error")

    def test_index_asset_embeds_pipeline_version_for_local_cache(self) -> None:
        html, mime = web_asset_response("index.html")

        self.assertEqual(mime, "text/html; charset=utf-8")
        self.assertIn(b"window.__MAP_BOUNDARY_PIPELINE_VERSION__ = \"pipeline-", html)
        self.assertNotIn(b'= "__MAP_BOUNDARY_PIPELINE_VERSION__";', html)

    def test_index_asset_cache_busts_frontend_bundle(self) -> None:
        html, _mime = web_asset_response("index.html")
        asset_version = web_asset_version().encode("utf-8")

        self.assertIn(b"/static/app.css?v=" + asset_version, html)
        self.assertIn(b"/static/app.js?v=" + asset_version, html)
        self.assertNotIn(b"__MAP_BOUNDARY_ASSET_VERSION__", html)

    def test_index_asset_skips_normalized_cache_lookup_for_ui_fresh_uploads(self) -> None:
        html, _mime = web_asset_response("index.html")

        self.assertIn(b'name="normalized_cache_lookup" value="0"', html)

    def test_frontend_eagerly_reschedules_generation_prewarm_after_image_selection(self) -> None:
        app_js, mime = web_asset_response("app.js")

        self.assertEqual(mime, "text/javascript; charset=utf-8")
        startup_schedule = app_js.index(b"scheduleGenerationRuntimePrewarm();")
        select_schedule = app_js.index(b"scheduleGenerationRuntimePrewarm({ eager: true });")
        function_start = app_js.index(b"function scheduleGenerationRuntimePrewarm(options = {}) {")
        scheduled_guard = app_js.index(b"if (generationRuntimePrewarmScheduled) {", function_start)
        eager_guard = app_js.index(b"if (!options.eager) return;", scheduled_guard)
        clear_pending = app_js.index(b"clearScheduledGenerationRuntimePrewarm();", eager_guard)
        idle_schedule = app_js.index(
            b"if (!options.eager && typeof window.requestIdleCallback === \"function\")",
            function_start,
        )
        eager_delay = app_js.index(b"const delayMs = options.eager ? 0 : 400;", idle_schedule)

        self.assertLess(startup_schedule, select_schedule)
        self.assertLess(function_start, scheduled_guard)
        self.assertLess(scheduled_guard, eager_guard)
        self.assertLess(eager_guard, clear_pending)
        self.assertLess(clear_pending, idle_schedule)
        self.assertLess(idle_schedule, eager_delay)

    def test_frontend_cancels_stale_scheduled_generation_prewarm_callbacks(self) -> None:
        app_js, mime = web_asset_response("app.js")

        self.assertEqual(mime, "text/javascript; charset=utf-8")
        self.assertIn(b"let generationRuntimePrewarmScheduleToken = 0;", app_js)
        self.assertIn(b"let generationRuntimePrewarmIdleCallbackId = null;", app_js)
        self.assertIn(b"let generationRuntimePrewarmTimeoutId = null;", app_js)

        function_start = app_js.index(b"function scheduleGenerationRuntimePrewarm(options = {}) {")
        token_increment = app_js.index(b"generationRuntimePrewarmScheduleToken += 1;", function_start)
        token_capture = app_js.index(b"const scheduleToken = generationRuntimePrewarmScheduleToken;", token_increment)
        stale_guard = app_js.index(
            b"if (scheduleToken !== generationRuntimePrewarmScheduleToken) return;",
            token_capture,
        )
        cancel_start = app_js.index(b"function clearScheduledGenerationRuntimePrewarm() {")
        cancel_token = app_js.index(b"generationRuntimePrewarmScheduleToken += 1;", cancel_start)
        cancel_idle = app_js.index(b"window.cancelIdleCallback(generationRuntimePrewarmIdleCallbackId);", cancel_token)
        clear_timeout = app_js.index(b"window.clearTimeout(generationRuntimePrewarmTimeoutId);", cancel_idle)
        submit_cancel = app_js.index(b"function cancelPendingGenerationRuntimePrewarm() {")
        clear_before_abort = app_js.index(b"clearScheduledGenerationRuntimePrewarm();", submit_cancel)
        abort_active = app_js.index(b"generationRuntimePrewarmAbortController.abort();", clear_before_abort)

        self.assertLess(function_start, token_increment)
        self.assertLess(token_increment, token_capture)
        self.assertLess(token_capture, stale_guard)
        self.assertLess(cancel_start, cancel_token)
        self.assertLess(cancel_token, cancel_idle)
        self.assertLess(cancel_idle, clear_timeout)
        self.assertLess(submit_cancel, clear_before_abort)
        self.assertLess(clear_before_abort, abort_active)

    def test_frontend_marks_unhinted_skipped_catalog_probe_as_missed(self) -> None:
        app_js, mime = web_asset_response("app.js")

        self.assertEqual(mime, "text/javascript; charset=utf-8")
        self.assertIn(
            b"if (!probeCandidate.file) return probeCandidate.skippedMiss ? { missed: true } : null;",
            app_js,
        )
        self.assertIn(
            b"return { file: null, skippedMiss: !looksServiceAreaLike, ...metadata };",
            app_js,
        )
        self.assertIn(
            b"return { file: null, skippedMiss: true, ...metadata };",
            app_js,
        )
        self.assertIn(
            b"hasHint,\n    hintText,\n    looksServiceAreaLike,\n    maxDimension,\n    sourceCanvas,",
            app_js,
        )
        self.assertIn(
            b'if (catalogProbeResult.lowIou) {\n        formData.set("catalog_probe_miss_low_iou", "1");',
            app_js,
        )
        self.assertIn(
            b"lowIou: miss.active_shape_iou_is_low === true,",
            app_js,
        )
        self.assertIn(
            b"const FAST_CATALOG_HANDOFF_MAX_DIMENSION = 1600;",
            app_js,
        )
        self.assertIn(
            b"const FAST_CATALOG_HANDOFF_PROVIDER_UI_MIN_CONFIDENCE = 0.70;",
            app_js,
        )
        self.assertIn(
            b"const CATALOG_PROBE_AREA_HINT_PATTERN =",
            app_js,
        )
        self.assertIn(
            b'if (name !== "image" && name !== "catalog_probe_miss_low_iou")',
            app_js,
        )
        self.assertIn(
            b'fastData.set("fast_catalog_handoff", "1");',
            app_js,
        )
        self.assertIn(
            b"const fastHandoffFilePromise = fastCatalogHandoffCandidate(file, probeCandidate);",
            app_js,
        )
        self.assertIn(
            b"if (shouldUseFastCatalogHandoff(result)) {\n        result.fastHandoffFile = await fastHandoffFilePromise;",
            app_js,
        )
        self.assertIn(
            b"summary.catalog_slug !== catalogProbeResult.bestActiveCatalogSlug",
            app_js,
        )
        self.assertIn(
            b"if (!isProviderUiCatalogHandoffPayload(summary)) return false;",
            app_js,
        )
        self.assertIn(
            b"const scale = Math.min(1, FAST_CATALOG_HANDOFF_MAX_DIMENSION / maxDimension);",
            app_js,
        )
        self.assertIn(
            b"catalogSlugMatchesHint(summary.catalog_slug, catalogProbeResult.catalogHintText)",
            app_js,
        )

    def test_frontend_races_browser_cache_hit_against_catalog_probe(self) -> None:
        app_js, mime = web_asset_response("app.js")

        self.assertEqual(mime, "text/javascript; charset=utf-8")
        cache_start = app_js.index(b"const cacheLookupPromise = buildRunCacheKeys(uploadFile, formData);")
        deferred_cache = app_js.index(b"const deferredCacheKeysPromise = cacheKeysFromLookupPromise(cacheLookupPromise);")
        race_start = app_js.index(b"const firstFastResult = await Promise.race([")
        probe_race = app_js.index(b'catalogProbePromise.then((result) => ({ type: "catalog-probe", result }))')
        cache_race = app_js.index(b"cachedHistoryEntryFromLookupPromise(cacheLookupPromise).then")
        cache_hit_branch = app_js.index(b'if (firstFastResult?.type === "cache-hit") {')
        cache_abort = app_js.index(b"catalogProbeAbortController?.abort();", cache_hit_branch)
        cache_restore = app_js.index(b"restoreCachedHistoryEntry(firstFastResult.cachedEntry);")
        probe_result = app_js.index(b'const catalogProbeResult = firstFastResult?.type === "catalog-probe"')
        first_inline = app_js.index(b"applyInlineRun(catalogProbeResult.payload, {")
        cache_await = app_js.index(b"const cacheLookup = await cacheLookupPromise;")
        cached_entry = app_js.index(
            b"const cachedEntry = await cachedHistoryEntryFromLookupPromise(cacheLookupPromise, {"
        )

        self.assertLess(cache_start, deferred_cache)
        self.assertLess(deferred_cache, race_start)
        self.assertLess(race_start, probe_race)
        self.assertLess(race_start, cache_race)
        self.assertLess(race_start, cache_hit_branch)
        self.assertLess(cache_hit_branch, cache_abort)
        self.assertLess(cache_abort, cache_restore)
        self.assertLess(cache_hit_branch, probe_result)
        self.assertLess(probe_result, first_inline)
        self.assertLess(first_inline, cache_await)
        self.assertLess(cache_await, cached_entry)
        self.assertIn(b"cacheKeysPromise: deferredCacheKeysPromise,", app_js)
        self.assertIn(b"async function cacheKeysFromLookupPromise(cacheLookupPromise) {", app_js)
        self.assertIn(
            b"async function cachedHistoryEntryFromLookupPromise(cacheLookupPromise, options = {}) {",
            app_js,
        )

    def test_frontend_checks_deferred_pixel_cache_before_full_upload(self) -> None:
        app_js, mime = web_asset_response("app.js")

        self.assertEqual(mime, "text/javascript; charset=utf-8")
        self.assertIn(b"const RUN_CACHE_DEFERRED_HISTORY_WAIT_MS = 180;", app_js)

        cache_await = app_js.index(b"const cacheLookup = await cacheLookupPromise;")
        deferred_cached_entry = app_js.index(
            b"const cachedEntry = await cachedHistoryEntryFromLookupPromise(cacheLookupPromise, {"
        )
        upload_status = app_js.index(b'markProgressStep("prepare", "running", "Uploading image.");')
        upload_call = app_js.index(b"const { response, payload } = await postRunUpload(formData, uploadFile);")

        self.assertLess(cache_await, deferred_cached_entry)
        self.assertLess(deferred_cached_entry, upload_status)
        self.assertLess(upload_status, upload_call)
        self.assertIn(
            b"includeDeferred: true,\n      deferredWaitMs: RUN_CACHE_DEFERRED_HISTORY_WAIT_MS,",
            app_js,
        )

        helper_start = app_js.index(
            b"async function cachedHistoryEntryFromLookupPromise(cacheLookupPromise, options = {}) {"
        )
        quick_hit = app_js.index(b"const cachedEntry = findCachedHistoryEntry(lookupKeys);", helper_start)
        fresh_guard = app_js.index(
            b"if (cachedEntry || !options.includeDeferred || !hasCachedRunHistoryEntries())",
            helper_start,
        )
        deferred_wait = app_js.index(
            b"const deferredWaitMs = Math.max(0, Number(options.deferredWaitMs || 0));",
            helper_start,
        )
        deferred_keys = app_js.index(b"cacheKeysFromPromise(lookup?.cacheKeysPromise),", helper_start)
        combined_lookup = app_js.index(b"return findCachedHistoryEntry([\n      ...lookupKeys,", helper_start)

        self.assertLess(helper_start, quick_hit)
        self.assertLess(quick_hit, fresh_guard)
        self.assertLess(fresh_guard, deferred_wait)
        self.assertLess(deferred_wait, deferred_keys)
        self.assertLess(deferred_keys, combined_lookup)
        self.assertIn(b"function hasCachedRunHistoryEntries() {", app_js)
        self.assertIn(b"entry?.geojson && entryCacheKeys(entry).length", app_js)

    def test_frontend_leaves_svgz_uploads_for_backend_rasterization(self) -> None:
        app_js, mime = web_asset_response("app.js")

        self.assertEqual(mime, "text/javascript; charset=utf-8")
        compressed_svg = app_js.index(b"if (isCompressedSvgFile(file)) {")
        plain_svg = app_js.index(b"if (isSvgFile(file)) {")
        compressed_status = app_js.index(b'setStatus("Uploading SVGZ map"', compressed_svg)
        compressed_return = app_js.index(b"return file;", compressed_svg)
        rasterize_call = app_js.index(b"const canvas = await svgFileToCanvas(file);", plain_svg)

        self.assertLess(compressed_svg, plain_svg)
        self.assertLess(compressed_svg, compressed_status)
        self.assertLess(compressed_status, compressed_return)
        self.assertLess(compressed_return, plain_svg)
        self.assertLess(plain_svg, rasterize_call)
        self.assertIn(b"function isCompressedSvgFile(file) {", app_js)
        self.assertIn(b'type === "image/svg+xml-compressed"', app_js)
        self.assertIn(b"/\\.svgz$/i.test(file?.name || \"\")", app_js)

    def test_frontend_preserves_failed_run_payload_for_reports(self) -> None:
        app_js, mime = web_asset_response("app.js")

        self.assertEqual(mime, "text/javascript; charset=utf-8")
        non_ok = app_js.index(b"if (!response.ok) {")
        failed_payload = app_js.index(b"if (isFailedRunPayload(payload)) {")
        finish_failed = app_js.index(b"finishWithFailedRunPayload(payload);")
        fallback_throw = app_js.index(b'throw new Error(payload?.error || uploadErrorMessage(response, "Run failed to start."));')

        self.assertLess(non_ok, failed_payload)
        self.assertLess(failed_payload, finish_failed)
        self.assertLess(finish_failed, fallback_throw)
        self.assertIn(b"latestRunId = payload.id || latestRunId;", app_js)
        self.assertIn(b"latestRunEvents = Array.isArray(payload.events) ? payload.events : [];", app_js)
        self.assertIn(b"latestRunProfile = payload.profile || null;", app_js)
        self.assertIn(b"function failedRunProgressStep(events) {", app_js)
        self.assertIn(b"formData.set(\"run_id\", latestRunId || \"\");", app_js)
        self.assertIn(b"formData.set(\"events\", JSON.stringify(latestRunEvents));", app_js)
        self.assertIn(b"formData.set(\"profile\", JSON.stringify(latestRunProfile || {}));", app_js)
        self.assertIn(b'if (event.status === "failed") {', app_js)
        self.assertIn(b"await loadFailureSnapshot(runId);", app_js)
        self.assertIn(b"async function loadFailureSnapshot(runId) {", app_js)
        self.assertIn(b"latestRunProfile = status.profile || latestRunProfile;", app_js)
        self.assertIn(b"function isFailureEvent(event) {", app_js)

    def test_normalized_cache_lookup_defaults_to_fast_path_but_can_opt_in(self) -> None:
        self.assertFalse(bool_field({}, "normalized_cache_lookup", default=False))
        self.assertFalse(bool_field({"normalized_cache_lookup": "0"}, "normalized_cache_lookup", default=False))
        self.assertTrue(bool_field({"normalized_cache_lookup": "1"}, "normalized_cache_lookup", default=False))

    def test_catalog_probe_requests_default_to_no_overlay(self) -> None:
        self.assertFalse(include_overlay_for_request({}, catalog_probe_only=True))
        self.assertTrue(include_overlay_for_request({}, catalog_probe_only=False))
        self.assertTrue(include_overlay_for_request({"include_overlay": "1"}, catalog_probe_only=True))
        self.assertFalse(include_overlay_for_request({"include_overlay": "0"}, catalog_probe_only=False))

    def test_api_can_disable_catalog_matching_for_controlled_no_catalog_runs(self) -> None:
        self.assertTrue(allow_catalog_for_request({}))
        self.assertTrue(allow_catalog_for_request({"allow_catalog": "1"}))
        self.assertFalse(allow_catalog_for_request({"allow_catalog": "0"}))
        self.assertFalse(allow_catalog_for_request({"no_catalog": "1"}))
        self.assertFalse(allow_catalog_for_request({"allow_catalog": "1", "no_catalog": "1"}))

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support required")
    def test_inline_overlay_uses_webp_for_typical_previews(self) -> None:
        with NamedTemporaryFile(suffix=".png") as handle:
            image = Image.effect_noise((240, 240), 64).convert("RGB")
            image.save(handle.name, format="PNG")
            self.assertGreater(Path(handle.name).stat().st_size, INLINE_OVERLAY_OPTIMIZE_BYTES)

            data_url = inline_overlay(Path(handle.name))

        self.assertIsNotNone(data_url)
        assert data_url is not None
        self.assertTrue(data_url.startswith("data:image/webp;base64,"))

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support required")
    def test_inline_overlay_uses_webp_for_large_previews(self) -> None:
        with NamedTemporaryFile(suffix=".png") as handle:
            image = Image.effect_noise((1400, 1400), 64).convert("RGB")
            image.save(handle.name, format="PNG")

            data_url = inline_overlay(Path(handle.name))

        self.assertIsNotNone(data_url)
        assert data_url is not None
        self.assertTrue(data_url.startswith("data:image/webp;base64,"))
        decoded = base64.b64decode(data_url.split(",", 1)[1])
        with Image.open(BytesIO(decoded)) as preview:
            self.assertLessEqual(max(preview.size), 1200)

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support required")
    def test_inline_overlay_preserves_existing_webp_preview(self) -> None:
        with NamedTemporaryFile(suffix=".webp") as handle:
            image = Image.effect_noise((240, 240), 64).convert("RGB")
            image.save(handle.name, format="WEBP", quality=82)

            data_url = inline_overlay(Path(handle.name))

        self.assertIsNotNone(data_url)
        assert data_url is not None
        self.assertTrue(data_url.startswith("data:image/webp;base64,"))


if __name__ == "__main__":
    unittest.main()
