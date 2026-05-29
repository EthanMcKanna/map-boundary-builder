import unittest
import base64
import gzip
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, PngImagePlugin, features

from api.index import (
    INLINE_OVERLAY_OPTIMIZE_BYTES,
    authorized_cron_request,
    cached_run_payload,
    cron_warm_generation_payload,
    event_stage_elapsed_seconds,
    generation_error_payload,
    generation_error_status,
    health_response_status,
    health_payload,
    inline_overlay,
    jpeg_commentless_run_result_cache_key,
    jpeg_commentless_sha256,
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
    run_result_cache_key,
    write_run_result_cache,
)
from map_boundary_builder.asset_response import web_asset_response
from map_boundary_builder.runner import (
    BoundaryBuildOptions,
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


class ApiRunCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        _RUN_RESULT_MEMORY_CACHE.clear()

    def test_catalog_matching_defaults_on_for_api_options_namespace(self) -> None:
        self.assertTrue(catalog_matching_enabled(SimpleNamespace()))
        self.assertFalse(catalog_matching_enabled(SimpleNamespace(allow_catalog=False)))

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
        changed_mask_options = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(write_mask_artifact=False),
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
        self.assertNotEqual(base, changed_mask_options)
        self.assertNotEqual(base, changed_overlay_mode)

    def test_run_cache_filename_hint_uses_semantic_basename(self) -> None:
        options = BoundaryBuildOptions(filename_hint="/tmp/uploads/Dallas.png")
        same_basename = BoundaryBuildOptions(filename_hint="Dallas.png")
        different_basename = BoundaryBuildOptions(filename_hint="Phoenix.png")
        same_context_cache_bust = BoundaryBuildOptions(
            filename_hint="Dallas pipeline-version-1780067151-e527924-ui.png"
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
        self.assertNotEqual(
            run_result_cache_key(b"image-a", None, options),
            run_result_cache_key(b"image-a", None, different_basename),
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

    def test_cached_run_payload_can_include_request_profile(self) -> None:
        cached = {
            "city": "Phoenix",
            "summary": {"city": "Phoenix"},
            "artifacts": {"geojson_inline": {"type": "FeatureCollection", "features": []}},
        }
        profile = {"cache_hit": "raw", "total_before_send_s": 0.012345}

        payload = cached_run_payload(cached, "run-id", "Phoenix.png", [], profile=profile)

        self.assertEqual(payload["profile"], profile)

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
        self.assertEqual(payload["events"], events)

    def test_health_payload_can_prewarm_generation_runtime(self) -> None:
        with patch("api.index.prewarm_generation_runtime", return_value={"status": "ok", "total_s": 0.1}) as prewarm:
            cold = health_payload()
            warm = health_payload(warm="ocr")

        self.assertEqual(cold["ocr"]["rapidocr_max_dimension"], 1600)
        self.assertEqual(cold["ocr"]["rapidocr_detector_limit_side_len"], 608)
        self.assertEqual(cold["ocr"]["rapidocr_large_image_detector_limit_side_len"], 608)
        self.assertEqual(cold["ocr"]["rapidocr_large_image_detector_limit_min_dimension"], 1000)
        self.assertEqual(cold["ocr"]["rapidocr_warm_detector_limit"], 608)
        self.assertEqual(cold["ocr"]["rapidocr_warm_detector_limits"], [608])
        self.assertTrue(cold["ocr"]["onnxruntime_enable_cpu_mem_arena"])
        self.assertTrue(cold["ocr"]["onnxruntime_allow_spinning"])
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

    def test_index_asset_skips_normalized_cache_lookup_for_ui_fresh_uploads(self) -> None:
        html, _mime = web_asset_response("index.html")

        self.assertIn(b'name="normalized_cache_lookup" value="0"', html)

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


if __name__ == "__main__":
    unittest.main()
