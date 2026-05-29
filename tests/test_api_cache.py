import unittest
import base64
import gzip
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, features

from api.index import (
    INLINE_OVERLAY_OPTIMIZE_BYTES,
    authorized_cron_request,
    cached_run_payload,
    cron_warm_generation_payload,
    event_stage_elapsed_seconds,
    health_payload,
    inline_overlay,
    json_response_body,
    normalized_image_sha256,
    raw_run_result_cache_key,
    read_run_result_cache,
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


class ApiRunCacheTests(unittest.TestCase):
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
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Phoenix", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Houston", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Bay Area", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Tesla Houston", allow_catalog=True))
        self.assertFalse(should_overlap_ocr_with_extraction(city_input="Zoox San Francisco", allow_catalog=True))
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
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Phoenix", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Houston", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Bay Area", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Tesla Houston", allow_catalog=True))
        self.assertTrue(should_try_pre_ocr_catalog(city_input="Zoox San Francisco", allow_catalog=True))
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
        self.assertNotEqual(base, changed_preview_options)
        self.assertNotEqual(base, changed_mask_options)
        self.assertNotEqual(base, changed_overlay_mode)

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

    def test_health_payload_can_prewarm_generation_runtime(self) -> None:
        with patch("api.index.prewarm_generation_runtime", return_value={"status": "ok", "total_s": 0.1}) as prewarm:
            cold = health_payload()
            warm = health_payload(warm="ocr")

        self.assertEqual(cold["ocr"]["rapidocr_max_dimension"], 1600)
        self.assertEqual(cold["ocr"]["rapidocr_detector_limit_side_len"], 608)
        self.assertEqual(cold["ocr"]["rapidocr_large_image_detector_limit_side_len"], 640)
        self.assertEqual(cold["ocr"]["rapidocr_large_image_detector_limit_min_dimension"], 1000)
        self.assertEqual(cold["ocr"]["rapidocr_warm_detector_limit"], 640)
        self.assertNotIn("warm", cold)
        self.assertEqual(warm["warm"], {"status": "ok", "total_s": 0.1})
        prewarm.assert_called_once_with()

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
