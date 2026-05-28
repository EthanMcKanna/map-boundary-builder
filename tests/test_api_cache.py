import unittest
import gzip
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, features

from api.index import (
    cached_run_payload,
    inline_overlay,
    json_response_body,
    normalized_image_sha256,
    raw_run_result_cache_key,
    read_run_result_cache,
    run_result_cache_key,
    write_run_result_cache,
)
from map_boundary_builder.runner import BoundaryBuildOptions


class ApiRunCacheTests(unittest.TestCase):
    def test_run_cache_key_depends_on_image_and_options(self) -> None:
        base = run_result_cache_key(b"image-a", None, BoundaryBuildOptions())
        changed_image = run_result_cache_key(b"image-b", None, BoundaryBuildOptions())
        changed_city = run_result_cache_key(b"image-a", "Miami", BoundaryBuildOptions())
        changed_options = run_result_cache_key(
            b"image-a",
            None,
            BoundaryBuildOptions(min_control_points=4),
        )

        self.assertNotEqual(base, changed_image)
        self.assertNotEqual(base, changed_city)
        self.assertNotEqual(base, changed_options)

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

    def test_json_response_body_gzips_large_payloads_when_supported(self) -> None:
        payload = {"data": "x" * 4096}

        encoded, headers = json_response_body(payload, accept_encoding="br, gzip")

        self.assertEqual(headers["Content-Encoding"], "gzip")
        self.assertLess(len(encoded), 512)
        self.assertEqual(json_response_body(payload)[0], gzip.decompress(encoded))

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support required")
    def test_inline_overlay_uses_webp_for_large_previews(self) -> None:
        with NamedTemporaryFile(suffix=".png") as handle:
            image = Image.effect_noise((900, 900), 64).convert("RGB")
            image.save(handle.name, format="PNG")

            data_url = inline_overlay(Path(handle.name))

        self.assertIsNotNone(data_url)
        assert data_url is not None
        self.assertTrue(data_url.startswith("data:image/webp;base64,"))


if __name__ == "__main__":
    unittest.main()
