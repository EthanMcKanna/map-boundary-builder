import os
import unittest
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import map_boundary_builder.web as web


class LocalWebHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        web.RUNS.clear()

    def tearDown(self) -> None:
        web.RUNS.clear()

    def test_background_run_preserves_original_filename_hint(self) -> None:
        request = web.BoundaryWebHandler.__new__(web.BoundaryWebHandler)
        request.parse_upload_request = lambda: (
            {"source_was_svg": "1"},
            {"image": ("Waymo Bay Area.png", b"image-bytes")},
        )
        captured_response: dict[str, object] = {}
        captured_threads: list[object] = []

        def send_json(payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            captured_response["payload"] = payload
            captured_response["status"] = status

        class FakeThread:
            def __init__(self, *, target: object, args: tuple[object, ...], daemon: bool) -> None:
                self.target = target
                self.args = args
                self.daemon = daemon
                captured_threads.append(self)

            def start(self) -> None:
                return None

        request.send_json = send_json

        with TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {"MAP_BOUNDARY_WEB_OUT": temp_dir}),
                patch("map_boundary_builder.web.secrets.token_hex", return_value="run-id"),
                patch("map_boundary_builder.web.get_pipeline_version", return_value="pipeline-test"),
                patch("map_boundary_builder.web.threading.Thread", FakeThread),
            ):
                request.handle_create_run()

        self.assertEqual(captured_response["status"], HTTPStatus.CREATED)
        self.assertEqual(captured_response["payload"], {"id": "run-id", "status_url": "/api/runs/run-id"})
        self.assertEqual(len(captured_threads), 1)
        state, options = captured_threads[0].args
        self.assertEqual(state.profile["pipeline_version"], "pipeline-test")
        self.assertEqual(state.original_filename, "Waymo Bay Area.png")
        self.assertEqual(options.filename_hint, "Waymo Bay Area.png")
        self.assertTrue(options.source_was_svg)

    def test_background_failure_records_terminal_failed_event(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state = web.RunState(
                run_id="run-id",
                city=None,
                original_filename="blank.png",
                run_dir=Path(temp_dir),
                image_path=Path(temp_dir) / "blank.png",
                output_path=Path(temp_dir) / "boundary.geojson",
                debug_dir=Path(temp_dir) / "debug",
                profile={"pipeline_version": "pipeline-test"},
            )
            state.image_path.write_bytes(b"not-an-image")

            def fail_build(*args, **kwargs):
                kwargs["progress"](
                    {
                        "stage": "extract",
                        "message": "Extracting service-area pixels",
                        "percent": 18,
                        "status": "running",
                        "details": {"width": 160, "height": 160},
                    }
                )
                raise ValueError("No service-area polygon could be extracted from the image.")

            with patch("map_boundary_builder.web.build_boundary", side_effect=fail_build):
                web.run_worker(state, web.BoundaryBuildOptions())

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.percent, 100)
        self.assertEqual(state.error, "No service-area polygon could be extracted from the image.")
        self.assertEqual(state.events[-1]["stage"], "failed")
        self.assertEqual(state.events[-1]["status"], "failed")
        self.assertEqual(state.events[-1]["message"], "Generation failed")
        self.assertEqual(
            state.events[-1]["details"],
            {"error": "No service-area polygon could be extracted from the image."},
        )
        self.assertIn("build_stage_elapsed_s", state.profile)

    def test_catalog_probe_miss_response_includes_terminal_event(self) -> None:
        request = web.BoundaryWebHandler.__new__(web.BoundaryWebHandler)
        request.parse_upload_request = lambda: (
            {"catalog_probe_only": "1"},
            {"image": ("Bay Area.png", b"image-bytes")},
        )
        captured_response: dict[str, object] = {}

        def send_json(payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            captured_response["payload"] = payload
            captured_response["status"] = status

        details = {
            "active_shape_iou_is_low": False,
            "best_active_catalog_slug": "bay-area-waymo",
            "best_active_catalog_iou": 0.91,
        }

        def miss_build(*args, **kwargs):
            kwargs["progress"](
                {
                    "stage": "extract",
                    "message": "Extracting service-area pixels",
                    "percent": 18,
                    "status": "running",
                }
            )
            raise web.CatalogProbeMiss("No known service-area shape matched the catalog probe.", details=details)

        request.send_json = send_json
        with TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {"MAP_BOUNDARY_WEB_OUT": temp_dir}),
                patch("map_boundary_builder.web.secrets.token_hex", return_value="run-id"),
                patch("map_boundary_builder.web.get_pipeline_version", return_value="pipeline-test"),
                patch("map_boundary_builder.web.build_boundary", side_effect=miss_build),
            ):
                request.handle_create_run()

        payload = captured_response["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(captured_response["status"], HTTPStatus.OK)
        self.assertEqual(payload["status"], "catalog_miss")
        self.assertEqual(payload["catalog_probe_miss"], details)
        self.assertEqual(payload["events"][-1]["stage"], "catalog_miss")
        self.assertEqual(payload["events"][-1]["status"], "catalog_miss")
        self.assertEqual(payload["events"][-1]["message"], "Catalog probe missed")
        self.assertEqual(payload["events"][-1]["details"], details)


if __name__ == "__main__":
    unittest.main()
