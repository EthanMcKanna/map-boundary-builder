import os
import unittest
from http import HTTPStatus
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
            {},
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


if __name__ == "__main__":
    unittest.main()
