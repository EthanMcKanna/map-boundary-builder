from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

os.environ.setdefault("MAP_BOUNDARY_CACHE_DIR", "/tmp/map-boundary-builder-cache")

from map_boundary_builder.extract import DEFAULT_SIMPLIFY_PX
from map_boundary_builder.github_reports import FailureReport, GithubReportError, create_failure_issue
from map_boundary_builder.runner import BoundaryBuildOptions, build_boundary
from map_boundary_builder.ocr import parse_client_ocr_labels
from map_boundary_builder.web import RequestError, float_field, int_field, safe_extension

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_INLINE_OVERLAY_BYTES = 1_800_000


class handler(BaseHTTPRequestHandler):
    server_version = "MapBoundaryVercel/0.1"

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path.startswith("/static/") or parsed.path == "/api/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self.send_json(
                    {
                        "ok": True,
                        "runtime": "vercel-python",
                        "tesseract": shutil.which("tesseract"),
                        "tmp_writable": os.access(tempfile.gettempdir(), os.W_OK),
                    }
                )
                return
            if parsed.path == "/":
                self.send_asset("index.html")
                return
            if parsed.path.startswith("/static/"):
                self.send_asset(unquote(parsed.path.removeprefix("/static/")))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except RequestError as exc:
            self.send_json({"error": str(exc)}, status=exc.status)
        except BrokenPipeError:
            return

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                self.handle_create_run()
                return
            if parsed.path == "/api/reports":
                self.handle_create_report()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except RequestError as exc:
            self.send_json({"error": str(exc)}, status=exc.status)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_create_run(self) -> None:
        fields, files = self.parse_multipart()
        city = fields.get("city", "").strip() or None
        upload = files.get("image")
        if upload is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Image upload is required.")
        original_filename, image_bytes = upload
        if not image_bytes:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Uploaded image is empty.")

        run_id = f"{int(time.time())}-{os.urandom(4).hex()}"
        run_dir = Path(tempfile.gettempdir()) / "map-boundary-builder" / run_id
        debug_dir = run_dir / "debug"
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / f"input{safe_extension(original_filename)}"
        output_path = run_dir / "boundary.geojson"
        image_path.write_bytes(image_bytes)

        events: list[dict[str, Any]] = [
            {"stage": "queued", "message": "Run queued", "percent": 1, "status": "queued"}
        ]

        def progress(event: dict[str, Any]) -> None:
            events.append({"timestamp": time.time(), **event})

        options = BoundaryBuildOptions(
            simplify_px=float_field(fields, "simplify_px", DEFAULT_SIMPLIFY_PX, 0.0, 10.0),
            min_confidence=float_field(fields, "min_confidence", 0.55, 0.0, 1.0),
            min_control_points=int_field(fields, "min_control_points", 3, 0, 12),
        )
        result = build_boundary(
            image_path,
            city,
            output_path,
            debug_dir=debug_dir,
            options=options,
            progress=progress,
            ocr_labels=parse_client_ocr_labels(fields.get("ocr_labels")),
        )
        payload = {
            "id": run_id,
            "city": result.summary["city"],
            "filename": Path(original_filename).name or "uploaded-image",
            "status": "complete",
            "percent": 100,
            "summary": result.summary,
            "events": events[-20:],
            "artifacts": {
                "geojson_inline": result.geojson,
                "overlay_data_url": inline_overlay(result.overlay_path),
            },
        }
        self.send_json(payload, status=HTTPStatus.CREATED)

    def handle_create_report(self) -> None:
        fields, files = self.parse_multipart()
        upload = files.get("image")
        if upload is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Image upload is required.")
        original_filename, image_bytes = upload
        try:
            events = json.loads(fields.get("events", "[]") or "[]")
        except json.JSONDecodeError:
            events = []
        try:
            settings = json.loads(fields.get("settings", "{}") or "{}")
        except json.JSONDecodeError:
            settings = {}
        try:
            result = create_failure_issue(
                FailureReport(
                    filename=original_filename,
                    image_bytes=image_bytes,
                    error=fields.get("error", "").strip() or "Generation failed without an error message.",
                    run_id=fields.get("run_id", "").strip() or None,
                    events=events if isinstance(events, list) else [],
                    user_agent=fields.get("user_agent", "").strip() or self.headers.get("User-Agent"),
                    page_url=fields.get("page_url", "").strip() or None,
                    settings=settings if isinstance(settings, dict) else {},
                )
            )
        except GithubReportError as exc:
            raise RequestError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
        self.send_json(result, status=HTTPStatus.CREATED)

    def parse_multipart(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        from map_boundary_builder.web import BoundaryWebHandler

        return BoundaryWebHandler.parse_multipart(self)

    def send_asset(self, name: str) -> None:
        if "/" in name or "\\" in name or name.startswith("."):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        asset = resources.files("map_boundary_builder").joinpath("web_assets", name)
        if not asset.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = asset.read_bytes()
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if name.endswith(".js"):
            mime = "text/javascript; charset=utf-8"
        elif name.endswith(".css"):
            mime = "text/css; charset=utf-8"
        elif name.endswith(".html"):
            mime = "text/html; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def inline_overlay(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    data = path.read_bytes()
    mime = "image/png"
    if len(data) > MAX_INLINE_OVERLAY_BYTES:
        try:
            from PIL import Image

            jpeg_path = path.with_suffix(".jpg")
            with Image.open(path) as image:
                image.convert("RGB").save(jpeg_path, format="JPEG", quality=82, optimize=True)
            data = jpeg_path.read_bytes()
            mime = "image/jpeg"
        except Exception:
            return None
    if len(data) > MAX_INLINE_OVERLAY_BYTES:
        return None
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
