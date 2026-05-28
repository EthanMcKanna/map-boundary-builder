from __future__ import annotations

import base64
import gzip
import hashlib
import json
import mimetypes
import os
import re
import shutil
import tempfile
import time
from io import BytesIO
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse

os.environ.setdefault("MAP_BOUNDARY_CACHE_DIR", "/tmp/map-boundary-builder-cache")

from map_boundary_builder.pipeline_version import get_pipeline_version

DEFAULT_SIMPLIFY_PX = 6.0
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_INLINE_OVERLAY_BYTES = 1_800_000
INLINE_OVERLAY_OPTIMIZE_BYTES = 64_000
INLINE_OVERLAY_MAX_DIMENSION = 1200
RUN_RESULT_CACHE_VERSION = "run-result-v3"
RUN_RESULT_CACHE_DIR = Path(os.environ["MAP_BOUNDARY_CACHE_DIR"]) / "run-results"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".svg", ".svgz"}


class RequestError(ValueError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


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
                        "pipeline_version": get_pipeline_version(),
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

        events: list[dict[str, Any]] = [
            {"stage": "queued", "message": "Run queued", "percent": 1, "status": "queued"}
        ]

        def progress(event: dict[str, Any]) -> None:
            events.append({"timestamp": time.time(), **event})

        options = SimpleNamespace(
            simplify_px=float_field(fields, "simplify_px", DEFAULT_SIMPLIFY_PX, 0.0, 10.0),
            min_confidence=float_field(fields, "min_confidence", 0.55, 0.0, 1.0),
            min_control_points=int_field(fields, "min_control_points", 3, 0, 12),
            preview_max_dimension=INLINE_OVERLAY_MAX_DIMENSION,
            write_mask_artifact=False,
        )
        run_id = f"{int(time.time())}-{os.urandom(4).hex()}"
        raw_cache_key = raw_run_result_cache_key(image_bytes, city, options)
        cached = read_run_result_cache(raw_cache_key)
        if cached is not None:
            self.send_json(
                cached_run_payload(cached, run_id, original_filename, events),
                status=HTTPStatus.CREATED,
            )
            return

        cache_key = run_result_cache_key(image_bytes, city, options)
        cached = read_run_result_cache(cache_key)
        if cached is not None:
            write_run_result_cache(raw_cache_key, cached)
            self.send_json(
                cached_run_payload(cached, run_id, original_filename, events),
                status=HTTPStatus.CREATED,
            )
            return

        run_dir = Path(tempfile.gettempdir()) / "map-boundary-builder" / run_id
        debug_dir = run_dir / "debug"
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / f"input{safe_extension(original_filename)}"
        output_path = run_dir / "boundary.geojson"
        image_path.write_bytes(image_bytes)

        try:
            from map_boundary_builder.runner import build_boundary

            result = build_boundary(
                image_path,
                city,
                output_path,
                debug_dir=debug_dir,
                options=options,
                progress=progress,
            )
        except Exception as exc:
            self.send_json({"error": str(exc), "events": events[-20:]}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
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
        write_run_result_cache(cache_key, payload)
        write_run_result_cache(raw_cache_key, payload)
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
            summary = json.loads(fields.get("summary", "{}") or "{}")
        except json.JSONDecodeError:
            summary = {}
        from map_boundary_builder.github_reports import FailureReport, GithubReportError, create_failure_issue

        try:
            result = create_failure_issue(
                FailureReport(
                    filename=original_filename,
                    image_bytes=image_bytes,
                    error=fields.get("error", "").strip() or "Generation failed without an error message.",
                    issue_type=fields.get("issue_type", "").strip() or "Generation issue",
                    generation_status=fields.get("generation_status", "").strip() or "unknown",
                    user_note=fields.get("user_note", "").strip() or None,
                    run_id=fields.get("run_id", "").strip() or None,
                    events=events if isinstance(events, list) else [],
                    user_agent=fields.get("user_agent", "").strip() or self.headers.get("User-Agent"),
                    page_url=fields.get("page_url", "").strip() or None,
                    settings=settings if isinstance(settings, dict) else {},
                    summary=summary if isinstance(summary, dict) else {},
                )
            )
        except GithubReportError as exc:
            raise RequestError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
        self.send_json(result, status=HTTPStatus.CREATED)

    def parse_multipart(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        content_type = self.headers.get("Content-Type", "")
        match = re.search(r'boundary="?([^";]+)"?', content_type)
        if "multipart/form-data" not in content_type or match is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data.")
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Request body is empty.")
        if length > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Upload is larger than {limit_mb} MB.")

        body = self.rfile.read(length)
        boundary = ("--" + match.group(1)).encode()
        fields: dict[str, str] = {}
        files: dict[str, tuple[str, bytes]] = {}
        for raw_part in body.split(boundary):
            part = raw_part
            if part.startswith(b"\r\n"):
                part = part[2:]
            if part.endswith(b"\r\n"):
                part = part[:-2]
            if not part or part == b"--":
                continue
            if part.endswith(b"--"):
                part = part[:-2].rstrip(b"\r\n")
            header_blob, separator, content = part.partition(b"\r\n\r\n")
            if not separator:
                continue
            headers = header_blob.decode("utf-8", "replace").split("\r\n")
            disposition = next(
                (header for header in headers if header.lower().startswith("content-disposition:")),
                "",
            )
            name_match = re.search(r'name="([^"]+)"', disposition)
            if name_match is None:
                continue
            field_name = name_match.group(1)
            filename_match = re.search(r'filename="([^"]*)"', disposition)
            if filename_match is not None:
                files[field_name] = (filename_match.group(1), content)
            else:
                fields[field_name] = content.decode("utf-8", "replace").strip()
        return fields, files

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
        data, extra_headers = json_response_body(
            payload,
            accept_encoding=self.headers.get("Accept-Encoding", ""),
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        for name, value in extra_headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def inline_overlay(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    data = path.read_bytes()
    mime = "image/png"
    if len(data) > INLINE_OVERLAY_OPTIMIZE_BYTES:
        optimized = optimized_overlay_bytes(path, original_size=len(data))
        if optimized is not None:
            mime, data = optimized
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


def optimized_overlay_bytes(path: Path, *, original_size: int) -> tuple[str, bytes] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            max_dimension = max(rgb.size)
            if max_dimension > INLINE_OVERLAY_MAX_DIMENSION:
                scale = max_dimension / INLINE_OVERLAY_MAX_DIMENSION
                size = (
                    max(1, int(round(rgb.width / scale))),
                    max(1, int(round(rgb.height / scale))),
                )
                rgb = rgb.resize(size, Image.Resampling.LANCZOS)
            webp = BytesIO()
            rgb.save(webp, format="WEBP", quality=90, method=4)
            webp_data = webp.getvalue()
            if webp_data and len(webp_data) < original_size:
                return "image/webp", webp_data
    except Exception:
        return None
    return None


def safe_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return ext if ext in SUPPORTED_IMAGE_EXTENSIONS else ".png"


def float_field(fields: dict[str, str], name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(fields.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def int_field(fields: dict[str, str], name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(fields.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def json_response_body(payload: dict[str, Any], *, accept_encoding: str = "") -> tuple[bytes, dict[str, str]]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(data) < 1024 or "gzip" not in accept_encoding.lower():
        return data, {}
    return gzip.compress(data, compresslevel=3), {
        "Content-Encoding": "gzip",
        "Vary": "Accept-Encoding",
    }


def run_result_cache_key(image_bytes: bytes, city: str | None, options: Any) -> str:
    return run_result_cache_key_for_hash("image_pixel_sha256", normalized_image_sha256(image_bytes), city, options)


def raw_run_result_cache_key(image_bytes: bytes, city: str | None, options: Any) -> str:
    return run_result_cache_key_for_hash("image_raw_sha256", hashlib.sha256(image_bytes).hexdigest(), city, options)


def run_result_cache_key_for_hash(
    image_hash_name: str,
    image_hash: str,
    city: str | None,
    options: Any,
) -> str:
    parts = {
        "version": RUN_RESULT_CACHE_VERSION,
        "pipeline_version": get_pipeline_version(),
        image_hash_name: image_hash,
        "city": city or "",
        "simplify_px": round(float(options.simplify_px), 4),
        "min_confidence": round(float(options.min_confidence), 4),
        "min_control_points": int(options.min_control_points),
        "preview_max_dimension": getattr(options, "preview_max_dimension", None) or "",
        "write_mask_artifact": bool(getattr(options, "write_mask_artifact", True)),
    }
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_image_sha256(image_bytes: bytes) -> str:
    try:
        from PIL import Image, ImageOps

        with Image.open(BytesIO(image_bytes)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGBA")
            digest = hashlib.sha256()
            digest.update(str(normalized.size).encode("ascii"))
            digest.update(normalized.mode.encode("ascii"))
            digest.update(normalized.tobytes())
            return digest.hexdigest()
    except Exception:
        return hashlib.sha256(image_bytes).hexdigest()


def read_run_result_cache(cache_key: str) -> dict[str, Any] | None:
    cache_path = RUN_RESULT_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_run_result_cache(cache_key: str, payload: dict[str, Any]) -> None:
    cached = {
        "city": payload.get("city"),
        "summary": payload.get("summary"),
        "artifacts": payload.get("artifacts"),
    }
    try:
        RUN_RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = RUN_RESULT_CACHE_DIR / f"{cache_key}.json"
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(cached, separators=(",", ":")))
        tmp_path.replace(cache_path)
    except OSError:
        return


def cached_run_payload(
    cached: dict[str, Any],
    run_id: str,
    original_filename: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(cached))
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    payload.update(
        {
            "id": run_id,
            "city": payload.get("city") or summary.get("city"),
            "filename": Path(original_filename).name or "uploaded-image",
            "status": "complete",
            "percent": 100,
            "cached": True,
            "events": [
                *events,
                {
                    "timestamp": time.time(),
                    "stage": "complete",
                    "message": "Boundary export ready from cache",
                    "percent": 100,
                    "status": "complete",
                    "details": summary,
                },
            ],
        }
    )
    return payload
