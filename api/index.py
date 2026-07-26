"""Vercel serverless handler for the unified boundary pipeline.

One pipeline, two cache keys. Each upload is looked up by its raw byte
digest, then by its decoded-pixel digest (which unifies every container
format re-encode of the same image), and only then generated. Results are
cached by outcome class: complete, needs_city, and deterministic failures
are memoized; transient failures never are.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

os.environ.setdefault("MAP_BOUNDARY_CACHE_DIR", "/tmp/map-boundary-builder-cache")

from map_boundary_builder.asset_response import web_asset_response
from map_boundary_builder.image_io import svg_rasterizer_diagnostics
from map_boundary_builder.pipeline_version import get_pipeline_version, pipeline_version_dependency_versions
from map_boundary_builder.request_options import (
    city_hint_for_request,
    extraction_hints_for_request,
    float_field,
    int_field,
)
from map_boundary_builder.runtime_config import ocr_runtime_config
from map_boundary_builder.runtime_warmup import (
    prewarm_generation_runtime,
    should_prewarm_generation_runtime,
)
from map_boundary_builder.upload_payload import (
    UploadPayloadError,
    json_upload_body_limit,
    parse_json_upload_body,
)

DEFAULT_SIMPLIFY_PX = 6.0
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_INLINE_OVERLAY_BYTES = 1_800_000
INLINE_OVERLAY_OPTIMIZE_BYTES = 64_000
INLINE_OVERLAY_MAX_DIMENSION = 1200
CRON_WARM_PATH = "/api/cron/warm-generation-v2"
LEGACY_CRON_WARM_PATH = "/api/cron/warm-generation"
CRON_WARM_PATHS = frozenset({CRON_WARM_PATH, LEGACY_CRON_WARM_PATH})
RUN_RESULT_CACHE_VERSION = "run-result-v9-unified-pipeline"
RUN_RESULT_CACHE_DIR = Path(os.environ["MAP_BOUNDARY_CACHE_DIR"]) / "run-results"
RUN_RESULT_MEMORY_CACHE_MAX = 64
RUN_RESULT_MEMORY_CACHE_MAX_BYTES = 512_000
SUPPORTED_IMAGE_EXTENSIONS = {
    ".avif",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".svg",
}

_RUN_RESULT_MEMORY_CACHE: OrderedDict[str, str] = OrderedDict()
_RUN_RESULT_MEMORY_CACHE_LOCK = threading.RLock()


class RequestError(ValueError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class handler(BaseHTTPRequestHandler):
    server_version = "MapBoundaryVercel/0.2"

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_response(health_response_status(health_payload()))
            self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
            self.end_headers()
            return
        if parsed.path == "/" or parsed.path.startswith("/static/"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in CRON_WARM_PATHS:
                payload, status = cron_warm_generation_payload(
                    authorization_header=self.headers.get("Authorization")
                )
                self.send_json(payload, status=status)
                return
            if parsed.path == "/api/health":
                query = parse_qs(parsed.query)
                payload = health_payload(warm=first_query_value(query, "warm"))
                self.send_json(payload, status=health_response_status(payload))
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
        request_started = time.perf_counter()
        fields, files, upload_encoding = self.parse_upload_request()
        profile: dict[str, Any] = {
            "pipeline_version": get_pipeline_version(),
            "parse_upload_s": elapsed_seconds(request_started),
            "upload_encoding": upload_encoding,
        }
        city = city_hint_for_request(fields)
        upload = files.get("image")
        if upload is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Image upload is required.")
        original_filename, image_bytes = upload
        if not image_bytes:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Uploaded image is empty.")
        profile["upload_bytes"] = len(image_bytes)

        from map_boundary_builder.pipeline import PipelineOptions, run_pipeline

        hints = extraction_hints_for_request(fields) or {}
        options = PipelineOptions(
            simplify_px=float_field(fields, "simplify_px", DEFAULT_SIMPLIFY_PX, 0.0, 10.0),
            min_confidence=float_field(fields, "min_confidence", 0.55, 0.0, 1.0),
            min_control_points=int_field(fields, "min_control_points", 3, 0, 12),
            seed_point=hints.get("seed_point"),  # type: ignore[arg-type]
            target_rgb=hints.get("target_rgb"),  # type: ignore[arg-type]
        )
        run_id = f"{int(time.time())}-{os.urandom(4).hex()}"
        identity = cache_identity(city, options)

        raw_cache_key = run_result_cache_key("image_raw_sha256", hashlib.sha256(image_bytes).hexdigest(), identity)
        cached = read_run_result_cache(raw_cache_key)
        cache_hit = "raw" if cached is not None else None
        visual_cache_key: str | None = None
        if cached is None:
            visual_cache_key = run_result_cache_key(
                "image_pixel_sha256", normalized_image_sha256(image_bytes), identity
            )
            cached = read_run_result_cache(visual_cache_key)
            cache_hit = "visual" if cached is not None else None
        if cached is not None:
            profile["cache_hit"] = cache_hit
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = {
                "id": run_id,
                "filename": Path(original_filename).name or "uploaded-image",
                "percent": 100,
                **cached,
                "profile": profile,
            }
            self.send_json(payload, status=response_status(payload.get("status"), cached=True))
            return

        run_dir = Path(tempfile.gettempdir()) / "map-boundary-builder" / run_id
        debug_dir = run_dir / "debug"
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / f"input{safe_extension(original_filename)}"
        image_path.write_bytes(image_bytes)
        output_path = run_dir / "boundary.geojson"

        events: list[dict[str, Any]] = [
            {"timestamp": time.time(), "stage": "queued", "message": "Run queued", "percent": 1, "status": "queued"}
        ]

        def progress(stage: str, percent: int, detail: str) -> None:
            events.append(
                {
                    "timestamp": time.time(),
                    "stage": stage,
                    "message": detail,
                    "percent": percent,
                    "status": "running",
                }
            )

        build_started = time.perf_counter()
        try:
            result = run_pipeline(
                image_path,
                city=city,
                output_path=output_path,
                debug_dir=debug_dir,
                options=options,
                progress=progress,
            )
        except Exception as exc:
            profile["build_boundary_s"] = elapsed_seconds(build_started)
            profile["cache_hit"] = "miss"
            profile["total_before_send_s"] = elapsed_seconds(request_started)
            payload = {
                "id": run_id,
                "filename": Path(original_filename).name or "uploaded-image",
                "status": "failed",
                "percent": 100,
                "error": str(exc),
                "events": events[-20:],
                "profile": profile,
            }
            self.send_json(payload, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        profile["build_boundary_s"] = elapsed_seconds(build_started)
        profile["build_stage_elapsed_s"] = event_stage_elapsed_seconds(events)

        payload = run_result_payload(result, run_id, original_filename, events)
        if result.cacheable:
            if visual_cache_key is None:
                visual_cache_key = run_result_cache_key(
                    "image_pixel_sha256", normalized_image_sha256(image_bytes), identity
                )
            write_run_result_cache(raw_cache_key, payload)
            write_run_result_cache(visual_cache_key, payload)
        profile["cache_hit"] = "miss"
        profile["total_before_send_s"] = elapsed_seconds(request_started)
        payload["profile"] = profile
        self.send_json(payload, status=response_status(result.status, cached=False))

    def handle_create_report(self) -> None:
        fields, files, _upload_encoding = self.parse_upload_request()
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
        try:
            profile = json.loads(fields.get("profile", "{}") or "{}")
        except json.JSONDecodeError:
            profile = {}
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
                    profile=profile if isinstance(profile, dict) else {},
                )
            )
        except GithubReportError as exc:
            raise RequestError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
        self.send_json(result, status=HTTPStatus.CREATED)

    def parse_upload_request(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]], str]:
        content_type = self.headers.get("Content-Type", "").lower()
        if "multipart/form-data" in content_type:
            fields, files = self.parse_multipart()
            return fields, files, "multipart"
        if "application/json" in content_type:
            fields, files = self.parse_json_upload()
            return fields, files, "json-base64"
        raise RequestError(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data or application/json.")

    def parse_json_upload(self) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Request body is empty.")
        if length > json_upload_body_limit(MAX_UPLOAD_BYTES):
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Upload is larger than {limit_mb} MB.")
        try:
            return parse_json_upload_body(self.rfile.read(length), max_upload_bytes=MAX_UPLOAD_BYTES)
        except UploadPayloadError as exc:
            raise RequestError(exc.status, str(exc)) from exc

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
        try:
            data, mime = web_asset_response(name)
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
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


def run_result_payload(result: Any, run_id: str, original_filename: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    filename = Path(original_filename).name or "uploaded-image"
    if result.status == "complete":
        events.append(
            {
                "timestamp": time.time(),
                "stage": "complete",
                "message": "Boundary complete",
                "percent": 100,
                "status": "complete",
            }
        )
        return {
            "id": run_id,
            "city": result.summary.get("city"),
            "filename": filename,
            "status": "complete",
            "percent": 100,
            "summary": result.summary,
            "events": events[-20:],
            "artifacts": {
                "geojson_inline": result.geojson,
                "overlay_data_url": inline_overlay(result.overlay_path),
            },
        }
    if result.status == "needs_city":
        events.append(
            {
                "timestamp": time.time(),
                "stage": "needs_city",
                "message": "Boundary extracted, but the city could not be determined.",
                "percent": 100,
                "status": "needs_city",
            }
        )
        return {
            "id": run_id,
            "filename": filename,
            "status": "needs_city",
            "percent": 100,
            "summary": result.summary,
            "needs_city": result.summary.get("needs_city"),
            "events": events[-20:],
            "artifacts": {
                "overlay_data_url": inline_overlay(result.overlay_path),
            },
        }
    message = result.summary.get("message") or result.reason or "Generation failed"
    events.append(
        {
            "timestamp": time.time(),
            "stage": "failed",
            "message": str(message),
            "percent": 100,
            "status": "failed",
            "details": {"error": str(message)},
        }
    )
    return {
        "id": run_id,
        "filename": filename,
        "status": "failed",
        "percent": 100,
        "error": str(message),
        "reason": result.reason,
        "summary": result.summary,
        "events": events[-20:],
    }


def response_status(status: str | None, *, cached: bool) -> HTTPStatus:
    if status == "complete":
        return HTTPStatus.OK if cached else HTTPStatus.CREATED
    if status == "needs_city":
        return HTTPStatus.OK
    return HTTPStatus.UNPROCESSABLE_ENTITY


def cache_identity(city: str | None, options: Any) -> dict[str, Any]:
    return {
        "version": RUN_RESULT_CACHE_VERSION,
        "pipeline_version": get_pipeline_version(),
        "city": (city or "").strip().lower(),
        "simplify_px": options.simplify_px,
        "min_confidence": options.min_confidence,
        "min_control_points": options.min_control_points,
        "seed_point": list(options.seed_point) if options.seed_point else None,
        "target_rgb": list(options.target_rgb) if options.target_rgb else None,
    }


def run_result_cache_key(hash_name: str, image_hash: str, identity: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({**identity, hash_name: image_hash}, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


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
    with _RUN_RESULT_MEMORY_CACHE_LOCK:
        cached_json = _RUN_RESULT_MEMORY_CACHE.get(cache_key)
        if cached_json is not None:
            _RUN_RESULT_MEMORY_CACHE.move_to_end(cache_key)
    if cached_json is not None:
        try:
            return json.loads(cached_json)
        except Exception:
            with _RUN_RESULT_MEMORY_CACHE_LOCK:
                _RUN_RESULT_MEMORY_CACHE.pop(cache_key, None)
            return None
    cache_path = RUN_RESULT_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        encoded = cache_path.read_text()
        payload = json.loads(encoded)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    remember_run_result_cache(cache_key, payload, encoded=encoded)
    return payload


def write_run_result_cache(cache_key: str, payload: dict[str, Any]) -> None:
    if payload.get("status") == "failed":
        cached = {
            "status": "failed",
            "error": payload.get("error"),
            "reason": payload.get("reason"),
        }
    else:
        cached = {
            "status": payload.get("status"),
            "city": payload.get("city"),
            "summary": payload.get("summary"),
            "needs_city": payload.get("needs_city"),
            "artifacts": payload.get("artifacts"),
        }
    encoded = remember_run_result_cache(cache_key, cached)
    cache_path = RUN_RESULT_CACHE_DIR / f"{cache_key}.json"
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        RUN_RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(encoded)
        tmp_path.replace(cache_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        return


def remember_run_result_cache(cache_key: str, payload: dict[str, Any], *, encoded: str | None = None) -> str:
    if encoded is None:
        encoded = json.dumps(payload, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > RUN_RESULT_MEMORY_CACHE_MAX_BYTES:
        with _RUN_RESULT_MEMORY_CACHE_LOCK:
            _RUN_RESULT_MEMORY_CACHE.pop(cache_key, None)
        return encoded
    with _RUN_RESULT_MEMORY_CACHE_LOCK:
        _RUN_RESULT_MEMORY_CACHE[cache_key] = encoded
        _RUN_RESULT_MEMORY_CACHE.move_to_end(cache_key)
        while len(_RUN_RESULT_MEMORY_CACHE) > RUN_RESULT_MEMORY_CACHE_MAX:
            _RUN_RESULT_MEMORY_CACHE.popitem(last=False)
    return encoded


def inline_overlay(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    data = path.read_bytes()
    mime = "image/webp" if path.suffix.lower() == ".webp" else "image/png"
    if mime == "image/png" and len(data) > INLINE_OVERLAY_OPTIMIZE_BYTES:
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


def json_response_body(payload: dict[str, Any], *, accept_encoding: str = "") -> tuple[bytes, dict[str, str]]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(data) < 1024 or "gzip" not in accept_encoding.lower():
        return data, {}
    return gzip.compress(data, compresslevel=3), {
        "Content-Encoding": "gzip",
        "Vary": "Accept-Encoding",
    }


def first_query_value(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    if not values:
        return None
    return values[0]


def health_payload(*, warm: str | None = None) -> dict[str, Any]:
    runtime_dependencies = dict(pipeline_version_dependency_versions())
    tmp_writable = os.access(tempfile.gettempdir(), os.W_OK)
    svg_rasterizer = svg_rasterizer_diagnostics()
    payload: dict[str, Any] = {
        "ok": runtime_health_ok(
            runtime_dependencies,
            tmp_writable=tmp_writable,
            svg_rasterizer_ok=svg_rasterizer.get("ok") is True,
        ),
        "runtime": "vercel-python",
        "tesseract": shutil.which("tesseract"),
        "tmp_writable": tmp_writable,
        "pipeline_version": get_pipeline_version(),
        "runtime_dependencies": runtime_dependencies,
        "svg_rasterizer": svg_rasterizer,
        "ocr": ocr_runtime_config(),
    }
    if should_prewarm_generation_runtime(warm):
        warm_payload = prewarm_generation_runtime()
        payload["warm"] = warm_payload
        if not warm_generation_ok(warm_payload):
            payload["ok"] = False
    return payload


def cron_warm_generation_payload(*, authorization_header: str | None) -> tuple[dict[str, Any], HTTPStatus]:
    if not authorized_cron_request(authorization_header):
        return {"ok": False, "error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED
    warm_payload = prewarm_generation_runtime()
    ok = warm_generation_ok(warm_payload)
    return {
        "ok": ok,
        "pipeline_version": get_pipeline_version(),
        "warm": warm_payload,
    }, HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE


def runtime_health_ok(
    runtime_dependencies: dict[str, str],
    *,
    tmp_writable: bool,
    svg_rasterizer_ok: bool = True,
) -> bool:
    if not tmp_writable:
        return False
    if not svg_rasterizer_ok:
        return False
    for dependency in ("numpy", "onnxruntime", "pillow", "rapidocr-onnxruntime", "shapely", "cv2"):
        if runtime_dependencies.get(dependency) in {None, "", "missing", "unknown"}:
            return False
    return True


def warm_generation_ok(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "ok"


def health_response_status(payload: dict[str, Any]) -> HTTPStatus:
    return HTTPStatus.OK if payload.get("ok") is True else HTTPStatus.SERVICE_UNAVAILABLE


def authorized_cron_request(authorization_header: str | None) -> bool:
    secret = os.environ.get("CRON_SECRET", "")
    if not secret:
        return False
    return hmac.compare_digest(authorization_header or "", f"Bearer {secret}")


def elapsed_seconds(started: float) -> float:
    return round(max(0.0, time.perf_counter() - started), 6)


def event_stage_elapsed_seconds(events: list[dict[str, Any]]) -> dict[str, float]:
    timestamped: list[tuple[str, float]] = []
    for event in events:
        stage = event.get("stage")
        timestamp = event.get("timestamp")
        if isinstance(stage, str) and isinstance(timestamp, (int, float)):
            timestamped.append((stage, float(timestamp)))

    totals: dict[str, float] = {}
    for (stage, timestamp), (_, next_timestamp) in zip(timestamped, timestamped[1:]):
        totals[stage] = totals.get(stage, 0.0) + max(0.0, next_timestamp - timestamp)
    return {stage: round(total, 6) for stage, total in totals.items()}
