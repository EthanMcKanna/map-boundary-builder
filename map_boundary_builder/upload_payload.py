from __future__ import annotations

import base64
import binascii
import json
from http import HTTPStatus
from pathlib import Path
from typing import Any


class UploadPayloadError(ValueError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def json_upload_body_limit(max_upload_bytes: int) -> int:
    return int(max_upload_bytes * 4 / 3) + 1_048_576


def parse_json_upload_body(
    body: bytes,
    *,
    max_upload_bytes: int,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UploadPayloadError(HTTPStatus.BAD_REQUEST, "Invalid JSON upload payload.") from exc
    if not isinstance(payload, dict):
        raise UploadPayloadError(HTTPStatus.BAD_REQUEST, "Expected a JSON object upload payload.")

    image_payload = payload.get("image")
    if not isinstance(image_payload, dict):
        raise UploadPayloadError(HTTPStatus.BAD_REQUEST, "Image upload is required.")

    filename = Path(str(image_payload.get("filename") or "uploaded-image")).name or "uploaded-image"
    image_base64 = image_payload.get("data_base64") or image_payload.get("base64") or image_payload.get("data")
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise UploadPayloadError(HTTPStatus.BAD_REQUEST, "Image upload data is required.")
    image_bytes = decode_json_upload_image(image_base64)
    if not image_bytes:
        raise UploadPayloadError(HTTPStatus.BAD_REQUEST, "Uploaded image is empty.")
    if len(image_bytes) > max_upload_bytes:
        limit_mb = max_upload_bytes // (1024 * 1024)
        raise UploadPayloadError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Upload is larger than {limit_mb} MB.")

    fields = json_upload_fields(payload)
    return fields, {"image": (filename, image_bytes)}


def decode_json_upload_image(image_base64: str) -> bytes:
    encoded = image_base64.strip()
    if encoded.startswith("data:"):
        _prefix, separator, encoded = encoded.partition(",")
        if not separator:
            raise UploadPayloadError(HTTPStatus.BAD_REQUEST, "Invalid data URL image upload.")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UploadPayloadError(HTTPStatus.BAD_REQUEST, "Invalid base64 image upload.") from exc


def json_upload_fields(payload: dict[str, Any]) -> dict[str, str]:
    field_payload: dict[str, Any] = {
        key: value for key, value in payload.items() if key not in {"image", "fields"}
    }
    nested_fields = payload.get("fields")
    if isinstance(nested_fields, dict):
        field_payload.update(nested_fields)
    fields: dict[str, str] = {}
    for key, value in field_payload.items():
        if value is None:
            continue
        fields[str(key)] = json_field_value(value)
    return fields


def json_field_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))
