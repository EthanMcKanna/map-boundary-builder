from __future__ import annotations

import math
import re

AUTO_CITY_TOKENS = {"auto", "automatic", "autodetect", "detect"}
FALSE_BOOLEAN_TOKENS = {"0", "false", "no", "off", ""}
EXPERIMENTAL_CLASSIFIER_TOKENS = {
    "experimental",
    "experimentalclassifier",
    "experimentalmodel",
    "experimentalonnx",
    "model",
    "onnx",
}
GENERALIZED_CLASSIFIER_TOKENS = {
    "generalized",
    "generalizedclassifier",
    "generalizedv11",
    "v11",
}


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


def bool_field(fields: dict[str, str], name: str, *, default: bool) -> bool:
    value = fields.get(name)
    if value is None:
        return default
    return value.strip().lower() not in FALSE_BOOLEAN_TOKENS


def city_hint_for_request(fields: dict[str, str]) -> str | None:
    city = fields.get("city", "").strip()
    if not city:
        return None
    if re.sub(r"[^a-z0-9]+", "", city.lower()) in AUTO_CITY_TOKENS:
        return None
    return city


def include_overlay_for_request(fields: dict[str, str], *, catalog_probe_only: bool) -> bool:
    return bool_field(fields, "include_overlay", default=not catalog_probe_only)


def allow_catalog_for_request(fields: dict[str, str]) -> bool:
    """Catalog substitution is intentionally disabled for public uploads.

    A newly uploaded service-area image must be interpreted from its own pixels.
    Request fields are ignored so stale clients cannot re-enable exact geometry
    replacement from the bundled historical catalog.
    """
    return False


def experimental_classifier_for_request(fields: dict[str, str]) -> bool:
    extractor = fields.get("extractor", "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "", extractor)
    return normalized in EXPERIMENTAL_CLASSIFIER_TOKENS


def extractor_for_request(fields: dict[str, str]) -> str:
    extractor = fields.get("extractor", "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "", extractor)
    if normalized in GENERALIZED_CLASSIFIER_TOKENS:
        return "generalized_v11"
    if normalized in EXPERIMENTAL_CLASSIFIER_TOKENS:
        return "experimental_classifier"
    return "deterministic"


def generalized_classifier_for_request(fields: dict[str, str]) -> bool:
    return extractor_for_request(fields) == "generalized_v11"


def extraction_hints_for_request(fields: dict[str, str]) -> dict[str, object] | None:
    hints: dict[str, object] = {}
    seed_x = fields.get("seed_x", "").strip()
    seed_y = fields.get("seed_y", "").strip()
    if seed_x and seed_y:
        try:
            point = (float(seed_x), float(seed_y))
            if all(math.isfinite(value) and value >= 0.0 for value in point):
                hints["seed_point"] = point
        except ValueError:
            pass

    target = fields.get("target_color", "").strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{6}", target):
        hints["target_rgb"] = tuple(int(target[index : index + 2], 16) for index in (0, 2, 4))
    return hints or None
