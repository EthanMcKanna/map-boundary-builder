from __future__ import annotations

import re

AUTO_CITY_TOKENS = {"auto", "automatic", "autodetect", "detect"}
FALSE_BOOLEAN_TOKENS = {"0", "false", "no", "off", ""}


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
    if bool_field(fields, "no_catalog", default=False):
        return False
    return bool_field(fields, "allow_catalog", default=True)
