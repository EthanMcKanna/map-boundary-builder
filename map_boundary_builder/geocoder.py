from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from importlib import resources
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .georef_transform import lonlat_to_mercator

_CACHE_ROOT = Path(os.environ.get("MAP_BOUNDARY_CACHE_DIR", ".cache/map-boundary-builder"))
CACHE_DIR = _CACHE_ROOT / "geocoder"
PHOTON_CACHE_DIR = _CACHE_ROOT / "geocoder-photon"
NOMINATIM_TIMEOUT_SECONDS = float(os.environ.get("MAP_BOUNDARY_NOMINATIM_TIMEOUT_SECONDS", "4.0"))
GEOCODER_SEED_FILE = "geocoder_seed.json"

_NO_SEED = object()
_GEOCODER_SEED: dict[str, object] | None = None


@dataclass(frozen=True)
class GeocodeResult:
    label: str
    lon: float
    lat: float
    display_name: str
    bbox: tuple[float, float, float, float] | None
    importance: float
    place_type: str = ""

    @property
    def mercator(self) -> tuple[float, float]:
        return lonlat_to_mercator(self.lon, self.lat)


def geocode(query: str, *, limit: int = 3, country_codes: str = "us") -> list[GeocodeResult]:
    return geocode_with_network(query, limit=limit, country_codes=country_codes, allow_network=True)


def geocode_cached_only(query: str, *, limit: int = 3, country_codes: str = "us") -> list[GeocodeResult]:
    return geocode_with_network(query, limit=limit, country_codes=country_codes, allow_network=False)


def geocode_with_network(
    query: str,
    *,
    limit: int = 3,
    country_codes: str = "us",
    allow_network: bool,
) -> list[GeocodeResult]:
    query = query.strip()
    if not query:
        return []
    requested_limit = max(1, int(limit))
    cache_limit = max(3, requested_limit)
    return list(_geocode_cached(query, cache_limit, country_codes, allow_network))[:requested_limit]


@lru_cache(maxsize=4096)
def _geocode_cached(query: str, limit: int, country_codes: str, allow_network: bool = True) -> tuple[GeocodeResult, ...]:
    cache_path = cache_file(query, limit, country_codes)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
    else:
        seeded = seed_cache_payload("nominatim", cache_path.stem)
        if seeded is not _NO_SEED:
            payload = seeded
        elif not allow_network:
            payload = []
        else:
            params = {
                "q": query,
                "format": "jsonv2",
                "limit": limit,
                "addressdetails": 0,
                "countrycodes": country_codes,
            }
            request = Request(
                f"https://nominatim.openstreetmap.org/search?{urlencode(params)}",
                headers={
                    "User-Agent": "map-boundary-builder/0.1 local georeferencer",
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=NOMINATIM_TIMEOUT_SECONDS) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception:
                payload = []
            else:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, indent=2) + "\n")

    results = parse_nominatim_payload(payload, query)
    if results:
        return tuple(results)
    return tuple(geocode_photon(query, limit=limit, country_codes=country_codes, allow_network=allow_network))


def parse_nominatim_payload(payload: object, query: str) -> list[GeocodeResult]:
    if not isinstance(payload, list):
        return []
    results: list[GeocodeResult] = []
    for item in payload:
        try:
            bbox = item.get("boundingbox")
            parsed_bbox = None
            if isinstance(bbox, list) and len(bbox) == 4:
                south, north, west, east = map(float, bbox)
                parsed_bbox = (west, south, east, north)
            results.append(
                GeocodeResult(
                    label=query,
                    lon=float(item["lon"]),
                    lat=float(item["lat"]),
                    display_name=str(item.get("display_name", query)),
                    bbox=parsed_bbox,
                    importance=float(item.get("importance", 0.0)),
                    place_type=str(item.get("addresstype") or item.get("type") or item.get("category") or ""),
                )
            )
        except Exception:
            continue
    return results


def geocode_photon(query: str, *, limit: int, country_codes: str, allow_network: bool = True) -> list[GeocodeResult]:
    cache_path = photon_cache_file(query, limit, country_codes)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
    else:
        seeded = seed_cache_payload("photon", cache_path.stem)
        if seeded is not _NO_SEED:
            payload = seeded
        elif not allow_network:
            return []
        else:
            params = {
                "q": query,
                "limit": limit,
                "lang": "en",
            }
            request = Request(
                f"https://photon.komoot.io/api/?{urlencode(params)}",
                headers={
                    "User-Agent": "map-boundary-builder/0.1 local georeferencer",
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception:
                return []
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=2) + "\n")

    if not isinstance(payload, dict):
        return []
    features = payload.get("features")
    if not isinstance(features, list):
        return []

    country_filter = {part.strip().upper() for part in country_codes.split(",") if part.strip()}
    results: list[GeocodeResult] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            continue
        country_code = str(properties.get("countrycode", "")).upper()
        if country_filter and country_code not in country_filter:
            continue
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        try:
            lon = float(coordinates[0])
            lat = float(coordinates[1])
        except Exception:
            continue
        bbox = parse_photon_extent(properties.get("extent"))
        name = str(properties.get("name") or query)
        display_name = photon_display_name(properties, name)
        place_type = str(properties.get("osm_value") or properties.get("type") or properties.get("osm_key") or "")
        results.append(
            GeocodeResult(
                label=query,
                lon=lon,
                lat=lat,
                display_name=display_name,
                bbox=bbox,
                importance=photon_importance(place_type),
                place_type=place_type,
            )
        )
    return results


def parse_photon_extent(extent: object) -> tuple[float, float, float, float] | None:
    if not isinstance(extent, list) or len(extent) != 4:
        return None
    try:
        west, north, east, south = map(float, extent)
        return west, south, east, north
    except Exception:
        return None


def photon_display_name(properties: dict, name: str) -> str:
    parts = [name]
    for key in ("district", "city", "county", "state", "country"):
        value = str(properties.get(key) or "").strip()
        if value and value.lower() not in {part.lower() for part in parts}:
            parts.append(value)
    return ", ".join(parts)


def photon_importance(place_type: str) -> float:
    return {
        "city": 0.72,
        "town": 0.64,
        "village": 0.56,
        "suburb": 0.48,
        "neighbourhood": 0.42,
        "locality": 0.38,
    }.get(place_type.lower(), 0.35)


def seed_cache_payload(provider: str, key: str) -> object:
    seed = load_geocoder_seed()
    provider_payloads = seed.get(provider)
    if not isinstance(provider_payloads, dict) or key not in provider_payloads:
        return _NO_SEED
    return provider_payloads[key]


def load_geocoder_seed() -> dict[str, object]:
    global _GEOCODER_SEED
    if _GEOCODER_SEED is not None:
        return _GEOCODER_SEED
    try:
        seed_file = resources.files("map_boundary_builder").joinpath(GEOCODER_SEED_FILE)
        payload = json.loads(seed_file.read_text())
    except Exception:
        payload = {}
    _GEOCODER_SEED = payload if isinstance(payload, dict) else {}
    return _GEOCODER_SEED


def cache_file(query: str, limit: int, country_codes: str) -> Path:
    key = hashlib.sha256(f"{query}|{limit}|{country_codes}".encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"


def photon_cache_file(query: str, limit: int, country_codes: str) -> Path:
    key = hashlib.sha256(f"photon|{query}|{limit}|{country_codes}".encode("utf-8")).hexdigest()[:24]
    return PHOTON_CACHE_DIR / f"{key}.json"
