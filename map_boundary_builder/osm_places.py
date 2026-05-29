from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
import json
import os
from importlib import resources
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .georef_transform import lonlat_to_mercator
from .network_policy import network_blocked

_CACHE_ROOT = Path(os.environ.get("MAP_BOUNDARY_CACHE_DIR", ".cache/map-boundary-builder"))
CACHE_DIR = _CACHE_ROOT / "overpass-places"
OSM_PLACES_SEED_FILE = "osm_places_seed.json.gz"

_NO_SEED = object()
_OSM_PLACES_SEED: dict[str, object] | None = None


@dataclass(frozen=True)
class PlacePoint:
    name: str
    place_type: str
    lon: float
    lat: float

    @property
    def mercator(self) -> tuple[float, float]:
        return lonlat_to_mercator(self.lon, self.lat)


@lru_cache(maxsize=256)
def load_place_points(bbox: tuple[float, float, float, float]) -> list[PlacePoint]:
    payload = load_overpass_places(bbox)
    places: list[PlacePoint] = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        center = element.get("center", {})
        lon = element.get("lon", center.get("lon"))
        lat = element.get("lat", center.get("lat"))
        if lon is None or lat is None:
            continue
        try:
            places.append(
                PlacePoint(
                    name=str(name),
                    place_type=str(tags.get("place", "")),
                    lon=float(lon),
                    lat=float(lat),
                )
            )
        except Exception:
            continue
    return places


@lru_cache(maxsize=256)
def load_overpass_places(bbox: tuple[float, float, float, float]) -> dict[str, object]:
    west, south, east, north = bbox
    cache_path = overpass_places_cache_file(bbox)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    seeded = seed_overpass_places_payload(cache_path.stem, bbox=bbox)
    if seeded is not _NO_SEED:
        return seeded if isinstance(seeded, dict) else {"elements": []}
    if network_blocked():
        return {"elements": []}
    query = f"""
[out:json][timeout:35];
(
  node[place~"city|town|village|suburb|neighbourhood|quarter|locality|hamlet"]({south},{west},{north},{east});
  way[place~"city|town|village|suburb|neighbourhood|quarter|locality|hamlet"]({south},{west},{north},{east});
  relation[place~"city|town|village|suburb|neighbourhood|quarter|locality|hamlet"]({south},{west},{north},{east});
);
out center tags;
"""
    request = Request(
        "https://overpass-api.de/api/interpreter",
        data=urlencode({"data": query}).encode("utf-8"),
        headers={"User-Agent": "map-boundary-builder/0.1 local place matcher"},
    )
    try:
        with urlopen(request, timeout=50) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {"elements": []}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload) + "\n")
    return payload


def overpass_places_cache_file(bbox: tuple[float, float, float, float]) -> Path:
    rounded = ",".join(f"{value:.4f}" for value in bbox)
    key = hashlib.sha256(rounded.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"


def seed_overpass_places_payload(key: str, bbox: tuple[float, float, float, float] | None = None) -> object:
    seed = load_osm_places_seed()
    place_payloads = seed.get("overpass_places")
    if not isinstance(place_payloads, dict):
        return _NO_SEED
    if key in place_payloads:
        return place_payloads[key]
    if bbox is None:
        return _NO_SEED
    return covering_seed_payload(place_payloads, bbox)


def covering_seed_payload(
    place_payloads: dict[str, object],
    bbox: tuple[float, float, float, float],
) -> object:
    best: tuple[float, object] | None = None
    for payload in place_payloads.values():
        if not isinstance(payload, dict):
            continue
        bounds = payload_bounds(payload)
        if bounds is None or not bounds_cover_bbox(bounds, bbox):
            continue
        west, south, east, north = bounds
        area = max(0.0, east - west) * max(0.0, north - south)
        if best is None or area < best[0]:
            best = (area, payload)
    return best[1] if best is not None else _NO_SEED


def payload_bounds(payload: dict[str, object]) -> tuple[float, float, float, float] | None:
    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        return None
    lons: list[float] = []
    lats: list[float] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        center = element.get("center", {})
        if not isinstance(center, dict):
            center = {}
        lon = element.get("lon", center.get("lon"))
        lat = element.get("lat", center.get("lat"))
        try:
            lons.append(float(lon))
            lats.append(float(lat))
        except (TypeError, ValueError):
            continue
    if not lons or not lats:
        return None
    return min(lons), min(lats), max(lons), max(lats)


def bounds_cover_bbox(
    bounds: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
    *,
    tolerance_degrees: float = 0.03,
) -> bool:
    west, south, east, north = bbox
    seed_west, seed_south, seed_east, seed_north = bounds
    return (
        west >= seed_west - tolerance_degrees
        and south >= seed_south - tolerance_degrees
        and east <= seed_east + tolerance_degrees
        and north <= seed_north + tolerance_degrees
    )


def load_osm_places_seed() -> dict[str, object]:
    global _OSM_PLACES_SEED
    if _OSM_PLACES_SEED is not None:
        return _OSM_PLACES_SEED
    try:
        seed_file = resources.files("map_boundary_builder").joinpath(OSM_PLACES_SEED_FILE)
        payload = json.loads(gzip.decompress(seed_file.read_bytes()).decode("utf-8"))
    except Exception:
        payload = {}
    _OSM_PLACES_SEED = payload if isinstance(payload, dict) else {}
    return _OSM_PLACES_SEED
