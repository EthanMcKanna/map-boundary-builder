from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .georef_transform import lonlat_to_mercator

_CACHE_ROOT = Path(os.environ.get("MAP_BOUNDARY_CACHE_DIR", ".cache/map-boundary-builder"))
CACHE_DIR = _CACHE_ROOT / "overpass-places"


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
