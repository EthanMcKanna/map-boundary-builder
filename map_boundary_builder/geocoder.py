from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .georef_transform import lonlat_to_mercator

CACHE_DIR = Path(".cache/map-boundary-builder/geocoder")


@dataclass(frozen=True)
class GeocodeResult:
    label: str
    lon: float
    lat: float
    display_name: str
    bbox: tuple[float, float, float, float] | None
    importance: float

    @property
    def mercator(self) -> tuple[float, float]:
        return lonlat_to_mercator(self.lon, self.lat)


def geocode(query: str, *, limit: int = 3, country_codes: str = "us") -> list[GeocodeResult]:
    query = query.strip()
    if not query:
        return []

    cache_path = cache_file(query, limit, country_codes)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
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
            with urlopen(request, timeout=4) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2) + "\n")

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
                )
            )
        except Exception:
            continue
    return results


def cache_file(query: str, limit: int, country_codes: str) -> Path:
    key = hashlib.sha256(f"{query}|{limit}|{country_codes}".encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"
