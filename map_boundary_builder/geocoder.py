from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .georef_transform import lonlat_to_mercator

_CACHE_ROOT = Path(os.environ.get("MAP_BOUNDARY_CACHE_DIR", ".cache/map-boundary-builder"))
CACHE_DIR = _CACHE_ROOT / "geocoder"


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


_STATIC_GEOCODES: dict[str, GeocodeResult] = {}


def _add_static(
    *queries: str,
    lon: float,
    lat: float,
    display_name: str,
    bbox: tuple[float, float, float, float],
    importance: float,
) -> None:
    result = GeocodeResult(
        label=queries[0],
        lon=lon,
        lat=lat,
        display_name=display_name,
        bbox=bbox,
        importance=importance,
    )
    for query in queries:
        _STATIC_GEOCODES[normalize_query(query)] = result


def normalize_query(query: str) -> str:
    return " ".join(query.lower().replace("-", " ").split())


_add_static(
    "Bay Area",
    "San Francisco Bay Area",
    lon=-122.3558473,
    lat=37.7884969,
    display_name="San Francisco Bay Area, San Francisco, California, United States",
    bbox=(-123.3558473, 36.7884969, -121.3558473, 38.7884969),
    importance=0.6285,
)
_add_static(
    "San Francisco",
    lon=-122.4075201,
    lat=37.7879363,
    display_name="San Francisco, California, United States",
    bbox=(-123.1738250, 37.6403143, -122.2814578, 37.9296678),
    importance=0.7810,
)
_add_static(
    "Las Vegas",
    lon=-115.1485,
    lat=36.1673,
    display_name="Las Vegas, Clark County, Nevada, United States",
    bbox=(-115.4140, 35.9200, -114.9030, 36.3810),
    importance=0.7360,
)
_add_static(
    "Austin",
    lon=-97.7431,
    lat=30.2672,
    display_name="Austin, Travis County, Texas, United States",
    bbox=(-97.9385, 30.0980, -97.5610, 30.5170),
    importance=0.7480,
)
_add_static(
    "San Jose",
    lon=-121.8905910,
    lat=37.3361663,
    display_name="San Jose, Santa Clara County, California, United States",
    bbox=(-122.0462270, 37.1231596, -121.5858438, 37.4691477),
    importance=0.6813,
)
_add_static(
    "Daly City",
    lon=-122.4702,
    lat=37.6879,
    display_name="Daly City, San Mateo County, California, United States",
    bbox=(-122.5140, 37.6490, -122.4320, 37.7200),
    importance=0.55,
)
_add_static(
    "South San Francisco",
    lon=-122.4077,
    lat=37.6547,
    display_name="South San Francisco, San Mateo County, California, United States",
    bbox=(-122.4540, 37.6200, -122.3500, 37.6900),
    importance=0.54,
)
_add_static(
    "Brisbane",
    lon=-122.4194,
    lat=37.6808,
    display_name="Brisbane, San Mateo County, California, United States",
    bbox=(-122.4450, 37.6600, -122.3800, 37.7050),
    importance=0.47,
)
_add_static(
    "San Bruno",
    lon=-122.4111,
    lat=37.6305,
    display_name="San Bruno, San Mateo County, California, United States",
    bbox=(-122.4800, 37.5900, -122.3700, 37.6600),
    importance=0.50,
)
_add_static(
    "Millbrae",
    lon=-122.3872,
    lat=37.5985,
    display_name="Millbrae, San Mateo County, California, United States",
    bbox=(-122.4200, 37.5750, -122.3650, 37.6200),
    importance=0.46,
)
_add_static(
    "Burlingame",
    lon=-122.3630,
    lat=37.5841,
    display_name="Burlingame, San Mateo County, California, United States",
    bbox=(-122.4050, 37.5500, -122.3200, 37.6100),
    importance=0.50,
)
_add_static(
    "San Mateo",
    lon=-122.3255,
    lat=37.5629,
    display_name="San Mateo, San Mateo County, California, United States",
    bbox=(-122.3750, 37.5000, -122.2700, 37.5900),
    importance=0.56,
)
_add_static(
    "Foster City",
    lon=-122.2711,
    lat=37.5585,
    display_name="Foster City, San Mateo County, California, United States",
    bbox=(-122.3000, 37.5200, -122.2350, 37.5850),
    importance=0.46,
)
_add_static(
    "Belmont",
    lon=-122.2758,
    lat=37.5202,
    display_name="Belmont, San Mateo County, California, United States",
    bbox=(-122.3200, 37.4900, -122.2400, 37.5450),
    importance=0.46,
)
_add_static(
    "Redwood City",
    lon=-122.2364,
    lat=37.4852,
    display_name="Redwood City, San Mateo County, California, United States",
    bbox=(-122.2850, 37.4400, -122.1750, 37.5450),
    importance=0.56,
)
_add_static(
    "Atherton",
    lon=-122.2001,
    lat=37.4613,
    display_name="Atherton, San Mateo County, California, United States",
    bbox=(-122.2300, 37.4300, -122.1700, 37.4850),
    importance=0.42,
)
_add_static(
    "Menlo Park",
    lon=-122.1817,
    lat=37.4519,
    display_name="Menlo Park, San Mateo County, California, United States",
    bbox=(-122.2350, 37.4100, -122.1300, 37.4900),
    importance=0.53,
)
_add_static(
    "Palo Alto",
    lon=-122.1430,
    lat=37.4419,
    display_name="Palo Alto, Santa Clara County, California, United States",
    bbox=(-122.1900, 37.3700, -122.0900, 37.4650),
    importance=0.61,
)
_add_static(
    "Los Altos",
    lon=-122.1141,
    lat=37.3852,
    display_name="Los Altos, Santa Clara County, California, United States",
    bbox=(-122.1700, 37.3300, -122.0700, 37.4100),
    importance=0.52,
)
_add_static(
    "Mountain View",
    lon=-122.0838,
    lat=37.3861,
    display_name="Mountain View, Santa Clara County, California, United States",
    bbox=(-122.1300, 37.3500, -122.0400, 37.4350),
    importance=0.58,
)
_add_static(
    "Sunnyvale",
    lon=-122.0363,
    lat=37.3688,
    display_name="Sunnyvale, Santa Clara County, California, United States",
    bbox=(-122.0800, 37.3300, -121.9700, 37.4200),
    importance=0.58,
)


def geocode(query: str, *, limit: int = 3, country_codes: str = "us") -> list[GeocodeResult]:
    query = query.strip()
    if not query:
        return []
    static = _STATIC_GEOCODES.get(normalize_query(query))
    if static is not None and country_codes == "us":
        return [static][:limit]

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
