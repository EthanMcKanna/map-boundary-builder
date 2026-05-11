# Map Boundary Builder

`map-boundary-builder` turns a service-map screenshot into a georeferenced
GeoJSON polygon. It ships as a CLI, a local browser workspace, and a hosted
Vercel app.

It does three things automatically:

1. Detects service-area fills across light, bright-blue, green, and dark map styles.
2. Repairs text, road shields, highway lines, and small rendering gaps in the mask.
3. Georeferences the pixel polygon from readable map labels and public map data.

The repo intentionally does not include provider screenshots or bundled example
maps. Bring your own crop from a map view you have permission to use.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .

.venv/bin/map-boundary \
  --image /path/to/service-map.png \
  --city "Las Vegas" \
  --output out/boundary.geojson \
  --debug-dir out/debug-boundary \
  --print-summary
```

The output is a GeoJSON `FeatureCollection` with the extracted polygon in
longitude/latitude coordinates and metadata describing the extraction strategy,
georeference fit, pixel coverage, and confidence.

## Requirements

- Python 3.11 or newer
- Internet access for OpenStreetMap/Nominatim lookups during georeferencing
- The hosted and local web apps run OCR in the browser.
- The CLI uses local Tesseract OCR when available (`brew install tesseract` on macOS), then fails closed/falls back if it cannot infer enough map evidence.

## Interactive Web Tool

The same pipeline is available as an end-to-end web workspace:

```bash
.venv/bin/python -m map_boundary_builder.web
```

Open `http://127.0.0.1:8765`, drop in a service-map screenshot, enter a city,
and run the builder. The web tool streams each stage as it happens, writes run
artifacts under `out/web-runs/<run-id>/`, previews the extracted mask overlay,
renders the generated boundary, and exposes the final GeoJSON for download or
copying.

The hosted Vercel app is available at
`https://map-boundary-builder.vercel.app`. It runs browser-side OCR plus the
same Python extraction/georeferencing backend as a serverless function. Large or
low-detail screenshots can still time out or fail closed if there is not enough
OCR/geocoded map evidence.

## Georeferencing Model

The final product has no city presets, manual georeference flags, or
ground-truth-reference fitting. It infers map position from OCR-detected labels,
matches labels against cached OpenStreetMap place names near the requested city,
and fits a rotation-aware Web Mercator transform only when there are enough
control points with low residual error. When that label fit is viable, a local
OpenStreetMap road-network refinement can tune scale, rotation, and origin
against visible road structure in the image.

If label control points are not available, the CLI can attempt a lower-confidence
city-context road search using the requested city and public OpenStreetMap road
data. For low-resolution map crops with visible street grids, it can rerank
candidate transforms by matching detected image line segments against projected
OpenStreetMap road segments. If the map does not contain enough readable labels
or public-map structure, the CLI fails instead of falling back to a hardcoded
city/provider boundary.
