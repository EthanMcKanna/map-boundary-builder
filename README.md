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
- The CLI uses local Tesseract OCR when available (`brew install tesseract` on macOS), then fails closed if it cannot infer enough map evidence.

## Interactive Web Tool

The same pipeline is available as an end-to-end web workspace:

```bash
.venv/bin/python -m map_boundary_builder.web
```

Open `http://127.0.0.1:8765`, drop in a service-map screenshot, and run the
builder. The web tool streams each stage as it happens, writes run
artifacts under `out/web-runs/<run-id>/`, previews the extracted mask overlay,
renders the generated boundary, and exposes the final GeoJSON for download or
copying.

The hosted Vercel app is available at
`https://map-boundary-builder.vercel.app`. It runs browser-side OCR plus the
same Python extraction/georeferencing backend as a serverless function. Large or
low-detail screenshots can still time out or fail closed if there is not enough
OCR/geocoded map evidence.

When a web generation fails, or when a completed boundary looks wrong, the UI
can create a GitHub debug report. Set `GITHUB_REPORT_TOKEN` in the deployment
environment with access to create issues and write repository contents. The app
stores the reported screenshot on the public `debug-reports` branch, embeds it
in the issue, and warns the user that the uploaded image will be public before
they report it. Optional overrides are `GITHUB_REPORT_REPOSITORY` and
`GITHUB_REPORT_BRANCH`.

## Georeferencing Model

The final product has no provider boundary presets, manual georeference flags,
or ground-truth-reference fitting. It infers map position from OCR-detected
labels, queries public OpenStreetMap-backed geocoders with a cached fallback,
clusters geocoded label candidates to infer the map city or region, matches
labels against cached OpenStreetMap place names near that inferred location, and
fits a rotation-aware Web Mercator transform only when there are enough control
points with low residual error. When multiple inferred contexts fit, the builder
scores them by residuals, control-point support, and visible-road alignment so
large regional screenshots are not forced into a single city scale. The CLI still
accepts `--city` as an optional override for unusually sparse screenshots.

If label control points are not available and a city override is supplied, the
CLI can attempt a lower-confidence city-context road search using public
OpenStreetMap road data. For low-resolution map crops with visible street grids,
it can rerank candidate transforms by matching detected image line segments
against projected OpenStreetMap road segments. If the map does not contain
enough readable labels or public-map structure, the tool fails instead of falling
back to a hardcoded city/provider boundary.

## Service-Area Benchmark

The local benchmark compares known service-area screenshots with reference
polygons from the Robotaxi Tracker service-area registry. By default it runs the
fast extraction benchmark: it extracts the pixel polygon, fits that shape into
the reference bounds, and reports IoU, area ratio, vertex count, and unmatched
inventory. Fixture-status overrides live in
`benchmarks/service-area-fixtures.json`, so known bad screenshot/reference pairs
stay visible as skipped data debt instead of weakening the model score.

```bash
.venv/bin/map-boundary-benchmark \
  --polygon-dir /Users/ethanmckanna/GitHub/av-coverage-checker/data/service-areas/polygons \
  --image-dir "/Users/ethanmckanna/Downloads/service area images"
```

For slower end-to-end georeferencing checks, run `--mode full`. The full mode
uses the same CLI output path as production and scores the exported GeoJSON
against the same references.
