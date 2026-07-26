# Map Boundary Builder

`map-boundary-builder` turns a service-map screenshot into a georeferenced
GeoJSON polygon. It ships as a CLI, a local browser workspace, and a hosted
Vercel app.

The pipeline is one linear path for every input:

```
load → segment → polygonize → read labels → locate → georeference → export
```

1. **Segment.** A single packaged ONNX model (`boundary_v1`, ~9 MB, trained on
   domain-randomized synthetic maps) predicts the service-area mask for any
   color, style, or provider — fills, translucent overlays, outline-only
   boundaries, dashed strokes, dark and light basemaps. A color-agnostic LAB
   clustering fallback covers the rare degenerate model output. There are no
   per-provider color rules.
2. **Locate.** OCR reads map labels (RapidOCR/ONNX, with local Tesseract as a
   CLI fallback), public OpenStreetMap-backed geocoders resolve them, and
   label clusters infer the city — no filename tricks, no provider presets.
3. **Georeference.** A rotation-aware Web Mercator transform is fitted from
   label control points with residual gates, optionally refined against OSM
   road geometry.

When the city cannot be inferred, the pipeline does not fail: it returns a
`needs_city` result carrying the extracted boundary and the labels it could
read, and the CLI/web UI ask you for the city and finish georeferencing
without re-extracting.

## Quick Start

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .

.venv/bin/map-boundary \
  --image /path/to/service-map.png \
  --output out/boundary.geojson \
  --debug-dir out/debug-boundary \
  --print-summary
```

The output is a GeoJSON `FeatureCollection` with the polygon in
longitude/latitude coordinates plus metadata describing extraction and
georeference confidence.

CLI exit codes: `0` success, `1` failure, `3` city needed (with `--no-input`;
on a terminal the CLI prompts for the city instead). `--city "Austin, TX"`
supplies it up front. `--seed-x/--seed-y` and `--target-color` optionally
steer which region is selected when a screenshot contains several candidates.

## Requirements

- Python 3.12
- Internet access for OpenStreetMap/Nominatim lookups during georeferencing
  (bundled read-only seed caches cover common lookups; `MAP_BOUNDARY_BLOCK_NETWORK=1`
  forces seed-only operation)
- Optional: local Tesseract (`brew install tesseract`) as a CLI OCR fallback

## Interactive Web Tool

```bash
.venv/bin/map-boundary-web
```

Open `http://127.0.0.1:8765`, drop in a screenshot, and run the builder. The
app streams each stage, previews the extracted mask, renders the boundary on
a map, and offers the GeoJSON for download. An optional **City** field
overrides auto-detection; when a run ends in *city needed*, the app shows
what it extracted and asks inline.

The hosted app at `https://map-boundary-builder.vercel.app` runs the same
pipeline as a serverless function. Results are cached by image content (raw
bytes and decoded pixels): re-uploads and format re-encodes of the same
screenshot return instantly. Transient failures (geocoder timeouts) are
never cached; a `needs_city` retry with a city is a fresh, separately cached
run.

When a web generation fails or looks wrong, the UI can create a GitHub debug
report (`GITHUB_REPORT_TOKEN` in the deployment environment; the reported
screenshot is stored publicly and the user is warned first).

## The Model

`models/boundary_v1.onnx` is a ~2.2 M-parameter ResUNet (3×384×384 RGB in,
mask logits out) trained by `tools/train_synthetic_model.py` on the
procedural generator in `map_boundary_builder/synthetic/`. The generator
domain-randomizes overlay hue, opacity, stroke style (solid/dashed,
outline-only), patterns, distractor overlays, UI chrome, JPEG artifacts, and
basemap style, so the model generalizes across providers instead of
memorizing palettes. The decision threshold is calibrated on a held-out
synthetic split and stored in the `.onnx.json` sidecar.

Retraining:

```bash
.venv/bin/python tools/train_synthetic_model.py \
  --dataset-dir out/train-boundary \
  --count 8192 --image-size 384 --base-channels 24 --arch resunet
```

## Benchmarks and the Promotion Gate

- `map-boundary-synthetic-benchmark` scores raw masks against exact synthetic
  truth (IoU, boundary F1, topology, latency). Generate-and-score:

  ```bash
  .venv/bin/map-boundary-synthetic-benchmark \
    --dataset-dir out/synthetic-eval --generate --count 256
  ```

- `map-boundary-real-benchmark` runs the full pipeline over the real
  screenshot manifest (`benchmarks/real-screenshot-stress.json`, images
  resolved from `benchmarks/real-screenshots/`) and scores durable
  expectations: status, inferred city, bounding-box error, control points.

- `tools/check_promotion.py` is the single gate: a fresh synthetic report
  must meet the committed pre-revamp v12 baseline, and the real manifest may
  not regress any previously-complete case to failure. Baselines live in
  `benchmarks/baselines/`.

## Georeferencing Model

There are no provider boundary presets and no ground-truth fitting. Position
is inferred from OCR labels geocoded against public OpenStreetMap services
(Nominatim, Photon, Overpass) with small bundled seed caches for cold
starts. A similarity transform is accepted only with enough control points
and low residual error; road-grid alignment can refine it. If the evidence
is insufficient, the result is `needs_city` (awaiting user input) or a
failure — never a hardcoded fallback boundary.
