# Latency R&D Notes

This log tracks evidence from the May 28, 2026 production-latency push. It is
not a benchmark contract; use the JSON reports under `out/` and production smoke
commands as the authoritative artifacts.

## Ground Truth Caveat

Houston Waymo, Miami Waymo, Bay Area Waymo, and Las Vegas Zoox are tracked as
`reference_mismatch` in `benchmarks/service-area-fixtures.json`. They remain
visible in reports but are not scored as model regressions until the reference
polygons and screenshots are refreshed.

## Shipped Changes

- `5c69590`: cache-only exploratory geocoder fanout, decisive OSM-place fast
  path, and seeded Phoenix/Nashville road points. Cold local Tesla cases moved
  to roughly half-second CLI runs; active full benchmark stayed green.
- `b8878ff`: removed `pyproj` by using the existing Web Mercator formulas.
  Vercel's bundle warning dropped from 361.06 MB to 329.50 MB.
- `63dbb1b`: included the pipeline hash in server run-cache keys so cached
  results cannot survive incompatible pipeline changes.
- `5f28d22`: increased road-refinement vectorized scoring batch size from 64
  to 256. Controlled local probes preserved Phoenix/Nashville road scores while
  reducing road-refinement-heavy CLI runs.
- Conservative RapidOCR input downscaling caps the OCR image's longest side at
  2000px by default, with coordinates scaled back to the original image. This
  keeps an env override through `MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION`.

## Current Validation

- `PYTHONPATH=. .venv/bin/pytest`: 39 passed.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-batch256-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/batch256-no-tess-full`: PASS 11/11 active, avg IoU 0.957, min IoU 0.896.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-default2000-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-default-2000-full`: PASS 11/11 active, avg IoU 0.961, min IoU 0.931.
- Focused no-downscale A/B (`Orlando`, `Phoenix`, `Nashville`, `San Antonio`):
  old path PASS 4/4 in 10.24s, avg IoU 0.946, min IoU 0.896.
- Focused default-2000 A/B on the same four fixtures: PASS 4/4 in 8.84s,
  avg IoU 0.958, min IoU 0.931.
- Focused Phoenix/Nashville with batch size 256: PASS 2/2, IoUs 0.978 and
  0.981, matching the high-accuracy baseline.

## Production Smoke Evidence

- Deployment `dpl_4xEKd4JZtjXhFgLHPLzkpkNTtv4q` is aliased to
  `https://mapboundary.app`.
- Health after deploy: `pipeline-fc5242d6c41d7742`, runtime `vercel-python`,
  `tesseract: null`.
- Warm production uncached examples after the latest deploy:
  - Phoenix Waymo: 5.506s, road score 0.715203.
  - Nashville Waymo: 4.850s, road score 0.764464.
  - Tesla Bay Area: 1.424s, confidence 0.955.

## Failed Or Rejected Experiments

- Narrowing the road-refinement grid improved speed but reduced Nashville IoU
  from 0.981 to 0.885. Rejected as an accuracy regression.
- A moderate grid reduction still lowered Nashville IoU to 0.942. Rejected
  because the goal requires no accuracy reduction, not merely passing thresholds.
- RapidOCR max dimension 2200 was faster locally but dropped Orlando IoU from
  0.930 to 0.863 in focused validation. Rejected.
- Local `vercel build --prod` remained blocked unless `uv` is on PATH; remote
  deploys are the reliable bundle-size evidence source for now.

## Remaining Bottlenecks

- Production still exceeds Vercel's Python bundle threshold at about 329.51 MB,
  so runtime dependency installation and cold starts remain the biggest
  production-only latency problem.
- The largest dependency weights are OpenCV and ONNX Runtime. Removing either
  would require a larger architecture change and must be proven against the full
  active fixture suite before production.
