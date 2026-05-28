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
- Road refinement now samples up to 6000 road points by default instead of
  12000, preserving the full active suite while reducing Phoenix/Nashville
  focused benchmark time.
- The Vercel API path now caps generated overlay previews at 1200px before
  encoding the inline artifact, while CLI/debug outputs keep full-size overlays.
  GeoJSON/extraction/georeferencing stay unchanged.
- The Vercel API path also skips writing the full-size mask PNG because the
  synchronous response only returns inline GeoJSON and overlay preview. The
  local async worker still writes downloadable mask artifacts.
- Road-refinement batch scoring now uses float32 arrays and bumped the
  road-refine cache version to avoid stale float64-scored cache hits.
- OCR fallback now reuses the initial RapidOCR words instead of rerunning the
  same RapidOCR pass when Tesseract is unavailable or when Tesseract fallback
  still produces too few labels.
- RapidOCR detector `limit_side_len` now defaults to 608 instead of the upstream
  736. The setting is included in OCR cache keys through
  `MAP_BOUNDARY_RAPIDOCR_DET_LIMIT_SIDE_LEN`.
- Road-refinement cache keys now hash the derived road-feature distance field
  instead of raw RGB bytes. Harmless pixel changes that do not affect road
  features can reuse the expensive OSM road alignment result, while the cache
  version bump prevents mixing old raw-image keyed entries with new results.

## Current Validation

- `PYTHONPATH=. .venv/bin/pytest`: 45 passed.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-batch256-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/batch256-no-tess-full`: PASS 11/11 active, avg IoU 0.957, min IoU 0.896.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-default2000-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-default-2000-full`: PASS 11/11 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road6000-full-XXXXXX) MAP_BOUNDARY_ROAD_MATCH_MAX_POINTS=6000 PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/road-points-6000-full`: PASS 11/11 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-preview-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/preview-max-default-full`: PASS 11/11 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-preview-nomask-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/preview-nomask-default-full`: PASS 11/11 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road-f32-cachev2-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/road-f32-cachev2-full`: PASS 11/11 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-rapid-dedupe-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-dedupe-full`: PASS 11/11 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-det640-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-det640-full`: PASS 11/11 active, avg IoU 0.962, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-det608-full-XXXXXX) MAP_BOUNDARY_RAPIDOCR_DET_LIMIT_SIDE_LEN=608 PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-det608-full`: PASS 11/11 active, avg IoU 0.962, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-det608-default-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-det608-default-full`: PASS 11/11 active, avg IoU 0.962, min IoU 0.931 with detector 608 as the code default.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road-feature-cache-focused-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --only phoenix --only nashville --out-dir out/road-feature-cache-focused`: PASS 2/2 active, avg IoU 0.982, min IoU 0.981; Phoenix stayed 0.983 and Nashville stayed 0.981.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road-feature-cache-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/road-feature-cache-full`: PASS 11/11 active, 4 skipped data-drift fixtures, avg IoU 0.962, min IoU 0.931.
- Focused no-downscale A/B (`Orlando`, `Phoenix`, `Nashville`, `San Antonio`):
  old path PASS 4/4 in 10.24s, avg IoU 0.946, min IoU 0.896.
- Focused default-2000 A/B on the same four fixtures: PASS 4/4 in 8.84s,
  avg IoU 0.958, min IoU 0.931.
- Focused Phoenix/Nashville with batch size 256: PASS 2/2, IoUs 0.978 and
  0.981, matching the high-accuracy baseline.
- Focused Phoenix/Nashville road-point cap sweep: 12000 points PASS 2/2 in
  6.52s, 6000 points PASS 2/2 in 5.04s with Phoenix IoU 0.983 and Nashville
  IoU 0.981.
- Warm local preview generation A/B after caches are populated: Orlando export
  stage dropped from 0.150s/full-size overlay to 0.047s/1200px overlay; Phoenix
  export dropped from 0.153s to 0.055s. Inline overlay encoding dropped from
  0.133s to 0.056s on Orlando and 0.107s to 0.060s on Phoenix.
- Warm local API artifact A/B on Orlando: full overlay+mask export took 0.146s,
  bounded overlay+mask took 0.052s, and bounded overlay without mask took 0.042s.
- Local Phoenix cProfile for road refinement dropped from 0.728s cumulative
  in `refine_transform_with_osm_roads` before float32 scoring to 0.689s after,
  with the same Phoenix/Nashville focused IoUs.
- Low-label Tesla Bay Area OCR measurement with Tesseract absent: the first
  RapidOCR pass took 0.244s, the duplicate warm pass cost another 0.111s, and
  dedupe found no additional unique OCR words. The fallback reuse change removes
  that duplicate model pass.
- RapidOCR detector-limit probes with identical sampled labels: direct repeated
  OCR on Orlando averaged 0.677s at 736 versus 0.480s at 640; Nashville averaged
  0.472s versus 0.451s; Phoenix was effectively flat in the repeated run. A
  focused full-pipeline A/B had equal IoUs but noisy wall time, so this is
  treated as a CPU reduction rather than a guaranteed wall-clock breakthrough.
- RapidOCR 608 detector-limit probes with identical sampled label counts:
  Orlando averaged 0.493s at 640 versus 0.464s at 608; Phoenix averaged 0.786s
  versus 0.774s; Tesla Bay Area was noise-flat at 0.069s versus 0.074s. The
  full-suite geometry stayed at avg IoU 0.962 and min IoU 0.931.
- Feature-distance road-refine cache proof: a synthetic near-duplicate whose
  raw RGB changed but whose road-feature distance field stayed identical reused
  the cached result. With an intentionally slow patched road search, the first
  call took 0.1708s, the second took 0.0005s, returned the same result, and
  produced one shared cache file.

## Production Smoke Evidence

- Deployment `dpl_7wU7ijL47rESEbkGT7LaLFpEzL35` is live on
  `https://mapboundary.app`.
- Health after deploy: `pipeline-8fd0c71ed10e3b96`, runtime `vercel-python`,
  `tesseract: null`.
- Deployment `dpl_F91Dk35znCEKB6eFmz1DL7EpWqio` deployed the
  feature-distance road-refine cache key to `https://mapboundary.app`.
- Health after deploy: `pipeline-691e8f992b33dd12`, runtime `vercel-python`,
  `tesseract: null`.
- Production Phoenix near-duplicate smoke after feature-distance road caching:
  - Original Phoenix upload: HTTP 201, 24.14s wall, 19.99s event span,
    confidence 0.917, 6 controls, road score 0.698507, georeference event
    3.19s, export 0.20s.
  - Single-pixel feature-equivalent variant: HTTP 201, 5.30s wall, 2.67s event
    span, confidence 0.917, 6 controls, identical road score 0.698507,
    georeference event 0.70s, export 0.21s. This confirms the warm
    near-duplicate production path is much faster, while OCR/model warm state
    also contributed to the total wall-clock drop.
- Production unique-image smoke after bounded previews and no API mask artifact:
  - Orlando Waymo: HTTP 201, 12.12s wall, 8.63s event span, confidence 0.909,
    6 controls, export 0.148s, summary mask `null`, returned overlay
    `image/webp` at 1200x1200.
  - Earlier bounded-preview Orlando before skipping the mask artifact: HTTP
    201, 13.55s wall, 9.69s event span, export 0.250s.
  - Phoenix Waymo: HTTP 201, 16.13s wall, 13.35s event span, confidence 0.917,
    6 controls, road score 0.69856, export 0.254s, returned overlay
    `image/webp` at 1200x1200.
  - Phoenix Waymo after float32 scoring: HTTP 201, 20.75s wall, 16.83s event
    span, confidence 0.917, 6 controls, road score 0.698507, georef 3.693s,
    export 0.191s, summary mask `null`, returned overlay `image/webp` at
    1200x1200. Production timing was noisy and not a breakthrough; local
    profiling is the cleaner evidence for this micro-optimization.
  - Tesla Bay Area after RapidOCR fallback reuse: HTTP 201, 4.33s wall, 2.35s
    event span, confidence 0.955, 5 controls, OCR 0.987s, export 0.105s,
    summary mask `null`.
  - Orlando Waymo after detector limit 640: HTTP 201, 12.63s wall, 9.01s event
    span, confidence 0.909, 6 controls, OCR 5.797s, export 0.172s, summary mask
    `null`, returned overlay `image/webp` at 1200x1200. Production OCR timing
    remained noisy on a single call; local repeated OCR and full-suite accuracy
    are the stronger evidence for this change.

## Failed Or Rejected Experiments

- Narrowing the road-refinement grid improved speed but reduced Nashville IoU
  from 0.981 to 0.885. Rejected as an accuracy regression.
- A moderate grid reduction still lowered Nashville IoU to 0.942. Rejected
  because the goal requires no accuracy reduction, not merely passing thresholds.
- RapidOCR max dimension 2200 was faster locally but dropped Orlando IoU from
  0.930 to 0.863 in focused validation. Rejected.
- Road-point cap 3000 was faster but dropped Nashville IoU to 0.796. Rejected.
- RapidOCR without the angle classifier matched normal focused fixtures but
  produced high-count gibberish on a 180-degree rotated Orlando stress image,
  while classifier-enabled OCR recovered readable labels. Rejected for
  robustness.
- Extraction downscaling sped up direct mask extraction, but even a 2000px cap
  moved Phoenix IoU from 0.983 to 0.982 and lower caps shifted additional
  stable fixtures. Rejected under the no-regression accuracy bar.
- Skipping road refinement is not safe: Nashville without road refinement fell
  to 0.796 IoU, and Phoenix without road refinement fell to 0.903 IoU. Rejected
  because the road stage is still carrying important georeference accuracy.
- RapidOCR detector 512 passed the coarse benchmark thresholds but dropped Bay
  Area Tesla to 0.862 IoU and Houston Tesla to 0.912 IoU. Detector 576 and 592
  also slipped Houston Tesla to 0.944 and 0.943 respectively. Rejected because
  the goal is no accuracy reduction, not just passing the floor.
- Local `vercel build --prod` remained blocked unless `uv` is on PATH; remote
  deploys are the reliable bundle-size evidence source for now.

## Remaining Bottlenecks

- Production still exceeds Vercel's Python bundle threshold at about 310.69 MB,
  so runtime dependency installation and cold starts remain the biggest
  production-only latency problem.
- The largest dependency weights are OpenCV and ONNX Runtime. Removing either
  would require a larger architecture change and must be proven against the full
  active fixture suite before production.
