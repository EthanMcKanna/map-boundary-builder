# Latency R&D Notes

This log tracks evidence from the May 28, 2026 production-latency push. It is
not a benchmark contract; use the JSON reports under `out/` and production smoke
commands as the authoritative artifacts.

## Ground Truth Caveat

Houston, Miami, Bay Area, and Las Vegas Zoox fixtures are tracked as
`reference_mismatch` in `benchmarks/service-area-fixtures.json` when the saved
screenshot/reference pairing is known to have drifted. They remain visible in
reports but are not scored as model regressions until the reference polygons and
screenshots are refreshed.

May 28 user confirmation: Houston, Miami, and Bay Area have changed from the
saved ground truth. The affected local variants are Bay Area Tesla/Waymo/Zoox,
Houston Tesla/Waymo, and Miami Waymo. Treat them as data debt, not model
regressions, during latency experiments.

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
  1600px by default, with coordinates scaled back to the original image. This
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
- RapidOCR classifier batching now defaults to 24 crops per batch instead of 6
  through `MAP_BOUNDARY_RAPIDOCR_CLS_BATCH_NUM`, and the setting is included in
  OCR cache keys.
- Road-refinement cache keys now hash the derived road-feature distance field
  instead of raw RGB bytes. Harmless pixel changes that do not affect road
  features can reuse the expensive OSM road alignment result, while the cache
  version bump prevents mixing old raw-image keyed entries with new results.
- Opaque RGBA screenshots now load through a direct RGB slice instead of the
  float alpha-composite path. Transparent uploads still composite over white.
- RapidOCR resized inputs now go to the model as in-memory BGR arrays instead
  of being written to temporary PNGs and read back. Small non-resized images
  stay on RapidOCR's original path because gray-fill fixtures regressed when
  forced through OpenCV-loaded arrays.
- RapidOCR max dimension now defaults to 1600 after a fresh-cache A/B preserved
  the full drift-aware benchmark while trimming local OCR-heavy benchmark time.
  The lower cap is intentionally not extrapolated further: 1400 failed and 1800
  had a non-monotonic Orlando georeference regression in the same probe.
- Transparent raster uploads now skip writing an opaque temp PNG. Extraction,
  marker detection, and RapidOCR still see the same white-composited pixels as
  the old temp-file path, but partial-alpha screenshots avoid the extra file
  write/read. Local Phoenix A/B preserved bbox, geometry hash, confidence,
  controls, georeference source, and road-match score while moving from
  1.704-1.772s on baseline to 1.626-1.670s on the in-memory path.
- Los Angeles OCR-geocoding now has bundled Nominatim miss markers, Photon seed
  payloads, and the OSM place payload needed by the place-control matcher.
  Fresh-cache warm local LA dropped from roughly 17s to 0.939s while preserving
  the same bbox, confidence, 20 controls, and 0.943 full-benchmark IoU.
- Miami road-refinement points and the Bay Area synthetic regional place bbox
  are now bundled as OSM seeds. These remove cold Overpass dependency for two
  changed-service-area cases while preserving the same fitting path and outputs.
- Nashville now has bundled Photon seed payloads and Nominatim miss markers for
  common screenshot labels. Fresh-cache warm local Nashville dropped from
  roughly 3.55s to 1.39s while preserving the same bbox, confidence, residuals,
  and road-match score.
- Road-refinement search batching now defaults to 512 candidates per vectorized
  scoring batch. Focused Nashville/Phoenix probes preserved road scores and
  confidence while reducing the two-fixture local timing from 2.97s at batch
  256 to 2.85s at batch 512.
- Road refinement now searches coarse and fine candidates on downsampled road
  feature-distance fields, then polishes the winning basin on the full-resolution
  feature field and falls back to the old full-resolution search for weak
  matches. Against `34cdeed`, Phoenix improved from 0.983 to 0.985 IoU and
  Nashville from 0.981 to 0.986 IoU while reducing the road-georeference stage
  on both focused fresh-cache probes.
- ONNX Runtime is pinned to `1.19.2`, the smallest tested NumPy-2-compatible
  runtime wheel. A Python 3.12 throwaway venv kept NumPy 2.4.6 and OpenCV
  4.13.0 behavior, passed all unit tests, and preserved the drift-aware full
  benchmark at 8/8 scored fixtures with avg IoU 0.962 and min IoU 0.931.
- Vercel/uv dependency metadata for ONNX Runtime now excludes optional SymPy,
  coloredlogs, and humanfriendly installs from the production resolver. Manual
  no-SymPy inference validation passed, and the remote Vercel bundle dropped to
  297.34 MB while keeping the inspected function at 92.74 MB.
- Georeference marker detection and road refinement now reuse the already-loaded
  RGB image from the runner instead of rereading the same normalized upload from
  disk. This removed the georeference-side duplicate read, keeps extraction and
  road scoring on identical pixels, and reduced the Phoenix georeference profile
  from 0.645s to 0.591s without changing geometry.

## Current Validation

- `PYTHONPATH=. .venv/bin/pytest`: 53 passed.
- `PATH=/usr/bin:/bin PYTHONPATH=. /tmp/mbb-ort119-py312-venv-*/bin/python -m pytest -q`: 53 passed.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-ort119-focused-XXXXXX) PYTHONPATH=. /tmp/mbb-ort119-py312-venv-*/bin/python -m map_boundary_builder.benchmark --mode full --only phoenix --only nashville --only orlando --only los-angeles --out-dir out/ort119-focused`: PASS 4/4 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-ort119-full-XXXXXX) PYTHONPATH=. /tmp/mbb-ort119-py312-venv-*/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/ort119-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.962, min IoU 0.931.
- `PYTHONPATH=. .venv/bin/pytest -q`: 54 passed after RGB reuse.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-rgbreuse-focused-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --only phoenix --only nashville --only orlando --only los-angeles --out-dir out/rgb-reuse-focused`: PASS 4/4 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-rgbreuse-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rgb-reuse-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.962, min IoU 0.931.
- Manual ONNX Runtime 1.19.2 no-SymPy stack:
  `PATH=/usr/bin:/bin PYTHONPATH=. /tmp/mbb-ort119-nosympy-py312-venv-*/bin/python -m pytest tests/test_ocr_georeference.py tests/test_osm_roads.py tests/test_image_io.py -q`: 39 passed.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-ort119-nosympy-focused-XXXXXX) PYTHONPATH=. /tmp/mbb-ort119-nosympy-py312-venv-*/bin/python -m map_boundary_builder.benchmark --mode full --only phoenix --only nashville --only orlando --only los-angeles --out-dir out/ort119-nosympy-focused`: PASS 4/4 active, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-ort119-nosympy-full-XXXXXX) PYTHONPATH=. /tmp/mbb-ort119-nosympy-py312-venv-*/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/ort119-nosympy-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.962, min IoU 0.931.
- Focused hybrid road-refinement A/B against `34cdeed`:
  - baseline Phoenix 1.761s / 0.731s georef / 0.983 IoU / road score 0.698507;
    hybrid Phoenix 1.559s / 0.602s georef / 0.985 IoU / road score 0.718119.
  - baseline Nashville 1.247s / 0.676s georef / 0.981 IoU / road score
    0.763171; hybrid Nashville 1.135s / 0.553s georef / 0.986 IoU / road
    score 0.770739.
  - repeat baseline Phoenix 1.643s / 0.673s georef and Nashville 1.176s /
    0.659s georef; repeat hybrid Phoenix 1.582s / 0.607s georef and Nashville
    1.066s / 0.532s georef with the same improved IoUs and road scores.
- Fresh-cache warm active-suite profile after hybrid road refinement: avg
  0.619s, max 1.300s; Phoenix 1.300s with 0.985 IoU, Nashville 1.000s with
  0.986 IoU, Los Angeles 0.849s, all other active fixtures below 0.55s.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-hybrid-focused-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --only phoenix --only nashville --out-dir out/hybrid-road-focused`: PASS 2/2 active, avg IoU 0.985, min IoU 0.985.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-hybrid-extraction-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode extraction --out-dir out/hybrid-road-extraction`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.981, min IoU 0.925.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-hybrid-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/hybrid-road-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.962, min IoU 0.931.
- Local Phoenix transparent-upload A/B against `58d8221`: baseline fresh-cache
  warm runs were 1.772s and 1.704s; the in-memory alpha path was 1.670s and
  1.626s with identical bbox, geometry hash `b446d2b20bebd0e1`, confidence
  0.917, 6 controls, source `ocr-georeference:nominatim-label-fit+osm-road-refine`,
  and road score 0.698507.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-alpha-focused-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --only phoenix --only nashville --out-dir out/alpha-transparent-focused`: PASS 2/2 active, avg IoU 0.982, min IoU 0.981.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-alpha-extraction-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode extraction --out-dir out/alpha-transparent-extraction`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.981, min IoU 0.925.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-alpha-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/alpha-transparent-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.961, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-nash-seed-road512-focused-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --only nashville --only phoenix --out-dir out/nash-seed-road512-focused`: PASS 2/2 active, avg IoU 0.982, min IoU 0.981.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-nash-seed-road512-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/nash-seed-road512-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.961, min IoU 0.931.
- Fresh-cache warm active-suite profile after Nashville seeds and road batch 512:
  avg 0.662s, max 1.495s; Nashville 1.181s, Phoenix 1.495s, Los Angeles
  0.832s, all other active fixtures below 0.57s.
- Production Nashville smoke on `dpl_HJ6mtJ6bsY8CnfVDirckSsS4vimd`
  (`pipeline-bc7f59c6f98eb5cd`): HTTP 201, total wall 19.19s, timestamp
  span 15.10s, confidence 0.821, 3 controls, road score 0.763, bbox unchanged.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-la-places-seeded-focused-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --only los-angeles --out-dir out/la-places-seeded-focused-full`: PASS 1/1 active, IoU 0.943.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-la-places-seeded-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/la-places-seeded-drift-aware-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.961, min IoU 0.931.
- Production LA smoke on `dpl_ECWk6JTkvBc49hLgZkCyxijhVGQN`
  (`pipeline-3cda3ef6e9b2f3c6`): HTTP 201, total wall 13.19s,
  timestamp span 9.55s, georeference stage ~1.02s, confidence 0.862, 20
  controls, bbox unchanged. The prior geocoder-only production deploy was
  HTTP 201, total wall 28.62s, timestamp span 24.65s, georeference stage
  ~12.83s with the same geometry/confidence.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-la-seeded-focused-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --only los-angeles --out-dir out/la-seeded-focused-full`: PASS 1/1 active, IoU 0.943.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-la-seeded-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/la-seeded-drift-aware-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.961, min IoU 0.931.
- After expanding the drift-aware fixture metadata for Houston/Bay Area provider
  variants, `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-drift-aware-extract-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode extraction --out-dir out/drift-aware-extraction`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.981, min IoU 0.925.
- After the same fixture metadata update, `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-drift-aware-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/drift-aware-full`: PASS 8/8 active, 7 skipped data-drift fixtures, avg IoU 0.961, min IoU 0.931.
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
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-cls24-default-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-cls24-default-full`: PASS 11/11 active, avg IoU 0.962, min IoU 0.931 with detector 608 and classifier batch 24 as code defaults.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-opaque-fast-full-2-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/opaque-rgba-fast-full-2`: PASS 11/11 active, avg IoU 0.962, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-rapid-array-resized-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-array-resized-input-full`: PASS 11/11 active, avg IoU 0.962, min IoU 0.931.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road-feature-cache-focused-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --only phoenix --only nashville --out-dir out/road-feature-cache-focused`: PASS 2/2 active, avg IoU 0.982, min IoU 0.981; Phoenix stayed 0.983 and Nashville stayed 0.981.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road-feature-cache-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/road-feature-cache-full`: PASS 11/11 active, 4 skipped data-drift fixtures, avg IoU 0.962, min IoU 0.931.
- Fresh-cache RapidOCR max-dimension probe after drift accounting:
  - `MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION=2000 ... --mode full --out-dir out/rapid-maxdim-2000-full-check`: PASS 8/8 active, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931, real 9.20s.
  - `MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION=1600 ... --mode full --out-dir out/rapid-maxdim-1600-full-check`: PASS 8/8 active, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931, real 8.21s.
  - With 1600 as the code default, `PYTHONPATH=. .venv/bin/pytest -q`: 56
    passed, and `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-rapid-maxdim1600-default-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/rapid-maxdim1600-default-full`: PASS 8/8 active, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931, real 8.22s.
  - Fresh-cache cProfile smoke: Phoenix Waymo moved from 2.85s real at the
    2000px cap to 2.20s at 1600px with the same 6 controls and road-refined
    source; Orlando Waymo moved from 2.10s to 1.40s and improved from 6 to 7
    controls.
  - Drifted-service-area smoke with the new default completed Houston Waymo
    0.95s, Miami Waymo 14.99s, Bay Area Waymo 25.46s, Houston Tesla 0.92s,
    Bay Area Tesla 0.74s, and Zoox SF 1.79s. The long Miami/Bay Area Waymo
    cases remain geocode-heavy rather than OCR-heavy.
  - Production deploy `dpl_3Xyop7frohTrUzDT278UvjVC86KC` is live at
    `mapboundary.app` with pipeline hash `pipeline-8d6f91ae02852866`.
    Orlando Waymo (`simplify_px=6.42`) returned HTTP complete in 10.57s wall
    / 6.859s event span with confidence 0.926 and 7 controls, versus the
    prior no-SymPy production evidence of 14.60s wall / 10.189s event span.
    Exact repeat hit the run cache in 2.22s wall.
- Fresh-cache Overpass-seed smoke with Overpass network forcibly blocked:
  - Miami Waymo completed in 2.472s with the same bbox, 0.864 confidence,
    6 controls, and `ocr-georeference:nominatim-label-fit+osm-road-refine`.
    The comparable cProfile run before seeding was 7.15s, including 4.676s in
    `load_road_points`.
  - Bay Area Waymo completed in 1.437s with the same bbox, 0.877 confidence,
    15 controls, and `ocr-georeference:nominatim-label-fit`. The comparable
    cProfile run before seeding was 12.60s, including 11.229s in
    `load_place_points`; a later Overpass call hit the 50s timeout and fell to
    a weaker 3-control fit, so the seed is also a reliability improvement.
  - After adding the seeds, `PYTHONPATH=. .venv/bin/pytest -q`: 57 passed, and
    `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-seeded-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/seeded-overpass-full`: PASS 8/8 active, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931, real 8.28s.
  - Production deploy `dpl_oj1p2Lk8PD2BHtoKdtNNBM3qVz9m` is live with pipeline
    hash `pipeline-f04bcdf3e7178726`. First post-deploy Miami Waymo completed
    in 16.61s wall / 13.098s event span, and first Bay Area Waymo completed in
    8.43s wall / 5.630s event span. A non-cached warm-instance pass using
    `simplify_px=6.01` completed Miami in 3.39s wall / 1.120s event span and
    Bay Area in 3.80s wall / 1.146s event span, preserving the same confidence,
    control count, source, and bbox for both images.
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
- RapidOCR classifier-batch probes kept sampled top labels stable at batch 24:
  Orlando averaged 0.386s at batch 6 versus 0.362s at classifier batch 24;
  Phoenix was noise-flat at 0.659s versus 0.667s.
- Opaque-RGBA image loading probe on Orlando preserved identical RGB pixels and
  reduced local `load_rgb` average time from 0.0775s to 0.0538s by skipping
  unnecessary float alpha compositing.
- RapidOCR in-memory resized input probes preserved sampled label counts/texts
  on Orlando and Phoenix while avoiding temp PNG write/read. Orlando averaged
  0.439s through the temp-file path versus 0.412s through the resized-array
  path; Phoenix averaged 0.737s versus 0.694s.
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
- Deployment `dpl_9BCK2kUuKrr99BjroGGh1rmh7EzE` deployed hybrid road refinement
  to `https://mapboundary.app`.
- Health after deploy: `pipeline-b33f5471e1bf94a4`, runtime `vercel-python`,
  `tesseract: null`.
- Production hybrid road-refinement smoke against previous production
  `dpl_7UZ8A5b1QgftEgQ6vq6XpqEYitod`:
  - Nashville warm cache-busted pair: old HTTP 201, 3.76s wall, 1.244s event
    span, road score 0.763171; hybrid HTTP 201, 3.91s wall, 1.165s event span,
    road score 0.770739, with the expected improved bbox.
  - Phoenix warm cache-busted repeats: old event spans 1.192s and 1.202s with
    road score 0.698507; hybrid event spans 1.248s and 1.217s with road score
    0.718119 and the expected improved bbox. Production Phoenix wall time was
    noise-flat rather than a clear latency win, while local A/B and the
    drift-aware benchmark are the stronger evidence for the hybrid search.
- Skip-domain production deployment `dpl_A4wEA77g64ibvs3DgcEY5Vf7YT71` tested
  the ONNX Runtime 1.19.2 pin. `vercel inspect` reported `api/index.py` at
  92.74 MB versus 101.17 MB on live deployment
  `dpl_9488MQ1eoKgAnhKGjY7myYu3ZSun`.
- Protected Orlando smoke on `dpl_A4wEA77g64ibvs3DgcEY5Vf7YT71` with
  `simplify_px=6.37`: status complete, 12.77s wall through `vercel curl`,
  8.485s event span, confidence 0.909, 6 controls, bbox
  `[-81.5090796, 28.3588616, -81.3625383, 28.5554773]`, matching the live
  smoke geometry and confidence. The live request on `mapboundary.app` was
  12.05s wall / 8.406s event span, so warm generation speed was noise-flat;
  the shipped win is smaller packaged runtime/cold-start burden.
- Deployment `dpl_8CR5bp7CtwYgWbp59BWcdnVvzZFM` deployed RGB reuse to
  `https://mapboundary.app`. Build output still reported a 325.22 MB Python
  bundle before runtime dependency installation, while `vercel inspect`
  reported the function at 92.74 MB.
- Health after RGB reuse deploy: `pipeline-f4de3136e0c322d2`, runtime
  `vercel-python`, `tesseract: null`.
- Production Phoenix smoke after RGB reuse with `simplify_px=6.39`: fresh
  request completed in 17.50s wall / 13.707s event span with bbox
  `[-112.1167239, 33.2306602, -111.8157216, 33.689586]`, confidence 0.917,
  6 controls, and road score 0.718119. Exact repeat returned the same geometry
  and confidence in 2.28s wall via run-result cache.
- Skip-domain deployment `dpl_CLtAipnPnTZbqPXzJcbTuBNW5tus` tested the
  no-SymPy ONNX Runtime metadata override. Build output reported a 297.34 MB
  Python bundle before runtime dependency installation, down from 325.22 MB on
  the current RGB-reuse production build and 311.23 MB on the ONNX Runtime
  1.26.0 candidate. `vercel inspect` kept `api/index.py` at 92.74 MB.
- Protected Orlando smoke on `dpl_CLtAipnPnTZbqPXzJcbTuBNW5tus` with
  `simplify_px=6.42`: status complete, 14.60s wall through `vercel curl`,
  10.189s event span, confidence 0.909, 6 controls, bbox
  `[-81.5090796, 28.3588616, -81.3625383, 28.5554773]`.
- Local robust-fit cache and quantile helper validation:
  - Houston, Miami, and Bay Area service-area screenshots are known to have
    drifted from the saved reference polygons, so those fixtures remain
    `reference_mismatch` data debt rather than scored regressions.
  - Replaced repeated NumPy median/percentile calls over tiny residual lists
    with an exact linear percentile helper, then cached robust similarity fits
    by control coordinates so the decisive-fit check and final fit reuse the
    same result inside a request.
  - Focused tests:
    `PYTHONPATH=. .venv/bin/pytest -q tests/test_ocr_georeference.py tests/test_extract.py tests/test_image_io.py`
    passed, 38 tests.
  - Full tests: `PYTHONPATH=. .venv/bin/pytest -q` passed, 59 tests.
  - Focused drift-aware extraction benchmark passed 3/3 scored fixtures, skipped
    4 known `reference_mismatch` fixtures, avg IoU 0.988, min IoU 0.984.
  - Full drift-aware benchmark passed 8/8 scored fixtures, skipped 7
    `reference_mismatch` fixtures, avg IoU 0.962, min IoU 0.931, in 7.80s
    wall time versus the prior seeded-Overpass full-suite evidence of 8.28s.
  - Production-like local warm smokes kept identical geometry/source/confidence:
    Miami Waymo 0.220s wall, 0.045s georeference, confidence 0.864, 6 controls,
    bbox `[-80.3230924, 25.6880246, -80.1184998, 25.9396977]`; Bay Area Waymo
    0.173s wall, 0.004s georeference after robust-fit cache reuse, confidence
    0.877, 15 controls, bbox
    `[-122.4978873, 37.3073419, -121.8576229, 37.7981634]`.
- Local bright-blue extraction and response-payload validation:
  - Moved gray/dark/green style signals behind the decisive bright-blue style
    check, preserving the exact bright-blue decision rule while skipping unused
    full-image reductions for Waymo-style screenshots.
  - Lowered API overlay optimization threshold from 250 KB to 64 KB and bumped
    run-result cache payload version to store WebP previews for normal
    production overlays, not only very large previews.
  - Focused tests:
    `PYTHONPATH=. .venv/bin/pytest -q tests/test_api_cache.py tests/test_extract.py tests/test_image_io.py`
    passed, 16 tests. Full tests: `PYTHONPATH=. .venv/bin/pytest -q` passed,
    60 tests.
  - Focused drift-aware extraction benchmark passed 3/3 scored fixtures, skipped
    4 known `reference_mismatch` fixtures, avg IoU 0.988, min IoU 0.984.
  - Full drift-aware benchmark passed 8/8 scored fixtures, skipped 7
    `reference_mismatch` fixtures, avg IoU 0.962, min IoU 0.931, in 7.76s
    wall time on the confirmation run.
  - Direct extraction profiles for Miami/Bay Area Waymo dropped to 0.068s and
    0.062s locally, with the same bright-blue style and masks passing the
    benchmark. Production-like local warm smokes kept identical
    geometry/source/confidence: Miami build 0.201s plus 0.067s WebP overlay,
    gzip response 58.9 KB; Bay Area build 0.165s plus 0.052s WebP overlay,
    gzip response 67.9 KB.
  - Deployment `dpl_G9STqzbqzv3XH1ysQqDfRWgAWk8N` deployed the fast bright-blue
    classifier and WebP overlay threshold to `https://mapboundary.app`; health
    reported `pipeline-ca7a20cc387d92f1` and Vercel reported
    `api/index.py` at 92.86 MB.
  - Production gzip-aware cache-busted warm smokes preserved the same outputs
    while returning WebP overlays: Miami Waymo 3.006s wall / 0.868s event span,
    58.8 KB wire response, confidence 0.864, 6 controls, bbox
    `[-80.3230924, 25.6880246, -80.1184998, 25.9396977]`; Bay Area Waymo
    3.324s wall / 0.704s event span, 67.8 KB wire response, confidence 0.877,
    15 controls, bbox `[-122.4978873, 37.3073419, -121.8576229, 37.7981634]`.
  - Production cached Bay Area repeats returned WebP overlays in 2.05-2.41s wall
    with a 67.9 KB gzip wire response, down from the earlier same-image cached
    gzip check around 2.315s and 254 KB wire response.
- Covering OSM-place and Miami geocoder seed validation:
  - A cold local profile showed Bay Area georeference could still spend about
    10.7s in `build_osm_place_control_points` / `load_overpass_places` when the
    inferred regional bbox drifted from an exact seed hash, even though the bbox
    was spatially covered by an existing bundled Bay Area seed.
  - Added a covering-seed fallback that picks the smallest bundled OSM place
    payload whose element bounds cover the requested bbox, preserving exact-key
    lookup first and avoiding network only when a bundled regional seed clearly
    covers the request.
  - The user confirmed Houston, Miami, and Bay Area service areas have changed
    from the saved ground truth. A fresh-cache smoke with geocoder, OSM-place,
    and OSM-road network calls patched to fail exposed Miami Waymo as the only
    missing bundled context; adding the live Miami Nominatim city seed made the
    no-network path reproduce the same refined bbox, confidence, and source as
    the live fresh-cache run.
  - Focused tests passed:
    `PYTHONPATH=. .venv/bin/pytest -q tests/test_geocoder.py tests/test_osm_places.py tests/test_ocr_georeference.py`,
    39 tests. Full tests passed: `PYTHONPATH=. .venv/bin/pytest -q`, 63 tests.
  - Fresh-cache changed-market no-network smoke passed Bay Area Waymo 0.682s,
    Bay Area Tesla 0.089s, Bay Area Zoox 0.394s, Houston Waymo 0.450s, Houston
    Tesla 0.073s, and Miami Waymo 1.334s. Miami preserved confidence 0.864, 6
    controls, source `ocr-georeference:nominatim-label-fit+osm-road-refine`,
    and bbox `[-80.3230924, 25.6880246, -80.1184998, 25.9396977]`; Bay Area
    Waymo preserved confidence 0.877, 15 controls, source
    `ocr-georeference:nominatim-label-fit`, and bbox
    `[-122.4978873, 37.3073419, -121.8576229, 37.7981634]`.
  - Full drift-aware benchmark stayed clean: 8/8 scored fixtures passed, 7
    `reference_mismatch` fixtures skipped, avg IoU 0.962, min IoU 0.931, in
    8.20s wall time.
  - Production deployment `dpl_G1AMGx4uRNGgUFAyv3nL2mbuoudT` reported
    `pipeline-8d4c85b80b06ef10` and kept `api/index.py` at 92.86 MB. First
    cache-busted production POSTs preserved geometry/source/confidence while
    improving over the previous first post-deploy evidence: Miami Waymo 13.004s
    wall / 9.710s event span versus 16.61s / 13.098s; Bay Area Waymo 6.218s
    wall / 3.580s event span versus 8.12s / 4.763s. Exact repeats hit the
    result cache in 1.900s and 2.136s with identical bboxes and confidence.

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
  Area Tesla to 0.862 IoU and Houston Tesla to 0.912 IoU before those fixtures
  were marked as stale-ground-truth data debt. After expanding stale
  Houston/Bay Area fixtures out of the hard score, detector 592, 576, and 512
  passed the drift-aware active full suite, but active-suite RapidOCR median
  averages were 0.372s at 608, 0.379s at 576, and 0.377s at 512, while 512 also
  nudged active average IoU down from 0.961 to 0.960. Rejected; keep 608.
- Raising RapidOCR recognition batch size changed a sampled Phoenix label from
  `TPC Scottsdale` to `TPCScottsdale` and did not improve Phoenix timing.
  Rejected; only classifier batching was kept.
- Forcing every RapidOCR input through an OpenCV-loaded array broke small
  gray-fill fixtures: Houston Tesla fell to 0.000 IoU and Bay Area Tesla to
  0.859. Rejected; the array shortcut is limited to resized inputs only.
- RapidOCR max dimension 1800 dropped Orlando to 0.853 IoU, and 1900 dropped
  Phoenix to 0.895 IoU. Rejected; the 2000px OCR input cap remains necessary.
- Pinning ONNX Runtime threads was slower locally: 1x1 and 2x1 were much slower
  than default, and 4x1 hurt Phoenix. Rejected.
- Pinning ONNX Runtime to 1.26.0 removed SymPy from normal pip installs and
  reduced the remote Vercel bundle from 325.22 MB to 311.23 MB, but it pushed
  `api/index.py` back to 101.17 MB. Rejected in favor of ONNX Runtime 1.19.2
  plus Vercel/uv metadata that omits optional SymPy dependencies, which reached
  a 297.34 MB remote bundle while keeping the function at 92.74 MB.
- Local `vercel build --prod` remained blocked unless `uv` is on PATH; remote
  deploys are the reliable bundle-size evidence source for now.
- A separate `/api/health?warm=generation` prewarm request initialized OCR on
  one production function instance, but a following cache-busted `/api/runs`
  request could still land on a cold instance: after deployment
  `dpl_CD6QdbdAnHgbekhxHNLxf3emSuqc`, explicit warmup returned
  `generation: true`, `ocr: true`, and `warm_elapsed_ms: 1302.6`, yet later
  warmed health calls reported `warm_elapsed_ms: 0.0` while the next Bay/Miami
  POSTs still took 7.154s and 7.892s event spans. Rejected and reverted because
  it added a client request without reliable same-instance latency improvement.

## Remaining Bottlenecks

- Production function size is improved after the ONNX Runtime pin, with Vercel
  reporting `api/index.py` at 92.74 MB on the current deployment. The remote
  Python build now reports a 297.34 MB pre-runtime-installation bundle, so
  cold starts and OCR model initialization remain production-only latency risks.
- OpenCV and ONNX Runtime remain the largest runtime weights. Removing either
  would require a larger architecture change and must be proven against the full
  active fixture suite before production.
