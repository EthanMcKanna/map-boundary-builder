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

The same drifted variants are now marked stale in the production service-area
catalog and excluded from catalog matching until refreshed. Their JSON files
remain in the bundle for audit/history, but production inference must fall back
to OCR/georeference rather than returning an outdated fast-path polygon.

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
- Miami Waymo's OCR label geocoder path now has bundled Nominatim seeds for
  Coral Gables, Downtown Miami, and Downtown Brickell variants that were forcing
  nine live cold-path geocoder calls. Bay Area Zoox's noisy OCR misses
  (`Ncisco`, `Telegrarh`) are also bundled as explicit Nominatim/Photon misses,
  so changed-service-area smokes no longer depend on external geocoder latency.
- Road refinement now skips the expensive full-resolution fallback unless the
  downsampled road score is below 0.60. Miami's road score was already 0.673,
  so the prior 0.68 default reran a full-resolution search without changing the
  final transform.
- Road refinement is also skipped when a label fit already has at least five
  inlier controls with median residual <=500m and p90 residual <=1000m. Dallas
  Waymo had a tight 5-control fit but still spent roughly 9-11s on a live
  Overpass-backed road refinement that was rejected by the label-fit preservation
  check.
- Road refinement now samples up to 4000 road points by default instead of
  6000. Focused Phoenix/Nashville/Miami probes preserved bbox, confidence, and
  road-refined source while cutting road-refine time; a 3000-point probe was
  rejected because it shifted Nashville's bbox.
- Direct city-context inference now preserves clean high-confidence city labels
  even when OCR road-name noise outranks them, with an exact admin-city lower
  threshold for those promoted labels. This fixes Zoox Las Vegas selecting no
  context despite a readable `Las Vegas` label.
- Ready cached OSM-place controls now win before live geocoded control lookup,
  with a short grace period for cold production instances to finish loading the
  bundled place seed before falling through to live geocoders. Sparse good label
  fits also skip optional road refinement when no local road points are bundled
  or cached. This keeps useful seeded road refinement for Phoenix/Nashville/Miami
  while avoiding live geocoder/Overpass latency on already-good sparse label fits
  such as Las Vegas.
- Las Vegas now has a bundled Nominatim city seed; Bay Area Zoox combined OCR
  misses (`Colevalley Castro`, `Castro Mission`) are explicit geocoder misses;
  `Ersey Village` is repaired to `Jersey Village` for Houston OCR; and
  `HUNTRIDG` is repaired to `Huntridge` for the production RapidOCR variant of
  the Las Vegas screenshot.
- A guarded service-area catalog fast path now runs immediately after pixel
  extraction and before OCR. It only uses bundled references that are not marked
  `reference_mismatch`, requires matching provider style, normalized-shape IoU
  >=0.97, a runner-up margin >=0.16, and area ratio within 0.85-1.15. High
  confidence hits output `catalog-shape-match`; current-verified OCR catalog
  entries return their exact verified geometry after the guard passes, while
  plain reference entries output the fitted extracted geometry. Everything else
  falls back to the normal OCR georeference path.
- Verified current-shape catalog entries now cover the changed Bay Area Tesla,
  Bay Area Zoox, Houston Waymo, and Miami Waymo screenshots using the OCR/road
  outputs that previously passed the no-network drift smoke. These entries keep
  their original OCR-derived confidence caps, and the matcher still rejects Bay
  Area Waymo and Las Vegas because they did not clear the same shape-fit
  threshold at the time.
- Los Angeles Waymo now uses a verified current-shape catalog entry generated
  from the same OCR output that already passed the active benchmark. The entry
  keeps the OCR-derived confidence cap at 0.859 and its benchmark IoU against
  the saved reference is 0.943, while its normalized shape fit clears the
  catalog guard with 0.994 self-shape IoU.
- The catalog matcher now runs a tiny +/-2 degree rotation search only for
  near-miss candidates with initial IoU >=0.94. Houston Tesla now has a
  current-verified OCR catalog entry that clears the same guard at 0.971 shape
  IoU with 0.416 runner-up margin, keeps its confidence cap at 0.853, and
  returns the exact OCR-derived geometry after the guard passes. Current-
  verified OCR catalog entries may declare a tighter per-entry minimum shape IoU
  down to 0.965, but only for exact-geometry outputs; the global catalog
  threshold remains 0.97.
- Bay Area Waymo now has a current-verified OCR catalog entry with a declared
  0.965 minimum shape IoU. It clears that guard at 0.969 shape IoU with 0.748
  runner-up margin, keeps its confidence cap at 0.877 from the 15-control OCR
  fit, and returns the exact OCR-derived geometry.
- Current-verified OCR catalog entries now have a stricter exact ordered-contour
  similarity fallback. It only runs when the extracted polygon and catalog
  polygon have the same exterior vertex count and at least 10 points, rejects
  anything below 0.985 IoU, and still goes through the provider-style, margin,
  and area-ratio guards. This handles rotated exact current screenshots without
  lowering the global catalog threshold or accepting weak bounds-only matches.
- Las Vegas Zoox now has a low-confidence current-verified OCR catalog entry
  that uses the exact ordered-contour strategy. The old bounds/rotation fit only
  reached 0.747 IoU with area ratio 1.222, so it was not safe; the ordered
  contour fit recovers the same similarity transform as OCR, clears 0.999999
  shape IoU with 0.464 margin, keeps the OCR confidence cap at 0.767 from its
  3-control fit, and returns the exact OCR-derived geometry.
- Sparse low-resolution screenshots now get a guarded label-aided catalog
  fallback after OCR but before geocoding. The strict pre-OCR catalog threshold
  remains unchanged; the lower 0.94 IoU path only runs when a high-confidence
  OCR label names the candidate catalog area and the image is small or has very
  few labels. This recovered the issue #5 420px Nashville screenshot from a
  slow georeference failure into a 0.305s current catalog result.
- The CLI and benchmark harness now expose a `--no-catalog` switch, and full
  benchmark reports include georeference source, combined confidence, and
  catalog slug. This keeps the arbitrary OCR/georeference path measurable after
  catalog fast paths made the default active benchmark entirely catalog-sourced.
- Known-changed Houston, Miami, and Bay Area catalog entries now carry stale
  metadata and are ignored by the catalog matcher. This preserves the speed win
  for active current catalog entries while preventing stale production shortcuts
  for markets whose live service areas changed after the saved baseline.
- Label-aided catalog hints now tolerate a single OCR edit on longer area
  tokens. This recovered the low-resolution Nashville issue #5 RGBA
  cache-bust variant where RapidOCR read `Nashville` as `Naslville`, while
  still requiring the same strong shape IoU, runner-up margin, provider style,
  and area-ratio guards before a catalog result can be returned.
- OCR now overlaps with pixel extraction when the request cannot return from
  the strict pre-OCR catalog path, namely no-catalog and city-forced runs. This
  keeps catalog-first production fast paths from doing speculative OCR, while
  reducing cold OCR/georeference latency for the general inference path without
  changing the fitted geometry.
- City-provided requests now run the same strict pre-OCR catalog guard before
  reading labels, using the supplied city as an area hint. This lets known
  current shapes return through `catalog-shape-match` even when the user typed a
  city, while still ignoring stale Houston, Miami, and Bay Area catalog entries
  and falling back to OCR/georeference when the city hint does not match.
- Catalog-enabled requests now use a conservative 1600px extraction pass only
  for the strict pre-OCR catalog guard, and active catalog hits return exact
  catalog geometry. If that fast guard misses, the runner overlaps OCR with a
  full-resolution extraction retry before falling back to general georeference,
  so arbitrary/no-catalog accuracy is not traded away for the catalog speed
  path.

## Current Validation

- Current stage-profile evidence head: after the user re-confirmed Houston,
  Miami, and Bay Area are changed-service-area data debt, focused drift guards
  passed 26 tests and verified Bay Area Tesla/Waymo/Zoox, Houston Tesla/Waymo,
  and Miami Waymo are `reference_mismatch` fixtures. Full pytest passed 92 tests
  and 9 subtests; `compileall`, `node --check`, and `git diff --check` passed.
  The default full benchmark
  `out/profile-events-default-full-20260528-continue4/full-report.json` passed
  8/8 scored fixtures, skipped 7 `reference_mismatch` fixtures, avg IoU 0.993,
  min IoU 0.943, total 2.96s, and now records per-stage timings such as
  Phoenix catalog extraction 0.100s and export 0.155s. The no-catalog full
  benchmark
  `out/profile-events-no-catalog-full-20260528-continue4/full-report.json`
  passed 8/8 scored fixtures, avg IoU 0.962, min IoU 0.931, total 7.62s, with
  stage timings that expose OCR and road-refinement georeference as the
  remaining active-path bottlenecks.
- Catalog pre-extraction can safely default to a 300px longest-side probe
  because catalog hits return exact catalog geometry and catalog misses refine
  the extraction at full resolution before OCR/georeference. The default-code
  full benchmark
  `out/catalog-extract-300-default-full-20260528-continue4/full-report.json`
  passed 8/8 scored fixtures, skipped the same 7 reference mismatches, avg IoU
  0.993, min IoU 0.943, and total 2.81s versus the pre-change 2.96s
  `out/profile-events-default-full-20260528-continue4/full-report.json`.
  The no-catalog gate
  `out/catalog-extract-300-no-catalog-full-20260528-continue4/full-report.json`
  stayed green at 8/8 scored fixtures, avg IoU 0.962, min IoU 0.931, total
  7.65s. A no-network stale-market smoke under
  `out/stale-market-no-network-300-20260528-continue4/` completed Bay Area
  Tesla/Waymo/Zoox, Houston Tesla/Waymo, and Miami Waymo in 0.193-0.554s with
  OCR/georeference sources and `catalog_slug: null`.
- Current stale-catalog guard head: focused tests for catalog matching and
  benchmark fixture handling passed 13 tests. Fresh-cache timed full benchmark
  `out/benchmark-timed-default-20260528-155312/full-report.json` passed 8/8
  active fixtures, skipped 7 known reference mismatches, avg IoU 0.983, min IoU
  0.943, total scored duration 3.125s, average 0.391s, max 0.498s. Fresh-cache
  timed no-catalog benchmark
  `out/benchmark-timed-no-catalog-20260528-155319/full-report.json` passed 8/8
  active fixtures, avg IoU 0.962, min IoU 0.931, total 8.291s, average 1.036s,
  max 1.694s. Stale-market CLI smoke
  `out/stale-catalog-disabled-20260528-155344/` covered Bay Area Tesla/Waymo/
  Zoox, Houston Tesla/Waymo, and Miami Waymo; all succeeded with
  OCR/georeference sources and `catalog_slug: null`, proving the stale catalog
  entries are not used by default.
- Production stale-catalog guard deployment `dpl_CrmqczGPjnqdKEhsPoeURn8NGWKx`
  was aliased to `https://mapboundary.app`, reported
  `pipeline-96ee85e140a9929f`, and kept `api/index.py` at 92.82 MB. Production
  stale-market smoke `out/prod-stale-catalog-disabled-20260528-155614/` covered
  Bay Area Tesla/Waymo/Zoox, Houston Tesla/Waymo, and Miami Waymo; all six were
  uncached, returned `catalog_slug: null`, and used OCR/georeference sources.
  Internal event spans were 1.752s, 4.832s, 3.588s, 0.556s, 4.305s, and 6.244s
  respectively. Active fast-path production smoke
  `out/prod-active-fastpath-20260528-155700/` kept the low-resolution Nashville
  issue #5 case on `catalog-shape-match:ocr-label-hint` in 0.862s wall /
  0.464s event span and Los Angeles on `catalog-shape-match` in 2.470s wall /
  0.244s event span.
- Current fuzzy-label hint head: the one-pixel RGBA cache-bust variant
  `/tmp/mbb-issue5-input-warm-bust.png` reproduced the production miss because
  OCR returned `Naslville`, then passed locally after the single-edit hint guard
  with the same Nashville bbox, source `catalog-shape-match:ocr-label-hint`,
  and confidence 0.949. Focused catalog tests passed 12 tests, full pytest
  passed 84 tests and 9 subtests, `compileall`, `node --check`, and
  `git diff --check` passed, and fresh-cache full benchmark
  `out/benchmark-fuzzy-label-hint-20260528-160018/full-report.json` passed 8/8
  active fixtures with avg IoU 0.983, min IoU 0.943, and total scored duration
  3.18s.
- Production fuzzy-label hint deployment `dpl_D4vAfWFR7GbLZKC7HZTedcEzWA9S`
  reported `pipeline-c44dde0142d2b891` and kept `api/index.py` at 92.82 MB.
  The exact RGBA cache-bust production repro then succeeded uncached through
  `catalog-shape-match:ocr-label-hint` with `nashville-waymo`, 0.949 confidence,
  2.550s wall, and 1.362s internal event span. A second warm uncached RGB pixel
  variant succeeded through the same source in 0.906s wall and 0.474s internal
  event span. Miami remained uncached on OCR/georeference with `catalog_slug:
  null`, preserving the stale-market guard.
- Current OCR/extraction overlap head: focused API-cache tests passed 10 tests.
  Fresh-cache in-process no-catalog profile moved the active OCR/georeference
  path to sub-second for 7/8 active fixtures, with Phoenix at 1.003s while
  preserving `ocr-georeference:nominatim-label-fit+osm-road-refine`, confidence
  0.908, and road score 0.706233. The subprocess benchmark still includes
  interpreter startup but improved its no-catalog total from the prior 8.291s
  report to 7.63s in
  `out/benchmark-ocr-overlap-no-catalog-20260528-160757/full-report.json`,
  with 8/8 active fixtures passed, avg IoU 0.962, min IoU 0.931. The city-forced
  no-catalog benchmark
  `out/benchmark-ocr-overlap-no-catalog-city-20260528-160815/full-report.json`
  passed 8/8 active fixtures at avg IoU 0.962, min IoU 0.931, total 7.36s.
  Default catalog benchmark
  `out/benchmark-ocr-overlap-default-20260528-160757/full-report.json` stayed
  green at 8/8 active fixtures, avg IoU 0.983, min IoU 0.943. Full pytest
  passed 85 tests and 9 subtests, and `compileall`, `node --check`, and
  `git diff --check` passed.
- Production OCR/extraction overlap deployment `dpl_93LnxNN4wAVRHAMBvbHe7hmw1Cai`
  reported `pipeline-c9d6bfac3de23dd9` and kept `api/index.py` at 92.82 MB.
  Cache-busted active Nashville stayed on `catalog-shape-match:ocr-label-hint`
  with `nashville-waymo`, 1.009s wall and 0.504s internal event span. A
  city-forced Phoenix smoke exercised the overlapped OCR/georeference path and
  returned the expected `ocr-georeference:nominatim-label-fit+osm-road-refine`
  output with `catalog_slug: null`, confidence 0.918, road score 0.806613, and
  the expected bbox; however Vercel CPU/runtime time remained high at 13.552s
  wall / 10.177s internal on the first uncached variant and 9.194s wall /
  6.937s internal on a warmed uncached variant, so this is a local/server-path
  latency win rather than a production sub-second breakthrough for the
  OCR-heavy Phoenix city-forced path.
- Current city-provided catalog head: focused API/cache and catalog tests passed
  23 tests, full pytest passed 86 tests and 9 subtests, `compileall`,
  `node --check map_boundary_builder/web_assets/app.js`, and `git diff --check`
  passed. Fresh-cache city-overrides benchmark
  `out/benchmark-city-catalog-fastpath-20260528-continue/full-report.json`
  passed 8/8 active fixtures, skipped 7 known `reference_mismatch` fixtures,
  avg IoU 0.983, min IoU 0.943, total 3.04s, with every scored fixture using
  `catalog-shape-match`; Phoenix city-provided local time was 0.44s. The
  explicit OCR/georeference gate
  `out/benchmark-city-fastpath-no-catalog-gate-20260528-continue/full-report.json`
  stayed green at 8/8 active, avg IoU 0.962, min IoU 0.931, total 7.47s. The
  default auto benchmark
  `out/benchmark-city-fastpath-default-gate-20260528-continue/full-report.json`
  stayed green at 8/8 active, avg IoU 0.983, min IoU 0.943, total 3.09s.
- Rejected city-context road-only early return. A downscaled probe still spent
  1.7-23.3s per active fixture and produced poor IoU on returned matches
  (Phoenix 0.225, Nashville 0.362, Los Angeles 0.257, Austin 0.539), so it is
  neither fast enough nor reliable enough for production.
- Rejected RapidOCR detector `limit_side_len=576` as a new default. It preserved
  the active no-catalog benchmark but took 7.82s total versus the matched 608
  baseline at 7.46s, and slightly reduced some gray-fill IoUs, so the current
  608 default remains better.
- Current catalog-scaled extraction head: focused extract/catalog/API tests
  passed 27 tests, full pytest passed 88 tests and 9 subtests, and `compileall`,
  `node --check map_boundary_builder/web_assets/app.js`, and `git diff --check`
  passed. A blunt global 1200px extraction cap was rejected even though
  extraction-only scoring passed, because the no-catalog full benchmark slightly
  reduced individual OCR/georeference IoUs and did not improve total time. The
  accepted version uses the 1600px scaled extraction only for catalog probing and
  exact catalog outputs. Default full benchmark
  `out/benchmark-catalog-scaled-default-20260528-continue/full-report.json`
  passed 8/8 active fixtures, skipped 7 known `reference_mismatch` fixtures,
  improved avg IoU from the prior 0.983 city-fastpath gate to 0.993, kept min
  IoU 0.943, and completed in 3.02s total. City-overrides benchmark
  `out/benchmark-catalog-scaled-city-20260528-continue/full-report.json`
  passed 8/8 active, avg IoU 0.993, min IoU 0.943, total 2.94s. The explicit
  no-catalog gates stayed on OCR/georeference with `catalog_slug: null`:
  `out/benchmark-catalog-scaled-no-catalog-20260528-continue/full-report.json`
  passed 8/8 active, avg IoU 0.962, min IoU 0.931, total 7.58s, and
  `out/benchmark-catalog-scaled-no-catalog-city-20260528-continue/full-report.json`
  passed 8/8 active, avg IoU 0.962, min IoU 0.931, total 7.50s.
- Production catalog-scaled deployment `dpl_C993qiU2JYypGVgtYrCjk8iTFZZh`
  was aliased to `https://mapboundary.app` and reported
  `pipeline-515414da4702cdfb`. Cache-busted Phoenix with `city=Phoenix` stayed
  on `catalog-shape-match` with `phoenix-waymo`, exact catalog bbox, and server
  event spans of 0.624s cold and 0.444s warm-busted, improving on the prior
  city-fastpath production spans of 0.688s and 0.519s. Production stale checks
  for Houston Tesla, Miami Waymo, and Bay Area Tesla returned `catalog_slug:
  null` and OCR/georeference sources, preserving the user-confirmed stale-market
  guard.
- Road-refinement search batches now default to 1024 candidates. This does not
  change the candidate grid or scoring function; it reduces Python batching
  overhead in the road-heavy arbitrary OCR/georeference path. Focused
  Phoenix/Nashville no-catalog probes preserved avg IoU 0.985 and min IoU
  0.984 while reducing total scored time from 2.85s at batch 512 to 2.77s at
  batch 1024. Full no-catalog benchmark
  `out/benchmark-roadbatch1024-no-catalog-20260528-continue/full-report.json`
  passed 8/8 active, skipped 7 known `reference_mismatch` fixtures, preserved
  avg IoU 0.962 and min IoU 0.931, and reduced total scored time from the
  current catalog-scaled no-catalog gate's 7.58s to 7.27s. With batch 1024 as
  the code default, `out/benchmark-roadbatch1024-defaultcode-no-catalog-20260528-continue/full-report.json`
  passed 8/8 active, avg IoU 0.962, min IoU 0.931, total 7.41s; default
  catalog benchmark `out/benchmark-roadbatch1024-defaultcode-default-20260528-continue/full-report.json`
  stayed green at 8/8 active, avg IoU 0.993, min IoU 0.943, total 2.96s. Full
  pytest passed 88 tests and 9 subtests, and `compileall`,
  `node --check map_boundary_builder/web_assets/app.js`, and `git diff --check`
  passed.
- Production road-batch deployment `dpl_BYrgHq7bSq9Bj4z22KZS1hknuzC2` was
  aliased to `https://mapboundary.app` and reported
  `pipeline-c1ba43b997adcc5b`. Cache-busted Miami Waymo, whose catalog entry is
  intentionally stale, stayed on
  `ocr-georeference:nominatim-label-fit+osm-road-refine` with `catalog_slug:
  null`, the same bbox, confidence 0.864, and road score 0.681518. The first
  request after deploy had a 7.235s event span; the warm-busted follow-up
  dropped to 3.894s, preserving output while exercising the road-refinement
  path in production.
- City-hinted requests now overlap OCR with extraction when the hint has no
  active catalog candidate, which covers the user-confirmed stale Houston,
  Miami, and Bay Area catalog entries plus unsupported cities. Active catalog
  hints still avoid OCR so the exact-shape fast path remains cheap. Local fresh
  Miami Waymo A/B with identical output improved from 1.253s when simulating
  the old no-overlap behavior to 0.921s with the new stale-city overlap,
  preserving source `ocr-georeference:nominatim-label-fit+osm-road-refine`,
  confidence 0.864, road score 0.681518, and bbox. Phoenix with `city=Phoenix`
  stayed on `catalog-shape-match` in 0.107s with no OCR event. Focused stale
  fixture/catalog tests passed 26 tests; full pytest passed 88 tests and 9
  subtests; `compileall`, `node --check`, and `git diff --check` passed. The
  default benchmark `out/stale-city-overlap-default-full-20260528-continue/full-report.json`
  passed 8/8 active, skipped 7 known `reference_mismatch` fixtures, avg IoU
  0.993, min IoU 0.943, total 3.00s; city-overrides
  `out/stale-city-overlap-city-full-20260528-continue/full-report.json` passed
  8/8 active, avg IoU 0.993, min IoU 0.943, total 2.91s.
- Production stale-city overlap deployment `dpl_ETkWceqgwERMGWXN9x5715vpZDk9`
  was aliased to `https://mapboundary.app` and reported
  `pipeline-b8d2af4019507275`. Phoenix with `city=Phoenix` stayed on
  `catalog-shape-match`, reported `phoenix-waymo`, had no OCR event, and
  completed with a 0.616s server event span. Cache-busted Miami with
  `city=Miami` stayed uncached on
  `ocr-georeference:nominatim-label-fit+osm-road-refine` with `catalog_slug:
  null`, confidence 0.864, road score 0.681518, and the same bbox. The first
  cold-ish stale run had a 7.085s span; the warm cache-busted follow-up had a
  0.893s server event span, down from the prior comparable warm-busted 3.894s
  road-batch production smoke, because OCR labels were ready immediately after
  full extraction.
- API run summaries now expose catalog observability fields directly:
  `catalog_slug`, `catalog_shape_iou`, `catalog_shape_margin`, and
  `catalog_area_ratio`. This makes production smokes less error-prone because
  active catalog hits and stale-market OCR fallbacks can be distinguished from
  the stable summary payload instead of progress events alone. Focused summary,
  API-cache, and catalog tests passed 26 tests. A local Phoenix/Miami smoke
  proved Phoenix summary now reports `catalog_slug: phoenix-waymo`, while stale
  Miami reports `catalog_slug: null` and preserves
  `ocr-georeference:nominatim-label-fit+osm-road-refine`, confidence 0.864, and
  road score 0.681518. Full pytest passed 90 tests and 9 subtests;
  `compileall`, `node --check`, and `git diff --check` passed. The default
  benchmark `out/summary-metadata-default-full-20260528-continue2/full-report.json`
  passed 8/8 active, skipped 7 known `reference_mismatch` fixtures, avg IoU
  0.993, min IoU 0.943, total 2.92s; the no-catalog benchmark
  `out/summary-metadata-no-catalog-full-20260528-continue2/full-report.json`
  passed 8/8 active, avg IoU 0.962, min IoU 0.931, total 7.36s.
- Production summary-metadata deployment `dpl_4FdkLbXRXwP6nJTN3FCf4uZBgJM8`
  was aliased to `https://mapboundary.app` and reported
  `pipeline-dd206b786b8b45aa`. Cache-busted Phoenix with `city=Phoenix`
  reported `catalog_slug: phoenix-waymo`, shape IoU 0.984044, margin 0.358165,
  no OCR event, and a 0.614s server event span. Cache-busted Miami with
  `city=Miami` reported null catalog fields, stayed on
  `ocr-georeference:nominatim-label-fit+osm-road-refine`, confidence 0.864,
  road score 0.681518, and the same bbox; the warm cache-busted follow-up
  preserved that output with a 0.900s server event span.
- Rejected RapidOCR detector-side reduction as a default after a fresh-cache
  full no-catalog gate. `MAP_BOUNDARY_RAPIDOCR_DET_LIMIT_SIDE_LEN=544` and
  `512` preserved pass/fail but slowed the total benchmark and reduced gray-fill
  IoU; `576` was effectively tied and not enough evidence to revisit the older
  rejection. Current 608 remains the safer default.
- Vercel code-side memory/CPU tuning is not currently a shippable repo change:
  current Vercel documentation says Fluid Compute memory/CPU is configured from
  project settings, not `vercel.json`; `maxDuration` can be configured for
  Python functions but affects timeout headroom rather than latency.
- Vercel packaging now excludes `benchmarks/` and `tests/` from both source
  upload (`.vercelignore`) and Python function bundling (`excludeFiles`). This
  keeps evidence logs and test files out of production packages, reduces the
  chance of notes-only commits affecting deployment payloads, and preserves the
  existing `promo-video/`, cache, output, and virtualenv exclusions. Focused
  pipeline/API cache tests passed 11 tests. Full pytest passed 90 tests and 9
  subtests; `compileall`, `node --check`, and `git diff --check` passed.
- Production packaging deployment `dpl_Bq9quUCwQygtoXFQFDzvLXk5CDhF` was
  aliased to `https://mapboundary.app` and kept health on
  `pipeline-dd206b786b8b45aa`. Vercel upload size for the config-only deploy
  dropped to 582B, deployment files dropped to 67 from the previous 78-file
  source surface, and the reported bundle size nudged from 298.10 MB to
  297.92 MB. The bundle still exceeds Vercel's inline limit because OpenCV/ONNX
  dominate the package; this is a hygiene win, not the cold-start breakthrough.
- Current non-catalog benchmark observability head: `PATH=/usr/bin:/bin
  PYTHONPATH=. .venv/bin/python -m pytest -q` passed 81 tests and 9 subtests;
  `compileall`, `node --check`, and `git diff --check` passed. The default
  full benchmark `out/benchmark-default-observability-20260528-154401/full-report.json`
  stayed green at 8/8 active fixtures, 7 skipped, avg IoU 0.983, min IoU 0.943,
  and now prints source columns proving the active suite is catalog-sourced.
  Both explicit non-catalog gates passed: image-only inference
  `out/benchmark-no-catalog-20260528-154327/full-report.json` and city-forced
  inference `out/benchmark-no-catalog-city-20260528-154327/full-report.json`
  each passed 8/8 active fixtures, skipped 7 known `reference_mismatch`
  fixtures, avg IoU 0.962, min IoU 0.931. The reports show OCR/georeference
  sources and `catalog_slug: null`, confirming the catalog path was bypassed.
- Post-deploy API compatibility fix: the API passes a `SimpleNamespace` options
  object instead of `BoundaryBuildOptions`, so `allow_catalog` now defaults to
  true when absent and has a focused regression test. After the fix,
  `PYTHONPATH=. .venv/bin/python -m pytest -q` passed 82 tests and 9 subtests;
  default full benchmark `out/benchmark-post-namespace-fix-20260528-154617/full-report.json`
  passed 8/8 active fixtures with avg IoU 0.983, min IoU 0.943; and
  `out/benchmark-no-catalog-post-fix-20260528-154617/full-report.json` passed
  the OCR/georeference-only gate at 8/8 active fixtures, avg IoU 0.962, min IoU
  0.931.
- Rejected mask-bounds OCR cropping as a default optimization. It reduced OCR
  time on representative city-forced cases, but Phoenix dropped from confidence
  0.908 with 6 controls to 0.873 with 5 controls, and Miami's road score moved
  from 0.681518 to 0.676386. That is not enough evidence for the no-regression
  bar even though several other cases were slightly faster.
- Rejected lower dynamic RapidOCR max dimensions for the general path. At 1500,
  Phoenix lost road refinement and dropped to confidence 0.854; at 1400,
  Nashville lost road refinement and dropped to confidence 0.672; at 1300/1200,
  Miami dropped to 3 controls with a substantially shifted bbox. Keep the 1600
  cap until a stronger validator exists.
- Current label-aided catalog head: `PATH=/usr/bin:/bin PYTHONPATH=.
  .venv/bin/python -m pytest -q` passed 81 tests and 9 subtests. `compileall`
  over `map_boundary_builder`, `api`, and `tests`, `node --check
  map_boundary_builder/web_assets/app.js`, and the full drift-aware benchmark
  passed.
- Fresh-cache full benchmark after the label-aided catalog change passed 8/8
  scored fixtures, skipped 7 known `reference_mismatch` fixtures, avg IoU
  0.983, min IoU 0.943, report
  `out/benchmark-label-hint-tight-20260528-153327/full-report.json`.
- Issue #5 Nashville debug artifact:
  `/tmp/mbb-issue5-input.png` is a 420x236 image with only one OCR label,
  `Nashville`. Before the change it failed after 18.503s with "Could not infer
  a reliable map location"; after the change it used
  `catalog-shape-match:ocr-label-hint`, confidence 0.949, catalog IoU 0.949070,
  runner-up margin 0.275103, bbox
  `[-86.8461084, 36.1089989, -86.6904545, 36.242681]`, and completed in 0.305s
  on the first post-change probe. The full-resolution companion still uses the
  strict pre-OCR `catalog-shape-match` path in 0.105s.
- Fresh-cache changed-service-area smoke for the user-confirmed drifted markets
  kept Bay Area, Houston, and Miami out of the hard reference score while still
  covering current outputs: Bay Area Tesla 0.027s, Bay Area Waymo 0.130s, Bay
  Area Zoox 0.094s, Houston Tesla 0.013s, Houston Waymo 0.116s, and Miami Waymo
  0.111s, all through `catalog-shape-match` with current-shape confidence caps
  and exact catalog bboxes.
- Production deployment `dpl_DBUxQoadhf5RVmotskymbHSggxts` reported
  `pipeline-d68b6e05753d3c01`, was aliased to `https://mapboundary.app`, and
  kept `api/index.py` at 92.83 MB. The first cache-busted low-resolution
  Nashville issue #5 POST succeeded through `catalog-shape-match:ocr-label-hint`
  in 2.729s wall while the function warmed. Three following cache-busted,
  uncached production repeats completed in 0.462s, 0.428s, and 0.466s wall with
  timestamped server event spans of 0.043s, 0.037s, and 0.037s. All returned
  catalog slug `nashville-waymo`, shape IoU 0.949070, margin 0.275103, and bbox
  `[-86.8461084, 36.1089989, -86.6904545, 36.242681]`.
- Production changed-market smoke after the same deployment covered the
  user-confirmed stale-ground-truth markets through current catalog outputs:
  Bay Area Tesla 0.969s wall, Bay Area Waymo 2.730s, Bay Area Zoox 2.660s,
  Houston Tesla 0.890s, Houston Waymo 2.845s, and Miami Waymo 2.405s. All were
  uncached, used `catalog-shape-match`, and returned the current catalog bboxes;
  the higher wall times reflect Vercel instance variance rather than OCR/geocoder
  fallback.
- Current Los Angeles catalog head: the full pytest suite passed 75 tests and 9
  subtests. `compileall`, `node --check map_boundary_builder/web_assets/app.js`,
  bundled catalog `json.tool`, and `git diff --check` passed. The mistaken
  old-path syntax check against `public/app.js` is ignored because that file is
  not present in this repo state.
- Fresh-cache full benchmark after the Los Angeles current-shape catalog entry:
  `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-la-current-catalog-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/la-current-catalog-full`
  passed 8/8 scored fixtures, skipped 7 known `reference_mismatch` fixtures,
  avg IoU 0.983, min IoU 0.942, and completed in 3.25s wall. Output sources
  showed Los Angeles Waymo moved from OCR to `catalog-shape-match` with
  confidence 0.859, while Houston, Miami, and Bay Area drifted fixtures remain
  treated as stale-ground-truth data debt rather than scored regressions.
- Focused fresh-cache Los Angeles full benchmark:
  `MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-la-current-single-XXXXXX) ... --mode full --only los-angeles`
  passed with IoU 0.942 and completed in 0.50s wall.
- Production deployment `dpl_DtDxTyqLy7L4hMUXAip9Zp5ScWeu` reported
  `pipeline-95677920c63d6cfd` and kept `api/index.py` at 92.8 MB. Cache-busted
  production Los Angeles calls used `catalog-shape-match` with confidence 0.859
  and internal event spans of 0.455s and 0.340s, down from the previous LA OCR
  fallback evidence at roughly 4.6-5.1s internal. Cache-busted production
  smokes also matched Miami Waymo in 0.559s internal, Houston Waymo in 0.540s,
  Bay Area Tesla in 0.056s, and Bay Area Zoox in 0.441s, all through
  `catalog-shape-match` with their capped current-shape confidences. Houston
  Tesla correctly stayed on `ocr-georeference:nominatim-label-fit`, confidence
  0.853, with no catalog slug.
- Current rotation-aware catalog head: the full pytest suite passed 77 tests and
  9 subtests. `compileall`, `node --check map_boundary_builder/web_assets/app.js`,
  bundled catalog `json.tool`, and `git diff --check` passed.
- Fresh-cache full benchmark after the rotation-aware catalog change passed 8/8
  scored fixtures, skipped 7 known `reference_mismatch` fixtures, avg IoU 0.983,
  min IoU 0.943, and completed in 3.29s wall.
- Fresh-cache changed-service-area no-network smoke had zero attempted network
  calls. Houston Tesla moved from OCR to `catalog-shape-match` in 0.014s with
  confidence 0.853, catalog shape IoU 0.970925, margin 0.415955, rotation
  -1.75 degrees, and exact bbox
  `[-95.6247125, 29.8572751, -95.5245324, 29.9718015]`. The catalog output had
  IoU 1.0 against the current OCR baseline because current-verified entries
  return exact verified geometry after matching. Bay Area Waymo and Las Vegas
  Zoox correctly stayed on OCR.
- Production deployment `dpl_HgroMZbD67NmTEeL4jabF6e9zEDp` reported
  `pipeline-72b0f2c4fe22282c` and kept `api/index.py` at 92.8 MB.
  Cache-busted Houston Tesla now uses `catalog-shape-match` with confidence
  0.853, catalog IoU 0.970925, and the exact current OCR bbox. The first live
  smoke returned in 1.909s wall / 0.166s internal generation, down from the
  previous OCR-path evidence at 2.084s wall / 1.302s internal; warmed
  cache-busted repeats returned in 0.899s and 0.741s wall with 0.056s internal
  generation spans. A cache-busted Bay Area Waymo guard check stayed on
  `ocr-georeference:nominatim-label-fit`, proving the rotation search did not
  force a weak Bay Area catalog match.
- Current Bay Area Waymo exact-catalog head: the full pytest suite passed 78
  tests and 9 subtests. `compileall`, `node --check
  map_boundary_builder/web_assets/app.js`, bundled catalog `json.tool`, and
  `git diff --check` passed.
- Fresh-cache full benchmark after the Bay Area Waymo exact-catalog change
  passed 8/8 scored fixtures, skipped 7 known `reference_mismatch` fixtures,
  avg IoU 0.983, min IoU 0.943, and completed in 3.21s wall.
- Fresh-cache changed-service-area no-network smoke had zero attempted network
  calls. Bay Area Waymo moved from OCR to `catalog-shape-match` in 0.134s with
  confidence 0.877, catalog shape IoU 0.969189, margin 0.747906, and exact bbox
  `[-122.4978873, 37.3073419, -121.8576229, 37.7981634]`; the catalog output
  had IoU 1.0 against the current OCR baseline. Las Vegas Zoox correctly stayed
  on OCR.
- Production deployment `dpl_AfSoK7X3k5xYzREkP2YdCXnrTrp9` reported
  `pipeline-e1892727b126c384` and `api/index.py` at 92.84 MB. Cache-busted
  Bay Area Waymo now uses `catalog-shape-match` with confidence 0.877, catalog
  IoU 0.969189, margin 0.747906, and the exact current OCR bbox. Live internal
  generation spans dropped from the previous OCR-path guard check at 5.807s to
  0.655s, 0.485s, 0.520s, and 0.501s. A cache-busted Las Vegas Zoox guard check
  stayed on `ocr-georeference:nominatim-label-fit`.
- Current exact ordered-contour head: the full pytest suite passed 79 tests and
  9 subtests. `compileall`, `node --check
  map_boundary_builder/web_assets/app.js`, bundled catalog `json.tool`, and
  `git diff --check` passed.
- Fresh-cache full benchmark after the exact ordered-contour catalog change
  passed 8/8 scored fixtures, skipped 7 known `reference_mismatch` fixtures,
  avg IoU 0.983, min IoU 0.943, and completed in 3.22s wall.
- Fresh-cache changed-service-area no-network smoke had zero attempted network
  calls. Las Vegas Zoox moved from OCR to `catalog-shape-match` in 0.017s with
  confidence 0.767, catalog shape IoU 0.999999, margin 0.464159, rotation
  -9.799 degrees, and exact bbox
  `[-115.3550119, 36.0353866, -115.1830059, 36.187696]`; the catalog output had
  IoU 1.0 against the current OCR baseline.
- Production deployment `dpl_BRqXmDNNrMkuy6nrUt7959q6nPRF` reported
  `pipeline-513196b335a53f9b` and `api/index.py` at 92.83 MB. Cache-busted
  Las Vegas Zoox now uses `catalog-shape-match` with confidence 0.767, catalog
  IoU 0.999999, margin 0.464159, and the exact current OCR bbox. Live internal
  generation spans dropped from the previous OCR-path check at 2.317s to
  0.176s, 0.079s, 0.072s, and 0.074s; warmed cache-busted end-to-end wall time
  reached 0.956s and 0.814s. Bay Area Waymo stayed healthy on
  `catalog-shape-match` in the same production smoke.
- Catalog fast-path head: `PATH=/usr/bin:/bin PYTHONPATH=. .venv/bin/pytest -q`
  passed 74 tests and 9 subtests. `compileall`, `node --check`,
  `json.tool` for bundled JSON, and `git diff --check` passed.
- Fresh-cache full benchmark:
  `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-catalog-final-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/catalog-fast-final-full`
  passed 8/8 scored fixtures, skipped 7 known `reference_mismatch` fixtures,
  improved avg IoU from 0.962 to 0.983, improved min IoU from 0.931 to 0.943,
  and reduced local wall time from the latest classifier-only 7.40s baseline to
  3.94s.
- Catalog fast-path output sources: Austin Tesla, Dallas Tesla, Dallas Waymo,
  Nashville Waymo, Orlando Waymo, Phoenix Waymo, and San Antonio Waymo used
  `catalog-shape-match`; Los Angeles Waymo stayed on
  `ocr-georeference:nominatim-label-fit` because its normalized shape match was
  below the high-confidence threshold.
- Fresh-cache changed-service-area no-network smoke still had zero attempted
  geocoder/Overpass calls and did not use catalog references for drifted
  Houston, Miami, Bay Area, or Las Vegas fixtures. Outputs preserved the current
  OCR/georeference sources and bboxes; Miami Waymo remained road-refined with
  confidence 0.864, 6 controls, bbox
  `[-80.323092, 25.688025, -80.1185, 25.939698]`, and road score 0.681518.
- Current-shape catalog probe: a no-network fresh-cache smoke matched Bay Area
  Tesla in 0.014s, Bay Area Zoox in 0.095s, Houston Waymo in 0.127s, and Miami
  Waymo in 0.117s with the same bboxes and confidence caps as their OCR outputs.
  Bay Area Waymo, Houston Tesla, and Las Vegas Zoox remained on OCR. The scored
  full benchmark stayed clean at 8/8 scored, 7 `reference_mismatch` skipped,
  avg IoU 0.983, min IoU 0.943, in 4.16s wall.
- Production deployment `dpl_BrUkaa4zLHGxJ3DorGnsVnc7JqAd` reported
  `pipeline-66de065f3955a437` and `api/index.py` at 92.8 MB. Cache-busted
  production smokes matched Miami Waymo via `catalog-shape-match` with
  confidence 0.864 and 0.626s internal span, Bay Area Tesla via
  `catalog-shape-match` with confidence 0.916 and 0.053s internal span, and
  Houston Waymo via `catalog-shape-match` with confidence 0.865 and 0.520s
  internal span. Houston Tesla correctly stayed on
  `ocr-georeference:nominatim-label-fit`.
- Current head: `PYTHONPATH=. .venv/bin/pytest -q`: 71 passed, 9 subtests
  passed. `compileall`, `node --check`, and `json.tool` passed.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-final3-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/final3-place-wait-full`: PASS 8/8 scored fixtures, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931.
- Fresh-cache changed-service-area no-network smoke after the Las Vegas/context
  changes had zero attempted geocoder/Overpass `urlopen` calls:
  - Bay Area Waymo 0.576s, confidence 0.877, 15 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Bay Area Tesla 0.108s, confidence 0.916, 5 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Bay Area Zoox 0.313s, confidence 0.930, 6 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Houston Waymo 0.424s, confidence 0.865, 7 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Houston Tesla 0.073s, confidence 0.853, 3 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Miami Waymo 0.708s, confidence 0.864, 6 controls,
    `ocr-georeference:nominatim-label-fit+osm-road-refine`, road score
    0.681518.
  - Las Vegas Zoox 0.164s, confidence 0.767, 3 controls,
    `ocr-georeference:nominatim-label-fit`.
- After confirming Houston, Miami, and Bay Area are changed-service-area data
  debt, `PYTHONPATH=. .venv/bin/pytest -q`: 65 passed, 9 subtests passed.
- `PYTHONPATH=. .venv/bin/python -m compileall -q api map_boundary_builder`,
  `node --check map_boundary_builder/web_assets/app.js`, and
  `git diff --check`: pass.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-miami-label-seed-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/miami-label-seed-full`: PASS 8/8 scored fixtures, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931.
- Fresh-cache changed-service-area no-network smoke with geocoder, OSM road, and
  OSM place `urlopen` patched to record and fail showed zero attempted network
  calls:
  - Bay Area Waymo 0.640s, confidence 0.877, 15 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Bay Area Tesla 0.473s, confidence 0.840, 4 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Bay Area Zoox 0.373s, confidence 0.930, 6 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Houston Waymo 0.519s, confidence 0.865, 7 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Houston Tesla 0.436s, confidence 0.845, 3 controls,
    `ocr-georeference:nominatim-label-fit`.
  - Miami Waymo 1.344s, confidence 0.864, 6 controls,
    `ocr-georeference:nominatim-label-fit+osm-road-refine`.
- Rejected replacing the Downtown Brickell live-result seed with either empty
  misses or a cleaner Brickell-neighborhood payload. Both preserved the same
  final Miami bbox/confidence, but they caused dozens of fallback geocoder
  attempts for lower-quality Miami label combinations, so the literal seeded
  current-result payload is the lower-latency and lower-variance option.
- Road full-fallback threshold probe: Miami Waymo with the old 0.68 threshold
  spent roughly 1.6s locally with bbox
  `[-80.3230924,25.6880246,-80.1184998,25.9396977]`, confidence 0.864,
  6 controls, and road score 0.673348. With the 0.60 default it completed in
  0.991s in a focused run and 0.887s in the six-fixture changed-market smoke
  with the same bbox, confidence, controls, source, and road score.
- After the 0.60 threshold change, `PYTHONPATH=. .venv/bin/pytest -q`: 65
  passed, 9 subtests passed. `PYTHONPATH=. .venv/bin/python -m compileall -q api
  map_boundary_builder`, `node --check map_boundary_builder/web_assets/app.js`,
  and `git diff --check`: pass.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road-fallback060-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/road-fallback060-full`: PASS 8/8 scored fixtures, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931.
- Fresh-cache changed-service-area no-network smoke after the 0.60 road fallback
  threshold: Bay Area Waymo 0.560s, Bay Area Tesla 0.423s, Bay Area Zoox 0.340s,
  Houston Waymo 0.422s, Houston Tesla 0.439s, Miami Waymo 0.887s; all had zero
  attempted geocoder/Overpass `urlopen` calls.
- Dallas Waymo road-gate trace before the tight-fit skip made one live Overpass
  road request, took 13.878s, and still returned
  `ocr-georeference:nominatim-label-fit` with bbox
  `[-96.8802629,32.7284439,-96.7280792,32.8668601]`, confidence 0.864, and
  5 controls. Instrumented gate metrics showed the road refine spent 9.331s,
  produced score 0.822784, then was rejected by label-fit preservation.
- After the tight-fit road skip, Dallas Waymo with geocoder, OSM road, and OSM
  place `urlopen` blocked completed with zero attempted network calls, the same
  bbox/confidence/controls/source, and 1.859s cold local wall time. In a warm OCR
  timing sweep, Dallas georeference dropped to 0.036s and total runtime to
  1.653s; the remaining Dallas time is RapidOCR-heavy rather than georef-heavy.
- After the tight-fit road skip, `PYTHONPATH=. .venv/bin/pytest -q`: 67 passed,
  9 subtests passed. `PYTHONPATH=. .venv/bin/python -m compileall -q api
  map_boundary_builder`, `node --check map_boundary_builder/web_assets/app.js`,
  and `git diff --check`: pass.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-tight-label-road-gate-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/tight-label-road-gate-full`: PASS 8/8 scored fixtures, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931.
- Rejected a Dallas-only OCR max-dimension sweep as too small and too risky to
  generalize. Dallas preserved bbox/confidence/controls/source from caps 1600
  down to 900, but cold local wall time only moved from 1.714s at 1600 to 1.536s
  at 1000, and prior full-suite cap probes found lower global caps can regress
  other screenshots.
- Road-point cap sweep after the tight-label road skip:
  - Default 6000 cap: Phoenix refine 0.477s / score 0.714768, Nashville refine
    0.440s / score 0.770404, Miami refine 0.446s / score 0.673348.
  - 4000 cap: Phoenix refine 0.328s / score 0.706233, Nashville refine 0.340s
    / score 0.772836, Miami refine 0.368s / score 0.681518; all preserved bbox,
    confidence, controls, and road-refined source.
  - 3000 cap: faster, but Nashville bbox shifted, so rejected for accuracy
    preservation.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road-points4000-full-XXXXXX) MAP_BOUNDARY_ROAD_MATCH_MAX_POINTS=4000 PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/road-points4000-full`: PASS 8/8 scored fixtures, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931.
- With 4000 as the code default, `PYTHONPATH=. .venv/bin/pytest -q`: 67
  passed, 9 subtests passed. `PYTHONPATH=. .venv/bin/python -m compileall -q
  api map_boundary_builder`, `node --check map_boundary_builder/web_assets/app.js`,
  and `git diff --check`: pass.
- `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d /tmp/mbb-road-points4000-default-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m map_boundary_builder.benchmark --mode full --out-dir out/road-points4000-default-full`: PASS 8/8 scored fixtures, 7 skipped `reference_mismatch`, avg IoU 0.962, min IoU 0.931.
- Fresh-cache changed-service-area no-network smoke with the 4000-point default:
  Bay Area Waymo 0.584s, Bay Area Tesla 0.435s, Bay Area Zoox 0.395s, Houston
  Waymo 0.488s, Houston Tesla 0.449s, Miami Waymo 0.785s; all had zero attempted
  geocoder/Overpass `urlopen` calls. Miami kept bbox
  `[-80.3230924,25.6880246,-80.1184998,25.9396977]`, confidence 0.864,
  6 controls, `ocr-georeference:nominatim-label-fit+osm-road-refine`, and road
  score 0.681518.
- Production deploy `dpl_8tVWMypKxGqmQnXZubpVuooxkqc2` is live at
  `mapboundary.app` with pipeline hash `pipeline-79b1bc6754d6e3bc`. After the
  first post-deploy warm-up, cache-busted production repeats returned Phoenix
  0.747s event span / road score 0.706232, Nashville 0.723s / road score
  0.772836, and Miami 0.726s / road score 0.681518, all preserving the expected
  road-refined source and bbox.
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
- Road score-image validation:
  - A fresh-cache no-network Miami profile showed road refinement dominated the
    run, with `refine_transform_with_osm_roads` at about 1.06s and
    `score_transform_batch` at about 0.95s. Precomputing the distance-to-road
    score image once per feature field preserved Miami's bbox, confidence,
    controls, source, and road score while reducing the profiled run from 1.849s
    to 1.455s.
  - Clean A/B from a detached `HEAD` worktree versus the score-image prototype
    preserved bboxes, confidence, and road scores on Miami/Phoenix/Nashville
    no-network smokes while reducing total wall time from 3.79s to 3.65s.
  - Clean full-suite A/B preserved the drift-aware benchmark at 8/8 scored
    fixtures, 7 `reference_mismatch` fixtures skipped, avg IoU 0.962, min IoU
    0.931, while moving wall time from 7.87s to 7.77s.
  - Validation passed: `PYTHONPATH=. .venv/bin/pytest -q`, 63 tests;
    `PYTHONPATH=. .venv/bin/python -m compileall -q api map_boundary_builder`;
    and `node --check map_boundary_builder/web_assets/app.js`.
  - Production deployment `dpl_CbPzvdUsFmhrLw434Y8HmUqFqYMw` reported
    `pipeline-782cc2ab7d027532` and kept `api/index.py` at 92.86 MB. First
    cache-busted post-deploy calls paid cold/cache-invalidation cost: Miami
    15.634s wall / 12.519s event span and Phoenix 12.114s wall / 9.410s event
    span, with identical geometry, confidence, and road scores. After warmup,
    new cache-busted calls hit sub-second server-side generation spans: Miami
    4.722s wall / 0.788s event span and Phoenix 3.427s wall / 0.791s event
    span. Exact repeats returned from the result cache in 2.293s and 2.217s.
- GeoJSON-first response validation:
  - Added an `include_overlay` API flag, defaulting to the previous overlay
    behavior for compatibility. The web app now requests `include_overlay=0`,
    so the first response returns GeoJSON, summary, and the rendered boundary
    without writing/encoding an inline overlay preview.
  - Local Miami no-network validation preserved the exact bbox, confidence,
    source, and GeoJSON path while cutting the gzip response from about 58.5 KB
    with inline overlay to about 2.1 KB without overlay. Result-cache keys now
    include `include_overlay` and `RUN_RESULT_CACHE_VERSION` moved to
    `run-result-v4`, so fast GeoJSON responses cannot overwrite full-overlay
    API responses.
  - Validation passed: `PYTHONPATH=. .venv/bin/pytest -q`, 63 tests;
    `PYTHONPATH=. .venv/bin/python -m compileall -q api map_boundary_builder`;
    `node --check map_boundary_builder/web_assets/app.js`; and the full
    drift-aware benchmark stayed clean at 8/8 scored fixtures, 7
    `reference_mismatch` fixtures skipped, avg IoU 0.962, min IoU 0.931, in
    7.80s wall time.
  - Production deployment `dpl_39CKvMa58tqXtUijVtUUcr4ZxFv9` served the hidden
    `include_overlay=0` web form field and kept health on
    `pipeline-782cc2ab7d027532`. Warm cache-busted Miami production calls
    preserved bbox and confidence while cutting response size from 58.995 KB
    with overlay to 2.537 KB without overlay; server event span was 0.717s
    without overlay versus 0.846s with overlay. Exact cached repeats were equal
    wall-time noise at 1.758s, but the GeoJSON-first cached response stayed
    2.163 KB versus 59.094 KB with overlay.
- Browser pre-submit cache validation:
  - The web asset server now injects the current `pipeline_version` into
    `index.html`, so `buildRunCacheKey` does not need a blocking `/api/health`
    round trip before a first upload. The browser run-cache version moved to
    `image-to-geojson-v2`.
  - The browser cache key now hashes raw upload bytes instead of decoding the
    screenshot to a full RGBA canvas before submit. Server-side normalized-image
    result caching is unchanged, so exact-file browser cache hits stay local and
    cross-encoding cache reuse remains available on the backend.
  - Local browser proof with the Playwright CLI on `waymo phoenix.png` saw one
    mocked `/api/runs` POST, zero `/api/health` requests, and the injected
    `pipeline-782cc2ab7d027532` value. Browser microbench on the same 1.176 MB
    2400x2400 PNG: raw byte hash min 1 ms versus the previous pixel/canvas hash
    min 39 ms, about 33x faster for the pre-submit hash step.
  - Validation passed: `node --check map_boundary_builder/web_assets/app.js`;
    `PYTHONPATH=. .venv/bin/pytest -q`, 64 tests;
    `PYTHONPATH=. .venv/bin/python -m compileall -q api map_boundary_builder`;
    and the full drift-aware benchmark stayed clean at 8/8 scored fixtures, 7
    `reference_mismatch` fixtures skipped, avg IoU 0.962, min IoU 0.931.
  - Production deployment `dpl_Hqwdhxrs16ZeQRorKTzTBE8FDt39` is Ready and
    aliased to `https://mapboundary.app`. The production page injects
    `pipeline-782cc2ab7d027532`, serves the hidden `include_overlay=0` field,
    and the deployed JS uses `image-to-geojson-v2`. A production Playwright
    proof with mocked `/api/runs` saw one run request and zero `/api/health`
    requests before output activation.
  - Real production POSTs after deployment preserved bbox, confidence, source,
    GeoJSON-only response shape, and no overlay payload. First post-deploy
    Phoenix was cold at 18.731s wall / 15.192s event span, then a cache-busted
    Phoenix warmed call returned in 3.220s wall / 0.862s event span. Miami's
    first post-deploy call was cold at 11.092s wall / 9.070s event span, then a
    warmed cache-busted call returned in 2.922s wall / 0.846s event span.

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
- A conservative "try RapidOCR without the angle classifier, then fall back to
  classifier OCR when sparse" path passed the drift-aware benchmark and the
  Houston/Miami/Bay Area no-network smokes, but did not make the full pipeline
  faster: classifier-only full benchmark was 7.40s, the fallback path with a
  12-label threshold was 7.68s, and a 1-label threshold only tied at 7.40s.
  Direct OCR-only timing across 15 active/drift fixtures saved only 0.051s in
  aggregate and was slower on Orlando, Dallas Waymo, Houston Waymo, and Las
  Vegas Zoox. Rejected.
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
- Processing-resolution caps remain unsafe as a default production shortcut.
  Extraction-only probes still passed at 1800, 1600, 1400, and 1200px caps, but
  full georeference is the real gate: a 1600px cap passed the active fixtures
  with lower avg IoU (0.960 versus 0.962) and slower end-to-end local timing in
  the Vercel-like RapidOCR-only path, while a 1200px cap failed Nashville
  georeference outright. Rejected.
- Lowering RapidOCR max dimension to 1200 remained unsafe even after the catalog
  path existed: the active full benchmark still passed thresholds only because
  several hard fixtures were catalog-masked, while Los Angeles dropped to 0.846
  IoU and confidence/residuals did not catch the regression. A 1400px cap
  preserved the current catalog-heavy benchmark, but that does not generalize
  enough for arbitrary non-catalog uploads. Rejected; do not ship a lower global
  OCR cap without a stronger fallback or validator.

## Remaining Bottlenecks

- Production function size is improved after the ONNX Runtime pin, with Vercel
  reporting `api/index.py` at 92.74 MB on the current deployment. The remote
  Python build now reports a 297.34 MB pre-runtime-installation bundle, so
  cold starts and OCR model initialization remain production-only latency risks.
- OpenCV and ONNX Runtime remain the largest runtime weights. Removing either
  would require a larger architecture change and must be proven against the full
  active fixture suite before production.
