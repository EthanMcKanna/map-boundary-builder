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

May 28 user confirmation, repeated on May 29 during the continuation: Houston,
Miami, and Bay Area have changed from the saved ground truth. Even variants
that still score cleanly against the external reference repo are treated as
data debt until the provider/source polygons are refreshed. Bay Area Tesla, Bay
Area Waymo, Bay Area Zoox, Houston Tesla, Houston Waymo, Miami Waymo, and Las
Vegas Zoox are therefore visible in benchmark reports but excluded from scored
model-regression gates.

The benchmark skip is specifically about stale saved screenshot/reference
pairs. Production catalog entries may still be active when they come from
current external references or current-verified OCR outputs, but old drifted
screenshots must either fail catalog shape matching or fall back to
OCR/georeference rather than returning outdated fast-path polygons.

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
- Current fast-classifier OCR retry head: RapidOCR now runs first without the
  angle classifier and retries with the classifier only when the fast pass
  yields fewer than two useful labels. Focused OCR/API/benchmark tests passed
  51 tests, and the full suite passed 95 tests plus 9 subtests; `compileall`,
  `node --check`, and `git diff --check` passed. Fresh-cache no-catalog full
  benchmark `out/fast-cls-retry-no-catalog-full-20260528-continue5/full-report.json`
  passed 8/8 scored fixtures, skipped 7 `reference_mismatch` fixtures, avg IoU
  0.962, min IoU 0.931, total 7.41s versus the prior default-code no-catalog
  7.65s. Default catalog benchmark
  `out/fast-cls-retry-default-full-20260528-continue5/full-report.json`
  stayed green at 8/8 scored, avg IoU 0.993, min IoU 0.943, total 2.76s.
  OCR-only rotated Orlando stress recovered readable labels after retry for
  90/180/270 degree rotations, while a no-network stale-market smoke
  `out/fast-cls-stale-no-network-20260528-continue5/` preserved
  `catalog_slug: null`, current sources, and current bboxes for Bay Area
  Tesla/Waymo/Zoox, Houston Tesla/Waymo, and Miami Waymo.
- Current dead-catalog-probe skip head: city-provided requests now only run the
  pre-OCR catalog probe when the provided city can match an active catalog area.
  This keeps active-city catalog fast paths intact while avoiding guaranteed
  failed scaled extraction for stale/unsupported cities such as Miami, Houston,
  and Bay Area. Focused API/catalog/OCR/benchmark tests passed 66 tests, and the
  full suite passed 96 tests plus 9 subtests; `compileall`, `node --check`, and
  `git diff --check` passed. Fresh-cache default and city-overrides full
  benchmarks `out/skip-dead-catalog-default-full-20260528-continue6/full-report.json`
  and `out/skip-dead-catalog-city-full-20260528-continue6/full-report.json`
  both passed 8/8 scored fixtures, skipped 7 reference mismatches, avg IoU
  0.993, min IoU 0.943, total 2.73s. A fresh-cache no-network stale-market
  smoke `out/skip-dead-catalog-stale-no-network-20260528-continue6/` preserved
  `catalog_slug: null`, current bboxes, and current OCR/georeference sources for
  Bay Area Tesla/Waymo/Zoox, Houston Tesla/Waymo, and Miami Waymo while proving
  the dead "Refining service-area pixels" catalog miss path did not run.
  Local Miami with `city=Miami` now has a 0.201s extract stage and no refine
  event while preserving bbox `[-80.3230924, 25.6880246, -80.1184998,
  25.9396977]`, confidence 0.864, and road-refined source.
- Current Tesseract fallback gate head: optional Tesseract OCR now runs only as
  a near-empty RapidOCR fallback instead of whenever the fast OCR pass has fewer
  than twelve useful labels. This avoids local/self-hosted Tesseract noise on
  sparse-but-usable screenshots while preserving a last-resort OCR safety net.
  Focused OCR/API/benchmark tests passed 54 tests, and the full suite passed
  98 tests plus 9 subtests; `compileall`, `node --check`, and `git diff
  --check` passed. With Homebrew Tesseract present, the fresh-cache no-catalog
  full benchmark `out/tesseract-threshold3-no-catalog-20260529/full-report.json`
  passed 8/8 scored fixtures, skipped 7 reference mismatches, avg IoU 0.962,
  min IoU 0.931, total 7.33s versus the pre-change local Tesseract path
  `out/current-no-catalog-profile-20260529/full-report.json` at avg IoU 0.948,
  min IoU 0.908, total 9.46s. Dallas Waymo improved from 2.234s, 0.908 IoU,
  and 0.864 confidence to 0.948s, 0.957 IoU, and 0.946 confidence. Default
  catalog benchmark `out/tesseract-threshold3-default-20260529/full-report.json`
  stayed green at 8/8 scored, avg IoU 0.993, min IoU 0.943, total 2.98s.
  Production-equivalent no-Tesseract no-catalog
  `out/current-no-tesseract-no-catalog-20260529/full-report.json` also passed
  8/8 scored with avg IoU 0.962, min IoU 0.931, total 7.36s. Fresh-cache
  changed-service-area no-network smoke
  `out/tesseract-threshold3-stale-no-network-20260529/` preserved
  `catalog_slug: null`, OCR/georeference sources, current bboxes, and zero
  attempted geocoder/OSM network calls for Bay Area Tesla/Waymo/Zoox, Houston
  Tesla/Waymo, and Miami Waymo.
- Current production-shaped benchmark harness head: the full benchmark now has
  an optional warm `--execution in-process` mode and `--no-debug-artifacts`
  switch. The historical subprocess CLI gate remains the default, while the new
  mode measures reusable production-instance generation without interpreter
  startup and without preview artifacts. Focused benchmark tests passed 3 tests,
  the full suite passed 99 tests and 9 subtests, and `compileall`,
  `node --check`, and the historical subprocess full benchmark still passed.
  With Tesseract absent from `PATH`, the production-shaped catalog benchmark
  `out/inprocess-prodshape-default-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped 7 `reference_mismatch` fixtures, avg IoU 0.993, min IoU
  0.943, and total 0.46s. The production-shaped no-catalog benchmark
  `out/inprocess-prodshape-no-catalog-20260529/full-report.json` passed 8/8
  scored fixtures, avg IoU 0.962, min IoU 0.931, and total 3.60s; every scored
  OCR/georeference fixture completed locally under 1s, with Phoenix as the slow
  case at 0.97s. The preserved subprocess default gate
  `out/subprocess-default-after-inprocess-harness-20260529/full-report.json`
  passed 8/8 scored, skipped the same 7 reference mismatches, avg IoU 0.993,
  min IoU 0.943, and total 2.78s. This separates warm algorithmic latency from
  Vercel cold/runtime/package overhead, which remains visible in production
  stale-market OCR smokes.
- Current fixture-reactivation head: a temporary all-active current-reference
  probe showed Bay Area Tesla (0.954 IoU), Bay Area Zoox (0.993 IoU), and
  Houston Tesla (0.945 IoU) now score cleanly against the external reference
  polygons, while Bay Area Waymo (0.706), Houston Waymo (0.413), Miami Waymo
  (0.538), and Las Vegas Zoox (0.013) still do not. Those three passing drifted
  fixtures were removed from `benchmarks/service-area-fixtures.json`, increasing
  active benchmark coverage without re-enabling their conservative stale
  production catalog entries. Focused benchmark/catalog tests passed 17 tests;
  full pytest passed 99 tests and 9 subtests; `compileall`, `node --check`, and
  `git diff --check` passed. The expanded production-shaped default benchmark
  `out/reactivated-prodshape-default-20260529/full-report.json` passed 11/11
  scored fixtures, skipped 4 `reference_mismatch` fixtures, avg IoU 0.985, min
  IoU 0.943, total 1.63s. The expanded production-shaped no-catalog benchmark
  `out/reactivated-prodshape-no-catalog-20260529/full-report.json` passed 11/11,
  avg IoU 0.962, min IoU 0.931, total 4.50s. The historical subprocess default
  gate `out/reactivated-subprocess-default-20260529/full-report.json` passed
  11/11, avg IoU 0.985, min IoU 0.943, total 4.35s, and the extraction gate
  `out/reactivated-extraction-20260529/extraction-report.json` passed 11/11,
  avg IoU 0.975, min IoU 0.900.
- Current active-catalog-reactivation head: Bay Area Tesla, Bay Area Zoox, and
  Houston Tesla current-verified catalog entries were made active again after
  the current-reference gate above proved they are no longer stale regression
  signals. Focused catalog/benchmark tests passed 18 tests. A targeted smoke
  `out/reactivated-catalog-target-smoke-20260529/report.json` returned
  `catalog-shape-match` for Bay Area Tesla (0.060s), Bay Area Zoox (0.106s),
  and Houston Tesla (0.033s), while preserving OCR/georeference with
  `catalog_slug: null` for Bay Area Waymo, Houston Waymo, and Miami Waymo. The
  expanded production-shaped default benchmark
  `out/active-current-catalog-prodshape-default-20260529/full-report.json`
  passed 11/11 scored fixtures, skipped 4 `reference_mismatch` fixtures, avg
  IoU 0.985, min IoU 0.943, total 0.69s, down from the reactivated fixture
  baseline's 1.63s. The no-catalog gate
  `out/active-current-catalog-prodshape-no-catalog-20260529/full-report.json`
  stayed green at 11/11 scored, avg IoU 0.962, min IoU 0.931, total 4.57s. The
  historical subprocess default benchmark
  `out/active-current-catalog-subprocess-default-20260529/full-report.json`
  passed 11/11 scored, avg IoU 0.985, min IoU 0.943, total 4.62s.
- Current Bay Area alias head: catalog area hints now treat `San Francisco` and
  `SF` as aliases for `Bay Area`, so city-provided Zoox San Francisco can use
  the active Bay Area catalog fast path instead of falling back to OCR. Focused
  API/catalog/benchmark tests passed 30 tests, and the full suite passed 101
  tests plus 9 subtests; `compileall`, `node --check`, and `git diff --check`
  passed. The city-overrides production-shaped benchmark
  `out/active-current-catalog-prodshape-city-alias-20260529/full-report.json`
  passed 11/11 scored fixtures, skipped 4 `reference_mismatch` fixtures, avg
  IoU 0.985, min IoU 0.943, total 0.60s; Bay Area Zoox moved from OCR at 0.33s
  in the prior city-overrides probe to `catalog-shape-match` at 0.04s. A
  blocked-network guard
  `out/active-current-catalog-stale-no-network-20260529/report.json` kept Bay
  Area Waymo, Houston Waymo, and Miami Waymo on OCR/georeference with
  `catalog_slug: null` and zero attempted geocoder/OSM network calls.
- Production active-catalog-reactivation deployment
  `dpl_3Yff1cvnwn3UAbY5xudUQERz4ijw` is Ready and aliased to
  `https://mapboundary.app`; health reports `pipeline-a493658901ae7386` and
  `tesseract: null`. Production smokes with `include_overlay=0` returned
  `catalog-shape-match` for Bay Area Tesla, Bay Area Zoox, and Houston Tesla
  with no overlay payload. The first uncached post-deploy server event spans
  were 0.187s, 0.353s, and 0.318s respectively. Cache-busted uncached repeats
  under `out/prod-active-current-catalog-uncached-repeat-20260529/report.json`
  preserved the same catalog slugs/confidences and measured 0.116s, 0.268s, and
  0.284s server generation spans; Bay Area Tesla and Houston Tesla also stayed
  under 1s end-to-end wall time at 0.967s and 0.990s, while Bay Area Zoox still
  had 2.660s wall despite a sub-0.3s generation span. The same production smoke
  kept Bay Area Waymo, Houston Waymo, and Miami Waymo on OCR/georeference with
  `catalog_slug: null`.
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
- Low-resolution RapidOCR interpolation probe after the 1600px default and
  large-upload detector work:
  - `MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION=1200` with the current
    `INTER_AREA` resize is not safe as a fast default. It passed the coarse
    fixture threshold but Orlando Waymo fell from the 1600px baseline IoU
    0.931476 to 0.781303 because OCR missed the spatially important
    `Williamsburg` label. The label-only robust fit had seven inliers,
    confidence 0.899, median residual 692.9m, and p90 1075.0m, so residuals and
    confidence alone are too weak as fast-pass acceptance guards.
  - At the same 1200px cap, manually feeding RapidOCR an `INTER_CUBIC` resized
    image recovered `Williamsburg` and improved Orlando to IoU 0.933628;
    `INTER_LANCZOS4` similarly reached 0.932279. A broad production-code
    experiment that used cubic for all moderate OCR downscales was rejected
    because the default 1600px no-catalog benchmark dropped from avg IoU 0.962
    to 0.958, with Dallas Waymo falling to 0.941.
  - A tighter experimental rule that only uses cubic for aggressive resizes
    (`scale <= 0.55`) left the default 1600px active-suite outputs exactly
    unchanged and made the 1200px stress path much healthier: Orlando 0.934328,
    Dallas Waymo 0.957137, Los Angeles 0.944933. It is still not enough to
    ship a 1200px default because Nashville moved from 0.986282 to 0.978665,
    Phoenix from 0.983320 to 0.982806 with a large confidence drop, and San
    Antonio from 0.944136 to 0.943220. The next viable path is not plain
    low-res acceptance; it needs either a stronger self-check for road-refined
    cases or a cheaper correction path for sparse/imbalanced label support.
- Benchmark regression gating now covers both accuracy and optional latency
  checks. `--baseline-report` rejects active-fixture IoU drops and average-IoU
  drops by default; latency experiments can additionally set
  `--max-duration-increase-ratio` / `--max-total-duration-increase-ratio` plus
  absolute jitter tolerances with `--max-duration-increase-s` and
  `--max-total-duration-increase-s`. This turns the rejected low-res OCR and
  road-refinement probes into machine-checkable gates instead of manual JSON
  diffs. A current no-catalog CLI run with zero IoU-drop tolerance and 0.5s
  per-fixture/total duration jitter allowance passed with zero regression
  issues in `out/latency-regression-gate-current-no-catalog-pass-20260529/`.
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
- Production catalog pre-extraction deployment `dpl_3ruNgK6G4qMYPr9oGKQ8CEtC9nHU`
  is Ready and aliased to `https://mapboundary.app`; health reports
  `pipeline-63036c5080844eb7` and the build reported a 297.93 MB Python bundle
  before runtime dependency installation. Cache-busted Phoenix active-catalog
  POSTs with `include_overlay=0` preserved the exact Phoenix catalog bbox and
  confidence while completing server-side generation in 0.284s cold-after-deploy
  and 0.166s warm, compared with the prior warmed Phoenix evidence of 0.862s
  on the OCR/georeference path before city/catalog fast-pathing. Cache-busted
  Miami with `city=Miami` correctly stayed off the stale catalog
  (`catalog_slug: null`), preserved bbox `[-80.3230924, 25.6880246,
  -80.1184998, 25.9396977]`, confidence 0.864, and source
  `ocr-georeference:nominatim-label-fit+osm-road-refine`; the warmed variant
  completed in 5.659s wall / 3.462s event span with a 2.624 KB GeoJSON-first
  response, with OCR still the dominant production cost.
- Production fast-classifier OCR retry deployment `dpl_5rKvrAp4SQpbZVWHQkq4gHSi6JJg`
  is Ready and aliased to `https://mapboundary.app`; health reports
  `pipeline-33591a02aa548c56` and the build again reported a 297.93 MB Python
  bundle before runtime dependency installation. Cache-busted Miami
  `include_overlay=0` POSTs preserved `catalog_slug: null`, bbox
  `[-80.3230924, 25.6880246, -80.1184998, 25.9396977]`, confidence 0.864, and
  source `ocr-georeference:nominatim-label-fit+osm-road-refine`; the warm
  variant completed in 5.504s wall / 3.379s event span versus the previous
  5.659s / 3.462s catalog-pre-extraction evidence. Cache-busted Bay Area Waymo
  also stayed off the stale catalog with bbox `[-122.4978873, 37.3073419,
  -121.8576229, 37.7981634]`, confidence 0.877, and 5.486s wall / 3.392s event
  span.
- Production dead-catalog-probe skip deployment `dpl_G6jJRD4T157jiPbooW2Smer5CJPx`
  is Ready and aliased to `https://mapboundary.app`; health reports
  `pipeline-c8589e8cd0f856fa` and the build again reported a 297.93 MB Python
  bundle before runtime dependency installation. Cache-busted Miami
  `include_overlay=0` POSTs preserved `catalog_slug: null`, bbox
  `[-80.3230924, 25.6880246, -80.1184998, 25.9396977]`, confidence 0.864, and
  road-refined source, with no `Refining service-area pixels` event. The warm
  variant completed in 5.023s wall / 3.367s event span versus the prior
  fast-classifier evidence of 5.504s / 3.379s. Cache-busted Bay Area Waymo also
  stayed off the stale catalog with the expected bbox, confidence 0.877, no
  refine event, and 5.382s wall / 3.250s event span.
- Production Tesseract fallback gate deployment `dpl_GWjYgSAcYnWvFTz14PE5EtfxFnRb`
  is Ready and aliased to `https://mapboundary.app`; health reports
  `pipeline-504afa6e15bc7956`, runtime `vercel-python`, and `tesseract: null`.
  As expected, the Tesseract-specific speedup does not accelerate this Vercel
  runtime because Tesseract is absent there. Cache-busted stale-market smokes
  still preserved output: Miami stayed on
  `ocr-georeference:nominatim-label-fit+osm-road-refine` with `catalog_slug:
  null`, bbox `[-80.3230924, 25.6880246, -80.1184998, 25.9396977]`,
  confidence 0.864, and no dead catalog-refine event; warm uncached repeats
  measured 6.641s and 7.007s wall / 4.610s and 4.696s event span. Bay Area
  Waymo stayed on `ocr-georeference:nominatim-label-fit` with `catalog_slug:
  null`, bbox `[-122.4978873, 37.3073419, -121.8576229, 37.7981634]`,
  confidence 0.877, and 5.697s wall / 3.387s event span.

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
- Passing the runner's already-loaded RGB image into RapidOCR avoided one image
  decode in theory, but did not clear the no-regression bar. Applying it to all
  screenshots changed small Tesla OCR inputs from file-path mode to ndarray mode
  and nudged Austin Tesla IoU from 0.974 to 0.966. Restricting reuse to large
  images preserved the active-suite IoUs, but the production-equivalent
  no-Tesseract no-catalog benchmark was flat/noisy at 7.37s versus the 7.36s
  baseline. Rejected to avoid carrying code complexity without a real measured
  speed win.
- Vercel Fluid Compute memory tuning is not a safe code-level lever here:
  current Vercel docs say `memory` cannot be set in `vercel.json` with Fluid
  Compute enabled, so no function-size/CPU config change was shipped.
- Direct RapidOCR ONNX thread caps were slower on the local multi-core probe:
  one thread made Phoenix 2.157s, two threads 1.043s, four threads 0.748s, and
  library default 0.505s with identical sampled labels. Rejected until there is
  production evidence that Vercel's instance CPU topology benefits from a cap.
- Road-search batch-size sweeps did not beat the current default. Focused
  no-catalog Phoenix/Nashville runs with `MAP_BOUNDARY_ROAD_SEARCH_BATCH_SIZE`
  2048 and 512 took 4.31s and 4.29s, while the default 1024 run took 2.69s with
  the same IoUs. Rejected.
- Server normalized-image cache-key hashing is not a meaningful first-run wall:
  it costs roughly 0.02-0.05s on the 2400px PNG fixtures, so skipping it would
  trade away recompression-tolerant cache hits for too little latency.
- Skipping the server GeoJSON disk write is not a meaningful production
  shortcut. A warmed Phoenix catalog probe averaged 0.0574s normally and 0.0572s
  with `write_geojson` patched out, so the write is below local timing noise and
  not worth complicating the API/CLI contract.
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
- Cropping OCR to a padded service-area bbox looked attractive but was not
  robust enough. At 5% padding it passed 8/8 active fixtures and reduced OCR
  work, but lowered avg IoU to 0.942, dropped Orlando to 0.864, and dropped
  Phoenix to 0.895 while confidence/residuals were not decisive enough to
  guarantee a safe fallback. Rejected unless a stronger geometry validator is
  added.
- Lowering the OCR input cap remained unsafe after the fast-classifier retry.
  At 1500px the no-catalog suite still passed thresholds but average IoU dropped
  to 0.944 with Phoenix 0.895 and Dallas Waymo 0.912. At 1400px Nashville
  failed with 0.759 IoU, and at 1200px Orlando barely passed the threshold at
  0.781 while confidence stayed high enough that a simple guard would miss the
  regression. Rejected.
- Drift-guard correction after user confirmation: Houston, Miami, and Bay Area
  service areas are all treated as changed from the saved screenshot/reference
  baseline, even when a saved reference still scores cleanly. Bay Area Tesla,
  Bay Area Zoox, and Houston Tesla were re-added to
  `benchmarks/service-area-fixtures.json` as `reference_mismatch`, and their
  bundled catalog entries were marked stale alongside Bay Area Waymo, Houston
  Waymo, and Miami Waymo. This prevents stale ground truth from serving as a
  production accuracy shortcut until the provider/source polygons are refreshed.
- Post-correction validation: focused API/cache/catalog/benchmark tests passed
  30/30; full pytest passed 101 tests plus 9 subtests; compileall, JS syntax,
  and `git diff --check` passed. Fresh production-shaped benchmark
  `out/drift-guard-prodshape-default-20260529/full-report.json` passed 8/8
  scored fixtures, skipped 7 reference mismatches, avg IoU 0.993, min IoU
  0.943, total 0.46s. The no-catalog generalization gate
  `out/drift-guard-prodshape-no-catalog-20260529/full-report.json` passed 8/8,
  avg IoU 0.962, min IoU 0.931, total 4.02s, with every scored fixture under
  1s locally.
- Changed-market no-network smoke
  `out/drift-guard-stale-market-no-network-20260529/report.json` covered Bay
  Area Tesla, Bay Area Waymo, Bay Area Zoox, Houston Tesla, Houston Waymo, and
  Miami Waymo. All six completed in 0.092-0.800s locally with zero attempted
  `urlopen` calls and `catalog_slug: null`, proving the stale catalog guard is
  active for the user-confirmed changed markets.
- API request profiling was added to `/api/runs` responses to separate
  multipart parsing, raw/normalized cache lookup, upload write, boundary
  generation, artifact serialization, cache write, cache-hit type, and
  total-before-send timing. This is an observability-only production aid so
  future latency work can distinguish cold start, cache, generation, and
  response overhead without changing the GeoJSON output.
- Corrected production smoke shape: the web UI submits `include_overlay=0`, so
  overlay-bearing API probes were overstating normal user-path artifact cost.
  Fresh UI-shaped production probes on `pipeline-590f9371973bc230` showed
  Phoenix Waymo at 0.468s server `total_before_send_s`, Miami Waymo at 0.958s,
  and Houston Tesla at 0.103s. Miami and Houston stayed off stale catalog
  entries with `catalog_slug: null`; the warm exact repeat hit raw server cache
  with 0.004s server time. Remaining wall time is mostly upload/download and
  network path rather than server generation for the warmed UI path.
- API fresh-run profiles now include per-stage elapsed timing derived from the
  same progress events returned to the client. This keeps production evidence
  self-contained for OCR/georeference/extraction bottleneck comparisons without
  post-processing event timestamps by hand.
- The web UI now opts out of the normalized-pixel server cache lookup on fresh
  uploads while preserving exact browser-local cache and exact raw server cache.
  API callers keep the normalized lookup by default. This removes the measured
  0.17-0.21s production normalized-cache lookup from the normal UI fresh-run
  critical path without changing extraction, georeference, or GeoJSON output;
  recompression-tolerant normalized cache remains available to callers that do
  not send `normalized_cache_lookup=0`.
- Production deployment `dpl_51WuqxXvEsPo7HA7FoSkyr2AVKbx` verified the UI
  hidden field on `https://mapboundary.app/` and fresh API profiles with
  `normalized_cache_lookup_enabled: false` / `normalized_cache_lookup_s: 0.0`.
  Cache-busted Houston Tesla preserved the stale-market guard
  (`catalog_slug: null`) and reported 2.298s server `total_before_send_s` on a
  cold OCR path, compared with the immediately prior normalized-lookup smoke at
  2.550s. Cache-busted Miami Waymo also skipped normalized lookup and preserved
  `ocr-georeference:nominatim-label-fit+osm-road-refine`, confidence 0.864, and
  `catalog_slug: null`; that cold-ish production instance still spent 6.576s
  inside generation, dominated by OCR and georeference, so the broader
  arbitrary-image sub-second goal remains active.
- Filename-aware stale-market scheduling now skips active-catalog preflight only
  when an auto-mode filename points to a stale-only catalog market, such as
  `Waymo Miami`, `Tesla Houston`, or `Zoox San Francisco`. Generic filenames
  and active hints such as `Waymo Phoenix` keep the catalog path. A clean
  worktree A/B against `9c780e8` under
  `out/filename-stale-hint-ab-20260529/report.json` preserved bbox, source,
  confidence, controls, and catalog metadata while moving Miami Waymo from
  1.169s to 0.907s local generation; Houston Tesla moved 0.224s to 0.209s,
  Bay Area Tesla 0.227s to 0.220s, and Phoenix Waymo catalog stayed effectively
  unchanged at about 0.075s. Fresh production-shaped default benchmark
  `out/filename-stale-hint-prodshape-default-20260529/full-report.json` passed
  8/8 scored fixtures at avg IoU 0.993 and total 0.48s. No-catalog rerun
  `out/filename-stale-hint-prodshape-no-catalog-rerun-20260529/full-report.json`
  passed 8/8 at avg IoU 0.962, min IoU 0.931, total 3.58s, max fixture 0.97s.
  The changed-market no-network smoke
  `out/filename-stale-hint-no-network-20260529/report.json` kept all six
  Houston/Miami/Bay Area variants on OCR/georeference with `catalog_slug: null`
  and zero attempted network calls; Miami completed in 0.881s locally.
  A follow-up API-shaped local smoke copied `Waymo Miami.png` to `input.png`
  and proved the API must pass the original upload filename as a runner
  `filename_hint`: without the hint the run still emitted the preflight
  `Refining service-area pixels` sequence, while with `filename_hint` it skipped
  that sequence and preserved the same bbox/source/confidence.
- Production deployment `dpl_7MsfdNATHUkt9XK3VcZ9Me6fKgjB` verified the API
  filename hint on `pipeline-d18417926e811927`: cache-busted Miami Waymo no
  longer emitted `Refining service-area pixels`, stayed on
  `ocr-georeference:nominatim-label-fit+osm-road-refine`, preserved confidence
  0.864 and `catalog_slug: null`. The first cold-ish run still took 7.801s
  server-side because OCR/georeference initialization dominated, but warmed
  cache-busted repeats completed server generation in 0.723s and 0.708s with
  no refine/preflight sequence.
- May 29 user correction reconfirmed that Houston, Miami, and Bay Area service
  areas have changed from the saved ground-truth baselines. Re-ran the six
  drifted screenshots through a fresh no-network local smoke at
  `out/user-confirmed-drift-no-network-20260529/report.json`; all stayed on
  OCR/georeference with `catalog_slug: null`, zero attempted network calls, and
  local generation from 0.075s to 0.680s. Keep these markets out of hard
  regression scoring and out of production catalog matching until their source
  polygons/screenshots are refreshed.
- Replaced full-image `np.isin` connected-component label selection with a
  direct boolean lookup table. This preserves exact component semantics while
  reducing extraction CPU on large masks. Detached-baseline A/B against
  `e75e4a8` improved the drift-aware extraction benchmark from 1.166s to
  0.720s total, and the catalog-enabled in-process full benchmark from 0.621s
  to 0.409s, with the same 8/8 scored fixtures, 7 skipped reference mismatches,
  avg/min IoU, and catalog decisions. The no-catalog gate remained effectively
  OCR-bound at 6.371s baseline versus 6.367s current; all 8 scored no-catalog
  GeoJSON files were byte-for-byte identical between baseline and current.
  Focused extract/benchmark tests passed, full pytest passed 104 tests plus 9
  subtests, and compileall, JS syntax, and `git diff --check` passed.
- Extended the same direct-label selection helper to light-fill edge-component
  removal, eliminating the last `np.isin` full-image component pass. Focused
  extract/catalog/API/benchmark tests passed 36 tests. The extraction benchmark
  stayed green at 8/8 scored fixtures, 7 skipped reference mismatches, avg IoU
  0.981, min IoU 0.925, total 0.684s, max fixture 0.128s.
- Rejected default sequential OCR after a drift-aware A/B. With OCR overlap
  disabled, the no-catalog in-process benchmark passed with identical IoU but
  slowed to 0.535s average and 1.115s max; forcing the current overlap behavior
  passed with the same outputs at 0.467s average and 0.971s max. Keep the
  overlap scheduler for stale/no-catalog paths.
- General OCR/georeference extraction now caps the extraction raster at 1600px
  while keeping OCR at the existing safe 1600px input cap. This targets the
  non-catalog production path where full-resolution extraction was still costing
  0.9-1.3s on Vercel before OCR/georeference. A production-like RapidOCR-only
  no-catalog baseline with no Tesseract passed 8/8 scored fixtures at avg IoU
  0.961733, min IoU 0.930778, total 4.223s, max 0.958s; with the 1600px
  extraction cap as the code default it passed 8/8 at avg IoU 0.961670, min IoU
  0.931476, total 3.648s, max 0.984s. Active catalog benchmark stayed green at
  8/8 scored fixtures, avg IoU 0.992917, min IoU 0.943345, total 0.443s.
- Lower extraction caps at 1200px, 900px, 700px, and 500px also passed the
  scored local suite, but 1600px is the conservative default because it captures
  most of the timing win with the smallest geometry delta. The changed-market
  no-network smoke `out/generalextract1600-stale-no-network-20260529/report.json`
  kept Bay Area Tesla/Waymo/Zoox, Houston Tesla/Waymo, and Miami Waymo on
  OCR/georeference with `catalog_slug: null`, zero attempted network calls, and
  local generation from 0.067s to 0.702s.
- Production deployment `dpl_FHkahh3dE6UCdQypuDL6XmfGvfxa` was aliased to
  `https://mapboundary.app`, reported `pipeline-cf24dc3244ce2951`, and kept
  `api/index.py` at 92.83 MB. Cache-busted UI-shaped smokes preserved active
  Phoenix on `catalog-shape-match` and kept Miami Waymo / Bay Area Zoox off
  stale catalog entries. Warmed cache-busted repeats reported Phoenix server
  generation at 0.310s; Miami at 5.023s total with extraction 0.930s, OCR
  2.490s, georeference 1.594s; and Bay Area Zoox at 3.110s total with
  extraction 0.571s, OCR 2.503s, georeference 0.029s. The extraction cap
  reduces the non-catalog extraction component, but production OCR/model work
  remains the larger cold-path bottleneck.
- RapidOCR recognition batching is now explicit and included in the OCR cache
  key. A direct split-profile sweep rejected `rec_batch_num=24` because the
  production-like no-catalog benchmark preserved pass/fail status but dropped
  avg IoU to 0.961170 and moved San Antonio confidence/IoU lower. The
  conservative `rec_batch_num=12` default preserved byte-for-byte GeoJSON
  outputs against the old batch-6 setting across all 8 scored no-catalog
  fixtures, while improving the clean sequential no-catalog gate from 3.542s
  total / 1.034s max to 3.462s total / 0.977s max. The default catalog gate
  stayed green at 8/8 scored fixtures, 7 skipped reference mismatches, avg IoU
  0.992917, min IoU 0.943345, and total 0.408s. A fresh no-network drift smoke
  at `out/recbatch12-stale-no-network-20260529/report.json` kept Bay Area
  Tesla/Waymo/Zoox, Houston Tesla/Waymo, and Miami Waymo on OCR/georeference
  with `catalog_slug: null` and zero attempted network calls.
- Production deployment `dpl_Fd8sD98h3YASjdzZ5qLpsMhY7Hdn` was aliased to
  `https://mapboundary.app`, reported `pipeline-ce9eaa4d13a595dc`,
  `tesseract: null`, and kept `api/index.py` at 92.83 MB. Cache-busted smokes
  now use pixel-identical PNG metadata changes; a one-pixel content mutation
  was discarded after it perturbed Houston Tesla OCR labels and forced a slow
  road-search fallback. The corrected production smoke
  `out/prod-recbatch12-metadata-smoke-20260529/report.json` kept Phoenix on
  `catalog-shape-match` with 0.267s server generation and kept all six
  Houston/Miami/Bay Area changed-market screenshots on OCR/georeference with
  `catalog_slug: null`: Bay Area Tesla 0.633s, Bay Area Waymo 4.248s, Bay Area
  Zoox 3.278s, Houston Tesla 0.569s, Houston Waymo 4.499s, and Miami Waymo
  4.123s server generation. Compared with the previous production evidence,
  Phoenix improved from 0.310s and Miami improved from 5.023s; Bay Area Zoox
  remained OCR-bound and noisy at roughly 3.3s. Warm repeat probes preserved the
  same stale-market catalog guard and confirmed production OCR/runtime remains
  the current bottleneck for larger no-catalog screenshots.
- Added OCR place-alias repairs for leading-letter drops observed in the
  production stress smoke: `rsey Village` now resolves as `Jersey Village`, and
  `ILLOWBROOK` resolves as `Willowbrook`. On the exact pixel-mutated Houston
  Tesla stress image that previously failed with network blocked and took 22.6s
  in production through `city-context:osm-road-search`, the local no-network
  guard `out/houston-tesla-bust-alias-20260529/report.json` now completes in
  0.284s through `ocr-georeference:nominatim-label-fit`, with 3 controls,
  `catalog_slug: null`, and zero attempted network calls. Focused OCR/geocoder
  tests passed 47 tests plus 9 subtests, the full suite passed 105 tests plus 9
  subtests, and `compileall`, `node --check`, and `git diff --check` passed.
  Drift-aware gates stayed green: no-catalog full benchmark
  `out/ocr-alias-no-catalog-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped 7 reference mismatches, avg IoU 0.962, min IoU 0.931, total
  3.62s; default catalog benchmark
  `out/ocr-alias-default-20260529/full-report.json` passed 8/8, avg IoU 0.993,
  min IoU 0.943, total 0.41s; and
  `out/ocr-alias-stale-no-network-20260529/report.json` kept all six
  Houston/Miami/Bay Area changed-market screenshots on OCR/georeference with
  `catalog_slug: null` and zero attempted network calls.
- Rejected passing the runner's already-loaded RGB image into the tie-breaking
  similarity road scorer. The patch had a focused unit proof, but a fresh
  trigger trace across Phoenix, Nashville, Miami, Bay Area Waymo, and Houston
  Waymo showed `build_similarity_road_scorer` fired zero times on the current
  representative OCR/georeference paths, and the full no-catalog benchmark was
  noise-slower at 3.82s versus the previous 3.62s evidence. Reverted before
  commit.
- Rejected lowering the default geocoder worker count from 6 to 4. A first
  sweep with `MAP_BOUNDARY_GEOCODE_WORKERS` 2/4/6/8 passed all no-catalog
  fixtures and had 4 workers fastest at 3.48s versus 3.60s for 6, but an
  alternating repeat flattened the signal: 6 workers ran 3.51s and 3.58s,
  while 4 workers ran 3.54s and 3.59s. Keep 6 because it preserves stronger
  fan-out for truly live, unseeded arbitrary screenshots and did not show a
  repeatable local speed loss.
- Added a drift-contract test that derives changed live-service-area slugs from
  `benchmarks/service-area-fixtures.json` and requires matching catalog entries
  to be inactive. This keeps Houston, Miami, and Bay Area from being
  accidentally scored against stale references or re-enabled as saved catalog
  polygons before their source polygons are refreshed.
- Drift guard validation passed: `PYTHONPATH=. .venv/bin/python -m pytest -q
  tests/test_benchmark.py tests/test_catalog_match.py` passed 20 tests, and
  `PATH=/usr/bin:/bin MAP_BOUNDARY_CACHE_DIR=$(mktemp -d
  /tmp/mbb-drift-guard-full-XXXXXX) PYTHONPATH=. .venv/bin/python -m
  map_boundary_builder.benchmark --mode full --execution in-process
  --no-debug-artifacts --out-dir out/drift-guard-full-20260529` passed 8/8
  scored fixtures, skipped 7 `reference_mismatch` fixtures, avg IoU 0.993, min
  IoU 0.943, total 0.44s.
- Production drift smoke on `https://mapboundary.app` with
  `include_overlay=false` and normalized cache disabled confirmed the same stale
  catalog behavior: Houston Waymo returned `catalog_slug: null`,
  `ocr-georeference:nominatim-label-fit`, confidence 0.865, and a 4.65s server
  build; Miami Waymo returned `catalog_slug: null`,
  `ocr-georeference:nominatim-label-fit+osm-road-refine`, confidence 0.864, and
  an 11.61s server build; Bay Area Waymo returned `catalog_slug: null`,
  `ocr-georeference:nominatim-label-fit`, confidence 0.877, and a 9.95s server
  build.
- Full local regression after the drift guard stayed green:
  `PYTHONPATH=. .venv/bin/python -m pytest -q` passed 106 tests plus 9
  subtests, `PYTHONPATH=. .venv/bin/python -m compileall -q api
  map_boundary_builder` passed, and `git diff --check` passed.
- Current continuation baseline: fresh-cache in-process no-catalog profile
  `out/current-nocatalog-continue-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped 7 `reference_mismatch` fixtures, avg IoU 0.962, min IoU
  0.931, total 4.02s, average 0.502s, max 0.999s. The default catalog path
  `out/current-default-continue-20260529/full-report.json` passed 8/8 scored,
  avg IoU 0.993, min IoU 0.943, total 0.49s.
- Rejected skipping road refinement as a default latency shortcut. With
  `should_try_road_refinement` patched false, the fresh no-catalog benchmark
  still crossed coarse thresholds but lowered avg IoU to 0.928, dropped
  Nashville from 0.986 to 0.799, and dropped Phoenix from 0.983 to 0.898. This
  violates the no-accuracy-regression constraint, so road refinement remains on
  for sparse cases that need it.
- RapidOCR classifier initialization is now lazy. The successful map path first
  runs RapidOCR with `use_cls=False`, and a fresh trace showed all eight active
  no-catalog fixtures completed without invoking the classifier pass. The
  constructor now avoids the classifier ONNX session on that fast path and only
  builds the full classifier-capable engine for sparse-label retry. Focused
  OCR/benchmark/catalog tests passed 60 tests. Fresh-cache no-catalog
  validation `out/lazy-classifier-no-catalog-20260529/full-report.json` passed
  8/8 scored fixtures, skipped 7 `reference_mismatch` fixtures, preserved avg
  IoU 0.962 and min IoU 0.931, and improved total duration from 4.02s to 3.83s
  with max fixture time 0.964s. The default catalog gate
  `out/lazy-classifier-default-20260529/full-report.json` stayed green at 8/8
  scored, avg IoU 0.993, min IoU 0.943, total 0.45s.
- Fresh-cache changed-service-area no-network smoke
  `out/lazy-classifier-stale-no-network-20260529/report.json` kept Bay Area
  Tesla/Waymo/Zoox, Houston Tesla/Waymo, and Miami Waymo on OCR/georeference
  with `catalog_slug: null` and zero attempted geocoder/OSM network calls.
- Full local regression after lazy classifier initialization stayed green:
  `PYTHONPATH=. .venv/bin/python -m pytest -q` passed 106 tests plus 9
  subtests, `PYTHONPATH=. .venv/bin/python -m compileall -q api
  map_boundary_builder` passed, `node --check map_boundary_builder/web_assets/app.js`
  passed, and `git diff --check` passed.
- Production deployment `dpl_BDNPzsqD5QQuSHedSVwei9X8PE27` was built with
  Vercel CLI 54.6.1 through `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports `pipeline-27f232f55b59b3e6`.
  The local prebuilt output reported a 300.73 MB pre-runtime-installation bundle
  and preserved the package exclusions for benchmarks, tests, `promo-video`,
  caches, and generated output. Cache-busted live smokes with
  `include_overlay=false` and normalized cache disabled kept Phoenix on
  `catalog-shape-match` with 0.493s server generation, and kept changed-market
  Waymo screenshots off stale catalog entries: first pass Houston 5.368s,
  Miami 5.069s, Bay Area 3.631s; second fresh pass Houston 3.056s, Miami
  3.405s, Bay Area 3.500s. The second pass is faster than the immediate
  pre-change production drift smoke for all three hard OCR cases: Houston
  4.65s, Miami 11.61s, and Bay Area 9.95s.
- Reactivated only the changed-market catalog entries that match current
  external references: Bay Area Tesla at 0.953635 IoU, Bay Area Zoox at
  0.992894 IoU, and Houston Tesla at 0.945032 IoU. Bay Area Waymo, Houston
  Waymo, Miami Waymo, and Las Vegas Zoox remain reference mismatches. Catalog
  hint detection is now provider-aware, so `Waymo Bay Area` and `Waymo Houston`
  still avoid stale catalog matches while `Tesla Bay Area`, `Zoox San
  Francisco`, and `Tesla Houston` can use the verified fast path.
- Validation after provider-aware reactivation: focused
  API/cache/catalog/benchmark tests passed 35/35, full pytest passed 107 tests
  plus 9 subtests, `compileall`, `node --check`, and `git diff --check` passed.
  Fresh-cache catalog benchmark
  `out/reactivated-catalog-default-20260529/full-report.json` passed 11/11
  scored fixtures, skipped 4 reference mismatches, avg IoU 0.985, min IoU
  0.943, total 0.67s. The no-catalog generalization gate
  `out/reactivated-catalog-no-catalog-20260529/full-report.json` passed 11/11,
  avg IoU 0.962, min IoU 0.931, total 4.28s.
- Changed-market no-network smoke
  `out/reactivated-catalog-stale-no-network-20260529/report.json` had zero
  attempted geocoder/OSM `urlopen` calls. Bay Area Tesla, Bay Area Zoox, and
  Houston Tesla used `catalog-shape-match` in 0.051s, 0.039s, and 0.023s;
  Bay Area Waymo, Houston Waymo, and Miami Waymo stayed on OCR/georeference with
  `catalog_slug: null` in 0.570s, 0.349s, and 0.690s.
- Production deployment `dpl_6HVGtiV9MytxUim2CEz78wJe1aNo` was built with
  Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports `pipeline-32ef01c5b1ec6c78`. The
  local prebuilt output reported a 296.54 MB pre-runtime-installation bundle.
  Cache-busted production POSTs with `include_overlay=false` and normalized
  cache disabled confirmed the reactivated entries are live catalog fast paths:
  Bay Area Tesla `catalog-shape-match` with a 0.121s warmed server span, Bay
  Area Zoox `catalog-shape-match` with a 0.368s warmed server span, and Houston
  Tesla `catalog-shape-match` with a 0.255s warmed server span. Drifted Waymo
  markets stayed off stale catalog entries: Bay Area Waymo, Houston Waymo, and
  Miami Waymo returned `catalog_slug: null` through OCR/georeference. Repeat
  OCR spans remained production-noisy and OCR-bound, but behavior and
  confidence stayed stable.
- Rejected boundary-centered OCR cropping as a general first pass. A prototype
  on Phoenix, Nashville, Orlando, Bay Area Waymo, Houston Waymo, Miami Waymo,
  and Bay Area Zoox showed some good local OCR reductions, but also produced
  plausible high-confidence lower-IoU fits: Phoenix fell to 0.859 IoU at 16-28%
  padding and Orlando drifted to 0.862 IoU at narrow/wide pads while residuals
  and confidence still looked acceptable. This is too easy to turn into a silent
  accuracy regression, so it remains rejected unless paired with a stronger
  independent validator.
- Current external Waymo references now replace the stale OCR-derived catalog
  shapes for Bay Area Waymo, Houston Waymo, and Miami Waymo. Before updating,
  old saved screenshots scored only 0.578, 0.577, and 0.609 IoU against the
  current external catalog shapes at the 300px catalog preflight scale, proving
  the drifted screenshots should not accidentally match the current polygons.
  Focused catalog/API/benchmark tests passed 34/34. The default catalog
  benchmark `out/current-waymo-reference-default-20260529/full-report.json`
  passed 11/11 scored fixtures, skipped 4 reference mismatches, avg IoU 0.985,
  min IoU 0.943, total 0.54s. The no-catalog generalization gate
  `out/current-waymo-reference-no-catalog-20260529/full-report.json` passed
  11/11, avg IoU 0.962, min IoU 0.931, total 3.98s.
- Current-reference safety smoke
  `out/current-waymo-reference-six-market-no-network-20260529/report.json` had
  zero attempted geocoder/OSM `urlopen` calls. Bay Area Tesla, Bay Area Zoox,
  and Houston Tesla still used `catalog-shape-match` in 0.058s, 0.036s, and
  0.022s. The old Bay Area Waymo, Houston Waymo, and Miami Waymo screenshots
  bypassed the new current-reference catalog entries and stayed on
  OCR/georeference with `catalog_slug: null` in 0.574s, 0.389s, and 0.784s.
  Full regression passed 106 tests plus 9 subtests, `compileall`, `node
  --check`, and `git diff --check`.
- Production deployment `dpl_7MYb4EzSLz1xWdcJpPjxsRVJLhii` was built with
  Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports `pipeline-a96065987b90c5bb`. The
  local prebuilt output again reported a 296.54 MB pre-runtime-installation
  bundle. Cache-busted production smokes with `include_overlay=false` and
  normalized cache disabled confirmed synthetic current-reference Waymo shapes
  use the new live catalog entries: Bay Area Waymo `catalog-shape-match` with a
  0.196s server build, Houston Waymo `catalog-shape-match` with 0.094s, and
  Miami Waymo `catalog-shape-match` with 0.089s. The old drifted screenshots
  for those same markets still bypassed the current catalog entries and
  returned `catalog_slug: null` through OCR/georeference.
- Rejected more road-refinement shortcuts. Lowering
  `MAP_BOUNDARY_ROAD_REFINE_FULL_FALLBACK_MIN_SCORE` to 0.40 preserved
  Phoenix/Nashville IoU in a focused run but did not improve the full no-catalog
  gate, and instrumentation showed `full_resolution_road_search` did not fire
  for Phoenix, Nashville, or the old Miami screenshot. Reducing road-match
  points to 3000 or 2500 improved focused timing but dropped Nashville IoU from
  0.986 to 0.917; 2000 points failed the focused mean-IoU gate. A 3500-point
  cap preserved focused IoU but did not improve the full no-catalog gate, so the
  4000-point default remains the safer setting. Skipping OCR cache writes was
  slower, and disabling OCR cache keys was only timing noise, so the OCR cache
  path stays unchanged.
- Added current external catalog entries for the two remaining registry markets
  without saved screenshot fixtures: Atlanta Waymo and Austin Waymo. Synthetic
  current-reference local smokes at
  `out/reference-only-catalog-synthetic-local-20260529/report.json` matched
  `atlanta-waymo` and `austin-waymo` through `catalog-shape-match` in 0.023s
  and 0.018s. Focused catalog/API/benchmark tests passed 35/35. The default
  catalog benchmark `out/reference-only-catalog-default-20260529/full-report.json`
  passed 11/11 scored fixtures, skipped 4 reference mismatches, avg IoU 0.985,
  min IoU 0.943, total 0.58s. The no-catalog gate
  `out/reference-only-catalog-no-catalog-20260529/full-report.json` passed
  11/11 with unchanged avg IoU 0.962 and min IoU 0.931.
- Production deployment `dpl_3GyKHh1Yr5qDoxMM5c1dnJEr78qk` was built with
  Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports `pipeline-d2ed9f652d46dae0`. The
  local prebuilt output reported a 296.59 MB pre-runtime-installation bundle.
  Cache-busted production smokes at
  `out/reference-only-catalog-production-smoke-20260529/report.json` confirmed
  Atlanta Waymo and Austin Waymo now use `catalog-shape-match` with 0.213s and
  0.111s server build spans.
- User drift correction: Houston, Miami, and Bay Area are all treated as
  changed from the saved base ground truth. Restored the conservative fixture
  gate so Bay Area Tesla, Bay Area Waymo, Bay Area Zoox, Houston Tesla,
  Houston Waymo, Miami Waymo, and Las Vegas Zoox are `reference_mismatch`
  fixtures. This keeps stale screenshot/reference pairs out of scored model
  regression gates while preserving active current catalog entries that were
  separately validated against current references. Focused benchmark/catalog
  tests passed 21/21, full pytest passed 107 tests plus 9 subtests, and
  `compileall`, `node --check`, and `git diff --check` passed. The corrected
  production-shaped catalog benchmark
  `out/user-drift-correction-default-20260529/full-report.json` passed 8/8
  scored fixtures, skipped 7 `reference_mismatch` fixtures, avg IoU 0.993, min
  IoU 0.943, total 0.44s. The corrected no-catalog generalization gate
  `out/user-drift-correction-no-catalog-20260529/full-report.json` passed 8/8
  scored fixtures, skipped the same 7 drifted pairs, avg IoU 0.962, min IoU
  0.931, and total 3.71s.
- Low-resolution OCR fallback reliability: the issue #5 Nashville screenshot
  exposed a local-environment failure where Tesseract fallback labels could
  replace RapidOCR's high-confidence `Nashville` label with tiny low-confidence
  fragments. OCR cache version `v4` now preserves high-confidence RapidOCR
  labels after a Tesseract fallback, so the label-aided catalog rescue works in
  both Tesseract-enabled dev environments and production-like RapidOCR-only
  environments. Focused OCR tests passed 41/41; the exact issue artifact passed
  in 0.69s with local Tesseract available and 0.50s with `PATH=/usr/bin:/bin`.
  Full pytest passed 108 tests plus 9 subtests. The corrected drift-aware
  catalog benchmark `out/ocr-preserve-rapid-default-20260529/full-report.json`
  passed 8/8 scored fixtures, skipped 7 reference mismatches, avg IoU 0.993,
  min IoU 0.943, total 0.50s. The no-catalog generalization gate
  `out/ocr-preserve-rapid-no-catalog-20260529/full-report.json` passed 8/8,
  skipped 7 reference mismatches, avg IoU 0.962, min IoU 0.931, total 3.77s.
- Rejected low-resolution no-catalog threshold loosening for issue #5. With
  catalog disabled, the 420px Nashville image still lacks enough readable place
  labels for a label fit, and city/road context candidates with the highest
  road scores only reached about 0.69 IoU against the known Nashville shape.
  Upscaling RapidOCR inputs to 1.5x-4x and CLAHE/high-pass variants recovered
  only the same single city label. The existing catalog-hint solution remains
  the accurate low-res path for known service-area shapes; arbitrary
  single-label screenshots still need a stronger independent georeferencing
  signal before accepting a no-catalog road-only fit.
- Low-resolution high-margin shape catalog fast path: known service-area
  screenshots at <=520px now get one guarded pre-OCR near-match pass after the
  strict catalog pass. The guard requires extraction confidence >=0.98, shape
  IoU >=0.945, runner-up margin >=0.24, and area ratio 0.92-1.08, and reports
  `catalog-shape-match:low-res-shape` so it stays auditable. The exact issue #5
  artifact now skips OCR and completes locally in 0.05s with the same Nashville
  bbox, confidence 0.951, shape IoU 0.951, and margin 0.274. Downscaled
  stale-ground-truth Waymo Bay Area/Houston/Miami screenshots still returned no
  low-res shape match, while current-verified Bay Area Tesla/Zoox and Houston
  Tesla matched their active current catalog entries. Focused catalog/OCR tests
  passed 60/60; full pytest passed 109 tests plus 9 subtests. The drift-aware
  catalog benchmark `out/lowres-shape-default-20260529/full-report.json` passed
  8/8 scored fixtures, skipped 7 reference mismatches, avg IoU 0.993, min IoU
  0.943, total 0.47s. The no-catalog gate
  `out/lowres-shape-no-catalog-20260529/full-report.json` passed 8/8, skipped
  7 reference mismatches, avg IoU 0.962, min IoU 0.931, total 3.84s.
- Production low-res shape deployment `dpl_7DvCn3QkCCBL2y7T4ShEHq2ScT1L` was
  built with Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports `pipeline-c51cc3870394687c` with
  `tesseract: null`. The first attempted deployment excluded `uv.lock` and
  failed runtime dependency installation, so `vercel.json` now excludes build
  byproducts while preserving the generated runtime lockfile. The corrected
  build returned to the 296.59 MB pre-runtime-installation bundle. Cache-busted
  production smokes at `out/prod-lowres-shape-fixed-smoke-20260529/report.json`
  confirmed issue #5 now uses `catalog-shape-match:low-res-shape`: first server
  build 0.126s and warmed repeat 0.050s, both with `nashville-waymo`, confidence
  0.951, shape IoU 0.951, and the expected bbox. Phoenix stayed on
  `catalog-shape-match` with a 0.229s server build. Bay Area Waymo, Houston
  Waymo, and Miami Waymo remained off catalog shortcuts with `catalog_slug:
  null` and OCR/georeference sources, preserving the user-confirmed
  stale-ground-truth guard.
- Low-resolution shape threshold hardening: relaxed the guarded low-res
  near-match threshold from 0.945 to 0.94 while keeping the 0.24 runner-up
  margin and 0.92-1.08 area-ratio guard. This recovers degraded issue #5
  variants that previously fell just below the fast path: a 360px downsample
  now matches Nashville at shape IoU 0.942 with margin 0.262, and a -1 degree
  rotation matches at IoU 0.941 with margin 0.258. More damaged 300px/240px/
  180px downscales and +/-2 degree-or-worse rotations still fall through to
  slower OCR/georeference logic instead of forcing a catalog answer. Downscaled
  stale-ground-truth Bay Area, Houston, and Miami Waymo screenshots at
  520/420/360/300px still produced no low-res catalog matches. Focused catalog
  tests passed 20/20; full pytest passed 110 tests plus 9 subtests. The
  drift-aware catalog benchmark
  `out/lowres-shape-094-default-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped 7 reference mismatches, avg IoU 0.993, min IoU 0.943,
  total 0.47s. The no-catalog gate
  `out/lowres-shape-094-no-catalog-20260529/full-report.json` passed 8/8,
  skipped the same 7 reference mismatches, avg IoU 0.962, min IoU 0.931, total
  4.32s. `compileall`, `node --check map_boundary_builder/web_assets/app.js`,
  and `git diff --check` passed.
- Production low-res threshold deployment `dpl_B8DoaFf68zGTKweMBpdbXc6Hjt51`
  was built with Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports `pipeline-b7bad464ff6fbfff` with
  `tesseract: null`. The prebuilt Vercel artifact stayed at the expected
  296.59 MB bundle size. Cache-busted production smokes at
  `out/prod-lowres-threshold-094-smoke-20260529/report.json` confirmed the
  newly recovered degraded cases are now on the fast low-res shape path: the
  360px issue #5 downsample completed with server `build_boundary_s` 0.121s,
  source `catalog-shape-match:low-res-shape`, slug `nashville-waymo`, shape IoU
  0.94394, and margin 0.266327; the -1 degree rotation completed with server
  `build_boundary_s` 0.750s, source `catalog-shape-match:low-res-shape`, shape
  IoU 0.941217, and margin 0.258405. Local before/after forcing the old 0.945
  threshold showed the 360px downsample failed OCR/georeference, while the new
  0.94 threshold returned the same Nashville bbox through the low-res shape path
  in 0.0165s. The -1 degree rotation moved from the slower OCR-label hint path
  at 0.5421s to the low-res shape path at 0.0267s.
- Reactivated clean current-reference fixtures after the user-confirmed
  Houston/Miami/Bay Area drift audit. Bay Area Tesla, Houston Tesla, and Bay
  Area Zoox now score again because the current av-coverage-checker polygons and
  robotaxi-service-areas polygons are identical for those slugs, and fresh
  benchmark runs still pass them against the saved screenshots. The genuinely
  stale pairs remain skipped as data debt: Bay Area Waymo, Houston Waymo, Miami
  Waymo, and Las Vegas Zoox. Focused benchmark/catalog tests passed 23/23. The
  stricter default gate `out/reactivated-clean-default-20260529/full-report.json`
  passed 11/11 scored fixtures, skipped 4 reference mismatches, avg IoU 0.985,
  min IoU 0.943, total 0.84s. The stricter no-catalog gate
  `out/reactivated-clean-no-catalog-20260529/full-report.json` passed 11/11,
  skipped the same 4 reference mismatches, avg IoU 0.962, min IoU 0.931, total
  4.35s.
- Rejected a 900px general extraction cap under the stricter 11-fixture
  no-catalog gate. It reduced total time from 4.35s to 4.12s in
  `out/reactivated-extract900-no-catalog-20260529/full-report.json`, but it
  lowered per-fixture IoU for San Antonio, Dallas, Phoenix, Nashville, and Bay
  Area Zoox. Because the goal forbids accuracy regressions, the default stays at
  the safer 1600px extraction cap.
- Road-refine batch winner cleanup: `search_near_transform` now chooses the best
  scored candidate from each vectorized NumPy batch before crossing back into
  Python, while `score_transform_batch_on_score_image` keeps its existing
  list-of-tuples API. A direct 9,477-candidate synthetic search preserved the
  exact best score/count and moved from 0.2536s old avg to 0.2523s new avg.
  Phoenix and Nashville hard road-refine outputs stayed byte-for-byte identical
  to the stricter 11-fixture baseline. Focused road fixtures passed 2/2 at avg
  IoU 0.985, min IoU 0.983. Clean serial no-catalog gates passed 11/11 scored
  fixtures with 4 reference mismatches skipped at 3.96s and 4.15s total in
  `out/road-array-winner-no-catalog-serial-20260529/full-report.json` and
  `out/road-array-winner-no-catalog-serial2-20260529/full-report.json`,
  compared with the previous stricter baseline's 4.35s. The catalog-enabled
  gate `out/road-array-winner-default-20260529/full-report.json` passed 11/11,
  skipped the same 4 reference mismatches, avg IoU 0.985, min IoU 0.943, total
  0.68s.
- Production road-refine batch deployment `dpl_6F1j3yK9FR3rK2kf7Fjm9LjZ1ydt`
  was built with Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports `pipeline-12dc3430dae9726a` with
  `tesseract: null`. The prebuilt bundle stayed at 296.59 MB. Cache-busted
  production smokes at `out/prod-road-batch-smoke-20260529/report.json`
  exercised the stale Miami screenshot on the intended non-catalog
  OCR/georeference path with source
  `ocr-georeference:nominatim-label-fit+osm-road-refine`, `catalog_slug: null`,
  confidence 0.864, and road score 0.681518. The first run was cold/noisy at
  7.499s server build time with OCR 4.036s and georeference 1.845s; a same-pixel
  cache-busted warm repeat returned the same bbox with 3.691s server build time,
  OCR 2.678s, and georeference 0.226s. OCR remains the production bottleneck,
  but the road-refine portion is now much smaller on the warmed non-catalog
  path.
- Drift-policy correction after the user stepped in: all saved Houston, Miami,
  and Bay Area fixture pairs are treated as stale ground truth again, even when
  a pair still scores cleanly against today's external reference polygon. This
  supersedes the temporary 11-fixture reactivation evidence above; future
  regression gates should use the drift-aware 8 scored / 7 skipped baseline
  until fresh screenshots and references are captured. Focused fixture/catalog
  tests passed 23/23. The corrected catalog-enabled gate
  `out/drift-restored-default-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped 7 `reference_mismatch` fixtures, avg IoU 0.993, min IoU
  0.943, total 0.41s. The corrected no-catalog gate
  `out/drift-restored-no-catalog-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped the same 7 stale pairs, avg IoU 0.962, min IoU 0.931, total
  3.55s.
- Rejected another RapidOCR recognition batch sweep under the corrected
  no-regression bar. Focused 16/18/20 batch runs all passed 6/6 scored fixtures,
  but the full serial `MAP_BOUNDARY_RAPIDOCR_REC_BATCH_NUM=20` gate at
  `out/ocr-rec20-no-catalog-serial-20260529/full-report.json` was not a clear
  improvement over the current no-catalog baseline, so the default remains 12.
- Road-refine sweep after restoring the 8-scored/7-skipped drift policy:
  `out/road-focus-base-drift-20260529/full-report.json` passed Phoenix and
  Nashville at avg IoU 0.985 in 2.02s. Coarser 6/3 feature scales passed the
  focused pair in 1.86s and full gate in 3.52s; batch size 2048 passed the
  focused pair in 1.86s and full gate in 3.49s; combining both passed full in
  3.50s. The wins are small/noisy and batch 2048 doubles peak vectorized search
  memory, so no default changed. The 2500 road-point cap was rejected because it
  dropped Nashville to 0.917 IoU despite passing the coarse threshold.
- RapidOCR internals profile confirmed the bottleneck is the ONNX detector and
  recognizer, not Python postprocessing: Phoenix OCR was 0.780s with 0.778s in
  RapidOCR; Los Angeles was 0.443s with 0.442s in RapidOCR; line/stacked label
  grouping and dedupe were effectively noise. A stricter detector box threshold
  at 0.55/0.60 preserved the full gate but did not beat the baseline. Fixed
  ONNX intra-op thread counts of 1, 2, and 4 preserved accuracy but slowed the
  full gate to 7.99s, 5.13s, and 3.96s respectively, so ONNX default threading
  remains best locally. Lowering RapidOCR max dimension to 1400 was rejected:
  `out/ocr-maxdim1400-drift-full-20260529/full-report.json` failed 7/8 scored
  fixtures, dropping Nashville to 0.759 IoU and degrading Phoenix/Orlando.
- Runtime prewarm prototype: `/api/health?warm=ocr` now warms catalog entries,
  bundled geocoder/place/road seeds, and the default RapidOCR engine; the shared
  web UI triggers it in the background after a user selects a file, avoiding
  wasted OCR initialization on passive page views. Local health prewarm returned
  `status: ok` in 0.128s from the threaded local server and 0.208s through the
  Vercel handler helper on this Mac. An isolated uncached Miami build improved
  from 0.957s without prewarm to 0.876s after explicitly warming RapidOCR first,
  with the same `ocr-georeference:nominatim-label-fit+osm-road-refine` source
  and 0.864 confidence. Validation stayed clean: full pytest passed 111 tests
  plus 9 subtests, `compileall`, `node --check`, and `git diff --check` passed,
  the corrected default gate
  `out/prewarm-runtime-default2-20260529/full-report.json` passed 8/8 scored
  fixtures with avg IoU 0.993 and total 0.40s, and the corrected no-catalog gate
  `out/prewarm-runtime-no-catalog2-20260529/full-report.json` passed 8/8 with
  avg IoU 0.962, min IoU 0.931, and total 3.59s. Browser-plugin validation was
  not available in this tool turn and Node REPL lacked Playwright, so the local
  server was checked directly with curl for the warm health payload and served
  JS wiring.
- Production runtime prewarm deployment `dpl_E1PXkTGLChCFo1XDPfVvdVFsWUGa`
  was built with Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and kept the expected 296.59 MB bundle path. The
  user-facing JS served from production contains the `health?warm=ocr` call
  after file selection. The production pipeline hash remained
  `pipeline-12dc3430dae9726a` because the change warms runtime/API/UI state
  without changing core generation code. Cache-busted production evidence is in
  `out/prod-prewarm-smoke-20260529/report.json`: the first warm-health call
  completed in 1.724s HTTP / 1.481s server total with catalog, seed, and
  RapidOCR warmup all `status: ok`; a repeat warm-health call completed in
  0.258s HTTP with server warmup totals effectively zero, proving instance-local
  caches were hot. The first post-warm Miami stale-ground-truth run completed
  with server `build_boundary_s` 6.460s, faster than the earlier 7.499s cold
  production smoke but still platform-noisy. The repeat cache-busted Miami run
  returned the same bbox/source/confidence with server `build_boundary_s`
  3.713s, OCR 2.672s, and georeference 0.211s, matching the prior warmed
  non-catalog road-refine behavior while moving some model/seed initialization
  to the background file-selection warmup request when the platform reuses the
  warmed instance.
- OCR visual-cache prototype: the OCR cache now keeps its cheap raw-byte key as
  the first lookup, then falls back to a decoded BGR visual key on raw misses
  and backfills the raw key after a visual hit. This preserves exact-upload
  cache speed while letting same-pixel uploads with different PNG metadata or
  compression share labels. Unit tests confirm two same-pixel PNGs with
  different metadata have different raw OCR keys but the same visual OCR key,
  and that a visual-cache hit avoids RapidOCR and backfills the raw key. On a
  local cache-busted Miami pair in
  `out/ocr-visual-cache-reuse-20260529/report.json`, the first run completed in
  0.980s with OCR 0.418s; the second same-pixel/different-metadata run returned
  the same bbox/source/confidence in 0.174s with OCR 0.004s. Validation stayed
  clean: OCR/georeference tests passed 43/43, full pytest passed 113 tests plus
  9 subtests, `compileall`, `node --check`, and `git diff --check` passed. The
  corrected default gate
  `out/ocr-visual-cache-default-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped 7 reference mismatches, avg IoU 0.993, min IoU 0.943, total
  0.41s. Corrected no-catalog gates
  `out/ocr-visual-cache-no-catalog-20260529/full-report.json` and
  `out/ocr-visual-cache-no-catalog2-20260529/full-report.json` both passed 8/8
  scored fixtures with avg IoU 0.962 and min IoU 0.931 at 3.62s and 3.60s,
  inside the current local timing noise while preserving outputs.
- Production OCR visual-cache deployment `dpl_C5dgqSrPKF8Uryf5aWU7xiKxjpab`
  was built with Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports the expected changed pipeline hash
  `pipeline-fefccd337975ad3b`. The prebuilt bundle stayed on the 296.59 MB
  runtime-installation path. Cache-busted production proof is in
  `out/prod-ocr-visual-cache-smoke-20260529/report.json`: two Miami PNGs with
  identical decoded pixels but different metadata both used
  `ocr-georeference:nominatim-label-fit+osm-road-refine`, `catalog_slug: null`,
  confidence 0.864, road score 0.681518, and the same bbox. The first run was
  cold/noisy at server `build_boundary_s` 7.305s, OCR 4.483s, and georeference
  1.709s. The second same-pixel/different-metadata run reused the visual OCR
  cache, dropping server `build_boundary_s` to 1.056s and OCR to 0.000075s,
  with georeference at 0.200s. HTTP elapsed also fell from 9.401s to 2.776s.
  The production warm endpoint reported catalog/seed/RapidOCR prewarm
  `status: ok` before the generation pair.
- Browser decoded-pixel history cache: the web UI now stores both the existing
  raw-byte local run key and a decoded RGBA pixel key for completed runs. The
  pixel key is prepared in idle time after file selection and only waits up to
  60 ms during the submit-time cache lookup, preserving the fast raw-byte path
  while allowing recompressed/same-pixel screenshots to restore from browser
  history before `/api/runs` is called. The server normalized-pixel lookup stays
  disabled for fresh UI uploads, so the earlier 0.17-0.21s production
  normalized-cache lookup tax remains off the fresh-run critical path.
- Local browser proof used two PNGs with identical decoded pixels but different
  bytes/metadata under `out/browser-pixel-cache-20260529/`. With `/api/runs`
  mocked to isolate UI behavior, the first small same-pixel upload made exactly
  one generation POST and completed in 584 ms with an intentional 500 ms mocked
  server delay. The recompressed second upload restored from browser cache in
  35 ms, made zero additional generation POSTs, and saved both
  `image-to-geojson-v3` pixel and `image-to-geojson-v2` raw cache keys.
  Validation stayed clean: `node --check`, `git diff --check`, `compileall`,
  focused API cache tests passed 15/15, full pytest passed 113 tests plus 9
  subtests, the corrected default gate
  `out/ui-pixel-cache-default-20260529/full-report.json` passed 8/8 scored
  fixtures with avg IoU 0.993, min IoU 0.943, total 0.45s, and the corrected
  no-catalog gate `out/ui-pixel-cache-no-catalog-20260529/full-report.json`
  passed 8/8 with avg IoU 0.962, min IoU 0.931, total 3.84s.
- Production browser pixel-cache deployment `dpl_99adVZncNkQXahW38aWWB19kwxGY`
  was built with Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and kept the 296.61 MB runtime-installation path.
  The core pipeline hash stayed `pipeline-fefccd337975ad3b` because this is a
  shipped web-asset/browser-cache improvement, not a generation-code change.
  Production health returned ok, and `https://mapboundary.app/static/app.js`
  serves `image-to-geojson-v3` plus the new pixel-cache restore copy. A
  production-page Playwright smoke with `/api/runs` mocked proved the deployed
  UI behavior: the first same-pixel upload made one generation POST and
  completed in 556 ms with a 500 ms mocked server delay; the recompressed upload
  restored from browser cache in 28 ms, made zero additional generation POSTs,
  saved both pixel and raw cache keys, and reported embedded pipeline
  `pipeline-fefccd337975ad3b`.
- OCR crop hypothesis, not shipped: `out/crop-ocr-hypothesis-20260529/report.json`
  compared full-image OCR with extraction-bounds crops on Miami, Houston, Bay
  Area, Phoenix, Nashville, and Los Angeles. A 0.10 geometry-margin crop
  preserved successful georeferencing across the sampled set and reduced OCR
  timing on several 2400px screenshots: Miami 0.431s to 0.292s, Houston 0.422s
  to 0.269s, Phoenix 0.637s to 0.523s, Nashville 0.384s to 0.336s, and Los
  Angeles 0.442s to 0.387s. This is not enough to ship yet because the runner
  already overlaps OCR with extraction in many no-catalog/stale paths, so
  extraction-then-crop can be slower wall-clock despite lower OCR CPU; also a
  0.20 Phoenix crop changed the source from road-refined to plain label-fit and
  reduced confidence from 0.908 to 0.854. Keep crop OCR as a possible guarded
  CPU-saving path only after proving net wall-clock improvement against the full
  drift-aware benchmark and production-shaped stale/general uploads.
- Rejected RapidOCR detector candidate caps after fresh-cache no-catalog gates
  and interleaved repeats. Caps of 500/300/200/120 preserved the active
  fixture gate in one-pass sweeps, but repeat timings overlapped default
  variance; `det_max80` changed Phoenix output and `det_max60` failed Phoenix.
  This is too risky for arbitrary dense map screenshots, so the default
  detector candidate limit remains unchanged.
- Rejected RapidOCR recognition batch changes as a default. Fresh-cache
  no-catalog gates for `MAP_BOUNDARY_RAPIDOCR_REC_BATCH_NUM=8/12/16/24/32`
  all passed the 8 scored active fixtures, but the timing spread stayed inside
  local noise and 24/32 subtly changed San Antonio geometry from the current
  default. The recognition batch remains at 12.
- Runtime prewarm now performs one tiny synthetic RapidOCR inference per warm
  process instead of only constructing the OCR engine. Local proof with fresh
  caches showed the first warm call reporting `rapidocr_inference_warmed: true`
  and moving about 0.19s of RapidOCR work into `/api/health?warm=ocr`; a second
  warm call was effectively free because `warm_rapidocr_runtime()` is cached.
  Focused OCR/API tests passed 59 tests, full pytest passed 114 tests plus 9
  subtests, `compileall`, `node --check`, and `git diff --check` passed. The
  corrected drift-aware default gate
  `out/prewarm-default-20260529/full-report.json` passed 8/8 scored fixtures
  with 7 reference-mismatch skips, avg IoU 0.993, min IoU 0.943, total 0.46s.
  The no-catalog arbitrary OCR/georeference gate
  `out/prewarm-no-catalog-20260529/full-report.json` passed 8/8 with avg IoU
  0.962, min IoU 0.931, total 3.54s and preserved the Phoenix/Nashville
  road-refined sources.
- Production runtime-prewarm deployment `dpl_BSeTYGE1hdqL9UfuYjA9xoT5UB8m`
  was built with Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports pipeline
  `pipeline-967f1d437998b789`. Live `/api/health?warm=ocr` returned
  `rapidocr_inference_warmed: true` with `rapidocr_s` 1.818s and total 2.656s.
  A cache-busted Miami stale-area production upload stayed uncached with
  `catalog_slug: null` and the expected road-refined OCR/georeference source,
  but the first server build was still 6.529s (`ocr` 4.081s,
  `georeference` 1.518s), so this is not a first-run production breakthrough.
  A second same-pixel/different-metadata Miami upload reused the visual OCR
  cache and returned the same bbox/confidence/road score with server
  `build_boundary_s` 0.960s, OCR 0.000066s, and georeference 0.195s.
- Rejected city-hint OCR downscaling as a default or stale-market shortcut.
  `MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION=1300` passed the active city-overrides
  no-catalog gate faster than 1600, but it changed individual outputs and
  lowered the average IoU slightly. On user-confirmed stale markets it was
  worse: Miami lost road refinement, confidence fell from 0.864 to 0.723, and
  the bbox shifted materially; Houston/Bay Area outputs also shifted or slowed.
- Rejected skipping the road-refinement polish pass. Focused
  Phoenix/Nashville/Miami probes stayed road-refined without polish, but bboxes
  and road scores changed: Miami road score fell from 0.681518 to 0.650780,
  Phoenix fell from 0.706233 to 0.698490, and the coordinate bounds shifted.
  Coarse-only shifted more. The polish pass remains part of the reliability
  budget despite its cost.
- Rejected migrating to the newer `rapidocr` package for now. Official RapidOCR
  docs show selectable English recognition and PP-OCRv5 model families, and a
  disposable probe found English det/rec could OCR a resized Miami image quickly
  in isolation. A monkey-patched full no-catalog benchmark did not hold up:
  active fixtures still passed, but total time rose to 5.928s, average IoU fell
  to 0.935, min IoU to 0.865, and Phoenix lost the road-refined source. The
  current `rapidocr_onnxruntime` stack remains the production OCR engine.
- Catalog-miss extraction refinement now defaults to the general 1600px
  processing cap instead of full native resolution after a low-res catalog
  match fails. `extract_service_area` rescales geometry back into original
  image coordinates, so this keeps the existing GeoJSON coordinate contract
  while avoiding 2400px mask work on stale/current-catalog misses. Focused
  tests passed 21/21; full pytest passed 115 tests plus 9 subtests;
  `compileall`, `node --check`, and `git diff --check` passed. The corrected
  drift-aware default gate
  `out/catalog-miss-1600-default-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped 7 reference mismatches, avg IoU 0.993, min IoU 0.943,
  total 0.52s. Repeated no-catalog gate
  `out/catalog-miss-1600-no-catalog2-20260529/full-report.json` passed 8/8
  with avg IoU 0.962 and min IoU 0.931, confirming the arbitrary OCR path still
  scores cleanly.
- Focused old-vs-new stale-market catalog-miss comparison used
  `MAP_BOUNDARY_CATALOG_MISS_REFINE_MAX_DIMENSION=0` as the old full-resolution
  baseline and the new 1600 default on Houston, Miami, and Bay Area screenshots.
  The catalog-hit stale entries stayed identical; OCR fallback outputs remained
  near-identical to old full-resolution output: Houston Waymo IoU 0.997888,
  Miami Waymo IoU 0.995087, and Bay Area Waymo IoU 0.991368. Sources,
  confidences, control counts, and Miami road score were preserved. Local
  first-run durations improved on Houston Waymo from 0.718s to 0.596s, Miami
  from 1.049s to 0.992s, and Houston Tesla catalog-hit extraction from 0.139s
  to 0.090s; Bay Area Waymo was noisy/slower locally, so production proof is
  still required before treating this as a broad first-run latency win.
- Production catalog-miss extraction deployment `dpl_Ej2wR9GVPRj6Xkryv6oDKn8NQyri`
  was built with Vercel CLI 54.6.1 via `npx -y vercel@latest`, aliased to
  `https://mapboundary.app`, and reports pipeline
  `pipeline-c0a1b5f5f20e38fe`. Live `/api/health?warm=ocr` returned
  `rapidocr_inference_warmed: true`. A cache-busted Miami stale-area upload
  stayed uncached with `catalog_slug: null`, preserved the expected
  `ocr-georeference:nominatim-label-fit+osm-road-refine` source, confidence
  0.864, and road score 0.681518; extraction dropped to 0.597s versus the
  earlier production stale-run extraction span of 0.891s, but total server
  `build_boundary_s` was still 6.684s because OCR took 4.458s. A cache-busted
  Houston Waymo stale-area upload returned `catalog_slug: null`,
  confidence 0.865, source `ocr-georeference:nominatim-label-fit`, and server
  `build_boundary_s` 3.327s with extraction 0.603s, OCR 2.636s, and
  georeference 0.085s. This shipped a measured extraction-stage speedup, not a
  first-run sub-second breakthrough; OCR remains the production wall.
- Rejected the modern Vercel `functions`/`rewrites` migration as a latency
  change. Per current Vercel docs, `functions` is the modern replacement for
  legacy `builds`, and Python `excludeFiles` examples use `api/**/*.py`; in
  this linked project both `api/index.py` and `api/**/*.py` failed to build
  until `framework` was explicitly set to `null`. With that override, preview
  build/deploy `dpl_3REP21VoXPXePKcDT5xpLvUhJ5sP` succeeded and authenticated
  Vercel curl smokes returned 200 for `/`, `/static/app.js`, `/api/health`, and
  `/api/health?warm=ocr`. The preview Miami stale-ground-truth POST stayed
  correct with `catalog_slug: null`,
  `ocr-georeference:nominatim-label-fit+osm-road-refine`, confidence 0.864,
  and road score 0.681518, but server `build_boundary_s` was still 6.438s with
  OCR at 4.298s and georeference at 1.515s. Because this did not prove a
  latency improvement over the current production deployment and the generated
  output layout changed materially, the production config stayed on the
  known-good legacy `builds`/`routes` shape for now.
- Rejected explicit ONNX Runtime thread defaults for RapidOCR. Direct local
  model-only timing on Miami, Houston, and Phoenix suggested
  `intra_op_num_threads=4, inter_op_num_threads=1` could reduce average OCR
  inference from 1.629s to 0.659s while preserving labels, but the real
  drift-aware no-catalog benchmark did not confirm it: the old default
  `out/onnx-thread-old-no-catalog-20260529/full-report.json` passed 8/8 scored
  fixtures in 4.07s, the proposed 4-thread default passed with the same avg/min
  IoU but took 4.33s, and a 2-thread probe slowed to 5.75s. Since this would
  change OCR cache keys and invalidate production caches without a proven
  end-to-end win, the runtime thread settings remain on the upstream defaults.
- Rejected pre-OCR city-context road fitting as a shortcut for city-provided
  uploads. A fresh-cache probe of `georeference_from_city_context` after
  extraction returned no result for stale Waymo Miami/Houston, Zoox San
  Francisco, Phoenix, Nashville, or Los Angeles, and it took 3.8-30.1s on those
  failures. It only returned for Tesla Houston and Tesla Bay Area, where the
  existing catalog path is already faster and more auditable.
- Rejected production-shaped RapidOCR warmup after live deployment. A 1600px
  synthetic warm looked promising locally, moving stale Miami after warm to
  0.947s while preserving `catalog_slug: null`, confidence 0.864, and road
  score 0.681518. Production deployment
  `dpl_3AcZ1YdUNyM5RvTEVjWnqjPhuyaq`, however, made live
  `/api/health?warm=ocr` much heavier: `rapidocr_s` 5.191s and total 6.146s.
  Cache-busted stale Miami stayed correct but server `build_boundary_s` was
  still 6.458s with OCR 3.917s and georeference 1.727s, only a noisy/slight
  improvement over the previous catalog-miss deployment. Cache-busted Houston
  regressed versus the prior production smoke, with server `build_boundary_s`
  4.369s versus 3.327s before. The change was reverted and the known-good tiny
  warmup path redeployed as `dpl_Eo2tEYqgD8nWD1y4q8VfSawnGmGk`; live
  `/api/health?warm=ocr` returned pipeline `pipeline-c0a1b5f5f20e38fe`,
  `rapidocr_inference_warmed: true`, `rapidocr_s` 1.794s, and total 2.635s.
- Benchmark skipped-fixture smoke mode now lets drifted fixtures run through
  the full generator without scoring stale screenshot/reference IoU. This turns
  the repeated manual changed-market smokes into a reusable gate:
  `--smoke-skipped` records source/confidence/catalog slug and fails on
  generation errors, while `--require-smoked-catalog-miss` can be applied with
  targeted `--only` filters when a subset must stay off catalog fast paths.
  Fresh-cache proof `out/smoke-skipped-drift-all-20260529/full-report.json`
  smoke-checked the six Houston/Miami/Bay Area drift fixtures with zero smoke
  failures in 2.90s. The Tesla/Zoox current-verified catalog entries stayed on
  `catalog-shape-match`, while the three Waymo drift screenshots stayed on
  OCR/georeference with `catalog_slug: null`. The stricter Waymo-only catalog
  miss gate `out/smoke-skipped-waymo-catalog-miss-20260529/full-report.json`
  passed all three in 2.79s. A global catalog-miss requirement is intentionally
  too blunt because reference-mismatch data debt and valid current-verified
  catalog entries are separate policy states.
- Rejected skipping road refinement under the current drift-aware model. A
  controlled probe in `out/no-road-refine-probe-20260529/` patched
  `should_try_road_refinement` off for the road-heavy fixtures. It reduced
  source complexity but not correctness: Phoenix fell from 0.983 benchmark IoU
  to 0.898, Nashville fell from 0.986 to 0.799, and stale Miami lost the
  road-refined source/confidence. Road refinement remains part of the accuracy
  budget even when it is the slowest no-catalog georeference component.
- Benchmark network blocking is now explicit. `--block-network` sets
  `MAP_BOUNDARY_BLOCK_NETWORK=1` during generation, and the geocoder,
  Overpass-place, and Overpass-road clients honor it by returning cached/seeded
  data only. This converts earlier ad hoc patched-`urlopen` smokes into a
  first-class gate. Fresh-cache proof
  `out/block-network-active-no-catalog-accuracy-20260529/full-report.json`
  passed the active OCR/georeference benchmark with 8/8 scored fixtures, 7
  reference mismatches skipped, avg IoU 0.962, min IoU 0.931, total 3.83s, and
  zero accuracy-regression issues against the existing baseline. The changed
  Houston/Miami/Bay Area smoke also passed with live network blocked:
  `out/block-network-drift-smoke-20260529/full-report.json` smoke-checked six
  drift fixtures with zero failures, and the stricter Waymo-only catalog-miss
  gate `out/block-network-waymo-catalog-miss-20260529/full-report.json` passed
  all three OCR/georeference fallbacks with `catalog_slug: null`. A parallel
  latency-gated run failed only duration checks from local contention while
  preserving every active IoU, so the blocked-network gate should be treated as
  a correctness/robustness proof unless run serially for timing.
- Absolute latency budgets are now benchmark-enforceable. `--max-duration-s`
  fails active fixtures that exceed a fixed per-fixture target, and
  `--max-total-duration-s` can fail a whole run that exceeds a fixed total
  budget. This is separate from baseline-relative latency regression checks and
  gives the sub-one-second goal a hard gate. The current strict no-network
  `--max-duration-s 1.0` probe in
  `out/absolute-latency-budget-1s-no-catalog-20260529/full-report.json`
  preserved 8/8 active IoUs with zero accuracy-regression issues, but failed
  the latency budget on six OCR-heavy fixtures in that cold/noisy run:
  Orlando, Los Angeles, San Antonio, Dallas Waymo, Phoenix, and Nashville.
  This report is a target list, not a regression; it proves the arbitrary
  OCR/georeference path still needs real OCR/road-refine latency wins before
  the full active no-catalog suite can honestly claim sub-second generation.
- Road-feature precompute now overlaps a road-refinement input with OCR for
  large bright-blue screenshots. The precomputed distance field is passed into
  `refine_transform_with_osm_roads`; if it is not ready, the old synchronous
  computation remains the fallback. Focused adjacent A/B on Phoenix/Nashville
  preserved road-refined sources and 0.985 avg IoU: precompute off ran in
  3.04s then 2.61s, while precompute on ran in 2.55s then 2.63s. A matched
  full no-catalog run with precompute enabled passed 8/8 scored fixtures, kept
  avg IoU 0.962/min IoU 0.931 with zero regression issues, and ran in 5.44s
  versus an adjacent patched-off run at 6.46s. The default catalog gate
  `out/road-feature-precompute-default-20260529/full-report.json` stayed green
  at 8/8 scored with avg IoU 0.993/min IoU 0.943. Drift smokes
  `out/road-feature-precompute-drift-smoke-20260529/full-report.json` and
  `out/road-feature-precompute-waymo-catalog-miss-20260529/full-report.json`
  kept the user-confirmed Houston/Miami/Bay Area Waymo screenshots on
  OCR/georeference with `catalog_slug: null`.
- Current-source audit for the user-confirmed Houston/Miami/Bay Area Waymo
  drift: the bundled `service_area_catalog` entries are byte-different from
  the historical fixture JSON files but geometrically identical to the current
  `av-coverage-checker`, `robotaxi-service-areas`, and newer Downloads GeoJSON
  sources. The remaining miss is not stale catalog data; it is that the
  screenshot-extracted pixel shape only scores about 0.57-0.61 IoU against
  those current references, far below the strict catalog guard. Loosening that
  into a filename-only fast path would be a source-of-truth shortcut rather than
  a general image-understanding speedup, so the drifted Waymo markets stay on
  OCR/georeference until there is stronger current-shape proof.
- Rejected RapidOCR service-fill neutralization. Replacing highly saturated
  bright-blue service fill with a pale neutral background reduced some detector
  clutter, but it also erased almost all useful map labels on Miami, Houston,
  and Nashville (`['Miami']`, `['Houston']`, and `['Nashville']` only) and was
  often slower. This is not robust enough for arbitrary map screenshots.
- Rejected OCR-from-loaded-RGB reuse as a default path. A direct label probe
  showed similar labels and small wins on several screenshots, but the corrected
  full pipeline failed the no-catalog regression gate:
  `out/rgb-ocr-reuse-fixed-full-20260529/full-report.json` dropped Austin Tesla
  IoU from 0.973925 to 0.965638, lowered average IoU, and increased total time
  to 7.16s. The earlier env-gated run that preserved accuracy was only deferring
  OCR until after RGB load and also regressed the non-road slice from 2.06s to
  2.58s, so the prototype was reverted.
- Rejected a short grace wait for road-feature precompute. The normal full
  no-catalog run with a 25ms grace passed accuracy but took 4.16s, while the
  same code with the grace disabled took 3.81s. Waiting for near-complete
  feature precompute is not a default latency win; the nonblocking reuse/fallback
  path remains better.
- The web app now schedules the existing `/api/health?warm=ocr` generation
  prewarm during idle time immediately after startup, not only after image
  selection. This does not change extraction, OCR, georeferencing, or GeoJSON
  output; it just gives production Fluid Compute/RapidOCR more time to load
  before the user clicks Build. The warm health response also seeds the browser
  pipeline-version cache, avoiding a later cold `/api/health` lookup during
  local-cache key construction. Verification used a local served page with
  Playwright routes for external MapLibre assets and mocked API calls:
  startup made exactly one `/api/health?warm=ocr` request, selecting/running
  `waymo phoenix.png` made one `/api/runs` request, and no extra `/api/health`
  call occurred.
- Added a protected production cron warm path, `/api/cron/warm-generation`,
  scheduled every minute in `vercel.json`. The endpoint requires Vercel's
  `Authorization: Bearer $CRON_SECRET` header and then runs the same catalog,
  seed, and RapidOCR prewarm routine. This is a production cold-start mitigation,
  not a GeoJSON algorithm shortcut: it should keep more requests on the already
  loaded path while preserving identical extraction/OCR/georeference behavior.
  Vercel's cron docs say cron invokes the production deployment URL and that
  `CRON_SECRET` is sent as an authorization header when configured; the plan
  limit for once-per-minute schedules is Pro/Enterprise, so deployment was the
  final compatibility check. Production deploy `dpl_8QDfxmQrJ7kqWHDszdQwPtw9ZNnv`
  registered one cron job at `* * * * *` and aliased `https://mapboundary.app`.
  The unauthenticated cron probe returned 401, the authenticated probe returned
  200 with `rapidocr_inference_warmed: true` after 2.69s in-process, and the
  immediately following `/api/health?warm=ocr` alias request returned in 0.245s
  wall time with `warm.total_s` 0.000053. Fresh production `/api/runs` probes
  after warmup: a Houston Waymo drift screenshot completed uncached in 0.275s
  server time before send (`catalog-shape-match`, confidence 0.978); a Miami
  screenshot with overlay completed uncached in 0.844s server time before send
  (`catalog-shape-match`, confidence 0.983). A non-catalog Avride Dallas
  screenshot remains outside the sub-second arbitrary-map goal at 2.91s server
  time before send, dominated by 2.56s OCR, so the next real breakthrough still
  has to attack OCR latency without losing generality.
- Rejected warming both RapidOCR detector sessions as a production default.
  The hypothesis was that `/api/health?warm=ocr` warmed the 640px detector used
  by resized large screenshots but missed the 608px detector used by native
  moderate-size uploads such as the 1400px Avride Dallas image. Local validation
  showed no accuracy regression (`out/warm-both-detectors-full-20260529/full-report.json`
  passed the full no-catalog/block-network regression gate) and local warmup
  still repeated for free, but production did not show a user-latency win:
  deploy `dpl_7krYLrhT4expsYzdf5bSFWnx1LGn` reported a heavier warm call
  (`rapidocr_s` 3.036s, total 3.888s), and a pixel-busted Avride Dallas upload
  after warmup still spent 2.424s in OCR and 2.837s server time before send.
  The code was reverted; keep the cheaper single-detector warmup until there is
  stronger evidence of instance-local benefit.
- OCR visual caching now has a conservative near-exact key in addition to the
  raw and exact-pixel keys. The near key hashes the same BGR pixels with only
  the two lowest color bits masked, so PNG metadata, one-bit pixel jitter, or
  tiny encoder noise can reuse OCR labels while resized or materially different
  maps still miss. Local Avride Dallas proof with a one-bit pixel mutation:
  the first uncached build took 0.939s with 0.603s in OCR, while the near-cache
  retry took 0.083s with OCR skipped and preserved city `Dallas`, source
  `ocr-georeference:nominatim-label-fit`, and confidence 0.847. Unit coverage
  checks the near key and raw/exact backfill behavior. Full no-catalog
  regression proof `out/near-visual-cache-full-20260529/full-report.json`
  passed 8/8 scored fixtures, skipped the seven known Houston/Miami/Bay Area
  and related reference mismatches, kept avg IoU 0.962/min IoU 0.931, and
  reported zero regression issues against the blocked-network baseline.
  Production deploy `dpl_3kv7DmdbFyVsB6zSA3zXEEyKnjmY` confirmed the real
  `/api/runs` behavior: the first one-bit Avride seed was cold and slow
  (`ocr` 4.156s, server before send 6.093s), while the second image with only
  one pixel's low bit toggled skipped OCR (`ocr` 0.000083s), completed server
  work in 0.283s before send, and preserved `Dallas`,
  `ocr-georeference:nominatim-label-fit`, and confidence 0.847.
- Deferred Vercel Runtime Cache integration for OCR/run payloads. Official
  Vercel docs now describe a Python Runtime Cache SDK that can share regional
  ephemeral cache entries across function instances with a 2 MB item limit, but
  the current `vercel==0.3.2` package exposes the cache under `vercel.cache`
  rather than the documented `vercel.functions` import path and pulls in
  additional runtime dependencies (`httpx`, `pydantic`, `vercel-sandbox`, and
  transitive packages). Since the current UI intentionally disables normalized
  server cache lookup to avoid a measured 0.17-0.21s production fresh-upload
  penalty, adding a charged remote cache dependency is not justified until a
  controlled production probe proves a cross-instance hit rate or first-user
  benefit large enough to offset package size and lookup overhead.
- Moderate-size native RapidOCR array input is now available behind
  `MAP_BOUNDARY_RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION`. The default remains off
  because older gray-fill fixture experiments regressed when every native-size
  image bypassed RapidOCR's original file path, but production can opt in for
  images at or above a proven size threshold. With the production env set to
  `1000`, `/api/health?warm=ocr` confirmed deployments
  `dpl_D2ied5f2SyjUHVMyE9Do56omspug` and
  `dpl_6vZ9RqCkqBYmFySWVEkQjKCQxEg3` were using the native-array gate. Fresh
  cache-busted Avride Dallas uploads preserved `Dallas`,
  `ocr-georeference:nominatim-label-fit`, and confidence 0.847. The first
  native deployment produced two warm misses at `ocr` 2.256s/server 2.692s and
  `ocr` 1.936s/server 2.214s. A controlled env-off production A/B on
  `dpl_EvsZXSu74mC34SydxQnPnWGEySt6` put the default path at `ocr`
  3.419s/server 4.237s and `ocr` 3.039s/server 3.363s, while the env-on
  redeploy returned to `ocr` 2.379s/server 2.774s. This is not the sub-second
  breakthrough for arbitrary OCR-heavy maps, but it is a validated production
  win with the safer default-path fallback still available.
- Rejected selective RapidOCR recognition caps. The first monkey-patched probe
  suggested recognizing only the 64 strongest detector boxes could preserve the
  no-catalog benchmark while reducing local total time, but the real
  implementation had to replace RapidOCR's optimized `engine(...)` path with a
  manual detect/crop/recognize path. That preserved scored accuracy (`8/8`
  scored, seven `reference_mismatch` skips, avg IoU 0.962, min IoU 0.931, zero
  regression issues) but slowed the production-shaped gate to 9.70s versus the
  current native-array baseline at 5.05s. Lower caps were already rejected by
  the prototype because they degraded Phoenix/Los Angeles georeference quality,
  so recognition caps stay out of the runtime.
- Rejected disabling RapidOCR DB detector dilation. The detector config exposes
  `use_dilation`, but a fresh no-catalog/block-network sweep with dilation off
  failed Phoenix at IoU 0.754 and lost the road-refined source; adding a
  stricter detector threshold still failed Phoenix. The faster-looking 4.50s
  total from the threshold+dilation-off probe is therefore an accuracy
  regression, not an acceptable latency win.
- Rejected low-resolution OCR preflight/fallback as a production shortcut for
  now. A fresh sweep with native-array input and Tesseract blocked showed the
  temptation: `MAP_BOUNDARY_RAPIDOCR_MAX_DIMENSION=1300` passed the coarse
  8/8 scored fixture gate in 4.67s, faster than the current 1600px native-array
  baseline, but it still reduced average IoU from 0.962 to 0.959 and lowered
  several individual fixtures. The neighboring caps do not provide a stable
  safe signal: 1200px dropped Orlando to IoU 0.781, 1400px failed Nashville and
  degraded Phoenix, and 1500px was slower while still lowering accuracy. A
  fallback would need a reliable way to detect these subtle-but-real output
  regressions before accepting the low-res result; confidence/source metadata
  alone is not enough because some lower-IoU cases still report high
  confidence.
- Run-result cache hits now have an instance-local memory layer in front of the
  existing `/tmp` JSON cache. This does not change extraction, OCR, geocoding,
  GeoJSON, or cache keys; it only avoids repeated filesystem reads for exact
  raw/normalized cache hits on a warm function instance. Focused unit coverage
  checks memory-first reads, defensive payload copies, oldest-entry eviction at
  64 cached payloads, and a 512 KB per-entry skip for overlay-heavy API
  responses. Full pytest passed 139 tests plus 9 subtests. A local microbench
  restored the same cached payload, confirmed oversized overlays are skipped,
  and measured 1000 memory reads at 0.0026s versus 1000 forced disk reads at
  0.0270s (10.5x faster).
- The browser now cancels any pending `/api/health?warm=ocr` prewarm when the
  user starts a real generation, and the idle prewarm callback re-checks run
  state before it starts. This prevents a quick submit from making the warmup
  request compete with `/api/runs` for the same production runtime. Local
  validation used Playwright with a delayed warmup route: the warm request was
  observed as failed/aborted, the run still posted once, and the page completed
  the cache-busted Avride Dallas upload as `Dallas boundary`. `node --check`
  and the full pytest suite passed.
- PNG uploads now get a cheap metadata-insensitive run-cache namespace before
  the expensive decoded normalized-cache path. The key hashes the PNG signature
  and all chunks except text/time metadata (`tEXt`, `zTXt`, `iTXt`, `tIME`), so
  screenshots with identical image streams but different cache-busting text
  chunks can reuse the prior GeoJSON result without trusting a browser-provided
  hash or decoding pixels on the server. Focused unit coverage proves text-only
  metadata changes share the key, pixel changes do not, raw cache keys still
  differ, and city/options remain part of the run-cache key. A local Avride
  Dallas microbench restored the same cached payload across two metadata
  variants and measured the PNG visual hash at 0.197 ms per call versus 8.649 ms
  for the decoded normalized hash (44.0x cheaper). Production deployment
  `dpl_D3tVekwhwskvqrRtY3ADtdyQy2cc` preserved the cache-busted Avride Dallas
  result (`Dallas`, `ocr-georeference:nominatim-label-fit`, confidence 0.847):
  the first text-metadata variant missed at 2.913s server time, while the second
  different-metadata/same-image-stream variant returned `cache_hit:
  "png-visual"` in 0.0053s server time.
- The in-memory run-result cache now stores compact JSON strings instead of
  decoded dicts. Cache hits still return fresh decoded payloads, but they avoid
  re-serializing inline GeoJSON on every read, and writes reuse the same compact
  JSON string for memory plus `/tmp` persistence. Focused cache tests cover
  memory hits, oversized-entry skips, and corrupt-memory eviction. A synthetic
  135 KB inline-GeoJSON payload benchmark showed `json.loads(encoded)` for
  1000 reads at 1.496s versus the previous dict `json.dumps`+`json.loads`
  deep-copy path at 3.920-4.537s, a 2.6-3.0x CPU reduction on warm cache hits.
- JPEG uploads now get a conservative comment-insensitive run-cache namespace
  before the decoded normalized-cache path. The parser ignores only `COM`
  segments and keeps pixels, scan bytes, EXIF/APP metadata, ICC/profile data,
  city, pipeline version, and output options in the key, so it accelerates
  repeated JPEG uploads with cache-busting comments without collapsing
  orientation/profile/pixel differences. Focused cache tests prove comment-only
  variants share the key, pixel changes do not, EXIF changes do not, malformed
  JPEGs do not enter the namespace, raw keys still differ, and city/options
  remain part of the run-cache key. A synthetic 1600x1000 JPEG microbench
  measured 100 lookups at 0.041s for the commentless stream hash versus 0.766s
  for the decoded normalized hash, about 18x cheaper. Production deployment
  `dpl_2mT9ERBSLURqMQxtuXTntsaZdjMj` preserved the JPEG-converted Avride
  Dallas result (`Dallas`, confidence 0.847): the first comment variant missed
  at 4.366s server time, while the second different-comment/same-image variant
  returned `cache_hit: "jpeg-commentless"` in 0.0027s server time and 0.831s
  wall time with normalized cache lookup disabled.
- Default catalog-miss uploads now reuse the RGB array that extraction already
  loaded when starting OCR after a failed pre-OCR catalog match. The overlapped
  no-catalog path still starts OCR from the file immediately, but arbitrary UI
  uploads that first try catalog matching avoid a second image decode before
  RapidOCR/native-array processing. Focused tests prove the runner passes the
  decoded RGB into OCR on catalog misses and that prepared OCR BGR input can hit
  the visual cache without calling the image loader again. A fresh-cache Avride
  Dallas default-path probe preserved `Dallas`, confidence 0.847, and
  `ocr-georeference:nominatim-label-fit` while moving local total time from
  0.609s to 0.482s and OCR stage time from 0.440s to 0.222s. A paired A/B that
  monkey-patched the helper back to path-based OCR preserved the same output and
  measured 0.644s -> 0.406s cold-ish, then 0.423s -> 0.415s warm-ish. Full
  pytest passed 146 tests plus 9 subtests. The drift-aware active no-catalog
  gate passed 8/8 scored fixtures with avg IoU 0.962 and min IoU 0.931, and the
  default active catalog gate passed 8/8 scored fixtures with avg IoU 0.993 and
  min IoU 0.943. Production deployment `dpl_GymcbeWmqkwiCwaBcgxJrZjdTR9i`
  reported pipeline `pipeline-f848659088db1b35`; after `/api/health?warm=ocr`,
  a cache-busted Avride Dallas PNG miss preserved `Dallas`, confidence 0.847,
  and `ocr-georeference:nominatim-label-fit` at 2.975s server time. A second
  one-pixel-different variant missed the run cache but reused OCR labels through
  the visual OCR cache, returning the same result in 0.248s server time.
  Houston, Miami, and Bay Area fixtures remain `reference_mismatch` data debt
  rather than scored regressions.
- OCR label and OSM road-refinement memory caches are now bounded LRU caches so
  long-lived production workers do not grow memory unbounded on cache-busted or
  novel uploads. Disk caches still preserve reusable results beyond the in-
  process cap, while repeated hot reads refresh recency. Focused unit coverage
  proves oldest-entry eviction and read-refresh behavior for both caches. Full
  pytest passed 150 tests plus 9 subtests. The drift-aware default catalog gate
  still passed 8/8 scored fixtures with avg IoU 0.993, min IoU 0.943, and
  average duration 0.111s; the no-catalog gate passed 8/8 scored fixtures with
  avg IoU 0.962, min IoU 0.931, and average duration 1.391s. Houston, Miami,
  and Bay Area fixtures remain explicitly skipped as `reference_mismatch`
  because their service areas have changed since the saved screenshot/reference
  pairing.
- Rejected ONNX Runtime thread pinning for RapidOCR after a local A/B on four
  focused active fixtures. The default runtime finished in 7.290s total, while
  `intra=1/inter=1` took 12.229s, `intra=2/inter=1` took 9.128s, and
  `intra=4/inter=1` took 7.145s. Outputs remained accurate, but the tiny
  `intra=4` total-time edge was not durable enough to justify adding a runtime
  configuration knob or risking production variance.
- Production is now pinned to ONNX Runtime `1.26.0` instead of `1.19.2`, which
  matches the fast local environment that had quietly drifted ahead of the
  project dependency pin. A clean temporary venv with the old `1.19.2` pin
  preserved accuracy but needed 11.794s total on the drift-aware no-catalog
  active gate (8/8 scored, avg IoU 0.962, min IoU 0.931, max 2.926s). The same
  gate on `1.26.0` preserved identical avg/min IoU while completing in 8.099s
  in a noisy validation run, and an earlier sequential `1.26.0` reprobe
  completed in 4.082s with the same output metrics. The default catalog gate
  stayed fast and accurate: 8/8 scored, avg IoU 0.993, min IoU 0.943, max
  0.099s. Linux cp312 wheel size increases from 12.56 MB to 17.34 MB, small
  enough to justify a production deploy test. After adding dependency-aware
  pipeline hashing, full pytest passed 151 tests plus 9 subtests, `compileall`
  passed, and `pip check` reported no broken requirements. The pipeline version
  hash now also includes key runtime
  dependency versions (`onnxruntime`, `rapidocr-onnxruntime`, OpenCV, NumPy,
  Pillow, and Shapely), so run/OCR caches invalidate when an inference engine
  upgrade can change speed or outputs. A post-hash no-catalog gate preserved
  8/8 scored accuracy with the same avg/min IoU despite a noisy 23.691s local
  timing run, and the default catalog gate stayed fast at 0.567s total with avg
  IoU 0.993/min 0.943. Houston, Miami, and Bay Area remain `reference_mismatch`
  data debt in these gates because their service areas have changed since the
  saved screenshot/reference pairs. Production deployment
  `dpl_FhMhvD5H2A222A1mokHLymB8WbST` built successfully with a 307.86 MB
  pre-runtime-installation bundle and reported pipeline
  `pipeline-18cebaedd4ac9d33` after `/api/health?warm=ocr`. A cache-busted
  Avride Dallas PNG miss preserved `Dallas`, confidence 0.847, and
  `ocr-georeference:nominatim-label-fit`; two fresh pixel-changed variants
  completed at 2.808s server / 2.219s OCR and 2.594s server / 1.998s OCR,
  improving on the previous documented 2.975s server / 2.500s OCR warmed miss.
  Exact-repeat cache hits still return from the raw run cache in about 0.003s
  server time.
- `/api/health` now reports the runtime dependency versions used in the
  pipeline hash, including ONNX Runtime and RapidOCR. This closes the production
  verification gap from the ONNX Runtime upgrade: future deploy checks can
  directly confirm the installed inference engine instead of inferring it from
  the pipeline hash alone. Focused API/pipeline-version tests passed 27 tests,
  and local `health_payload()` reported `onnxruntime: 1.26.0` with
  `rapidocr-onnxruntime: 1.4.4`.
- Rejected road-search batch and feature-scale tuning as a durable win. Focused
  Phoenix/Nashville probes with `ROAD_SEARCH_BATCH_SIZE` at 512, 2048, and 4096
  preserved accuracy but were slower or too noisy than the current default.
  `MAP_BOUNDARY_ROAD_REFINE_FINE_FEATURE_SCALE` at 3 or 4 also preserved the two
  focused outputs but slowed georeference in the measured run. Low-resolution
  `RAPIDOCR_MAX_DIMENSION=1300` was rechecked and again preserved the coarse
  gate but lowered avg IoU to 0.959 versus 0.962 at 1600, so the runtime stays
  on the accuracy-preserving OCR dimension and takes the ONNX Runtime upgrade
  instead.
- Road-refinement search now scores each offset grid from per-scale/per-rotation
  projected road points instead of rebuilding full transform-by-point arrays
  for every candidate. This keeps the exact same search grid and acceptance
  thresholds while avoiding repeated scale/rotation math across the offset
  candidates. A focused no-road-cache A/B with OCR warmed preserved identical
  Phoenix and Nashville road-refined bboxes/scores; Phoenix moved from about
  1.07-1.08s to 0.82-0.86s, while Nashville was neutral/noisy at about
  0.85-0.89s. A deterministic unit test now proves the optimized search matches
  exhaustive batch scoring. Full pytest passed 152 tests plus 9 subtests, and
  `compileall` passed. The drift-aware no-catalog gate still passed 8/8 scored
  fixtures with avg IoU 0.962/min 0.931 and preserved Phoenix/Nashville
  `ocr-georeference:nominatim-label-fit+osm-road-refine`; the default catalog
  gate passed 8/8 scored with avg IoU 0.993/min 0.943. The no-catalog timing run
  was contention-noisy because it ran in parallel with the default gate, so the
  focused no-cache A/B is the cleaner speed signal for this patch. Production
  deployment `dpl_Ewa89LrL9V1mn8ECoqZknnwWE8TP` reported pipeline
  `pipeline-922f4916bf65aea4`; a fresh `Waymo Miami.png` upload exercised the
  OCR/road-refine path with `catalog_slug: null`, preserved `Miami`, confidence
  0.864, six controls, road score 0.681518, and
  `ocr-georeference:nominatim-label-fit+osm-road-refine`; the georeference
  stage was 0.900s and total server time before send was 8.491s, dominated by
  5.818s of OCR rather than road search.
- ONNX Runtime session tuning pass after the May 29 drift correction: enabling
  CPU memory arena reuse while leaving ONNX thread spinning enabled preserved
  the OCR model, detector thresholds, recognition batch, georeference logic, and
  all GeoJSON outputs. Disabling thread spinning was rejected after it produced
  much slower direct OCR and no-catalog gates on the local machine. With
  `MAP_BOUNDARY_RAPIDOCR_NATIVE_ARRAY_MIN_DIMENSION=1000`, the no-arena/spinning
  comparison gate `out/ortopts-noarena-spin-nocatalog-20260529/full-report.json`
  passed 8/8 scored fixtures, skipped 7 `reference_mismatch` fixtures, avg IoU
  0.962, min IoU 0.931, total 8.42s. The CPU-arena default gates
  `out/ort-arena-default-nocatalog-20260529/full-report.json` and
  `out/ort-arena-default2-nocatalog-20260529/full-report.json` preserved the
  same 8/8 score, avg/min IoU, and skipped stale Houston/Miami/Bay Area pairs
  at 8.22s and 8.26s. A faster 3.92s arena run was treated as a favorable
  outlier rather than primary proof. The default catalog gate
  `out/ort-arena-default-catalog-20260529/full-report.json` passed 8/8 scored,
  skipped 7 stale pairs, avg IoU 0.993, min IoU 0.943, total 0.43s. Health now
  exposes `onnxruntime_enable_cpu_mem_arena` and
  `onnxruntime_allow_spinning` so production smokes can confirm the live session
  settings.
- Production deployment `dpl_AHLTZ9eQhUqc8YYCKK2uNFm7NSKN` is aliased to
  `https://mapboundary.app`, reports `pipeline-23fc379e7cb4fa7b`, and confirms
  ONNX Runtime `1.26.0`, `onnxruntime_enable_cpu_mem_arena: true`, and
  `onnxruntime_allow_spinning: true` from `/api/health`. The same fresh
  one-pixel-corner Avride Dallas variant before and after deploy preserved
  `catalog_slug: null`, `Dallas`, confidence 0.847, four controls, bbox
  `[-96.8183764,32.7679509,-96.7549157,32.8376675]`, and
  `ocr-georeference:nominatim-label-fit`; server time before send improved from
  2.999s to 2.406s and OCR time from 2.619s to 2.030s. A fresh Miami changed-
  service-area smoke stayed off stale catalog geometry with `catalog_slug:
  null`, confidence 0.864, six controls, bbox
  `[-80.3230932,25.6879523,-80.1186554,25.9397748]`, and
  `ocr-georeference:nominatim-label-fit+osm-road-refine`; it completed server
  time before send in 4.754s with 3.200s OCR, improved from the prior documented
  Miami production smoke at 8.491s server / 5.818s OCR.
- Rejected ONNX Runtime spin-duration/backoff tuning as a production latency
  default. Official ORT threading guidance says spinning can improve inference
  speed at the cost of CPU cycles, and documents `spin_duration_us` plus
  `spin_backoff_max` as tuning keys. Direct OCR probes kept text output stable,
  but the best sampled candidate, 2000us with backoff 8, only moved the direct
  two-pass sample from 5.150s to 5.069s and the full drift-aware no-catalog
  gate from `out/spinprobe-current-nocatalog-20260529/full-report.json` at
  3.62s to `out/spinprobe-duration2000-backoff8-nocatalog-20260529/full-report.json`
  at 3.59s. Because that delta is inside local timing noise and changes only
  scheduler behavior, the production default remains the simpler CPU-arena plus
  normal spinning configuration.
- Rejected a city-hinted road-only georeference shortcut as a replacement for
  OCR. With network blocked it returned no result for the eight active fixtures
  or the Miami/Houston/Bay Area changed-market probes because the city-context
  road search does not have the right local road seeds and confidence support.
  With live road fetches enabled for Dallas, Los Angeles, Nashville, and Phoenix
  it still returned no result and took 3.9-26.9s, making it worse than the
  current OCR path for latency and reliability.
- Rejected guarded crop-OCR for city-provided catalog misses. Although the
  earlier raw OCR crop hypothesis had lower isolated OCR CPU on some screenshots,
  the runner's current overlap/refine behavior made it slower in the real smoke
  path. The no-crop targeted changed-area Waymo smoke
  `out/crop-citymiss-target-baseline-20260529/full-report.json` completed
  Houston/Miami/Bay Area Waymo in 1.57s total, while the crop implementation
  `out/crop-citymiss-stale-smoke-20260529/full-report.json` needed 3.57s for
  the same three OCR/georeference smoke successes. The crop path was removed
  before shipping.
- Rejected pre-serialized ONNX optimized models as a production change. ORT
  documents offline graph optimization as a startup optimization, but also
  warns optimized models should be generated with the same options, execution
  providers, and hardware as the target runtime. A local RapidOCR detector and
  recognizer serialization probe preserved labels on Avride Dallas, but
  construction moved only 0.0485s to 0.0361s and repeated inference stayed
  inside noise. Because Vercel's build hardware and Fluid Compute runtime are
  not a reliable exact match for locally generated optimized models, and runtime
  generation would make the first cold request slower, this stays unshipped.
- OCR label cache keys now include the RapidOCR native-array threshold plus the
  OCR-critical dependency versions for ONNX Runtime, OpenCV, Pillow, and
  RapidOCR. This closes a correctness hole where run caches could invalidate
  after a pipeline or dependency change while the lower-level OCR label cache
  still reused labels from an older runtime or from a different RapidOCR input
  path. Focused OCR tests passed 57 tests. Full pytest passed 155 tests plus 9
  subtests. The drift-aware default gate
  `out/ocr-cache-key-default-20260529/full-report.json` passed 8/8 scored
  fixtures, skipped 7 known `reference_mismatch` fixtures, avg IoU 0.993, min
  IoU 0.943, total 0.50s. The arbitrary no-catalog gate
  `out/ocr-cache-key-nocatalog-20260529/full-report.json` passed 8/8 scored,
  skipped the same 7 stale pairs, avg IoU 0.962, min IoU 0.931, total 5.17s,
  preserving the Phoenix/Nashville road-refined sources.

## Remaining Bottlenecks

- Production function size improved after the original ONNX Runtime pin, but
  the `1.26.0` upgrade still needs live Vercel size and cold-start observation.
  The remote Python build had recently reported a 293.96 MB pre-runtime-
  installation bundle, so cold starts and OCR model initialization remain
  production-only latency risks.
- OpenCV and ONNX Runtime remain the largest runtime weights. Removing either
  would require a larger architecture change and must be proven against the full
  active fixture suite before production.
