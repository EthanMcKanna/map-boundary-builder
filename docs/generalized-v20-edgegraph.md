# Generalized v20 EdgeGraph

Generalized v20 treats service-area extraction as a geometry problem, not a
full-frame mask upscaling problem. The prior BoundaryField path predicted a
320px semantic field and polygonized its resized threshold. On a 2400px upload,
one selector pixel represented 7.5 source pixels; sharp corners had already
been discarded before vectorization began.

## Architecture

v20 has two source-native lanes.

### SVG vector lane

When a supported SVG contains one visible service-area path, v20 parses the
path directly. It supports `M/L/H/V/C/S/Q/T/A/Z`, relative commands, inherited
CSS fills, affine transforms, viewBox mapping, and nonzero/evenodd fills. Line
vertices remain exact. Curves and arcs are adaptively flattened to at most
0.15 source-pixel deviation. The learned selector is never invoked for this
geometry.

### Raster EdgeGraph lane

The packaged five-channel 320px selector remains useful for semantic target
selection and topology. Its result is never exported as geometry. Before
localization, a guarded source-native proposal stage can recover a deep narrow
stem or notch that vanished during selector downsampling. Every proposal must
have persistent image-local appearance support, retain the guided component,
preserve topology, pass area and terminal-cap checks, and improve native
boundary energy.

For each selector ring, EdgeGraph:

1. resamples the coarse contour at source-pixel spacing;
2. constructs normal profiles from native RGB, luminance, Scharr fields, target
   similarity, source-validity, normalized native offset, and a repeated
   inside-direction sign inferred from the selector profile; raw selector
   probability is not a model input;
3. runs a small depthwise-separable EdgeStripNet over the contour ribbon; its
   profile dilations `1/2/4` span 14 source pixels, so even the configured 6px
   centered stroke can be observed on both sides without choosing one edge;
4. keeps vector-v3 as the single localization owner because target similarity,
   native gradients, explicit coordinates, and inside direction are already
   model inputs; the
   deterministic semantic/phase estimator remains a fail-closed legacy and
   no-refiner fallback instead of double-counting those cues;
5. decodes a globally consistent closed sequence of subpixel normal offsets;
6. first attempts a rotated-axis, topology-guarded rectilinear graph fit with
   robust line levels and sharp intersections, then uses protected learned
   corner anchors and generic straight-run fitting for non-orthogonal shapes;
7. falls back exactly to the dense subpixel ring when either vector fit would
   change topology, lose a short parallel step, or self-intersect.

The production refiner tensor has ten channels and shape
`[batch, 10, contour_length, 49]`, in this exact order: red, green, blue,
luminance, p99-normalized Scharr magnitude, p99-normalized normal-projected
Scharr, target-RGB similarity, source-valid mask, normalized native offset, and
inside-direction sign. Its heads predict an unconstrained offset
distribution, corner probability, and boundary reliability. Invalid source
samples are explicitly masked. Runtime cost scales with boundary length instead
of full-frame output resolution. Runtime dispatch remains compatible with the
diagnostic seven-channel and vector-v2 nine-channel ONNX contracts by selecting
the assembly from each session's declared input-channel count.

## Training contract

`tools/train_edgegraph_model.py` trains against exact source-pixel GeoJSON
segments after deliberately degrading the semantic proposal through the frozen
production selector. Signed targets come from continuous normal-ray/segment
intersections; corner targets come from exact vector vertices rather than a
raster approximation. It samples long closed-contour chunks rather than one
random 2D patch. During augmentation, each sampled contour chunk receives one
global normal shift drawn uniformly from `[-6px, +6px]` before both feature and
target extraction. The collapsed inside-direction sign is measured once at the
unshifted selector contour and carried with the augmented profile; it cannot
silently reverse when a large shift moves both sign probes to the same side of
the selector transition. This prevents the network from learning that the
selector center is the answer without introducing contradictory orientation
labels. The loss keeps every exported head active:

- 50% offset-distribution cross entropy;
- 25% subpixel offset Smooth L1;
- 12% corner focal loss;
- 8% reliability calibration;
- 5% supervised corner-aware smoothness.

The old hard center prior, raw coarse-probability input, and export-time zeroing
of the learned offset head are removed. Promotion rejects a legacy
seven-/nine-channel or stabilized artifact, missing shifted-center/unshifted-
direction-reference metadata,
and a profile tower whose receptive field cannot span the full centered-stroke
contract.

The v20 generator includes true angular, rectilinear, road-following, and
radial families. Angular fixtures no longer fall through to the radial sampler.

## Promotion contract

`tools/check_v20_promotion.py` fails closed when any required metric or artifact
is absent. Promotion evaluates fully automatic extraction and requires:

- no extraction failures, invalid geometry, or topology errors;
- balanced per-family synthetic coverage;
- balanced 0/1/2/3px stroke-width coverage and stroke-invariant quality;
- one-pixel boundary F1 and p95/p99/maximum symmetric boundary distance;
- one-to-one corner F1 plus angular error;
- straight-run RMS and minimum complexity at a fixed geometric error budget;
- native-pixel real/vector-derived references without fit-to-bounds scoring;
- exact SVG topology, at least 0.98 corner F1, and at most 0.25px deviation;
- warm p95 and hard maximum latency budgets;
- combined model artifacts below 15 MiB;
- the exact ten-channel vector-v3 metadata contract, selector-prior removal,
  shifted-center augmentation, and learned offset logits intact;
- a paired real-suite improvement over the promoted baseline.

Raw vertex count is intentionally not a gate. Dense staircase noise is not
more accurate geometry.

## Runtime selection

The public selector value is:

```text
generalized_v20_edgegraph
```

Optional model overrides are:

```text
MAP_BOUNDARY_EDGEGRAPH_SELECTOR_PATH
MAP_BOUNDARY_EDGEGRAPH_REFINER_PATH
MAP_BOUNDARY_EXTRACTOR_MODEL_INPUT_SIZE
MAP_BOUNDARY_EXTRACTOR_MODEL_THRESHOLD
```

The default EdgeGraph selector uses the already promoted semantic selector.
The source-native proposal, phase/refiner graph, and vector decoder own all
exported raster geometry. When catalog matching supplies georeferencing, v20
preserves the uploaded source-native shape and uses the catalog only to fit it
into geographic coordinates; it does not replace sharp uploaded edges with a
static catalog polygon.
