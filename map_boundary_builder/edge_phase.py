"""Source-pixel edge phase estimation for normal boundary profiles.

Semantic color answers *which* observed transition belongs to the requested
region.  It must not answer *where inside a rendered stroke* the vector path
lives.  This module keeps those two questions separate:

* transition locations come only from native RGB/luminance/gradient evidence;
* target color and inside orientation rank the observed candidates;
* two coherent transitions are interpreted as the two sides of a centered
  stroke, so their midpoint is the vector phase;
* one visible transition is used directly.

The public logits have shape ``[sample, offset]`` and can be added to a learned
strip posterior.  ``estimate_edge_phase`` also exposes the selected centers,
stroke widths, and reliability for diagnostics and gating.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EdgePhaseConfig:
    """Tuning for native transition localization.

    The defaults cover the 0--6 px centered outlines normally produced by map,
    SVG, and Canvas renderers while retaining subpixel phase on a 0.5 px strip.
    """

    maximum_stroke_width_px: float = 6.0
    # The second side of a centered stroke can be deliberately low contrast
    # when its stroke and fill colors are close.  Keep it as an observed
    # candidate, then let persistence/semantics reject unrelated weak edges.
    minimum_transition_contrast: float = 0.006
    minimum_pair_band_contrast: float = 0.010
    minimum_pair_strength_ratio: float = 0.03
    maximum_transition_candidates: int = 10
    semantic_probe_distances_px: tuple[float, ...] = (1.5, 3.0, 5.0)
    phase_sigma_px: float = 0.45
    peak_logit: float = 6.0
    pair_preference: float = 0.75
    orientation_override_margin: float = 0.03
    nonconsensus_pair_penalty: float = 4.0
    stroke_consensus_minimum_fraction: float = 0.18
    stroke_consensus_minimum_share: float = 0.50
    stroke_consensus_disagreement_penalty: float = 2.0

    def __post_init__(self) -> None:
        if self.maximum_stroke_width_px <= 0.0:
            raise ValueError("maximum stroke width must be positive")
        if self.minimum_transition_contrast < 0.0:
            raise ValueError("minimum transition contrast must be non-negative")
        if self.minimum_pair_band_contrast < 0.0:
            raise ValueError("minimum pair band contrast must be non-negative")
        if not 0.0 <= self.minimum_pair_strength_ratio <= 1.0:
            raise ValueError("minimum pair strength ratio must be in [0, 1]")
        if self.maximum_transition_candidates < 1:
            raise ValueError("maximum transition candidates must be positive")
        if not self.semantic_probe_distances_px or any(
            distance <= 0.0 for distance in self.semantic_probe_distances_px
        ):
            raise ValueError("semantic probe distances must be positive")
        if self.phase_sigma_px <= 0.0 or self.peak_logit <= 0.0:
            raise ValueError("phase sigma and peak logit must be positive")
        if self.orientation_override_margin < 0.0:
            raise ValueError("orientation override margin must be non-negative")
        if self.nonconsensus_pair_penalty < 0.0:
            raise ValueError("nonconsensus pair penalty must be non-negative")
        if not 0.0 <= self.stroke_consensus_minimum_fraction <= 1.0:
            raise ValueError("stroke consensus fraction must be in [0, 1]")
        if not 0.0 <= self.stroke_consensus_minimum_share <= 1.0:
            raise ValueError("stroke consensus share must be in [0, 1]")
        if self.stroke_consensus_disagreement_penalty < 0.0:
            raise ValueError("stroke consensus disagreement penalty must be non-negative")


@dataclass(frozen=True)
class EdgePhaseResult:
    """Phase posterior and diagnostics for a batch of normal profiles."""

    logits: np.ndarray
    centers_px: np.ndarray
    stroke_widths_px: np.ndarray
    reliability: np.ndarray
    paired: np.ndarray


@dataclass(frozen=True)
class _Transition:
    position: float
    strength: float


@dataclass(frozen=True)
class _PhaseCandidate:
    center: float
    width: float
    score: float
    semantic_score: float
    phase_strength: float
    paired: bool
    target_gain: float = 0.0
    local_target_separation: float = 0.0
    band_contrast: float = 0.0


def centered_edge_phase_logits(
    rgb_profiles: np.ndarray,
    luminance_profiles: np.ndarray,
    normal_gradient_profiles: np.ndarray,
    offsets: np.ndarray,
    *,
    target_rgb: Any,
    inside_sign: int | np.ndarray,
    config: EdgePhaseConfig | None = None,
) -> np.ndarray:
    """Return centered native-edge phase logits with shape ``[N, B]``.

    ``inside_sign`` is ``+1`` when increasing offsets point inside the target
    region and ``-1`` when decreasing offsets do.  RGB inputs may use either
    ``[0, 1]`` floats or ``[0, 255]`` values.
    """

    return estimate_edge_phase(
        rgb_profiles,
        luminance_profiles,
        normal_gradient_profiles,
        offsets,
        target_rgb=target_rgb,
        inside_sign=inside_sign,
        config=config,
    ).logits


def estimate_edge_phase(
    rgb_profiles: np.ndarray,
    luminance_profiles: np.ndarray,
    normal_gradient_profiles: np.ndarray,
    offsets: np.ndarray,
    *,
    target_rgb: Any,
    inside_sign: int | np.ndarray,
    config: EdgePhaseConfig | None = None,
) -> EdgePhaseResult:
    """Estimate the vector phase of one or more native normal profiles.

    Transition positions are computed before target color is consulted.  The
    target is used only by ``_semantic_selection_score`` to rank those fixed
    candidates, preventing a broad fill-color plateau from pulling the phase
    toward the inner side of an outline.
    """

    cfg = config or EdgePhaseConfig()
    rgb, luminance, normal_gradient, native_offsets = _validate_inputs(
        rgb_profiles,
        luminance_profiles,
        normal_gradient_profiles,
        offsets,
    )
    target = _normalize_target_rgb(target_rgb)
    signs = _normalize_inside_sign(inside_sign, len(rgb))
    signs = _target_ranked_inside_sign(
        rgb,
        native_offsets,
        signs,
        target=target,
        margin=cfg.orientation_override_margin,
    )
    negative = signs < 0
    oriented_rgb_batch = np.array(rgb, copy=True)
    oriented_luminance_batch = np.array(luminance, copy=True)
    oriented_gradient_batch = np.array(normal_gradient, copy=True)
    if np.any(negative):
        oriented_rgb_batch[negative] = rgb[negative, ::-1]
        oriented_luminance_batch[negative] = luminance[negative, ::-1]
        oriented_gradient_batch[negative] = -normal_gradient[negative, ::-1]
    transition_response = _transition_response_batch(
        oriented_rgb_batch,
        oriented_luminance_batch,
        oriented_gradient_batch,
    )

    logits = np.zeros((len(rgb), len(native_offsets)), dtype=np.float32)
    centers = np.zeros(len(rgb), dtype=np.float32)
    stroke_widths = np.zeros(len(rgb), dtype=np.float32)
    reliability = np.zeros(len(rgb), dtype=np.float32)
    paired = np.zeros(len(rgb), dtype=bool)

    candidate_rows: list[list[_PhaseCandidate]] = []
    row_signs: list[int] = []
    for row in range(len(rgb)):
        sign = int(signs[row])
        if sign > 0:
            oriented_offsets = native_offsets
        else:
            # Work in an oriented coordinate that always grows from outside to
            # inside. Convert the selected center back to native offset space.
            oriented_offsets = -native_offsets[::-1]
        oriented_rgb = oriented_rgb_batch[row]
        oriented_luminance = oriented_luminance_batch[row]
        oriented_gradient = oriented_gradient_batch[row]

        transitions = _transition_candidates(
            oriented_rgb,
            oriented_luminance,
            oriented_gradient,
            oriented_offsets,
            config=cfg,
            _response=transition_response[row],
        )
        candidates = _phase_candidates(
            oriented_rgb,
            oriented_offsets,
            transitions,
            target=target,
            config=cfg,
        )
        candidate_rows.append(candidates)
        row_signs.append(sign)

    selected_rows = _select_with_stroke_width_consensus(candidate_rows, config=cfg)
    for row, selected in enumerate(selected_rows):
        if selected is None:
            continue
        sign = row_signs[row]
        ordered = sorted(candidate_rows[row], key=lambda candidate: candidate.score, reverse=True)
        second_score = ordered[1].score if len(ordered) > 1 else selected.score - 1.0
        native_center = float(sign) * selected.center
        centers[row] = native_center
        stroke_widths[row] = selected.width
        paired[row] = selected.paired
        reliability[row] = _candidate_reliability(selected, second_score)
        logits[row] = np.asarray(
            cfg.peak_logit
            * reliability[row]
            * np.exp(-0.5 * ((native_offsets - native_center) / cfg.phase_sigma_px) ** 2),
            dtype=np.float32,
        )

    return EdgePhaseResult(
        logits=logits,
        centers_px=centers,
        stroke_widths_px=stroke_widths,
        reliability=reliability,
        paired=paired,
    )


def _select_with_stroke_width_consensus(
    candidate_rows: list[list[_PhaseCandidate]],
    *,
    config: EdgePhaseConfig,
) -> list[_PhaseCandidate | None]:
    """Use ring-wide renderer consistency to disambiguate local stroke pairs.

    A centered outline has one nearly constant width around a contour.  Road,
    label, and basemap transitions produce accidental pairs at varying widths.
    The consensus changes only which fixed candidate wins; it never averages
    or shifts transition positions.
    """

    initial = [
        max(candidates, key=lambda candidate: candidate.score) if candidates else None
        for candidates in candidate_rows
    ]
    if len(candidate_rows) < 16:
        return initial
    paired = [candidate for candidate in initial if candidate is not None and candidate.paired]
    if len(paired) < 4:
        return initial
    widths = np.asarray([candidate.width for candidate in paired], dtype=np.float32)
    grid = np.arange(0.5, config.maximum_stroke_width_px + 0.051, 0.10, dtype=np.float32)
    density = np.sum(
        np.exp(-0.5 * ((widths[:, None] - grid[None, :]) / 0.28) ** 2),
        axis=0,
    )
    consensus_width = float(grid[int(np.argmax(density))])
    coherent = np.abs(widths - consensus_width) <= 0.55
    coherent_fraction = float(np.count_nonzero(coherent)) / max(1, len(candidate_rows))
    coherent_share = float(np.mean(coherent))
    consensus_present = (
        coherent_fraction >= config.stroke_consensus_minimum_fraction
        and coherent_share >= config.stroke_consensus_minimum_share
    )

    selected_rows: list[_PhaseCandidate | None] = []
    for candidates in candidate_rows:
        if not candidates:
            selected_rows.append(None)
            continue

        def adjusted_score(candidate: _PhaseCandidate) -> float:
            if not candidate.paired:
                return candidate.score
            if not consensus_present:
                return candidate.score - config.nonconsensus_pair_penalty
            distance = abs(candidate.width - consensus_width)
            agreement = math.exp(-0.5 * (distance / 0.42) ** 2)
            disagreement = float(np.clip((distance - 0.55) / 0.90, 0.0, 1.0))
            return (
                candidate.score
                + 1.00 * agreement
                - config.stroke_consensus_disagreement_penalty * disagreement
            )

        selected_rows.append(max(candidates, key=adjusted_score))
    return selected_rows


def _validate_inputs(
    rgb_profiles: np.ndarray,
    luminance_profiles: np.ndarray,
    normal_gradient_profiles: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb_profiles, dtype=np.float32)
    if rgb.ndim == 2 and rgb.shape[-1] == 3:
        rgb = rgb[np.newaxis, ...]
    luminance = np.asarray(luminance_profiles, dtype=np.float32)
    if luminance.ndim == 1:
        luminance = luminance[np.newaxis, ...]
    gradient = np.asarray(normal_gradient_profiles, dtype=np.float32)
    if gradient.ndim == 1:
        gradient = gradient[np.newaxis, ...]
    native_offsets = np.asarray(offsets, dtype=np.float32).reshape(-1)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb profiles must have shape [N, B, 3], got {rgb.shape}")
    if luminance.shape != rgb.shape[:2] or gradient.shape != rgb.shape[:2]:
        raise ValueError("luminance and normal-gradient profiles must match RGB [N, B]")
    if len(native_offsets) != rgb.shape[1] or len(native_offsets) < 5:
        raise ValueError("offsets must match the profile width and contain at least five bins")
    if not np.all(np.isfinite(rgb)) or not np.all(np.isfinite(luminance)):
        raise ValueError("RGB and luminance profiles must be finite")
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(native_offsets)):
        raise ValueError("gradient profiles and offsets must be finite")
    if np.any(np.diff(native_offsets) <= 0.0):
        raise ValueError("offsets must be strictly increasing")

    if float(np.max(rgb, initial=0.0)) > 1.5:
        rgb = rgb / 255.0
    return (
        np.clip(rgb, 0.0, 1.0),
        luminance,
        gradient,
        native_offsets,
    )


def _normalize_target_rgb(target_rgb: Any) -> np.ndarray:
    target = np.asarray(target_rgb, dtype=np.float32).reshape(-1)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_rgb must contain three finite channels")
    if float(np.max(target, initial=0.0)) > 1.5:
        target = target / 255.0
    return np.clip(target, 0.0, 1.0)


def _normalize_inside_sign(inside_sign: int | np.ndarray, count: int) -> np.ndarray:
    signs = np.asarray(inside_sign, dtype=np.int8)
    if signs.ndim == 0:
        signs = np.full(count, int(signs), dtype=np.int8)
    else:
        signs = signs.reshape(-1)
    if len(signs) != count or np.any((signs != -1) & (signs != 1)):
        raise ValueError("inside_sign must be +1/-1 per profile")
    return signs


def _target_ranked_inside_sign(
    rgb: np.ndarray,
    offsets: np.ndarray,
    signs: np.ndarray,
    *,
    target: np.ndarray,
    margin: float,
) -> np.ndarray:
    """Correct locally reversed coarse normals without changing edge phase.

    Coarse selectors can put an individual normal on the wrong side near a
    corner, omission, or hole.  Three distant native probes determine which
    half-profile persistently resembles the requested region.  The supplied
    topology orientation remains authoritative unless the target evidence is
    decisively better in the opposite direction.
    """

    radius = min(abs(float(offsets[0])), abs(float(offsets[-1])))
    if radius <= 1e-6 or margin == float("inf"):
        return np.ascontiguousarray(signs, dtype=np.int8)
    distances = radius * np.asarray([0.25, 0.50, 0.75], dtype=np.float32)
    positive_indices = np.asarray(
        [int(np.argmin(np.abs(offsets - distance))) for distance in distances],
        dtype=np.int32,
    )
    negative_indices = np.asarray(
        [int(np.argmin(np.abs(offsets + distance))) for distance in distances],
        dtype=np.int32,
    )
    positive_distance = np.median(
        np.linalg.norm(rgb[:, positive_indices, :] - target[None, None, :], axis=2)
        / math.sqrt(3.0),
        axis=1,
    )
    negative_distance = np.median(
        np.linalg.norm(rgb[:, negative_indices, :] - target[None, None, :], axis=2)
        / math.sqrt(3.0),
        axis=1,
    )
    target_sign = np.where(positive_distance <= negative_distance, 1, -1).astype(np.int8)
    current_distance = np.where(signs > 0, positive_distance, negative_distance)
    opposite_distance = np.where(signs > 0, negative_distance, positive_distance)
    decisive = current_distance - opposite_distance >= float(margin)
    return np.ascontiguousarray(np.where(decisive, target_sign, signs), dtype=np.int8)


def _transition_candidates(
    rgb: np.ndarray,
    luminance: np.ndarray,
    normal_gradient: np.ndarray,
    offsets: np.ndarray,
    *,
    config: EdgePhaseConfig,
    _response: np.ndarray | None = None,
) -> list[_Transition]:
    positions = 0.5 * (offsets[:-1] + offsets[1:])
    response = (
        np.asarray(_response, dtype=np.float32)
        if _response is not None
        else _transition_response_batch(
            rgb[np.newaxis, ...],
            luminance[np.newaxis, ...],
            normal_gradient[np.newaxis, ...],
        )[0]
    )
    maximum = float(np.max(response, initial=0.0))
    if maximum < config.minimum_transition_contrast:
        return []
    threshold = max(config.minimum_transition_contrast, maximum * 0.025)

    peaks: list[int] = []
    index = 0
    while index < len(response):
        if response[index] < threshold:
            index += 1
            continue
        end = index
        while end + 1 < len(response) and abs(float(response[end + 1] - response[index])) <= 1e-7:
            end += 1
        center = (index + end) // 2
        left = float(response[index - 1]) if index > 0 else -float("inf")
        right = float(response[end + 1]) if end + 1 < len(response) else -float("inf")
        if float(response[center]) >= left and float(response[center]) >= right:
            peaks.append(center)
        index = end + 1
    if not peaks:
        peaks = [int(np.argmax(response))]

    transitions: list[_Transition] = []
    median_step = float(np.median(np.diff(offsets)))
    for peak in peaks:
        position = float(positions[peak])
        if 0 < peak < len(response) - 1:
            before = float(response[peak - 1])
            center = float(response[peak])
            after = float(response[peak + 1])
            denominator = before - 2.0 * center + after
            if denominator < -1e-9:
                displacement = 0.5 * (before - after) / denominator
                position += float(np.clip(displacement, -0.5, 0.5)) * median_step
        transitions.append(_Transition(position=position, strength=float(response[peak])))

    # Non-maximum suppression retains both sides of a one-pixel stroke while
    # removing duplicate peaks created by antialiasing shoulders.
    minimum_separation = max(0.30, median_step * 0.70)
    selected: list[_Transition] = []
    for transition in sorted(transitions, key=lambda item: item.strength, reverse=True):
        if all(abs(transition.position - other.position) >= minimum_separation for other in selected):
            selected.append(transition)
        if len(selected) >= config.maximum_transition_candidates:
            break
    return sorted(selected, key=lambda item: item.position)


def _transition_response_batch(
    rgb: np.ndarray,
    luminance: np.ndarray,
    normal_gradient: np.ndarray,
) -> np.ndarray:
    """Compute native transition evidence for every profile in one pass."""

    rgb_jump = np.linalg.norm(np.diff(rgb, axis=1), axis=2) / math.sqrt(3.0)
    luminance_jump = np.abs(np.diff(luminance, axis=1))
    gradient_support = 0.5 * (
        np.abs(normal_gradient[:, :-1]) + np.abs(normal_gradient[:, 1:])
    )
    gradient_scale = np.maximum(
        np.quantile(gradient_support, 0.90, axis=1, keepdims=True),
        1e-6,
    )
    normalized_gradient = np.clip(gradient_support / gradient_scale, 0.0, 2.0)
    # Native color/luminance changes establish the candidate phase. Gradient is
    # only a bounded multiplicative support term, so a high Scharr response
    # without an observed color step cannot invent a candidate.
    response = np.hypot(rgb_jump, 0.35 * luminance_jump)
    response *= 1.0 + 0.12 * normalized_gradient
    return np.ascontiguousarray(response, dtype=np.float32)


def _phase_candidates(
    rgb: np.ndarray,
    offsets: np.ndarray,
    transitions: list[_Transition],
    *,
    target: np.ndarray,
    config: EdgePhaseConfig,
) -> list[_PhaseCandidate]:
    if not transitions:
        return []
    positions = np.asarray([item.position for item in transitions], dtype=np.float32)
    strengths = np.asarray([item.strength for item in transitions], dtype=np.float32)
    maximum_strength = max(1e-9, float(np.max(strengths)))
    normalized_strength = strengths / maximum_strength
    semantics = _semantic_selection_scores(
        rgb,
        offsets,
        centers=positions,
        half_widths=np.zeros_like(positions),
        target=target,
        config=config,
    )
    candidates: list[_PhaseCandidate] = [
        _PhaseCandidate(
            center=float(position),
            width=0.0,
            score=float(semantic + 0.45 * strength),
            semantic_score=float(semantic),
            phase_strength=float(strength),
            paired=False,
        )
        for position, semantic, strength in zip(positions, semantics, normalized_strength)
    ]

    outer_indices, inner_indices = np.triu_indices(len(transitions), k=1)
    widths = positions[inner_indices] - positions[outer_indices]
    keep = widths <= config.maximum_stroke_width_px
    outer_indices = outer_indices[keep]
    inner_indices = inner_indices[keep]
    widths = widths[keep]
    if not len(widths):
        return candidates
    outer_strength = strengths[outer_indices]
    inner_strength = strengths[inner_indices]
    strength_ratio = np.minimum(outer_strength, inner_strength) / np.maximum(
        np.maximum(outer_strength, inner_strength),
        1e-9,
    )
    keep = strength_ratio >= config.minimum_pair_strength_ratio
    outer_indices = outer_indices[keep]
    inner_indices = inner_indices[keep]
    widths = widths[keep]
    strength_ratio = strength_ratio[keep]
    outer_positions = positions[outer_indices]
    inner_positions = positions[inner_indices]
    if not len(widths):
        return candidates
    band_contrast = _pair_band_contrasts(
        rgb,
        offsets,
        outer=outer_positions,
        inner=inner_positions,
    )
    keep = band_contrast >= config.minimum_pair_band_contrast
    outer_indices = outer_indices[keep]
    inner_indices = inner_indices[keep]
    widths = widths[keep]
    strength_ratio = strength_ratio[keep]
    outer_positions = outer_positions[keep]
    inner_positions = inner_positions[keep]
    band_contrast = band_contrast[keep]
    if not len(widths):
        return candidates

    centers = 0.5 * (outer_positions + inner_positions)
    semantics = _semantic_selection_scores(
        rgb,
        offsets,
        centers=centers,
        half_widths=0.5 * widths,
        target=target,
        config=config,
    )
    target_gain, local_target_separation = _pair_target_evidence_batch(
        rgb,
        offsets,
        outer=outer_positions,
        inner=inner_positions,
        target=target,
        config=config,
    )
    phase_strength = np.sqrt(strengths[outer_indices] * strengths[inner_indices]) / maximum_strength
    # Pair preference expresses the renderer contract, not target color: when
    # both sides of a coherent band are visible, its midpoint is the only phase
    # consistent across stroke widths. Target evidence only ranks these fixed
    # observed candidates; it never changes either transition position.
    scores = (
        semantics
        + 0.55 * phase_strength
        + 0.20 * strength_ratio
        + config.pair_preference * np.clip(band_contrast / 0.05, 0.0, 1.0)
        + 1.20 * np.clip(target_gain / 0.08, -1.0, 1.0)
        - 1.50 * (1.0 - np.clip(target_gain / 0.006, 0.0, 1.0))
        - 2.00 * (1.0 - np.clip(local_target_separation / 0.03, 0.0, 1.0))
    )
    candidates.extend(
        _PhaseCandidate(
            center=float(center),
            width=float(width),
            score=float(score),
            semantic_score=float(semantic),
            phase_strength=float(phase),
            paired=True,
            target_gain=float(gain),
            local_target_separation=float(separation),
            band_contrast=float(contrast),
        )
        for center, width, score, semantic, phase, gain, separation, contrast in zip(
            centers,
            widths,
            scores,
            semantics,
            phase_strength,
            target_gain,
            local_target_separation,
            band_contrast,
        )
    )
    return candidates


def _semantic_selection_score(
    rgb: np.ndarray,
    offsets: np.ndarray,
    *,
    center: float,
    half_width: float,
    target: np.ndarray,
    config: EdgePhaseConfig,
) -> float:
    return float(
        _semantic_selection_scores(
            rgb,
            offsets,
            centers=np.asarray([center], dtype=np.float32),
            half_widths=np.asarray([half_width], dtype=np.float32),
            target=target,
            config=config,
        )[0]
    )


def _semantic_selection_scores(
    rgb: np.ndarray,
    offsets: np.ndarray,
    *,
    centers: np.ndarray,
    half_widths: np.ndarray,
    target: np.ndarray,
    config: EdgePhaseConfig,
) -> np.ndarray:
    distances = np.asarray(config.semantic_probe_distances_px, dtype=np.float32)
    outside_positions = centers[:, None] - half_widths[:, None] - distances[None, :]
    inside_positions = centers[:, None] + half_widths[:, None] + distances[None, :]
    outside = _sample_rgb_grid(rgb, offsets, outside_positions)
    inside = _sample_rgb_grid(rgb, offsets, inside_positions)
    outside_distance = np.median(
        np.linalg.norm(outside - target[None, None, :], axis=2) / math.sqrt(3.0),
        axis=1,
    )
    inside_distance = np.median(
        np.linalg.norm(inside - target[None, None, :], axis=2) / math.sqrt(3.0),
        axis=1,
    )
    separation = np.clip(outside_distance - inside_distance, -0.5, 0.5)
    inside_support = np.clip((0.38 - inside_distance) / 0.38, 0.0, 1.0)
    # Persistent target support makes a true boundary beat a thin distractor;
    # neither term is allowed to alter the already observed candidate center.
    return np.asarray(2.5 * separation + 0.9 * inside_support, dtype=np.float32)


def _pair_band_contrast(
    rgb: np.ndarray,
    offsets: np.ndarray,
    *,
    outer: float,
    inner: float,
) -> float:
    return float(
        _pair_band_contrasts(
            rgb,
            offsets,
            outer=np.asarray([outer], dtype=np.float32),
            inner=np.asarray([inner], dtype=np.float32),
        )[0]
    )


def _pair_band_contrasts(
    rgb: np.ndarray,
    offsets: np.ndarray,
    *,
    outer: np.ndarray,
    inner: np.ndarray,
) -> np.ndarray:
    width = np.maximum(1e-6, inner - outer)
    probe = np.clip(width * 0.45, 0.50, 1.50)
    positions = np.stack([outer - probe, 0.5 * (outer + inner), inner + probe], axis=1)
    sampled = _sample_rgb_grid(rgb, offsets, positions)
    outside_contrast = np.linalg.norm(sampled[:, 1] - sampled[:, 0], axis=1) / math.sqrt(3.0)
    inside_contrast = np.linalg.norm(sampled[:, 1] - sampled[:, 2], axis=1) / math.sqrt(3.0)
    return np.asarray(np.minimum(outside_contrast, inside_contrast), dtype=np.float32)


def _pair_target_evidence(
    rgb: np.ndarray,
    offsets: np.ndarray,
    *,
    outer: float,
    inner: float,
    target: np.ndarray,
    config: EdgePhaseConfig,
) -> tuple[float, float]:
    """Return band-vs-fill gain and local outside-vs-inside separation.

    The locations are fixed before this function is called.  Target color is
    deliberately limited to candidate ranking: it compares the observed band
    to several samples farther inside, but never modifies either transition.
    """

    gain, separation = _pair_target_evidence_batch(
        rgb,
        offsets,
        outer=np.asarray([outer], dtype=np.float32),
        inner=np.asarray([inner], dtype=np.float32),
        target=target,
        config=config,
    )
    return float(gain[0]), float(separation[0])


def _pair_target_evidence_batch(
    rgb: np.ndarray,
    offsets: np.ndarray,
    *,
    outer: np.ndarray,
    inner: np.ndarray,
    target: np.ndarray,
    config: EdgePhaseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    width = np.maximum(1e-6, inner - outer)
    center = 0.5 * (outer + inner)
    # Keep every band probe away from an antialiased transition shoulder.  A
    # single center probe is most stable for one-pixel strokes; wider bands get
    # three probes so a crossing label cannot masquerade as a coherent stroke.
    inset = np.where(width < 1.5, 0.0, np.minimum(width * 0.22, 0.75))
    band_positions = np.stack([center - inset, center, center + inset], axis=1)
    inside_positions = inner[:, None] + np.asarray(
        config.semantic_probe_distances_px,
        dtype=np.float32,
    )[None, :]
    band = _sample_rgb_grid(rgb, offsets, band_positions)
    inside = _sample_rgb_grid(rgb, offsets, inside_positions)
    nearest_probe = float(min(config.semantic_probe_distances_px))
    outside_near = _sample_rgb_grid(
        rgb,
        offsets,
        outer - nearest_probe,
    )
    band_distance = np.median(
        np.linalg.norm(band - target[None, None, :], axis=2) / math.sqrt(3.0),
        axis=1,
    )
    inside_distance = np.median(
        np.linalg.norm(inside - target[None, None, :], axis=2) / math.sqrt(3.0),
        axis=1,
    )
    outside_near_distance = np.linalg.norm(outside_near - target[None, :], axis=1) / math.sqrt(3.0)
    inside_near_distance = np.linalg.norm(inside[:, 0] - target[None, :], axis=1) / math.sqrt(3.0)
    return (
        np.asarray(band_distance - inside_distance, dtype=np.float32),
        np.asarray(outside_near_distance - inside_near_distance, dtype=np.float32),
    )


def _sample_rgb(rgb: np.ndarray, offsets: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.interp(positions, offsets, rgb[:, channel])
            for channel in range(3)
        ],
        axis=1,
    ).astype(np.float32)


def _sample_rgb_grid(rgb: np.ndarray, offsets: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions_array = np.asarray(positions, dtype=np.float32)
    sampled = _sample_rgb(rgb, offsets, positions_array.reshape(-1))
    return sampled.reshape(*positions_array.shape, 3)


def _candidate_reliability(candidate: _PhaseCandidate, second_score: float) -> float:
    margin = max(0.0, candidate.score - second_score)
    semantic = max(0.0, candidate.semantic_score)
    evidence = (
        0.45 * float(np.clip(candidate.phase_strength, 0.0, 1.0))
        + 0.35 * float(np.clip(semantic / 1.6, 0.0, 1.0))
        + 0.20 * float(np.clip(margin / 0.8, 0.0, 1.0))
    )
    return float(np.clip(evidence, 0.05, 1.0))


__all__ = [
    "EdgePhaseConfig",
    "EdgePhaseResult",
    "centered_edge_phase_logits",
    "estimate_edge_phase",
]
