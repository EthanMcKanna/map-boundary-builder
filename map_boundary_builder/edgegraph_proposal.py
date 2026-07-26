"""Source-native structural proposals for EdgeGraph selectors.

The global selector deliberately solves *which* map region is the service
area at a small, fixed input size.  It is therefore a useful semantic prior,
but it cannot preserve every narrow stem or notch after downsampling.  A
normal-profile refiner cannot recover geometry which is absent from that
prior: there is no contour sample from which to search.

This module repairs that specific failure before contour localization.  It
learns tiny, image-local inside/outside appearance dictionaries from pixels
well away from the selector boundary, evaluates them at source resolution,
and proposes only deep, elongated additions or removals.  Proposals must be
connected to the selected region/exterior, retain the guided seed, preserve
digital topology, remain local in area, and have persistent native appearance
support.  Ordinary one-to-three-pixel boundary phase is left for EdgeGraph's
normal-strip decoder.

The implementation is intentionally independent of :mod:`edgegraph` so the
proposal stage can be tested, benchmarked, and promoted separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SourceNativeProposalConfig:
    """Controls conservative structural recovery at source resolution."""

    coarse_threshold: float = 0.45
    training_margin_px: int = 10
    prototype_count: int = 8
    maximum_training_samples: int = 12_000
    appearance_blur_px: int = 5
    target_median_px: int = 5
    search_radius_px: float = 96.0
    # Native branch/notch proposals remain independently gated by terminal
    # support, boundary energy, area, seed containment, and digital topology.
    # A high global rectilinearity cutoff hid real long orthogonal structure
    # when the rest of an authored service area contained roads/curves.
    minimum_rectilinear_fraction: float = 0.50
    rectilinear_tolerance_degrees: float = 10.0
    minimum_structural_depth_px: float = 4.0
    minimum_core_area_px: int = 4
    minimum_long_axis_px: int = 8
    minimum_short_axis_px: int = 4
    maximum_short_axis_px: int = 56
    proposal_padding_px: int = 10
    minimum_support_fraction: float = 0.10
    minimum_mean_confidence: float = 0.45
    minimum_contraction_confidence: float = 0.70
    minimum_terminal_contrast: float = 0.05
    minimum_terminal_support_fraction: float = 0.55
    minimum_boundary_energy_ratio: float = 0.72
    maximum_boundary_energy_drop: float = 0.06
    # A global median boundary-energy statistic can veto a real narrow stem:
    # its long, deliberately soft side edges can outweigh a strong terminal
    # cap even when source-native appearance is unambiguous.  The only
    # exception is a unique, non-viewport, deep-and-thin expansion with
    # independent native evidence.  Depth, slenderness, and area are
    # normalized by selector scale so this remains resolution independent.
    deep_thin_expansion_energy_exception_enabled: bool = True
    deep_thin_expansion_minimum_native_confidence: float = 0.80
    deep_thin_expansion_minimum_terminal_contrast: float = 0.30
    deep_thin_expansion_minimum_terminal_support_fraction: float = 0.95
    deep_thin_expansion_minimum_rectilinear_fraction: float = 0.90
    deep_thin_expansion_minimum_normalized_depth: float = 0.10
    deep_thin_expansion_minimum_slenderness_ratio: float = 6.0
    deep_thin_expansion_maximum_selector_area_fraction: float = 0.0025
    maximum_proposal_area_fraction: float = 0.06
    maximum_total_area_change_fraction: float = 0.12
    support_probability_threshold: float = 0.50
    viewport_continuation_enabled: bool = True
    minimum_viewport_support_fraction: float = 0.30
    minimum_viewport_terminal_confidence: float = 0.60
    minimum_viewport_terminal_support_fraction: float = 0.90
    topology_guard: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.coarse_threshold < 1.0:
            raise ValueError("coarse_threshold must be between zero and one")
        if self.training_margin_px < 2:
            raise ValueError("training_margin_px must be at least two")
        if self.prototype_count < 1:
            raise ValueError("prototype_count must be positive")
        if self.maximum_training_samples < self.prototype_count:
            raise ValueError("maximum_training_samples must cover all prototypes")
        for name in ("appearance_blur_px", "target_median_px"):
            value = int(getattr(self, name))
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.search_radius_px <= self.minimum_structural_depth_px:
            raise ValueError("search radius must exceed structural depth")
        if self.minimum_core_area_px < 1 or self.minimum_long_axis_px < 2:
            raise ValueError("proposal size limits must be positive")
        if self.minimum_short_axis_px < 1:
            raise ValueError("minimum_short_axis_px must be positive")
        if self.maximum_short_axis_px < 2:
            raise ValueError("maximum_short_axis_px must be at least two")
        if self.minimum_short_axis_px > self.maximum_short_axis_px:
            raise ValueError("minimum_short_axis_px cannot exceed its maximum")
        if not 0.0 < self.minimum_rectilinear_fraction <= 1.0:
            raise ValueError("minimum_rectilinear_fraction must be in (0, 1]")
        if not 0.0 < self.rectilinear_tolerance_degrees <= 45.0:
            raise ValueError("rectilinear_tolerance_degrees must be in (0, 45]")
        if self.proposal_padding_px < 1:
            raise ValueError("proposal_padding_px must be positive")
        for name in (
            "minimum_support_fraction",
            "minimum_mean_confidence",
            "minimum_contraction_confidence",
            "minimum_terminal_contrast",
            "minimum_terminal_support_fraction",
            "minimum_boundary_energy_ratio",
            "maximum_boundary_energy_drop",
            "deep_thin_expansion_minimum_native_confidence",
            "deep_thin_expansion_minimum_terminal_contrast",
            "deep_thin_expansion_minimum_terminal_support_fraction",
            "deep_thin_expansion_minimum_rectilinear_fraction",
            "deep_thin_expansion_maximum_selector_area_fraction",
            "maximum_proposal_area_fraction",
            "maximum_total_area_change_fraction",
            "support_probability_threshold",
            "minimum_viewport_support_fraction",
            "minimum_viewport_terminal_confidence",
            "minimum_viewport_terminal_support_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "deep_thin_expansion_minimum_normalized_depth",
            "deep_thin_expansion_minimum_slenderness_ratio",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SourceNativeProposalResult:
    """A topology-guarded selector proposal and its source-native evidence."""

    mask: np.ndarray
    probabilities: np.ndarray
    diagnostics: dict[str, object]
    native_inside_probability: np.ndarray | None = None


def recover_source_native_rectilinear_proposals(
    rgb: np.ndarray,
    coarse_probabilities: np.ndarray,
    *,
    target_rgb: Any,
    seed_point: tuple[float, float] | None = None,
    coarse_mask: np.ndarray | None = None,
    config: SourceNativeProposalConfig | None = None,
) -> SourceNativeProposalResult:
    """Recover supported thin stems and notches missing from a coarse mask.

    The returned probabilities retain the selector values everywhere except
    accepted proposal pixels.  Added pixels are raised above the configured
    threshold and removed pixels are lowered below it, allowing the ordinary
    EdgeGraph contour/localization path to consume the result unchanged.

    This stage is conservative by construction.  It does not attempt general
    source-native segmentation and it does not change component or hole
    counts.  Unsupported or ambiguous images return the original selector.
    """

    cfg = config or SourceNativeProposalConfig()
    image = np.asarray(rgb)
    probabilities = np.asarray(coarse_probabilities, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must have shape (height, width, 3), got {image.shape}")
    if probabilities.shape != image.shape[:2]:
        raise ValueError(
            "coarse probabilities must match the source image, got "
            f"{probabilities.shape} and {image.shape[:2]}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("coarse probabilities must be finite")
    target = np.asarray(target_rgb, dtype=np.float32).reshape(-1)
    if target.size != 3 or not np.isfinite(target).all():
        raise ValueError("target_rgb must contain three finite channels")
    target = np.clip(target, 0.0, 255.0)

    original = (
        np.asarray(coarse_mask, dtype=bool).copy()
        if coarse_mask is not None
        else probabilities >= cfg.coarse_threshold
    )
    if original.shape != image.shape[:2]:
        raise ValueError("coarse_mask must match the source image")
    original = _select_seed_component(original, seed_point)
    if not original.any() or original.all():
        return _unchanged_result(
            original,
            probabilities,
            reason="empty-or-full-selector",
            config=cfg,
        )

    rectilinear_fraction = _rectilinear_fraction(
        original,
        tolerance_degrees=cfg.rectilinear_tolerance_degrees,
    )
    if rectilinear_fraction < cfg.minimum_rectilinear_fraction:
        result = _unchanged_result(
            original,
            probabilities,
            reason="non-rectilinear-selector",
            config=cfg,
        )
        result.diagnostics["rectilinear_fraction"] = round(rectilinear_fraction, 6)
        return result

    distance_outside = cv2.distanceTransform(
        (~original).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    distance_inside = cv2.distanceTransform(
        original.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    inside_training = distance_inside > float(cfg.training_margin_px)
    outside_training = distance_outside > float(cfg.training_margin_px)
    search_band = (distance_inside + distance_outside) <= float(cfg.search_radius_px)
    if (
        int(inside_training.sum()) < cfg.prototype_count * 4
        or int(outside_training.sum()) < cfg.prototype_count * 4
    ):
        return _unchanged_result(
            original,
            probabilities,
            reason="insufficient-training-support",
            config=cfg,
        )

    rgb_float = image.astype(np.float32) / 255.0
    native = _native_inside_probability(
        rgb_float,
        original,
        inside_training=inside_training,
        outside_training=outside_training,
        search_band=search_band,
        target_rgb=target,
        config=cfg,
    )
    inside_supported = _connected_inside_support(
        native,
        original,
        threshold=cfg.support_probability_threshold,
    )
    outside_supported = _connected_outside_support(
        native,
        original,
        threshold=1.0 - cfg.support_probability_threshold,
    )

    expansion_core = (
        inside_supported
        & ~original
        & search_band
        & (distance_outside >= cfg.minimum_structural_depth_px)
    )
    contraction_core = (
        outside_supported
        & original
        & search_band
        & (distance_inside >= cfg.minimum_structural_depth_px)
    )

    proposals = _collect_proposals(
        expansion_core,
        kind="expansion",
        support_mask=inside_supported,
        confidence=native,
        structural_depth=distance_outside,
        rgb_float=rgb_float,
        config=cfg,
    )
    viewport_proposals: list[_StructuralProposal] = []
    expansion_frontier = expansion_core & (
        distance_outside >= cfg.search_radius_px - cfg.proposal_padding_px
    )
    if cfg.viewport_continuation_enabled and bool(np.any(expansion_frontier)):
        # A real service area may continue beyond the local proposal band and
        # terminate at the screenshot edge. Re-evaluate native appearance over
        # the frame only when a supported expansion actually reaches that
        # band frontier; ordinary samples retain the bounded local search.
        expanded_search_band = np.ones_like(search_band, dtype=bool)
        expanded_native = _native_inside_probability(
            rgb_float,
            original,
            inside_training=inside_training,
            outside_training=outside_training,
            search_band=expanded_search_band,
            target_rgb=target,
            config=cfg,
        )
        expanded_inside_supported = _connected_inside_support(
            expanded_native,
            original,
            threshold=cfg.support_probability_threshold,
        )
        expanded_core = (
            expanded_inside_supported
            & ~original
            & (distance_outside >= cfg.minimum_structural_depth_px)
        )
        viewport_proposals = _collect_proposals(
            expanded_core,
            kind="expansion",
            support_mask=expanded_inside_supported,
            confidence=expanded_native,
            structural_depth=distance_outside,
            rgb_float=rgb_float,
            config=cfg,
            required_core_overlap=expansion_frontier,
            require_viewport_terminal=True,
        )
        if viewport_proposals:
            native = expanded_native
            search_band = expanded_search_band
    proposals.extend(viewport_proposals)
    proposals.extend(
        _collect_proposals(
            contraction_core,
            kind="contraction",
            support_mask=outside_supported,
            confidence=1.0 - native,
            structural_depth=distance_inside,
            rgb_float=rgb_float,
            config=cfg,
        )
    )
    proposals.sort(key=lambda item: (-item.maximum_depth, -item.core_area, item.kind))

    current = original.copy()
    baseline_topology = _digital_topology(original)
    maximum_single_area = max(
        8,
        int(round(float(original.sum()) * cfg.maximum_proposal_area_fraction)),
    )
    maximum_total_change = max(
        12,
        int(round(float(original.sum()) * cfg.maximum_total_area_change_fraction)),
    )
    accepted: list[_StructuralProposal] = []
    rejected_topology = 0
    rejected_area = 0
    rejected_seed = 0
    rejected_energy = 0
    changed_total = 0
    for proposal in proposals:
        candidate = current.copy()
        if proposal.kind == "expansion":
            candidate[proposal.change_mask] = True
        else:
            candidate[proposal.change_mask] = False
        changed = int(np.count_nonzero(candidate != current))
        if changed == 0:
            continue
        if changed > maximum_single_area or changed_total + changed > maximum_total_change:
            rejected_area += 1
            continue
        if seed_point is not None and not _contains_seed(candidate, seed_point):
            rejected_seed += 1
            continue
        if cfg.topology_guard and _digital_topology(candidate) != baseline_topology:
            rejected_topology += 1
            continue
        current = candidate
        changed_total += changed
        accepted.append(proposal)

    edge_magnitude = _normalized_edge_magnitude(image)
    original_boundary_energy = _boundary_edge_energy(edge_magnitude, original)
    proposed_boundary_energy = _boundary_edge_energy(edge_magnitude, current)
    boundary_energy_ratio = proposed_boundary_energy / max(1e-6, original_boundary_energy)
    boundary_energy_drop = original_boundary_energy - proposed_boundary_energy
    evaluated_boundary_energy = proposed_boundary_energy
    evaluated_boundary_energy_ratio = boundary_energy_ratio
    evaluated_boundary_energy_drop = boundary_energy_drop
    energy_veto_triggered = bool(
        accepted
        and boundary_energy_ratio < cfg.minimum_boundary_energy_ratio
        and boundary_energy_drop > cfg.maximum_boundary_energy_drop
    )
    energy_exception = _deep_thin_expansion_energy_exception_evidence(
        proposals,
        accepted,
        selector_area_px=int(original.sum()),
        changed_pixel_count=changed_total,
        rectilinear_fraction=rectilinear_fraction,
        config=cfg,
    )
    energy_exception["global_energy_veto_triggered"] = energy_veto_triggered
    energy_exception["applied"] = bool(
        energy_veto_triggered and energy_exception["eligible"]
    )
    if energy_veto_triggered and not energy_exception["applied"]:
        rejected_energy = len(accepted)
        current = original.copy()
        accepted = []
        changed_total = 0
        proposed_boundary_energy = original_boundary_energy
        boundary_energy_ratio = 1.0

    output_probabilities = probabilities.copy()
    raised = current & ~original
    lowered = original & ~current
    high = min(0.995, max(cfg.coarse_threshold + 0.20, 0.80))
    low = max(0.005, min(cfg.coarse_threshold - 0.20, 0.20))
    output_probabilities[raised] = np.maximum(output_probabilities[raised], high)
    output_probabilities[lowered] = np.minimum(output_probabilities[lowered], low)
    output_probabilities = np.ascontiguousarray(output_probabilities, dtype=np.float32)

    expansion = [item for item in accepted if item.kind == "expansion"]
    contraction = [item for item in accepted if item.kind == "contraction"]
    return SourceNativeProposalResult(
        mask=np.ascontiguousarray(current, dtype=bool),
        probabilities=output_probabilities,
        diagnostics={
            "source_native_proposals": True,
            "reason": "evaluated",
            "candidate_count": len(proposals),
            "accepted_count": len(accepted),
            "accepted_expansion_count": len(expansion),
            "accepted_contraction_count": len(contraction),
            "accepted_viewport_expansion_count": sum(
                item.viewport_terminal for item in expansion
            ),
            "changed_pixel_count": changed_total,
            "expanded_pixel_count": int(raised.sum()),
            "contracted_pixel_count": int(lowered.sum()),
            "maximum_depth_px": round(
                max((item.maximum_depth for item in accepted), default=0.0),
                6,
            ),
            "mean_support_fraction": round(
                float(np.mean([item.support_fraction for item in accepted]))
                if accepted
                else 0.0,
                6,
            ),
            "minimum_support_fraction": round(
                min((item.support_fraction for item in accepted), default=0.0),
                6,
            ),
            "mean_proposal_confidence": round(
                float(np.mean([item.mean_confidence for item in accepted]))
                if accepted
                else 0.0,
                6,
            ),
            "minimum_proposal_confidence": round(
                min((item.mean_confidence for item in accepted), default=0.0),
                6,
            ),
            "mean_terminal_contrast": round(
                float(np.mean([item.terminal_contrast for item in accepted]))
                if accepted
                else 0.0,
                6,
            ),
            "minimum_terminal_contrast": round(
                min((item.terminal_contrast for item in accepted), default=0.0),
                6,
            ),
            "mean_terminal_support_fraction": round(
                float(np.mean([item.terminal_support_fraction for item in accepted]))
                if accepted
                else 0.0,
                6,
            ),
            "minimum_terminal_support_fraction": round(
                min(
                    (item.terminal_support_fraction for item in accepted),
                    default=0.0,
                ),
                6,
            ),
            "rejected_topology_count": rejected_topology,
            "rejected_area_count": rejected_area,
            "rejected_seed_count": rejected_seed,
            "rejected_energy_count": rejected_energy,
            "topology_preserved": _digital_topology(current) == baseline_topology,
            "native_probability_mean": round(float(np.mean(native)), 6),
            "rectilinear_fraction": round(rectilinear_fraction, 6),
            "original_boundary_energy": round(original_boundary_energy, 6),
            "proposed_boundary_energy": round(proposed_boundary_energy, 6),
            "boundary_energy_ratio": round(boundary_energy_ratio, 6),
            "evaluated_boundary_energy": round(evaluated_boundary_energy, 6),
            "evaluated_boundary_energy_ratio": round(evaluated_boundary_energy_ratio, 6),
            "evaluated_boundary_energy_drop": round(evaluated_boundary_energy_drop, 6),
            "deep_thin_expansion_energy_exception": energy_exception,
            "shortcut_suppression_recommended": bool(
                accepted
                and _digital_topology(current) == baseline_topology
            ),
        },
        # The appearance map is an independent source-native witness for
        # downstream structural arbitration. Pixels outside the evaluated
        # selector band are deliberately unavailable rather than filled with
        # the selector prior: a structural edit may only consume observed
        # native evidence.
        native_inside_probability=np.ascontiguousarray(
            np.where(search_band, native, np.nan),
            dtype=np.float32,
        ),
    )


@dataclass(frozen=True)
class _StructuralProposal:
    kind: str
    change_mask: np.ndarray
    core_area: int
    maximum_depth: float
    support_fraction: float
    mean_confidence: float
    terminal_contrast: float
    terminal_support_fraction: float
    viewport_terminal: bool = False


def _deep_thin_expansion_energy_exception_evidence(
    proposals: list[_StructuralProposal],
    accepted: list[_StructuralProposal],
    *,
    selector_area_px: int,
    changed_pixel_count: int,
    rectilinear_fraction: float,
    config: SourceNativeProposalConfig,
) -> dict[str, object]:
    """Return fail-closed evidence for the sole global-energy exception.

    ``maximum_depth / sqrt(selector_area)`` is invariant to uniform image
    scaling.  The slenderness statistic estimates depth / effective width,
    where effective width is changed area / depth.  Together with changed
    area / selector area, these distinguish a missing narrow stem from a
    broad segmentation correction without any fixture-sized pixel constant.
    """

    proposal = accepted[0] if len(accepted) == 1 else None
    selector_area = max(0, int(selector_area_px))
    changed_area = max(0, int(changed_pixel_count))
    maximum_depth = float(proposal.maximum_depth) if proposal is not None else 0.0
    changed_area_fraction = (
        float(changed_area) / float(selector_area) if selector_area > 0 else 0.0
    )
    normalized_depth = (
        maximum_depth / math.sqrt(float(selector_area)) if selector_area > 0 else 0.0
    )
    slenderness_ratio = (
        maximum_depth * maximum_depth / float(changed_area)
        if changed_area > 0
        else 0.0
    )
    native_confidence = (
        float(proposal.mean_confidence) if proposal is not None else 0.0
    )
    terminal_contrast = (
        float(proposal.terminal_contrast) if proposal is not None else 0.0
    )
    terminal_support_fraction = (
        float(proposal.terminal_support_fraction) if proposal is not None else 0.0
    )
    checks = (
        (
            "disabled",
            bool(config.deep_thin_expansion_energy_exception_enabled),
        ),
        ("ambiguous-candidate-count", len(proposals) == 1),
        ("not-single-accepted-proposal", proposal is not None),
        (
            "not-expansion",
            proposal is not None and proposal.kind == "expansion",
        ),
        (
            "viewport-expansion",
            proposal is not None and not proposal.viewport_terminal,
        ),
        ("invalid-area-evidence", selector_area > 0 and changed_area > 0),
        (
            "insufficient-native-confidence",
            native_confidence
            >= config.deep_thin_expansion_minimum_native_confidence,
        ),
        (
            "insufficient-terminal-contrast",
            terminal_contrast
            >= config.deep_thin_expansion_minimum_terminal_contrast,
        ),
        (
            "insufficient-terminal-support",
            terminal_support_fraction
            >= config.deep_thin_expansion_minimum_terminal_support_fraction,
        ),
        (
            "insufficient-rectilinearity",
            rectilinear_fraction
            >= config.deep_thin_expansion_minimum_rectilinear_fraction,
        ),
        (
            "insufficient-normalized-depth",
            normalized_depth
            >= config.deep_thin_expansion_minimum_normalized_depth,
        ),
        (
            "insufficient-slenderness",
            slenderness_ratio
            >= config.deep_thin_expansion_minimum_slenderness_ratio,
        ),
        (
            "excessive-selector-area-fraction",
            changed_area_fraction
            <= config.deep_thin_expansion_maximum_selector_area_fraction,
        ),
    )
    failed = [reason for reason, passed in checks if not passed]
    return {
        "enabled": bool(config.deep_thin_expansion_energy_exception_enabled),
        "eligible": not failed,
        "applied": False,
        "reason": failed[0] if failed else "strong-native-deep-thin-expansion",
        "candidate_count": len(proposals),
        "accepted_count_before_energy_veto": len(accepted),
        "native_confidence": round(native_confidence, 6),
        "terminal_contrast": round(terminal_contrast, 6),
        "terminal_support_fraction": round(terminal_support_fraction, 6),
        "rectilinear_fraction": round(float(rectilinear_fraction), 6),
        "maximum_depth_px": round(maximum_depth, 6),
        "normalized_depth": round(normalized_depth, 6),
        "slenderness_ratio": round(slenderness_ratio, 6),
        "changed_pixel_count": changed_area,
        "selector_area_px": selector_area,
        "changed_selector_area_fraction": round(changed_area_fraction, 8),
    }


def _native_inside_probability(
    rgb_float: np.ndarray,
    coarse_mask: np.ndarray,
    *,
    inside_training: np.ndarray,
    outside_training: np.ndarray,
    search_band: np.ndarray,
    target_rgb: np.ndarray,
    config: SourceNativeProposalConfig,
) -> np.ndarray:
    # RGB is intentional here.  The overlay hint and selector are both RGB
    # conditioned, and avoiding a second full-frame color transform keeps this
    # native pass cheap enough to sit in front of contour refinement.
    appearance = rgb_float
    inside_values = _sample_training_values(
        appearance[inside_training],
        maximum=config.maximum_training_samples,
    )
    outside_values = _sample_training_values(
        appearance[outside_training],
        maximum=config.maximum_training_samples,
    )
    inside_centers = _appearance_prototypes(inside_values, config.prototype_count)
    outside_centers = _appearance_prototypes(outside_values, config.prototype_count)
    band_values = np.ascontiguousarray(appearance[search_band], dtype=np.float32)
    band_inside_distance = _nearest_squared_distance(band_values, inside_centers)
    band_outside_distance = _nearest_squared_distance(band_values, outside_centers)
    inside_training_score = _nearest_squared_distance(inside_values, outside_centers) - _nearest_squared_distance(
        inside_values,
        inside_centers,
    )
    outside_training_score = _nearest_squared_distance(outside_values, outside_centers) - _nearest_squared_distance(
        outside_values,
        inside_centers,
    )
    inside_floor = float(np.quantile(inside_training_score, 0.05))
    outside_ceiling = float(np.quantile(outside_training_score, 0.95))
    appearance_score = np.where(
        coarse_mask,
        float(np.median(inside_training_score)),
        float(np.median(outside_training_score)),
    ).astype(np.float32)
    appearance_score[search_band] = band_outside_distance - band_inside_distance
    if config.appearance_blur_px > 1:
        appearance_score = cv2.GaussianBlur(
            appearance_score,
            (config.appearance_blur_px, config.appearance_blur_px),
            0.0,
        )
    midpoint = 0.5 * (inside_floor + outside_ceiling)
    score_span = max(1e-4, inside_floor - outside_ceiling)
    appearance_probability = _sigmoid((appearance_score - midpoint) * (8.0 / score_span))

    inside_rgb = _sample_training_values(
        rgb_float[inside_training],
        maximum=config.maximum_training_samples,
    )
    outside_rgb = _sample_training_values(
        rgb_float[outside_training],
        maximum=config.maximum_training_samples,
    )
    target_normalized = target_rgb.reshape(1, 3) / 255.0
    target_denominator = math.sqrt(3.0)
    target_inside = np.linalg.norm(inside_rgb - target_normalized, axis=1) / target_denominator
    target_outside = np.linalg.norm(outside_rgb - target_normalized, axis=1) / target_denominator
    target_distance = np.where(
        coarse_mask,
        float(np.median(target_inside)),
        float(np.median(target_outside)),
    ).astype(np.float32)
    target_distance[search_band] = np.linalg.norm(
        rgb_float[search_band] - target_normalized,
        axis=1,
    ) / target_denominator
    if config.target_median_px > 1:
        target_distance = cv2.medianBlur(
            target_distance.astype(np.float32),
            config.target_median_px,
        )
    inside_ceiling = float(np.quantile(target_inside, 0.95))
    outside_floor = float(np.quantile(target_outside, 0.10))
    target_midpoint = 0.5 * (inside_ceiling + outside_floor)
    target_span = max(0.01, outside_floor - inside_ceiling)
    target_probability = _sigmoid((target_midpoint - target_distance) * (6.0 / target_span))

    # Either local appearance prototypes or direct target-color agreement may
    # identify an addition.  A confident rejection from either source may
    # identify a notch, so retain both tails instead of averaging them away.
    inside_probability = np.maximum(appearance_probability, 0.92 * target_probability)
    outside_probability = np.maximum(1.0 - appearance_probability, 0.92 * (1.0 - target_probability))
    normalization = np.maximum(1e-6, inside_probability + outside_probability)
    probability = inside_probability / normalization
    return np.asarray(np.clip(probability, 0.0, 1.0), dtype=np.float32)


def _sample_training_values(values: np.ndarray, *, maximum: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if len(values) <= maximum:
        return np.ascontiguousarray(values)
    indices = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return np.ascontiguousarray(values[indices])


def _appearance_prototypes(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return np.ascontiguousarray(values, dtype=np.float32)
    # Deterministic farthest-point prototypes avoid global OpenCV RNG state and
    # retain rare road/UI modes which a mean-only color model would discard.
    mean = values.mean(axis=0)
    first = int(np.argmax(np.sum((values - mean) ** 2, axis=1)))
    selected = [first]
    nearest = np.sum((values - values[first]) ** 2, axis=1)
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((values - values[index]) ** 2, axis=1))
    return np.ascontiguousarray(values[selected], dtype=np.float32)


def _nearest_squared_distance(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    value_norm = np.einsum("ij,ij->i", values, values)
    center_norm = np.einsum("ij,ij->i", centers, centers)
    distances = value_norm[:, None] + center_norm[None, :] - 2.0 * (values @ centers.T)
    return np.maximum(0.0, np.min(distances, axis=1)).astype(np.float32)


def _connected_inside_support(
    probability: np.ndarray,
    coarse_mask: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    supported = probability >= threshold
    supported = cv2.morphologyEx(
        supported.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    count, labels = cv2.connectedComponents(supported.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(coarse_mask)
    deep = cv2.erode(coarse_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    overlap = np.bincount(labels[deep].ravel(), minlength=count)
    overlap[0] = 0
    selected = int(np.argmax(overlap))
    return labels == selected if overlap[selected] > 0 else np.zeros_like(coarse_mask)


def _connected_outside_support(
    probability: np.ndarray,
    coarse_mask: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    supported = probability <= threshold
    supported = cv2.morphologyEx(
        supported.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    count, labels = cv2.connectedComponents(supported.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(coarse_mask)
    deep = cv2.erode((~coarse_mask).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    overlap = np.bincount(labels[deep].ravel(), minlength=count)
    overlap[0] = 0
    selected_labels = np.flatnonzero(overlap >= max(12, int(0.001 * deep.sum())))
    if not len(selected_labels):
        selected_labels = np.asarray([int(np.argmax(overlap))], dtype=np.int32)
    return np.isin(labels, selected_labels)


def _collect_proposals(
    core: np.ndarray,
    *,
    kind: str,
    support_mask: np.ndarray,
    confidence: np.ndarray,
    structural_depth: np.ndarray,
    rgb_float: np.ndarray,
    config: SourceNativeProposalConfig,
    required_core_overlap: np.ndarray | None = None,
    require_viewport_terminal: bool = False,
) -> list[_StructuralProposal]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        core.astype(np.uint8),
        connectivity=8,
    )
    proposals: list[_StructuralProposal] = []
    padding = int(config.proposal_padding_px)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * padding + 1, 2 * padding + 1),
    )
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        long_axis = max(width, height)
        short_axis = min(width, height)
        if area < config.minimum_core_area_px or long_axis < config.minimum_long_axis_px:
            continue
        if short_axis < config.minimum_short_axis_px:
            continue
        if short_axis > config.maximum_short_axis_px:
            continue
        component = labels == label
        if (
            required_core_overlap is not None
            and not bool(np.any(component & required_core_overlap))
        ):
            continue
        influence = cv2.dilate(component.astype(np.uint8), kernel, iterations=1) > 0
        change = influence & support_mask
        if not change.any():
            continue
        support_fraction = float(np.mean(support_mask[influence]))
        mean_confidence = float(np.mean(confidence[change]))
        if support_fraction < config.minimum_support_fraction:
            continue
        if (
            require_viewport_terminal
            and support_fraction < config.minimum_viewport_support_fraction
        ):
            continue
        required_confidence = (
            max(config.minimum_mean_confidence, config.minimum_contraction_confidence)
            if kind == "contraction"
            else config.minimum_mean_confidence
        )
        if mean_confidence < required_confidence:
            continue
        maximum_depth = float(np.max(structural_depth[component], initial=0.0))
        # A component clipped by the ordinary search band has no observed
        # terminal cap. The separate viewport-continuation lane may cross that
        # frontier only when it reaches a real image edge with persistent
        # independently classified inside support.
        if (
            not require_viewport_terminal
            and maximum_depth > config.search_radius_px - config.proposal_padding_px
        ):
            continue
        terminal_contrast, terminal_support_fraction = _terminal_cap_evidence(
            change,
            horizontal=width >= height,
            rgb_float=rgb_float,
            structural_depth=structural_depth,
        )
        viewport_terminal = False
        if require_viewport_terminal:
            viewport_confidence, viewport_support = _viewport_terminal_evidence(
                change,
                horizontal=width >= height,
                confidence=confidence,
                threshold=config.support_probability_threshold,
            )
            if viewport_confidence < config.minimum_viewport_terminal_confidence:
                continue
            if viewport_support < config.minimum_viewport_terminal_support_fraction:
                continue
            viewport_terminal = True
            terminal_support_fraction = viewport_support
        else:
            if terminal_contrast < config.minimum_terminal_contrast:
                continue
            if terminal_support_fraction < config.minimum_terminal_support_fraction:
                continue
        proposals.append(
            _StructuralProposal(
                kind=kind,
                change_mask=change,
                core_area=area,
                maximum_depth=maximum_depth,
                support_fraction=support_fraction,
                mean_confidence=mean_confidence,
                terminal_contrast=terminal_contrast,
                terminal_support_fraction=terminal_support_fraction,
                viewport_terminal=viewport_terminal,
            )
        )
    return proposals


def _select_seed_component(mask: np.ndarray, seed_point: tuple[float, float] | None) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    if seed_point is not None:
        x, y = _clipped_seed(seed_point, mask.shape)
        label = int(labels[y, x])
        if label > 0:
            return labels == label
    sizes = np.bincount(labels.ravel(), minlength=count)
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _contains_seed(mask: np.ndarray, seed_point: tuple[float, float]) -> bool:
    x, y = _clipped_seed(seed_point, mask.shape)
    return bool(mask[y, x])


def _clipped_seed(seed_point: tuple[float, float], shape: tuple[int, int]) -> tuple[int, int]:
    x = int(round(float(np.clip(seed_point[0], 0, shape[1] - 1))))
    y = int(round(float(np.clip(seed_point[1], 0, shape[0] - 1))))
    return x, y


def _digital_topology(mask: np.ndarray) -> tuple[int, int]:
    foreground_count, _foreground = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    background_count, background, stats, _centroids = cv2.connectedComponentsWithStats(
        (~mask).astype(np.uint8),
        connectivity=8,
    )
    holes = 0
    height, width = mask.shape
    for label in range(1, background_count):
        x, y, component_width, component_height, _area = (int(value) for value in stats[label])
        if x > 0 and y > 0 and x + component_width < width and y + component_height < height:
            holes += 1
    del background
    return max(0, foreground_count - 1), holes


def _terminal_cap_evidence(
    change_mask: np.ndarray,
    *,
    horizontal: bool,
    rgb_float: np.ndarray,
    structural_depth: np.ndarray,
) -> tuple[float, float]:
    directions = ((-1, 0), (1, 0)) if horizontal else ((0, -1), (0, 1))
    observations: list[tuple[float, float, float]] = []
    for direction_x, direction_y in directions:
        neighbor = _shift_mask(change_mask, dx=-direction_x, dy=-direction_y)
        cap = change_mask & ~neighbor & (structural_depth >= 4.0)
        ys, xs = np.where(cap)
        if not len(xs):
            continue

        def sample(step: int) -> np.ndarray:
            sample_x = np.clip(xs + direction_x * step, 0, rgb_float.shape[1] - 1)
            sample_y = np.clip(ys + direction_y * step, 0, rgb_float.shape[0] - 1)
            return rgb_float[sample_y, sample_x]

        inside = 0.5 * (sample(-2) + sample(-4))
        outside = 0.5 * (sample(2) + sample(4))
        contrast = np.linalg.norm(inside - outside, axis=1) / math.sqrt(3.0)
        observations.append(
            (
                float(np.median(structural_depth[cap])),
                float(np.median(contrast)),
                float(np.mean(contrast >= 0.025)),
            )
        )
    if not observations:
        return 0.0, 0.0
    # The attachment cap lies near the old selector.  The terminal cap is the
    # observation farther from it and must be a real image transition, not the
    # arbitrary point at which a road-like color classifier stopped firing.
    _depth, contrast, support = max(observations, key=lambda item: item[0])
    return contrast, support


def _viewport_terminal_evidence(
    change_mask: np.ndarray,
    *,
    horizontal: bool,
    confidence: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Score a structural continuation that legitimately ends at the viewport.

    Only one edge in the continuation direction may be touched. Perpendicular
    or multiple-edge contact is ambiguous clipping and fails closed.
    """

    contacts = {
        "top": change_mask[0, :],
        "bottom": change_mask[-1, :],
        "left": change_mask[:, 0],
        "right": change_mask[:, -1],
    }
    allowed = ("left", "right") if horizontal else ("top", "bottom")
    perpendicular = ("top", "bottom") if horizontal else ("left", "right")
    touched = [name for name in allowed if bool(np.any(contacts[name]))]
    if len(touched) != 1 or any(bool(np.any(contacts[name])) for name in perpendicular):
        return 0.0, 0.0
    name = touched[0]
    if name == "top":
        values = confidence[0, contacts[name]]
    elif name == "bottom":
        values = confidence[-1, contacts[name]]
    elif name == "left":
        values = confidence[contacts[name], 0]
    else:
        values = confidence[contacts[name], -1]
    if not len(values):
        return 0.0, 0.0
    return float(np.mean(values)), float(np.mean(values >= threshold))


def _shift_mask(mask: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    target_y = slice(max(0, dy), mask.shape[0] + min(0, dy))
    target_x = slice(max(0, dx), mask.shape[1] + min(0, dx))
    source_y = slice(max(0, -dy), mask.shape[0] - max(0, dy))
    source_x = slice(max(0, -dx), mask.shape[1] - max(0, dx))
    shifted[target_y, target_x] = mask[source_y, source_x]
    return shifted


def _normalized_edge_magnitude(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb.astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(gradient_x, gradient_y)
    scale = max(1e-6, float(np.quantile(magnitude, 0.99)))
    return np.asarray(np.clip(magnitude / scale, 0.0, 1.0), dtype=np.float32)


def _boundary_edge_energy(edge_magnitude: np.ndarray, mask: np.ndarray) -> float:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    boundary = mask & ~eroded
    if not boundary.any():
        return 0.0
    return float(np.median(edge_magnitude[boundary]))


def _rectilinear_fraction(mask: np.ndarray, *, tolerance_degrees: float) -> float:
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    approximation = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)[:, 0, :].astype(np.float32)
    if len(approximation) < 4:
        return 0.0
    edges = np.roll(approximation, -1, axis=0) - approximation
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths >= 2.0
    if int(np.count_nonzero(keep)) < 4:
        return 0.0
    edges = edges[keep]
    lengths = lengths[keep]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    weighted_cosine = float(np.sum(lengths * np.cos(4.0 * angles)))
    weighted_sine = float(np.sum(lengths * np.sin(4.0 * angles)))
    dominant_axis = 0.25 * math.atan2(weighted_sine, weighted_cosine)
    errors = np.abs(((angles - dominant_axis + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
    aligned = errors <= math.radians(tolerance_degrees)
    return float(np.sum(lengths[aligned]) / max(1e-6, float(np.sum(lengths))))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _unchanged_result(
    mask: np.ndarray,
    probabilities: np.ndarray,
    *,
    reason: str,
    config: SourceNativeProposalConfig,
) -> SourceNativeProposalResult:
    return SourceNativeProposalResult(
        mask=np.ascontiguousarray(mask, dtype=bool),
        probabilities=np.ascontiguousarray(probabilities, dtype=np.float32),
        diagnostics={
            "source_native_proposals": True,
            "reason": reason,
            "candidate_count": 0,
            "accepted_count": 0,
            "accepted_expansion_count": 0,
            "accepted_contraction_count": 0,
            "accepted_viewport_expansion_count": 0,
            "changed_pixel_count": 0,
            "expanded_pixel_count": 0,
            "contracted_pixel_count": 0,
            "maximum_depth_px": 0.0,
            "mean_support_fraction": 0.0,
            "minimum_support_fraction": 0.0,
            "mean_proposal_confidence": 0.0,
            "minimum_proposal_confidence": 0.0,
            "mean_terminal_contrast": 0.0,
            "minimum_terminal_contrast": 0.0,
            "mean_terminal_support_fraction": 0.0,
            "minimum_terminal_support_fraction": 0.0,
            "rejected_topology_count": 0,
            "rejected_area_count": 0,
            "rejected_seed_count": 0,
            "rejected_energy_count": 0,
            "topology_preserved": True,
            "original_boundary_energy": 0.0,
            "proposed_boundary_energy": 0.0,
            "boundary_energy_ratio": 1.0,
            "evaluated_boundary_energy": 0.0,
            "evaluated_boundary_energy_ratio": 1.0,
            "evaluated_boundary_energy_drop": 0.0,
            "deep_thin_expansion_energy_exception": {
                "enabled": bool(
                    config.deep_thin_expansion_energy_exception_enabled
                ),
                "eligible": False,
                "applied": False,
                "reason": "not-evaluated",
                "global_energy_veto_triggered": False,
            },
            "shortcut_suppression_recommended": False,
        },
    )


__all__ = [
    "SourceNativeProposalConfig",
    "SourceNativeProposalResult",
    "recover_source_native_rectilinear_proposals",
]
