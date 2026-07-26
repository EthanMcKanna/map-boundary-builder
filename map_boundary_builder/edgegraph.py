"""Source-native contour localization and vector reconstruction for v20.

The global selector is semantic only.  Its rings are resampled in source-pixel
coordinates and every sample receives a normal profile with these channels:

The internal native strip retains the original seven-channel diagnostic
contract (RGB, selector probability, luminance, and Scharr fields). Runtime
adapts it by ONNX input width: legacy seven- and nine-channel models keep their
original tensors, while vector-v3 receives ten channels and never receives raw
selector probability.

Normal offsets are ordered from negative to positive and use the contour's left
normal ``(-dy, +dx)``.  The default 49 bins cover ``[-12, +12]`` source pixels
at 0.5px spacing.  EdgeStripNet predicts an offset distribution plus optional
corner and reliability logits.  A cyclic decoder makes the offsets globally
consistent before line/curve-aware vector reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Protocol

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .angle_lattice import snap_geometry_to_angle_lattice
from .edge_phase import estimate_edge_phase
from .edgegraph_proposal import recover_source_native_rectilinear_proposals
from .evaluation import geometry_corner_f1
from .extract import (
    AUTO_FILL_STYLE,
    ExtractionHints,
    extraction_confidence,
    orient_pixel_polygon,
)
from .rectilinear_fit import fit_rectilinear_ring


class EdgeGraphSessionLike(Protocol):
    def get_inputs(self) -> list[Any]: ...

    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class EdgeGraphConfig:
    coarse_threshold: float = 0.45
    strip_radius_px: float = 12.0
    strip_step_px: float = 0.5
    contour_sample_step_px: float = 2.0
    smoothness_weight: float = 0.18
    maximum_bin_jump: int = 5
    source_native_proposals_enabled: bool = True
    wide_recenter_enabled: bool = False
    wide_recenter_max_radius_px: float = 48.0
    wide_recenter_step_px: float = 1.0
    wide_recenter_radius_per_selector_scale: float = 16.0
    wide_recenter_minimum_move_px: float = 8.0
    wide_recenter_minimum_evidence_gain: float = 0.10
    wide_recenter_maximum_center_evidence: float = 0.80
    wide_recenter_minimum_run_samples: int = 6
    wide_recenter_smoothness_weight: float = 0.10
    wide_recenter_maximum_bin_jump: int = 8
    # The selector/proposal lane owns semantic structure. A chord contraction
    # is eligible only when an independently learned source-native appearance
    # map classifies the removed region as outside the selected overlay.
    # Geometry/topology and chord-edge support alone are insufficient: an
    # internal road can otherwise erase a real thin branch.
    shortcut_enabled: bool = True
    shortcut_minimum_native_evidence_coverage: float = 0.90
    shortcut_maximum_removed_inside_mean: float = 0.60
    shortcut_maximum_removed_inside_p90: float = 0.70
    shortcut_anchor_tolerance_px: float = 6.0
    shortcut_minimum_chord_length_px: float = 24.0
    shortcut_minimum_detour_ratio: float = 1.12
    shortcut_minimum_depth_px: float = 10.0
    shortcut_maximum_depth_fraction: float = 0.16
    shortcut_maximum_anchor_span: int = 14
    shortcut_minimum_support_fraction: float = 0.90
    shortcut_minimum_target_separation: float = 0.035
    shortcut_minimum_edge_contrast: float = 0.030
    shortcut_alignment_radius_px: float = 12.0
    shortcut_alignment_step_px: float = 0.5
    flat_fill_recovery_enabled: bool = True
    flat_fill_maximum_reliability: float = 0.75
    flat_fill_training_margin_px: float = 10.0
    flat_fill_minimum_target_separation: float = 8.0
    flat_fill_maximum_target_radius: float = 16.0
    flat_fill_closing_fraction: float = 0.08
    flat_fill_maximum_closing_kernel_px: int = 61
    flat_fill_maximum_closed_addition_fraction: float = 0.08
    flat_fill_minimum_candidate_iou: float = 0.99
    flat_fill_minimum_candidate_area_ratio: float = 0.985
    flat_fill_maximum_candidate_area_ratio: float = 1.015
    flat_fill_maximum_boundary_median_drop: float = 0.005
    flat_fill_minimum_boundary_mean_gain: float = 0.015
    flat_fill_vector_phase_px: float = 0.75
    flat_fill_minimum_vector_iou: float = 0.985
    flat_fill_minimum_vector_area_ratio: float = 0.98
    flat_fill_maximum_vector_area_ratio: float = 1.02
    flat_fill_structural_maximum_reliability: float = 0.50
    flat_fill_structural_minimum_target_separation: float = 12.0
    flat_fill_structural_maximum_closed_addition_fraction: float = 0.06
    flat_fill_structural_minimum_candidate_iou: float = 0.95
    flat_fill_structural_minimum_candidate_area_ratio: float = 0.95
    flat_fill_structural_maximum_candidate_area_ratio: float = 1.04
    flat_fill_structural_minimum_boundary_median_gain: float = 0.015
    flat_fill_structural_minimum_boundary_mean_gain: float = 0.035
    flat_fill_structural_minimum_vector_iou: float = 0.95
    flat_fill_structural_minimum_vector_area_ratio: float = 0.95
    flat_fill_structural_maximum_vector_area_ratio: float = 1.04
    graphcut_recovery_enabled: bool = True
    graphcut_minimum_maximum_displacement_px: float = 11.5
    graphcut_sure_foreground_margin_px: float = 24.0
    graphcut_sure_background_margin_px: float = 72.0
    graphcut_minimum_candidate_iou: float = 0.95
    graphcut_minimum_candidate_area_ratio: float = 0.95
    graphcut_maximum_candidate_area_ratio: float = 1.05
    graphcut_maximum_boundary_median_drop: float = 0.005
    graphcut_minimum_boundary_mean_gain: float = 0.05
    graphcut_minimum_vector_iou: float = 0.99
    graphcut_minimum_vector_area_ratio: float = 0.98
    graphcut_maximum_vector_area_ratio: float = 1.02
    # A bounded GraphCut may recover local structure outside the refiner's
    # normal-strip support, but its raster contour is the outside edge of the
    # rendered stroke.  This lane estimates the native paired-edge center and
    # phases the vector inward before it may replace current geometry.
    graphcut_centered_recovery_enabled: bool = True
    graphcut_centered_maximum_current_reliability: float = 0.95
    graphcut_centered_minimum_candidate_iou: float = 0.98
    graphcut_centered_minimum_candidate_area_ratio: float = 0.98
    graphcut_centered_maximum_candidate_area_ratio: float = 1.02
    graphcut_centered_maximum_boundary_median_drop: float = 0.01
    graphcut_centered_minimum_boundary_mean_gain: float = 0.10
    graphcut_centered_minimum_vector_iou: float = 0.995
    graphcut_centered_minimum_vector_area_ratio: float = 0.99
    graphcut_centered_maximum_vector_area_ratio: float = 1.01
    graphcut_centered_minimum_paired_fraction: float = 0.75
    graphcut_centered_minimum_reliable_paired_fraction: float = 0.60
    graphcut_centered_minimum_coherent_paired_fraction: float = 0.55
    graphcut_centered_minimum_phase_reliability_p10: float = 0.55
    graphcut_centered_minimum_positive_center_fraction: float = 0.90
    graphcut_centered_positive_center_threshold_px: float = 0.25
    graphcut_centered_coherent_center_tolerance_px: float = 0.75
    graphcut_centered_minimum_center_px: float = 0.45
    graphcut_centered_maximum_center_px: float = 1.25
    graphcut_centered_maximum_center_mad_px: float = 0.30
    graphcut_centered_maximum_center_p90_deviation_px: float = 0.65
    graphcut_centered_minimum_stroke_width_px: float = 1.75
    graphcut_centered_maximum_stroke_width_px: float = 3.50
    graphcut_centered_maximum_stroke_width_mad_px: float = 0.35
    graphcut_centered_maximum_stroke_width_p90_deviation_px: float = 0.85
    graphcut_centered_minimum_center_to_half_width_ratio: float = 0.25
    graphcut_centered_maximum_center_to_half_width_ratio: float = 0.90
    graphcut_centered_minimum_phased_current_iou: float = 0.99
    graphcut_centered_minimum_phased_current_area_ratio: float = 0.985
    graphcut_centered_maximum_phased_current_area_ratio: float = 1.015
    graphcut_centered_maximum_vector_current_vertex_ratio: float = 1.0
    graphcut_centered_maximum_phased_current_vertex_ratio: float = 1.0
    graphcut_centered_maximum_vector_vertex_count: int = 48
    graphcut_centered_maximum_vector_vertex_density: float = 0.012
    graphcut_centered_minimum_structural_p99_px: float = 1.40
    graphcut_centered_minimum_structural_max_px: float = 2.0
    # High-reliability geometry is normally complete.  It is eligible only
    # when an independent phase probe on the current vector observes a
    # coherent inward correction.  These gates are intentionally stricter
    # than the candidate-contour phase gate and are mutually exclusive with
    # the low-reliability structural route above.
    graphcut_current_phase_minimum_paired_fraction: float = 0.80
    graphcut_current_phase_minimum_reliable_paired_fraction: float = 0.60
    graphcut_current_phase_minimum_coherent_paired_fraction: float = 0.55
    graphcut_current_phase_minimum_positive_center_fraction: float = 0.75
    graphcut_current_phase_minimum_center_px: float = 0.35
    graphcut_current_phase_maximum_center_px: float = 0.65
    graphcut_current_phase_maximum_center_p90_deviation_px: float = 0.90
    graphcut_current_phase_sample_step_px: float = 7.0
    # Structural recovery is a separate ownership layer from the learned
    # normal-strip localizer.  A bounded source-native raster may own missing
    # structure only when a multiscale line graph is stable, source-supported,
    # topology-preserving, and economical per removed edge.
    multiscale_line_graph_enabled: bool = True
    multiscale_line_graph_minimum_candidate_iou: float = 0.99
    multiscale_line_graph_minimum_candidate_area_ratio: float = 0.98
    multiscale_line_graph_maximum_candidate_area_ratio: float = 1.02
    multiscale_line_graph_maximum_boundary_median_drop: float = 0.01
    multiscale_line_graph_minimum_boundary_mean_gain: float = 0.015
    multiscale_line_graph_minimum_structural_p99_px: float = 2.0
    multiscale_line_graph_minimum_structural_max_px: float = 3.0
    multiscale_line_graph_maximum_default_vertex_ratio: float = 1.55
    multiscale_line_graph_maximum_tiny_hole_area_px: float = 64.0
    multiscale_line_graph_maximum_tiny_hole_area_fraction: float = 0.0005
    multiscale_line_graph_minimum_source_iou: float = 0.995
    multiscale_line_graph_minimum_source_area_ratio: float = 0.99
    multiscale_line_graph_maximum_source_area_ratio: float = 1.01
    multiscale_line_graph_minimum_output_current_iou: float = 0.99
    multiscale_line_graph_minimum_output_area_ratio: float = 0.98
    multiscale_line_graph_maximum_output_area_ratio: float = 1.02
    multiscale_line_graph_minimum_base_boundary_mean_gain: float = 0.01
    multiscale_line_graph_maximum_vertex_count: int = 48
    multiscale_line_graph_maximum_vertex_ratio: float = 1.55
    multiscale_line_graph_maximum_added_vertices: int = 4
    multiscale_line_graph_minimum_corner_agreement_f1: float = 0.60
    multiscale_line_graph_minimum_corner_agreement_recall: float = 0.72
    multiscale_line_graph_maximum_corner_agreement_p95_degrees: float = 18.0
    multiscale_line_graph_maximum_source_loss: float = 0.00020
    multiscale_line_graph_maximum_source_loss_per_removed_vertex: float = 0.00004
    multiscale_line_graph_phase_minimum_center_px: float = 0.35
    multiscale_line_graph_phase_maximum_center_px: float = 1.50
    multiscale_line_graph_phase_minimum_paired_fraction: float = 0.45
    multiscale_line_graph_phase_minimum_reliable_fraction: float = 0.35
    multiscale_line_graph_phase_minimum_coherent_fraction: float = 0.25
    multiscale_line_graph_phase_minimum_positive_fraction: float = 0.60
    multiscale_line_graph_phase_maximum_p90_deviation_px: float = 3.10
    multiscale_line_graph_phase_maximum_stroke_width_px: float = 4.50
    multiscale_line_graph_lattice_skip_p99_px: float = 4.0
    multiscale_line_graph_lattice_skip_max_px: float = 5.0
    # The final geometry-only lattice is intentionally near-identity.  It can
    # sharpen long runs and infer a short perpendicular connector, but cannot
    # repair semantic structure or materially move the raster boundary.
    angle_lattice_enabled: bool = True
    angle_lattice_simplification_tolerance_px: float = 0.75
    angle_lattice_minimum_segment_length_px: float = 8.0
    angle_lattice_maximum_segment_residual_degrees: float = 5.0
    angle_lattice_maximum_p90_residual_degrees: float = 1.0
    angle_lattice_maximum_mean_residual_degrees: float = 2.0
    angle_lattice_maximum_vertex_displacement_px: float = 3.5
    angle_lattice_minimum_current_iou: float = 0.992
    angle_lattice_maximum_boundary_median_drop: float = 0.005
    angle_lattice_maximum_boundary_mean_drop: float = 0.01
    angle_lattice_minimum_material_p90_residual_degrees: float = 0.15
    angle_lattice_minimum_material_displacement_px: float = 0.25
    target_semantic_logit_scale: float = 6.0
    vector_v2_learned_localization_owner: bool = True
    vector_v3_learned_localization_owner: bool = True
    edge_phase_logit_weight: float = 0.1
    edge_phase_minimum_reliability: float = 0.65
    edge_phase_unpaired_agreement_px: float = 1.0
    line_alignment_radius_px: float = 0.0
    line_alignment_step_px: float = 0.25
    line_alignment_minimum_support_fraction: float = 0.55
    line_alignment_minimum_score_gain: float = 0.015
    corner_probability_threshold: float = 0.78
    corner_anchor_minimum_reliability: float = 0.75
    learned_corner_anchors_enabled: bool = False
    reliability_threshold: float = 0.30
    line_anchor_tolerance_px: float = 5.00
    line_fit_tolerance_px: float = 2.25
    line_fit_inlier_quantile: float = 0.90
    curve_tolerance_px: float = 1.00
    minimum_line_length_px: float = 8.0
    maximum_intersection_displacement_px: float = 12.0
    maximum_corner_connector_length_px: float = 12.0
    maximum_corner_connector_neighbor_ratio: float = 0.10
    minimum_corner_connector_angle_degrees: float = 15.0
    minimum_reconstructed_corner_angle_degrees: float = 20.0
    rectilinear_fit_enabled: bool = True
    rectilinear_native_phase_enabled: bool = True
    rectilinear_native_phase_radius_px: float = 4.0
    rectilinear_native_phase_step_px: float = 0.5
    rectilinear_native_phase_sample_step_px: float = 4.0
    rectilinear_native_phase_minimum_line_length_px: float = 8.0
    rectilinear_native_phase_minimum_shift_px: float = 0.85
    rectilinear_native_phase_maximum_shift_px: float = 2.5
    rectilinear_native_phase_maximum_edge_disagreement_px: float = 0.65
    rectilinear_native_phase_minimum_proposal_coverage: float = 0.40
    rectilinear_native_phase_minimum_validation_coverage: float = 0.30
    rectilinear_native_phase_minimum_validation_gain: float = 0.20
    rectilinear_native_phase_minimum_stroke_width_px: float = 1.25
    rectilinear_native_phase_maximum_stroke_width_mad_px: float = 0.75
    rectilinear_native_phase_minimum_current_iou: float = 0.99
    rectilinear_native_phase_minimum_area_ratio: float = 0.985
    rectilinear_native_phase_maximum_area_ratio: float = 1.015
    rectilinear_connector_enabled: bool = False
    rectilinear_minimum_aligned_length_fraction: float = 0.72
    rectilinear_alignment_tolerance_degrees: float = 8.0
    rectilinear_connector_maximum_length_px: float = 30.0
    rectilinear_connector_neighbor_ratio: float = 0.55
    rectilinear_connector_minimum_axis_error_degrees: float = 12.0
    rectilinear_connector_minimum_corner_angle_degrees: float = 70.0
    rectilinear_connector_minimum_corner_response: float = 0.12
    inference_chunk_length: int = 1024
    inference_context: int = 48
    minimum_component_area_px: float = 24.0
    topology_guard: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.coarse_threshold < 1.0:
            raise ValueError("coarse_threshold must be between 0 and 1")
        if self.strip_radius_px <= 0.0 or self.strip_step_px <= 0.0:
            raise ValueError("strip radius and step must be positive")
        width = (2.0 * self.strip_radius_px) / self.strip_step_px
        if abs(width - round(width)) > 1e-6:
            raise ValueError("strip radius must be divisible by strip step")
        if self.contour_sample_step_px <= 0.0:
            raise ValueError("contour sample step must be positive")
        if self.maximum_bin_jump < 1:
            raise ValueError("maximum_bin_jump must be positive")
        if self.wide_recenter_max_radius_px < self.strip_radius_px:
            raise ValueError("wide recenter radius must cover the fine strip")
        if self.wide_recenter_step_px <= 0.0:
            raise ValueError("wide recenter step must be positive")
        if self.wide_recenter_radius_per_selector_scale <= 0.0:
            raise ValueError("wide recenter scale must be positive")
        if self.wide_recenter_minimum_run_samples < 2:
            raise ValueError("wide recenter runs require at least two samples")
        if self.wide_recenter_maximum_center_evidence <= 0.0:
            raise ValueError("wide recenter center-evidence ceiling must be positive")
        if self.wide_recenter_maximum_bin_jump < 1:
            raise ValueError("wide recenter maximum bin jump must be positive")
        if self.shortcut_anchor_tolerance_px <= 0.0:
            raise ValueError("shortcut anchor tolerance must be positive")
        if self.shortcut_minimum_chord_length_px <= 0.0:
            raise ValueError("shortcut chord length must be positive")
        if self.shortcut_minimum_detour_ratio <= 1.0:
            raise ValueError("shortcut detour ratio must exceed one")
        if self.shortcut_minimum_depth_px <= 0.0:
            raise ValueError("shortcut depth must be positive")
        if not 0.0 < self.shortcut_maximum_depth_fraction <= 0.5:
            raise ValueError("shortcut maximum depth fraction must be in (0, 0.5]")
        if self.shortcut_maximum_anchor_span < 2:
            raise ValueError("shortcut anchor span must cover at least two runs")
        if self.shortcut_alignment_radius_px < 0.0 or self.shortcut_alignment_step_px <= 0.0:
            raise ValueError("shortcut alignment radius and step must be non-negative/positive")
        for name in (
            "shortcut_minimum_native_evidence_coverage",
            "shortcut_maximum_removed_inside_mean",
            "shortcut_maximum_removed_inside_p90",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.flat_fill_maximum_reliability <= 1.0:
            raise ValueError("flat-fill maximum reliability must be in [0, 1]")
        if self.flat_fill_training_margin_px < 2.0:
            raise ValueError("flat-fill training margin must be at least two pixels")
        if (
            self.flat_fill_minimum_target_separation <= 0.0
            or self.flat_fill_maximum_target_radius <= 0.0
        ):
            raise ValueError("flat-fill target separation and radius must be positive")
        if not 0.0 < self.flat_fill_closing_fraction <= 0.25:
            raise ValueError("flat-fill closing fraction must be in (0, 0.25]")
        if (
            self.flat_fill_maximum_closing_kernel_px < 3
            or self.flat_fill_maximum_closing_kernel_px % 2 == 0
        ):
            raise ValueError("flat-fill maximum closing kernel must be an odd integer >= 3")
        for name in (
            "flat_fill_maximum_closed_addition_fraction",
            "flat_fill_minimum_candidate_iou",
            "flat_fill_minimum_candidate_area_ratio",
            "flat_fill_maximum_candidate_area_ratio",
            "flat_fill_minimum_vector_iou",
            "flat_fill_minimum_vector_area_ratio",
            "flat_fill_maximum_vector_area_ratio",
            "flat_fill_structural_maximum_reliability",
            "flat_fill_structural_maximum_closed_addition_fraction",
            "flat_fill_structural_minimum_candidate_iou",
            "flat_fill_structural_minimum_candidate_area_ratio",
            "flat_fill_structural_maximum_candidate_area_ratio",
            "flat_fill_structural_minimum_vector_iou",
            "flat_fill_structural_minimum_vector_area_ratio",
            "flat_fill_structural_maximum_vector_area_ratio",
            "graphcut_minimum_candidate_iou",
            "graphcut_minimum_candidate_area_ratio",
            "graphcut_maximum_candidate_area_ratio",
            "graphcut_minimum_vector_iou",
            "graphcut_minimum_vector_area_ratio",
            "graphcut_maximum_vector_area_ratio",
            "graphcut_centered_maximum_current_reliability",
            "graphcut_centered_minimum_candidate_iou",
            "graphcut_centered_minimum_candidate_area_ratio",
            "graphcut_centered_maximum_candidate_area_ratio",
            "graphcut_centered_minimum_vector_iou",
            "graphcut_centered_minimum_vector_area_ratio",
            "graphcut_centered_maximum_vector_area_ratio",
            "graphcut_centered_minimum_paired_fraction",
            "graphcut_centered_minimum_reliable_paired_fraction",
            "graphcut_centered_minimum_coherent_paired_fraction",
            "graphcut_centered_minimum_phase_reliability_p10",
            "graphcut_centered_minimum_positive_center_fraction",
            "graphcut_centered_minimum_center_to_half_width_ratio",
            "graphcut_centered_maximum_center_to_half_width_ratio",
            "graphcut_centered_minimum_phased_current_iou",
            "graphcut_centered_minimum_phased_current_area_ratio",
            "graphcut_centered_maximum_phased_current_area_ratio",
            "graphcut_centered_maximum_vector_current_vertex_ratio",
            "graphcut_centered_maximum_phased_current_vertex_ratio",
            "graphcut_current_phase_minimum_paired_fraction",
            "graphcut_current_phase_minimum_reliable_paired_fraction",
            "graphcut_current_phase_minimum_coherent_paired_fraction",
            "graphcut_current_phase_minimum_positive_center_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.1:
                raise ValueError(f"{name} must be in (0, 1.1]")
        if (
            self.flat_fill_minimum_candidate_area_ratio
            > self.flat_fill_maximum_candidate_area_ratio
            or self.flat_fill_minimum_vector_area_ratio
            > self.flat_fill_maximum_vector_area_ratio
            or self.flat_fill_structural_minimum_candidate_area_ratio
            > self.flat_fill_structural_maximum_candidate_area_ratio
            or self.flat_fill_structural_minimum_vector_area_ratio
            > self.flat_fill_structural_maximum_vector_area_ratio
            or self.graphcut_minimum_candidate_area_ratio
            > self.graphcut_maximum_candidate_area_ratio
            or self.graphcut_minimum_vector_area_ratio
            > self.graphcut_maximum_vector_area_ratio
            or self.graphcut_centered_minimum_candidate_area_ratio
            > self.graphcut_centered_maximum_candidate_area_ratio
            or self.graphcut_centered_minimum_vector_area_ratio
            > self.graphcut_centered_maximum_vector_area_ratio
            or self.graphcut_centered_minimum_phased_current_area_ratio
            > self.graphcut_centered_maximum_phased_current_area_ratio
        ):
            raise ValueError("flat-fill minimum area ratios cannot exceed their maxima")
        if (
            self.flat_fill_maximum_boundary_median_drop < 0.0
            or self.flat_fill_minimum_boundary_mean_gain < 0.0
            or self.flat_fill_vector_phase_px < 0.0
            or self.flat_fill_structural_minimum_target_separation < 0.0
            or self.flat_fill_structural_minimum_boundary_median_gain < 0.0
            or self.flat_fill_structural_minimum_boundary_mean_gain < 0.0
            or self.graphcut_minimum_maximum_displacement_px < 0.0
            or self.graphcut_sure_foreground_margin_px <= 0.0
            or self.graphcut_sure_background_margin_px <= 0.0
            or self.graphcut_maximum_boundary_median_drop < 0.0
            or self.graphcut_minimum_boundary_mean_gain < 0.0
            or self.graphcut_centered_maximum_boundary_median_drop < 0.0
            or self.graphcut_centered_minimum_boundary_mean_gain < 0.0
            or self.graphcut_centered_positive_center_threshold_px < 0.0
            or self.graphcut_centered_coherent_center_tolerance_px <= 0.0
            or self.graphcut_centered_minimum_center_px < 0.0
            or self.graphcut_centered_maximum_center_mad_px < 0.0
            or self.graphcut_centered_maximum_center_p90_deviation_px < 0.0
            or self.graphcut_centered_minimum_stroke_width_px < 0.0
            or self.graphcut_centered_maximum_stroke_width_mad_px < 0.0
            or self.graphcut_centered_maximum_stroke_width_p90_deviation_px < 0.0
            or self.graphcut_centered_minimum_structural_p99_px < 0.0
            or self.graphcut_centered_minimum_structural_max_px < 0.0
            or self.graphcut_current_phase_minimum_center_px < 0.0
            or self.graphcut_current_phase_maximum_center_p90_deviation_px < 0.0
            or self.graphcut_current_phase_sample_step_px <= 0.0
        ):
            raise ValueError("flat-fill energy and phase limits must be non-negative")
        if (
            self.graphcut_centered_minimum_center_px
            > self.graphcut_centered_maximum_center_px
            or self.graphcut_centered_minimum_stroke_width_px
            > self.graphcut_centered_maximum_stroke_width_px
            or self.graphcut_centered_minimum_center_to_half_width_ratio
            > self.graphcut_centered_maximum_center_to_half_width_ratio
            or self.graphcut_current_phase_minimum_center_px
            > self.graphcut_current_phase_maximum_center_px
        ):
            raise ValueError("centered GraphCut phase limits are inverted")
        if (
            self.graphcut_centered_maximum_vector_vertex_count < 3
            or self.graphcut_centered_maximum_vector_vertex_density <= 0.0
        ):
            raise ValueError("centered GraphCut complexity limits are invalid")
        for name in (
            "multiscale_line_graph_minimum_candidate_iou",
            "multiscale_line_graph_minimum_candidate_area_ratio",
            "multiscale_line_graph_maximum_candidate_area_ratio",
            "multiscale_line_graph_maximum_tiny_hole_area_fraction",
            "multiscale_line_graph_minimum_source_iou",
            "multiscale_line_graph_minimum_source_area_ratio",
            "multiscale_line_graph_maximum_source_area_ratio",
            "multiscale_line_graph_minimum_output_current_iou",
            "multiscale_line_graph_minimum_output_area_ratio",
            "multiscale_line_graph_maximum_output_area_ratio",
            "multiscale_line_graph_minimum_corner_agreement_f1",
            "multiscale_line_graph_minimum_corner_agreement_recall",
            "angle_lattice_minimum_current_iou",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.1:
                raise ValueError(f"{name} must be in (0, 1.1]")
        if (
            self.multiscale_line_graph_minimum_candidate_area_ratio
            > self.multiscale_line_graph_maximum_candidate_area_ratio
            or self.multiscale_line_graph_minimum_source_area_ratio
            > self.multiscale_line_graph_maximum_source_area_ratio
            or self.multiscale_line_graph_minimum_output_area_ratio
            > self.multiscale_line_graph_maximum_output_area_ratio
        ):
            raise ValueError("multiscale line-graph area limits are inverted")
        if (
            self.multiscale_line_graph_maximum_default_vertex_ratio < 1.0
            or self.multiscale_line_graph_maximum_vertex_ratio < 1.0
            or self.multiscale_line_graph_maximum_vertex_count < 3
            or self.multiscale_line_graph_maximum_added_vertices < 0
        ):
            raise ValueError("multiscale line-graph complexity limits are invalid")
        for name in (
            "multiscale_line_graph_maximum_boundary_median_drop",
            "multiscale_line_graph_minimum_boundary_mean_gain",
            "multiscale_line_graph_minimum_structural_p99_px",
            "multiscale_line_graph_minimum_structural_max_px",
            "multiscale_line_graph_maximum_tiny_hole_area_px",
            "multiscale_line_graph_minimum_base_boundary_mean_gain",
            "multiscale_line_graph_maximum_corner_agreement_p95_degrees",
            "multiscale_line_graph_maximum_source_loss",
            "multiscale_line_graph_maximum_source_loss_per_removed_vertex",
            "multiscale_line_graph_phase_minimum_center_px",
            "multiscale_line_graph_phase_maximum_center_px",
            "multiscale_line_graph_phase_maximum_p90_deviation_px",
            "multiscale_line_graph_phase_maximum_stroke_width_px",
            "multiscale_line_graph_lattice_skip_p99_px",
            "multiscale_line_graph_lattice_skip_max_px",
            "angle_lattice_simplification_tolerance_px",
            "angle_lattice_maximum_p90_residual_degrees",
            "angle_lattice_maximum_mean_residual_degrees",
            "angle_lattice_maximum_segment_residual_degrees",
            "angle_lattice_maximum_boundary_median_drop",
            "angle_lattice_maximum_boundary_mean_drop",
            "angle_lattice_minimum_material_p90_residual_degrees",
            "angle_lattice_minimum_material_displacement_px",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.multiscale_line_graph_phase_minimum_center_px
            > self.multiscale_line_graph_phase_maximum_center_px
            or self.angle_lattice_minimum_segment_length_px <= 0.0
            or self.angle_lattice_maximum_vertex_displacement_px <= 0.0
        ):
            raise ValueError("line-graph phase or lattice limits are invalid")
        if self.target_semantic_logit_scale < 0.0:
            raise ValueError("target semantic logit scale must be non-negative")
        if self.edge_phase_logit_weight < 0.0:
            raise ValueError("edge phase logit weight must be non-negative")
        if not 0.0 <= self.edge_phase_minimum_reliability <= 1.0:
            raise ValueError("edge phase minimum reliability must be in [0, 1]")
        if not 0.0 <= self.corner_anchor_minimum_reliability <= 1.0:
            raise ValueError("corner anchor reliability must be in [0, 1]")
        if self.edge_phase_unpaired_agreement_px < 0.0:
            raise ValueError("edge phase agreement tolerance must be non-negative")
        if self.line_alignment_radius_px < 0.0 or self.line_alignment_step_px <= 0.0:
            raise ValueError("line alignment radius and step must be non-negative/positive")
        if not 0.5 <= self.line_fit_inlier_quantile <= 1.0:
            raise ValueError("line fit inlier quantile must be between 0.5 and 1")
        if not 0.0 < self.maximum_corner_connector_neighbor_ratio <= 1.0:
            raise ValueError("corner connector neighbor ratio must be between 0 and 1")
        if self.maximum_intersection_displacement_px <= 0.0:
            raise ValueError("maximum intersection displacement must be positive")
        if self.maximum_corner_connector_length_px <= 0.0:
            raise ValueError("maximum corner connector length must be positive")
        if not 0.0 <= self.minimum_corner_connector_angle_degrees <= 90.0:
            raise ValueError("corner connector angle must be between 0 and 90")
        if not 0.0 < self.rectilinear_minimum_aligned_length_fraction <= 1.0:
            raise ValueError("rectilinear aligned-length fraction must be in (0, 1]")
        if (
            self.rectilinear_native_phase_radius_px <= 0.0
            or self.rectilinear_native_phase_step_px <= 0.0
            or self.rectilinear_native_phase_sample_step_px <= 0.0
            or self.rectilinear_native_phase_minimum_line_length_px <= 0.0
        ):
            raise ValueError(
                "rectilinear native phase radii, steps, and line length "
                "must be positive"
            )
        if (
            self.rectilinear_native_phase_minimum_shift_px < 0.0
            or self.rectilinear_native_phase_maximum_shift_px
            < self.rectilinear_native_phase_minimum_shift_px
            or self.rectilinear_native_phase_maximum_edge_disagreement_px < 0.0
            or self.rectilinear_native_phase_minimum_validation_gain < 0.0
            or self.rectilinear_native_phase_minimum_stroke_width_px < 0.0
            or self.rectilinear_native_phase_maximum_stroke_width_mad_px < 0.0
        ):
            raise ValueError(
                "rectilinear native phase displacement and evidence limits "
                "are invalid"
            )
        for name in (
            "rectilinear_native_phase_minimum_proposal_coverage",
            "rectilinear_native_phase_minimum_validation_coverage",
            "rectilinear_native_phase_minimum_current_iou",
            "rectilinear_native_phase_minimum_area_ratio",
            "rectilinear_native_phase_maximum_area_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.1:
                raise ValueError(f"{name} must be in [0, 1.1]")
        if (
            self.rectilinear_native_phase_minimum_area_ratio
            > self.rectilinear_native_phase_maximum_area_ratio
        ):
            raise ValueError("rectilinear native phase area ratio limits are inverted")
        if not 0.0 < self.rectilinear_connector_neighbor_ratio <= 1.0:
            raise ValueError("rectilinear connector neighbor ratio must be in (0, 1]")
        if not 0.0 <= self.minimum_reconstructed_corner_angle_degrees <= 90.0:
            raise ValueError("reconstructed corner angle must be between 0 and 90")


@dataclass(frozen=True)
class EdgeGraphResult:
    mask: np.ndarray
    pixel_geometry: Polygon | MultiPolygon
    contour_count: int
    confidence: float
    diagnostics: dict[str, object]


def automatic_edgegraph_hints(
    rgb: np.ndarray,
    coarse_probabilities: np.ndarray,
    *,
    threshold: float = 0.45,
) -> tuple[ExtractionHints, dict[str, object]]:
    """Derive oracle-free native guidance from the semantic selector itself.

    Automatic v20 must not depend on the heuristic color extractor to decide
    which region is semantic foreground.  The selector owns that decision.
    This helper chooses its strongest connected component, places the seed at
    the deepest supported interior pixel, and estimates the rendered overlay
    appearance from pixels safely inside that component.
    """

    image = np.asarray(rgb)
    coarse = np.asarray(coarse_probabilities, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must have shape (height, width, 3), got {image.shape}")
    if coarse.shape != image.shape[:2]:
        raise ValueError(
            "coarse probabilities must match the source image, got "
            f"{coarse.shape} and {image.shape[:2]}"
        )
    if not np.isfinite(coarse).all():
        raise ValueError("coarse probabilities must be finite")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("selector threshold must be between zero and one")

    cleaned = _clean_selector_mask(coarse >= float(threshold))
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        raise ValueError("EdgeGraph global selector did not identify a service area")

    component_scores = np.asarray(
        [
            float(np.sum(coarse[labels == label], dtype=np.float64))
            for label in range(1, component_count)
        ],
        dtype=np.float64,
    )
    selected_label = int(np.argmax(component_scores)) + 1
    selected = labels == selected_label
    distance = cv2.distanceTransform(
        selected.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    seed_y, seed_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    maximum_depth = float(distance[seed_y, seed_x])
    interior_floor = max(2.0, min(10.0, maximum_depth * 0.25))
    interior = selected & (distance >= interior_floor)
    if not interior.any():
        interior = selected
    target = tuple(
        int(round(float(value)))
        for value in np.median(image[interior].astype(np.float32), axis=0)
    )

    stability: dict[str, float] = {}
    for offset in (-0.05, 0.05):
        alternate = _clean_selector_mask(
            coarse >= float(np.clip(threshold + offset, 0.01, 0.99))
        )
        alternate = _select_guided_mask(
            alternate,
            ExtractionHints(seed_point=(float(seed_x), float(seed_y))),
        )
        intersection = int(np.count_nonzero(selected & alternate))
        union = int(np.count_nonzero(selected | alternate))
        stability[f"{threshold + offset:.2f}"] = (
            round(intersection / union, 6) if union else 1.0
        )

    hints = ExtractionHints(
        seed_point=(float(seed_x), float(seed_y)),
        target_rgb=target,
    )
    diagnostics: dict[str, object] = {
        "route": "selector-bootstrap-v1",
        "selector_threshold": float(threshold),
        "selector_component_count": int(component_count - 1),
        "selected_component_area_px": int(stats[selected_label, cv2.CC_STAT_AREA]),
        "selected_component_coverage": round(float(selected.mean()), 6),
        "selected_component_score": round(float(component_scores[selected_label - 1]), 6),
        "seed_depth_px": round(maximum_depth, 6),
        "target_sample_count": int(np.count_nonzero(interior)),
        "threshold_stability_iou": stability,
    }
    return hints, diagnostics


@dataclass(frozen=True)
class NormalStrip:
    features: np.ndarray
    points: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True)
class _NativeImageFields:
    rgb_float: np.ndarray
    luminance: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    magnitude: np.ndarray
    native_inside_probability: np.ndarray | None = None


@dataclass(frozen=True)
class _RectilinearPhaseEvidence:
    sample_count: int
    paired_count: int
    shift_px: float
    coverage: float
    reliability: float
    stroke_width_px: float
    score_gain: float


@dataclass(frozen=True)
class _RectilinearNativeAlignment:
    candidate: Polygon | None
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _RefinedRing:
    coordinates: list[tuple[float, float]]
    sample_count: int
    corner_count: int
    line_run_count: int
    mean_reliability: float
    mean_entropy: float
    mean_displacement_px: float
    maximum_displacement_px: float
    wide_recenter_count: int
    wide_recenter_maximum_px: float
    shortcut_count: int
    shortcut_maximum_depth_px: float
    phase_reliability: float
    phase_paired_fraction: float
    phase_applied_fraction: float
    phase_mean_stroke_width_px: float
    rectilinear_fit_accepted: bool
    rectilinear_fit_reason: str
    rectilinear_fit_run_count: int
    rectilinear_fit_support_p95_px: float | None
    rectilinear_native_alignment_diagnostics: dict[str, object] | None = None
    shortcut_native_diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class _TargetFillReconstruction:
    geometry: Polygon | MultiPolygon | None
    mask: np.ndarray | None
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _LineGraphCandidate:
    key: str
    geometry: Polygon
    mask: np.ndarray
    source_iou: float
    source_area_ratio: float
    current_iou: float
    current_area_ratio: float
    boundary_energy_median: float
    boundary_energy_mean: float
    vertex_count: int
    corner_agreement: dict[str, float | int]
    line_count: int


def refine_boundary_with_edgegraph(
    rgb: np.ndarray,
    coarse_probabilities: np.ndarray,
    session: EdgeGraphSessionLike | None = None,
    *,
    hints: Any = None,
    config: EdgeGraphConfig | None = None,
) -> EdgeGraphResult:
    cfg = config or EdgeGraphConfig()
    requested_cfg = cfg
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (height, width, 3), got {rgb.shape}")
    if coarse_probabilities.shape != rgb.shape[:2]:
        raise ValueError(
            "coarse probabilities must match the source image, got "
            f"{coarse_probabilities.shape} and {rgb.shape[:2]}"
        )
    coarse = np.asarray(coarse_probabilities, dtype=np.float32)
    selector_coarse = coarse
    coarse_mask = _clean_selector_mask(coarse >= cfg.coarse_threshold)
    coarse_mask = _select_guided_mask(coarse_mask, hints)
    if not coarse_mask.any():
        raise ValueError("EdgeGraph global selector did not identify a service area")

    proposal_diagnostics: dict[str, object] = {
        "source_native_proposals": False,
        "reason": "disabled-or-unguided",
        "accepted_count": 0,
    }
    proposal_native_inside: np.ndarray | None = None
    target_rgb = (
        hints.get("target_rgb")
        if isinstance(hints, dict)
        else getattr(hints, "target_rgb", None)
    )
    seed_point = (
        hints.get("seed_point")
        if isinstance(hints, dict)
        else getattr(hints, "seed_point", None)
    )
    if cfg.source_native_proposals_enabled and target_rgb is not None:
        proposal = recover_source_native_rectilinear_proposals(
            rgb,
            coarse,
            target_rgb=target_rgb,
            seed_point=seed_point,
            coarse_mask=coarse_mask,
        )
        coarse = proposal.probabilities
        coarse_mask = proposal.mask
        proposal_diagnostics = proposal.diagnostics
        proposal_native_inside = proposal.native_inside_probability
        # Once a source-native structural proposal has supplied the missing
        # branch/notch geometry, the older selector-chord shortcut becomes a
        # competing second topology edit.  In particular it can erase a real
        # narrow stem that the proposal just recovered.  Keep localization and
        # vector fitting active, but make structural recovery single-owner.
        if bool(proposal_diagnostics.get("shortcut_suppression_recommended", False)):
            cfg = replace(cfg, shortcut_enabled=False)

    contours, hierarchy = cv2.findContours(
        coarse_mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    if hierarchy is None:
        raise ValueError("EdgeGraph selector did not produce polygonal topology")
    hierarchy = hierarchy[0]
    height, width = coarse_mask.shape
    minimum_area = max(float(cfg.minimum_component_area_px), height * width * 0.00001)
    native_fields = replace(
        _prepare_native_image_fields(rgb),
        native_inside_probability=proposal_native_inside,
    )
    polygons: list[Polygon] = []
    ring_diagnostics: list[_RefinedRing] = []
    topology_fallbacks = 0

    for index, contour in enumerate(contours):
        if int(hierarchy[index][3]) != -1 or cv2.contourArea(contour) < minimum_area:
            continue
        exterior = _refine_ring(
            rgb,
            coarse,
            contour[:, 0, :],
            session=session,
            hints=hints,
            config=cfg,
            native_fields=native_fields,
        )
        ring_diagnostics.append(exterior)
        holes: list[list[tuple[float, float]]] = []
        child = int(hierarchy[index][2])
        while child != -1:
            if cv2.contourArea(contours[child]) >= max(8.0, minimum_area * 0.20):
                refined_hole = _refine_ring(
                    rgb,
                    coarse,
                    contours[child][:, 0, :],
                    session=session,
                    hints=hints,
                    config=cfg,
                    native_fields=native_fields,
                )
                ring_diagnostics.append(refined_hole)
                holes.append(refined_hole.coordinates)
            child = int(hierarchy[child][0])

        polygon = Polygon(exterior.coordinates, holes)
        coarse_polygon = _coarse_polygon_for_contour(contours, hierarchy, index)
        if not polygon.is_valid or polygon.is_empty or polygon.area < minimum_area:
            topology_fallbacks += 1
            polygon = coarse_polygon
        elif cfg.topology_guard and not _topology_compatible(polygon, coarse_polygon):
            topology_fallbacks += 1
            polygon = coarse_polygon
        if polygon is None or polygon.is_empty:
            continue
        if isinstance(polygon, Polygon):
            polygons.append(orient_pixel_polygon(polygon))
        elif isinstance(polygon, MultiPolygon):
            polygons.extend(orient_pixel_polygon(part) for part in polygon.geoms if part.area >= minimum_area)

    if not polygons:
        raise ValueError("EdgeGraph could not construct a valid service-area polygon")
    geometry = unary_union(polygons)
    proposal_accepted_count = int(
        proposal_diagnostics.get("accepted_count", 0)
        if isinstance(proposal_diagnostics, dict)
        else 0
    )
    if (
        cfg.source_native_proposals_enabled
        and proposal_accepted_count > 0
        and (topology_fallbacks > 0 or not geometry.is_valid)
    ):
        # A structural proposal may suggest a missing branch, but it cannot own
        # the result after its refined geometry fails validity/topology. Retry
        # the original selector proposal once, keeping localization and vector
        # fitting active while removing only the rejected structural edit.
        retry = refine_boundary_with_edgegraph(
            rgb,
            selector_coarse,
            session,
            hints=hints,
            config=replace(
                requested_cfg,
                source_native_proposals_enabled=False,
            ),
        )
        return replace(
            retry,
            diagnostics={
                **retry.diagnostics,
                "source_native_proposal_retry": {
                    "attempted": True,
                    "accepted": True,
                    "reason": (
                        "proposal-union-invalid"
                        if not geometry.is_valid
                        else "proposal-topology-fallback"
                    ),
                    "rejected_proposal": proposal_diagnostics,
                },
            },
        )
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if isinstance(geometry, Polygon):
        geometry = orient_pixel_polygon(geometry)
    elif isinstance(geometry, MultiPolygon):
        geometry = MultiPolygon([orient_pixel_polygon(part) for part in geometry.geoms])
    else:
        raise ValueError("EdgeGraph output did not form polygonal geometry")

    mask = rasterize_geometry_mask(geometry, width=width, height=height)
    if not mask.any():
        raise ValueError("EdgeGraph output rasterized to an empty mask")
    reliabilities = [item.mean_reliability for item in ring_diagnostics]
    entropies = [item.mean_entropy for item in ring_diagnostics]
    target_fill = _recover_source_native_target_fill_geometry(
        rgb,
        current_mask=mask,
        target_rgb=target_rgb,
        seed_point=seed_point,
        proposal_diagnostics=proposal_diagnostics,
        config=cfg,
        native_fields=native_fields,
    )
    flat_fill = _TargetFillReconstruction(
        None,
        None,
        {
            "evaluated": False,
            "accepted": False,
            "reason": "existing-target-fill-accepted",
        },
    )
    if target_fill.geometry is not None and target_fill.mask is not None:
        geometry = target_fill.geometry
        mask = target_fill.mask
    else:
        flat_fill = _recover_source_native_flat_fill_geometry(
            rgb,
            current_mask=mask,
            target_rgb=target_rgb,
            seed_point=seed_point,
            proposal_diagnostics=proposal_diagnostics,
            shortcut_count=sum(item.shortcut_count for item in ring_diagnostics),
            mean_reliability=(
                float(np.mean(reliabilities))
                if reliabilities
                else 0.0
            ),
            config=cfg,
            native_fields=native_fields,
        )
        if flat_fill.geometry is not None and flat_fill.mask is not None:
            geometry = flat_fill.geometry
            mask = flat_fill.mask
    graphcut = _TargetFillReconstruction(
        None,
        None,
        {
            "evaluated": False,
            "accepted": False,
            "reason": "existing-source-native-fill-accepted",
            "mode": "strip-saturation-graphcut",
        },
    )
    if (
        target_fill.geometry is None
        and flat_fill.geometry is None
    ):
        graphcut = _recover_source_native_graphcut_geometry(
            rgb,
            current_mask=mask,
            current_geometry=geometry,
            target_rgb=target_rgb,
            seed_point=seed_point,
            proposal_diagnostics=proposal_diagnostics,
            mean_reliability=(
                float(np.mean(reliabilities))
                if reliabilities
                else 0.0
            ),
            maximum_displacement_px=max(
                (item.maximum_displacement_px for item in ring_diagnostics),
                default=0.0,
            ),
            config=cfg,
            native_fields=native_fields,
        )
        if graphcut.geometry is not None and graphcut.mask is not None:
            geometry = graphcut.geometry
            mask = graphcut.mask
    angle_lattice = _recover_angle_lattice_geometry(
        geometry,
        current_mask=mask,
        config=cfg,
        native_fields=native_fields,
    )
    line_graph = _TargetFillReconstruction(
        None,
        None,
        {
            "evaluated": False,
            "accepted": False,
            "reason": "existing-source-native-owner",
        },
    )
    if (
        target_fill.geometry is None
        and flat_fill.geometry is None
        and graphcut.geometry is None
    ):
        line_graph = _recover_multiscale_line_graph_geometry(
            rgb,
            current_mask=mask,
            current_geometry=geometry,
            target_rgb=target_rgb,
            seed_point=seed_point,
            direct_lattice_accepted=(
                angle_lattice.geometry is not None
                and angle_lattice.mask is not None
            ),
            direct_lattice_diagnostics=angle_lattice.diagnostics,
            config=cfg,
            native_fields=native_fields,
        )
        if line_graph.geometry is not None and line_graph.mask is not None:
            geometry = line_graph.geometry
            mask = line_graph.mask
            # The source-backed graph owns structure.  Re-evaluate the cheap
            # near-identity lattice on that final graph instead of applying a
            # preview built from the superseded learned geometry.
            angle_lattice = _recover_angle_lattice_geometry(
                geometry,
                current_mask=mask,
                config=cfg,
                native_fields=native_fields,
            )
    if angle_lattice.geometry is not None and angle_lattice.mask is not None:
        geometry = angle_lattice.geometry
        mask = angle_lattice.mask
    contour_count = 1 if isinstance(geometry, Polygon) else len(geometry.geoms)
    base_confidence = extraction_confidence(mask, AUTO_FILL_STYLE, contour_count)
    edge_confidence = float(np.mean(reliabilities)) if reliabilities else 0.0
    entropy_confidence = max(0.0, 1.0 - float(np.mean(entropies))) if entropies else 0.0
    geometry_confidence = 1.0 if topology_fallbacks == 0 else max(0.55, 1.0 - topology_fallbacks / len(polygons))
    # Selector confidence remains responsible for semantic correctness; edge
    # confidence describes only the local observability of the chosen ring.
    confidence = min(base_confidence, max(0.60, 0.45 * edge_confidence + 0.35 * entropy_confidence + 0.20 * geometry_confidence))
    rectilinear_ring_diagnostics = [
        {
            "accepted": item.rectilinear_fit_accepted,
            "reason": item.rectilinear_fit_reason,
            "run_count": item.rectilinear_fit_run_count,
            "support_p95_px": (
                round(item.rectilinear_fit_support_p95_px, 6)
                if item.rectilinear_fit_support_p95_px is not None
                else None
            ),
            "native_phase_alignment": (
                item.rectilinear_native_alignment_diagnostics
                or {
                    "evaluated": False,
                    "accepted": False,
                    "reason": "unavailable",
                }
            ),
        }
        for item in ring_diagnostics
    ]
    rectilinear_accepted = [
        item for item in ring_diagnostics if item.rectilinear_fit_accepted
    ]
    rectilinear_attempted_count = sum(
        item.rectilinear_fit_reason != "disabled" for item in ring_diagnostics
    )
    rectilinear_reason_counts: dict[str, int] = {}
    rectilinear_native_phase_reason_counts: dict[str, int] = {}
    rectilinear_native_phase_accepted_count = 0
    rectilinear_native_phase_evaluated_count = 0
    for item in ring_diagnostics:
        reason = item.rectilinear_fit_reason
        rectilinear_reason_counts[reason] = rectilinear_reason_counts.get(reason, 0) + 1
        alignment = item.rectilinear_native_alignment_diagnostics or {}
        alignment_reason = str(alignment.get("reason", "unavailable"))
        rectilinear_native_phase_reason_counts[alignment_reason] = (
            rectilinear_native_phase_reason_counts.get(alignment_reason, 0) + 1
        )
        rectilinear_native_phase_accepted_count += int(
            alignment.get("accepted") is True
        )
        rectilinear_native_phase_evaluated_count += int(
            alignment.get("evaluated") is True
        )
    rectilinear_enabled = cfg.rectilinear_fit_enabled
    if not rectilinear_enabled:
        rectilinear_result = "disabled"
    elif len(rectilinear_accepted) == len(ring_diagnostics):
        rectilinear_result = "accepted"
    elif rectilinear_accepted:
        rectilinear_result = "partial"
    else:
        rectilinear_result = "fallback"
    session_channels = _session_input_channel_count(session) if session is not None else None
    localization_fusion = _localization_fusion(
        session_channels,
        session_present=session is not None,
        config=cfg,
    )
    return EdgeGraphResult(
        mask=mask,
        pixel_geometry=geometry,
        contour_count=contour_count,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        diagnostics={
            "edgegraph": True,
            "source_native": True,
            "selector_geometry_exported": False,
            "source_native_proposal": proposal_diagnostics,
            "source_native_target_fill": target_fill.diagnostics,
            "source_native_flat_fill": flat_fill.diagnostics,
            "source_native_graphcut": graphcut.diagnostics,
            "source_native_multiscale_line_graph": line_graph.diagnostics,
            "angle_lattice": angle_lattice.diagnostics,
            "refiner_enabled": session is not None,
            "refiner_input_channels": session_channels,
            "localization_fusion": localization_fusion,
            "strip_radius_px": cfg.strip_radius_px,
            "strip_step_px": cfg.strip_step_px,
            "strip_bin_count": len(_strip_offsets(cfg)),
            "contour_sample_step_px": cfg.contour_sample_step_px,
            "ring_count": len(ring_diagnostics),
            "sample_count": sum(item.sample_count for item in ring_diagnostics),
            "corner_count": sum(item.corner_count for item in ring_diagnostics),
            "learned_corner_anchors_enabled": cfg.learned_corner_anchors_enabled,
            "line_run_count": sum(item.line_run_count for item in ring_diagnostics),
            "mean_reliability": round(float(np.mean(reliabilities)), 6) if reliabilities else 0.0,
            "mean_normalized_entropy": round(float(np.mean(entropies)), 6) if entropies else 1.0,
            "mean_displacement_px": round(
                float(np.mean([item.mean_displacement_px for item in ring_diagnostics])), 6
            ),
            "maximum_displacement_px": round(
                max((item.maximum_displacement_px for item in ring_diagnostics), default=0.0), 6
            ),
            "wide_recenter_count": sum(item.wide_recenter_count for item in ring_diagnostics),
            "wide_recenter_maximum_px": round(
                max((item.wide_recenter_maximum_px for item in ring_diagnostics), default=0.0),
                6,
            ),
            "shortcut_count": sum(item.shortcut_count for item in ring_diagnostics),
            "shortcut_maximum_depth_px": round(
                max((item.shortcut_maximum_depth_px for item in ring_diagnostics), default=0.0),
                6,
            ),
            "shortcut_native_guard": _summarize_shortcut_native_guard(
                ring_diagnostics,
                enabled=cfg.shortcut_enabled,
                evidence_available=proposal_native_inside is not None,
            ),
            "phase_mean_reliability": round(
                float(np.mean([item.phase_reliability for item in ring_diagnostics])), 6
            ),
            "phase_paired_fraction": round(
                float(np.mean([item.phase_paired_fraction for item in ring_diagnostics])), 6
            ),
            "phase_applied_fraction": round(
                float(np.mean([item.phase_applied_fraction for item in ring_diagnostics])), 6
            ),
            "phase_mean_stroke_width_px": round(
                float(np.mean([item.phase_mean_stroke_width_px for item in ring_diagnostics])), 6
            ),
            "rectilinear_fit": {
                "enabled": rectilinear_enabled,
                "result": rectilinear_result,
                "attempted_count": rectilinear_attempted_count,
                "accepted_count": len(rectilinear_accepted),
                "fallback_count": rectilinear_attempted_count - len(rectilinear_accepted),
                "accepted_run_count": sum(
                    item.rectilinear_fit_run_count for item in rectilinear_accepted
                ),
                "maximum_support_p95_px": round(
                    max(
                        (
                            item.rectilinear_fit_support_p95_px
                            for item in rectilinear_accepted
                            if item.rectilinear_fit_support_p95_px is not None
                        ),
                        default=0.0,
                    ),
                    6,
                ),
                "reason_counts": rectilinear_reason_counts,
                "native_phase_alignment": {
                    "enabled": cfg.rectilinear_native_phase_enabled,
                    "evaluated_count": rectilinear_native_phase_evaluated_count,
                    "accepted_count": rectilinear_native_phase_accepted_count,
                    "reason_counts": rectilinear_native_phase_reason_counts,
                },
                "rings": rectilinear_ring_diagnostics,
            },
            "topology_fallback_count": topology_fallbacks,
            "geometry_valid": bool(geometry.is_valid),
        },
    )


def _refine_ring(
    rgb: np.ndarray,
    coarse: np.ndarray,
    contour: np.ndarray,
    *,
    session: EdgeGraphSessionLike | None,
    hints: Any,
    config: EdgeGraphConfig,
    native_fields: _NativeImageFields,
) -> _RefinedRing:
    points = resample_closed_contour(contour, step_px=config.contour_sample_step_px)
    shortcut_native_diagnostics: dict[str, object] = {}
    points, shortcut_count, shortcut_maximum_depth = _shortcut_supported_protrusion(
        native_fields.rgb_float,
        points,
        hints=hints,
        config=config,
        native_inside_probability=native_fields.native_inside_probability,
        diagnostics=shortcut_native_diagnostics,
    )
    points, wide_recenter_offsets = _adaptive_wide_recenter(
        rgb,
        coarse,
        points,
        hints=hints,
        config=config,
        native_fields=native_fields,
    )
    strip = sample_normal_strip(
        rgb,
        coarse,
        points,
        config=config,
        _native_fields=native_fields,
    )
    session_channels = _session_input_channel_count(session) if session is not None else None
    learned_vector_owner = _learned_localization_owner(session_channels, config)
    deterministic_logits = (
        np.zeros((len(points), len(strip.offsets)), dtype=np.float32)
        if learned_vector_owner
        else deterministic_edge_logits(strip.features, strip.offsets)
    )
    target_rgb = (
        hints.get("target_rgb")
        if isinstance(hints, dict)
        else getattr(hints, "target_rgb", None)
    )
    semantic_logits = (
        target_boundary_logits(
            strip.features,
            strip.offsets,
            target_rgb=target_rgb,
            scale=config.target_semantic_logit_scale,
        )
        if target_rgb is not None and not learned_vector_owner
        else np.zeros_like(deterministic_logits)
    )
    phase = (
        estimate_edge_phase(
            strip.features[:3].transpose(1, 2, 0),
            strip.features[4],
            strip.features[6],
            strip.offsets,
            target_rgb=target_rgb,
            inside_sign=_inside_sign_from_coarse_profiles(strip.features[3], strip.offsets),
        )
        if (
            target_rgb is not None
            and config.edge_phase_logit_weight > 0.0
            and not learned_vector_owner
        )
        else None
    )
    phase_logits = np.zeros_like(deterministic_logits)
    phase_applied = np.zeros(len(points), dtype=bool)
    if session is None:
        if phase is not None:
            phase_logits, phase_applied = _gated_phase_logits(
                phase,
                strip.offsets,
                learned_logits=None,
                config=config,
            )
        offset_logits = deterministic_logits + semantic_logits + phase_logits
        corner_logits = np.full(len(points), -4.0, dtype=np.float32)
        reliability_logits = _deterministic_reliability_logits(deterministic_logits)
    else:
        learned, corner_logits, reliability_logits = _run_edgegraph_session(
            _session_features(
                strip,
                session=session,
                image_shape=rgb.shape[:2],
                target_rgb=target_rgb,
            ),
            session,
            config=config,
        )
        if phase is not None:
            phase_logits, phase_applied = _gated_phase_logits(
                phase,
                strip.offsets,
                learned_logits=learned,
                config=config,
            )
        if learned_vector_owner:
            # Vector-v2/v3 already receive the native evidence needed for
            # localization. Adding handcrafted posteriors double-counts those
            # cues and biases wide strokes toward one rendered side. Keep one
            # semantic owner.
            offset_logits = learned
        else:
            selector_scale = max(rgb.shape[:2]) / 320.0
            evidence_weight = float(
                np.clip(0.03 + 0.05 * max(0.0, selector_scale - 2.0), 0.03, 0.30)
            )
            offset_logits = (
                learned
                + evidence_weight * deterministic_logits
                + semantic_logits
                + phase_logits
            )

    corner_probability = _sigmoid(corner_logits)
    reliability = _sigmoid(reliability_logits)
    # Corner probabilities remain continuous inputs to the cyclic decoder, so
    # a real corner can still relax local offset smoothness.  Only discrete
    # vector anchors are reliability-gated: a high corner logit on an
    # occluded/ambiguous profile otherwise freezes a false spike into the final
    # polygon and can make an otherwise accurate ring self-intersect.
    corner_anchor_probability = _vector_corner_anchor_probability(
        corner_probability,
        reliability,
        learned_vector_owner=learned_vector_owner,
        config=config,
    )
    # OpenCV chooses an arbitrary first contour pixel. Anchoring the cyclic DP
    # there lets a label crossing or an actual corner force the entire ring's
    # mode through the closure term. Rotate to the most reliable, lowest-
    # entropy non-corner sample, solve, then restore source order.
    unary_probability = _softmax(offset_logits, axis=1)
    unary_entropy = -np.sum(
        unary_probability * np.log(np.clip(unary_probability, 1e-9, 1.0)),
        axis=1,
    ) / max(1e-9, math.log(unary_probability.shape[1]))
    anchor_score = reliability * (1.0 - corner_probability) * (1.0 - unary_entropy)
    anchor = int(np.argmax(anchor_score))
    rotation = -anchor
    decoded_rotated = decode_cyclic_offsets(
        np.roll(offset_logits, rotation, axis=0),
        strip.offsets,
        corner_probability=np.roll(corner_probability, rotation),
        reliability=np.roll(reliability, rotation),
        smoothness_weight=config.smoothness_weight,
        maximum_bin_jump=config.maximum_bin_jump,
    )
    offsets = np.roll(decoded_rotated, anchor)
    refined = strip.points + strip.normals * offsets[:, None]
    rectilinear = (
        fit_rectilinear_ring(refined)
        if config.rectilinear_fit_enabled
        else None
    )
    rectilinear_native_alignment = _RectilinearNativeAlignment(
        None,
        {
            "evaluated": False,
            "accepted": False,
            "reason": "rectilinear-fit-not-accepted",
        },
    )
    if rectilinear is not None and rectilinear.candidate is not None:
        rectilinear_candidate = rectilinear.candidate
        if (
            config.rectilinear_native_phase_enabled
            and target_rgb is not None
        ):
            rectilinear_native_alignment = (
                _align_rectilinear_candidate_to_native_phase(
                    native_fields.rgb_float,
                    rectilinear_candidate,
                    target_rgb=target_rgb,
                    image_shape=rgb.shape[:2],
                    config=config,
                )
            )
            if rectilinear_native_alignment.candidate is not None:
                rectilinear_candidate = rectilinear_native_alignment.candidate
        elif config.rectilinear_native_phase_enabled:
            rectilinear_native_alignment = _RectilinearNativeAlignment(
                None,
                {
                    "evaluated": False,
                    "accepted": False,
                    "reason": "target-color-unavailable",
                },
            )
        else:
            rectilinear_native_alignment = _RectilinearNativeAlignment(
                None,
                {
                    "evaluated": False,
                    "accepted": False,
                    "reason": "disabled",
                },
            )
        coordinates = [
            (float(x), float(y))
            for x, y in rectilinear_candidate.exterior.coords
        ]
        line_runs = rectilinear.diagnostics.run_count
        protected_corners = len(
            _corner_peak_indices(
                corner_anchor_probability,
                config.corner_probability_threshold,
            )
        )
    else:
        coordinates, line_runs, protected_corners = fit_corner_protected_ring(
            refined,
            corner_probability=corner_anchor_probability,
            rgb=rgb,
            target_rgb=target_rgb,
            config=config,
            _rgb_float=native_fields.rgb_float,
        )
    probability = unary_probability
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-9, 1.0)), axis=1)
    normalized_entropy = entropy / max(1e-9, math.log(probability.shape[1]))
    return _RefinedRing(
        coordinates=coordinates,
        sample_count=len(points),
        corner_count=protected_corners,
        line_run_count=line_runs,
        mean_reliability=float(np.mean(reliability)),
        mean_entropy=float(np.mean(normalized_entropy)),
        mean_displacement_px=float(np.mean(np.abs(offsets))),
        maximum_displacement_px=float(np.max(np.abs(offsets), initial=0.0)),
        wide_recenter_count=int(np.count_nonzero(np.abs(wide_recenter_offsets) > 1e-4)),
        wide_recenter_maximum_px=float(np.max(np.abs(wide_recenter_offsets), initial=0.0)),
        shortcut_count=shortcut_count,
        shortcut_maximum_depth_px=shortcut_maximum_depth,
        phase_reliability=float(np.mean(phase.reliability)) if phase is not None else 0.0,
        phase_paired_fraction=float(np.mean(phase.paired)) if phase is not None else 0.0,
        phase_applied_fraction=float(np.mean(phase_applied)),
        phase_mean_stroke_width_px=(
            float(np.mean(phase.stroke_widths_px[phase.paired]))
            if phase is not None and bool(np.any(phase.paired))
            else 0.0
        ),
        rectilinear_fit_accepted=(
            rectilinear is not None and rectilinear.candidate is not None
        ),
        rectilinear_fit_reason=(
            rectilinear.diagnostics.reason
            if rectilinear is not None
            else "disabled"
        ),
        rectilinear_fit_run_count=(
            rectilinear.diagnostics.run_count
            if rectilinear is not None
            else 0
        ),
        rectilinear_fit_support_p95_px=(
            rectilinear.diagnostics.support_p95_px
            if rectilinear is not None and rectilinear.candidate is not None
            else None
        ),
        rectilinear_native_alignment_diagnostics=(
            rectilinear_native_alignment.diagnostics
        ),
        shortcut_native_diagnostics=shortcut_native_diagnostics,
    )


def _gated_phase_logits(
    phase: Any,
    offsets: np.ndarray,
    *,
    learned_logits: np.ndarray | None,
    config: EdgeGraphConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse native transition phase only when its evidence is trustworthy.

    With a learned refiner present, every native phase candidate must agree
    with that model's modal offset before it can move the cyclic decode. This
    prevents a coherent-looking label, road, or wrong transition pair from
    becoming a competing phase signal while retaining subpixel localization
    on clean source pixels.
    """

    reliability = np.asarray(phase.reliability, dtype=np.float32).reshape(-1)
    centers = np.asarray(phase.centers_px, dtype=np.float32).reshape(-1)
    gate = reliability >= float(config.edge_phase_minimum_reliability)
    if learned_logits is not None:
        learned = np.asarray(learned_logits, dtype=np.float32)
        if learned.ndim != 2 or learned.shape[0] != len(gate) or learned.shape[1] != len(offsets):
            raise ValueError("learned phase-fusion logits do not match the normal strip")
        learned_centers = np.asarray(offsets, dtype=np.float32)[np.argmax(learned, axis=1)]
        agreement = np.abs(learned_centers - centers) <= float(
            config.edge_phase_unpaired_agreement_px
        )
        # A coherent-looking two-sided stroke can still be a label, road, or
        # the wrong pair of transitions. The learned vector posterior is the
        # semantic owner, so native phase may sharpen either paired or single
        # edges only when their modes agree spatially.
        gate &= agreement
    logits = (
        float(config.edge_phase_logit_weight)
        * np.asarray(phase.logits, dtype=np.float32)
        * gate[:, None]
    )
    return logits.astype(np.float32, copy=False), gate


def resample_closed_contour(contour: np.ndarray, *, step_px: float = 1.5) -> np.ndarray:
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("closed contour requires at least three points")
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    closed = np.concatenate([points, points[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = lengths > 1e-6
    if int(keep.sum()) < 3:
        raise ValueError("closed contour has insufficient distinct points")
    if not keep.all():
        points = points[keep]
        closed = np.concatenate([points, points[:1]], axis=0)
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    perimeter = float(cumulative[-1])
    count = max(8, int(math.ceil(perimeter / max(0.25, float(step_px)))))
    targets = np.linspace(0.0, perimeter, count, endpoint=False, dtype=np.float32)
    segments = np.minimum(np.searchsorted(cumulative, targets, side="right") - 1, len(points) - 1)
    fractions = (targets - cumulative[segments]) / np.maximum(lengths[segments], 1e-6)
    return np.ascontiguousarray(
        closed[segments] + (closed[segments + 1] - closed[segments]) * fractions[:, None],
        dtype=np.float32,
    )


def sample_normal_strip(
    rgb: np.ndarray,
    coarse_probabilities: np.ndarray,
    points: np.ndarray,
    *,
    config: EdgeGraphConfig | None = None,
    _native_fields: _NativeImageFields | None = None,
) -> NormalStrip:
    cfg = config or EdgeGraphConfig()
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    tangents = np.roll(points, -2, axis=0) - np.roll(points, 2, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1).astype(np.float32)
    offsets = _strip_offsets(cfg)

    native_fields = _native_fields or _prepare_native_image_fields(rgb)
    sampled_rgb = _remap_profiles(native_fields.rgb_float, points, normals, offsets)
    sampled_coarse = _remap_profiles(
        np.asarray(coarse_probabilities, dtype=np.float32),
        points,
        normals,
        offsets,
    )
    sampled_luminance = _remap_profiles(native_fields.luminance, points, normals, offsets)
    sampled_magnitude = _remap_profiles(native_fields.magnitude, points, normals, offsets)
    sampled_gx = _remap_profiles(native_fields.grad_x, points, normals, offsets)
    sampled_gy = _remap_profiles(native_fields.grad_y, points, normals, offsets)
    normal_gradient = sampled_gx * normals[:, 0:1] + sampled_gy * normals[:, 1:2]
    features = np.concatenate(
        [
            sampled_rgb.transpose(2, 0, 1),
            sampled_coarse[np.newaxis],
            sampled_luminance[np.newaxis],
            sampled_magnitude[np.newaxis],
            np.clip(normal_gradient, -1.0, 1.0)[np.newaxis],
        ],
        axis=0,
    )
    return NormalStrip(
        features=np.ascontiguousarray(features, dtype=np.float32),
        points=points,
        tangents=tangents.astype(np.float32),
        normals=normals,
        offsets=offsets,
    )


def _inside_sign_from_coarse_profiles(coarse: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return the source-normal direction that points into the selected region.

    Ring orientation is consistent in the common case, but holes, clipped
    contours, and local selector uncertainty can reverse or obscure individual
    profiles.  Use each native profile when it is decisive and the ring-wide
    orientation only for ambiguous rows.
    """

    probabilities = np.asarray(coarse, dtype=np.float32)
    native_offsets = np.asarray(offsets, dtype=np.float32).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(native_offsets):
        raise ValueError("coarse profiles must have shape [sample, offset]")
    center = int(np.argmin(np.abs(native_offsets)))
    step = max(1e-6, float(np.median(np.diff(native_offsets))))
    probe = max(1, int(round(3.0 / step)))
    negative = probabilities[:, max(0, center - probe)]
    positive = probabilities[:, min(len(native_offsets) - 1, center + probe)]
    delta = positive - negative
    global_sign = 1 if float(np.mean(delta)) > 0.0 else -1
    signs = np.where(delta > 0.0, 1, -1)
    signs = np.where(np.abs(delta) >= 0.025, signs, global_sign)
    return np.ascontiguousarray(signs, dtype=np.int8)


def _learned_localization_owner(
    session_channels: int | None,
    config: EdgeGraphConfig,
) -> bool:
    return bool(
        (session_channels == 10 and config.vector_v3_learned_localization_owner)
        or (session_channels == 9 and config.vector_v2_learned_localization_owner)
    )


def _localization_fusion(
    session_channels: int | None,
    *,
    session_present: bool,
    config: EdgeGraphConfig,
) -> str:
    if session_channels == 10 and config.vector_v3_learned_localization_owner:
        return "learned-vector-v3"
    if session_channels == 9 and config.vector_v2_learned_localization_owner:
        return "learned-vector-v2"
    return "hybrid" if session_present else "native-deterministic"


def _session_features(
    strip: NormalStrip,
    *,
    session: EdgeGraphSessionLike,
    image_shape: tuple[int, int],
    target_rgb: Any,
) -> np.ndarray:
    """Adapt the native seven-channel strip to a versioned model contract.

    The first v20 diagnostic model used seven channels.  Production vector-v2
    adds target similarity and an explicit source-valid mask. Vector-v3 removes
    raw coarse probability and adds explicit offset coordinates plus a repeated
    inside-direction sign. Supporting all three lets existing ONNX graphs keep
    their exact tensors while selecting the new contract by declared width.
    """

    expected_channels = _session_input_channel_count(session)
    base = strip.features
    if expected_channels in (None, base.shape[0]):
        return base
    if expected_channels not in (9, 10):
        raise ValueError(
            f"EdgeGraph refiner expects {expected_channels} channels; runtime supports 7, 9, or 10"
        )

    rgb_profiles = base[:3].transpose(1, 2, 0)
    if target_rgb is None:
        selected = rgb_profiles[base[3] >= 0.75]
        target = (
            np.median(selected, axis=0)
            if len(selected)
            else np.median(rgb_profiles[:, len(strip.offsets) // 2], axis=0)
        )
    else:
        target = np.asarray(target_rgb, dtype=np.float32).reshape(-1)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_rgb must contain three finite channels")
        if float(np.max(target, initial=0.0)) > 1.5:
            target = target / 255.0
    target = np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0)
    similarity = np.clip(
        1.0 - np.linalg.norm(rgb_profiles - target.reshape(1, 1, 3), axis=2) / math.sqrt(3.0),
        0.0,
        1.0,
    )
    height, width = image_shape
    map_x = strip.points[:, 0:1] + strip.normals[:, 0:1] * strip.offsets[None, :]
    map_y = strip.points[:, 1:2] + strip.normals[:, 1:2] * strip.offsets[None, :]
    valid = (
        (map_x >= 0.0)
        & (map_x <= float(width - 1))
        & (map_y >= 0.0)
        & (map_y <= float(height - 1))
    )
    if expected_channels == 9:
        return np.ascontiguousarray(
            np.concatenate([base, similarity[np.newaxis], valid[np.newaxis]], axis=0),
            dtype=np.float32,
        )

    radius = max(1e-6, float(np.max(np.abs(strip.offsets), initial=0.0)))
    normalized_offset = np.broadcast_to(
        strip.offsets[None, :] / radius,
        base[3].shape,
    )
    inside_direction_sign = np.broadcast_to(
        _inside_sign_from_coarse_profiles(base[3], strip.offsets)[:, None],
        base[3].shape,
    )
    return np.ascontiguousarray(
        np.concatenate(
            [
                base[:3],
                base[4:7],
                similarity[np.newaxis],
                valid[np.newaxis],
                normalized_offset[np.newaxis],
                inside_direction_sign[np.newaxis],
            ],
            axis=0,
        ),
        dtype=np.float32,
    )


def _session_input_channel_count(session: EdgeGraphSessionLike) -> int | None:
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("EdgeGraph refiner declares no inputs")
    shape = getattr(inputs[0], "shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        value = shape[1]
        if isinstance(value, (int, np.integer)):
            return int(value)
    return None


def _prepare_native_image_fields(rgb: np.ndarray) -> _NativeImageFields:
    rgb_float = rgb.astype(np.float32) / 255.0
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Scharr(luminance, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(luminance, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(grad_x, grad_y)
    scale = max(1e-6, float(np.quantile(magnitude, 0.99)))
    grad_x /= scale
    grad_y /= scale
    magnitude = np.clip(magnitude / scale, 0.0, 1.0)
    return _NativeImageFields(
        rgb_float=np.ascontiguousarray(rgb_float, dtype=np.float32),
        luminance=np.ascontiguousarray(luminance, dtype=np.float32),
        grad_x=np.ascontiguousarray(grad_x, dtype=np.float32),
        grad_y=np.ascontiguousarray(grad_y, dtype=np.float32),
        magnitude=np.ascontiguousarray(magnitude, dtype=np.float32),
    )


def deterministic_edge_logits(features: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    luminance = features[4]
    magnitude = features[5]
    normal_gradient = np.abs(features[6])
    # An overlay boundary is supported by a local step, not just a thin road
    # line. Compare profiles several source pixels to each side of each bin.
    step_bins = max(1, int(round(2.0 / max(1e-6, float(offsets[1] - offsets[0])))))
    left = np.take(luminance, np.maximum(0, np.arange(len(offsets)) - step_bins), axis=1)
    right = np.take(luminance, np.minimum(len(offsets) - 1, np.arange(len(offsets)) + step_bins), axis=1)
    step_contrast = np.abs(right - left)
    selector_prior = np.exp(-0.5 * (offsets / 4.0) ** 2)[None, :]
    edge = 2.2 * normal_gradient + 1.2 * magnitude + 1.5 * step_contrast
    edge /= np.maximum(np.max(edge, axis=1, keepdims=True), 1e-6)
    # A rendered stroke creates two opposing edge responses around the actual
    # vector boundary. Score their midpoint; choosing either outer edge is the
    # exact mechanism that turns a correct sharp polygon into an expanded or
    # contracted blob.
    paired = np.zeros_like(edge)
    signed = features[6]
    maximum_half_width = min(12, (len(offsets) - 1) // 2)
    for half_width in range(1, maximum_half_width + 1):
        left = edge[:, : -2 * half_width]
        right = edge[:, 2 * half_width :]
        opposite = (signed[:, : -2 * half_width] * signed[:, 2 * half_width :]) < 0.0
        support = np.minimum(left, right) * np.where(opposite, 1.0, 0.35)
        paired[:, half_width:-half_width] = np.maximum(
            paired[:, half_width:-half_width],
            support,
        )
    score = 0.55 * edge + 1.20 * paired + 0.08 * selector_prior
    return np.asarray(score * 6.0, dtype=np.float32)


def target_boundary_logits(
    features: np.ndarray,
    offsets: np.ndarray,
    *,
    target_rgb: Any,
    scale: float = 60.0,
) -> np.ndarray:
    """Score persistent target-colored inside versus non-target outside.

    Thin map strokes can have a larger gradient than the service boundary.
    Sampling several pixels on both sides makes this score insensitive to a
    one-pixel road while retaining an opaque or translucent fill transition.
    """

    rgb = features[:3].transpose(1, 2, 0)
    coarse = features[3]
    step = max(1e-6, float(offsets[1] - offsets[0]))
    inside_sign = _inside_sign_from_coarse_profiles(coarse, offsets)
    indices = np.arange(len(offsets))[None, :]
    target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 1, 3)
    if float(np.max(target, initial=0.0)) > 1.5:
        target = target / 255.0
    rows = np.arange(len(rgb))[:, None]
    inside_distances: list[np.ndarray] = []
    outside_distances: list[np.ndarray] = []
    for distance_px in (2.0, 4.0, 6.0):
        distance_bins = max(1, int(round(distance_px / step)))
        inside_indices = np.clip(
            indices + inside_sign[:, None] * distance_bins,
            0,
            len(offsets) - 1,
        )
        outside_indices = np.clip(
            indices - inside_sign[:, None] * distance_bins,
            0,
            len(offsets) - 1,
        )
        inside_distances.append(
            np.linalg.norm(rgb[rows, inside_indices] - target, axis=2) / math.sqrt(3.0)
        )
        outside_distances.append(
            np.linalg.norm(rgb[rows, outside_indices] - target, axis=2) / math.sqrt(3.0)
        )
    inside_distance = np.median(inside_distances, axis=0)
    outside_distance = np.median(outside_distances, axis=0)
    target_support = np.clip((0.35 - inside_distance) / 0.35, 0.0, 1.0)
    separation = np.clip(outside_distance - inside_distance, 0.0, 0.30)
    evidence = separation * target_support

    # Target color answers the semantic question (which transition belongs to
    # the selected overlay), but it cannot answer the subpixel phase question.
    # In particular, a rendered outline makes target separation peak at the
    # inner stroke/fill transition even though the vector path is centered in
    # the stroke. Convert the score into a broad candidate gate and let native
    # one-/two-change-point evidence determine phase inside that gate.
    peak = np.maximum(np.max(evidence, axis=1, keepdims=True), 1e-6)
    relative = evidence / peak
    gate = np.clip((relative - 0.30) / 0.45, 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    observable = np.clip((peak - 0.006) / 0.035, 0.0, 1.0)
    return np.asarray(float(scale) * gate * observable, dtype=np.float32)


def _shortcut_supported_protrusion(
    rgb_float: np.ndarray,
    points: np.ndarray,
    *,
    hints: Any,
    config: EdgeGraphConfig,
    native_inside_probability: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> tuple[np.ndarray, int, float]:
    """Replace one selector-only bulge with an observed native straight edge.

    The broad anchor pass preserves the existing ability to recover deep,
    semantically missing branches.  If it finds no supported chord, a finer
    anchor pass may recover a small local protrusion, but its maximum depth is
    capped at five percent of the image span.  That asymmetric second-pass
    gate prevents fine polygonization from turning a real deep rectilinear
    branch into a false shortcut.
    """

    broad = _shortcut_supported_protrusion_at_scale(
        rgb_float,
        points,
        hints=hints,
        config=config,
        native_inside_probability=native_inside_probability,
        diagnostics=diagnostics,
    )
    if broad[1] > 0:
        return broad
    fine_config = replace(
        config,
        shortcut_anchor_tolerance_px=min(
            3.0,
            config.shortcut_anchor_tolerance_px,
        ),
        shortcut_maximum_depth_fraction=min(
            0.05,
            config.shortcut_maximum_depth_fraction,
        ),
    )
    if (
        fine_config.shortcut_anchor_tolerance_px
        == config.shortcut_anchor_tolerance_px
        and fine_config.shortcut_maximum_depth_fraction
        == config.shortcut_maximum_depth_fraction
    ):
        return broad
    return _shortcut_supported_protrusion_at_scale(
        rgb_float,
        points,
        hints=hints,
        config=fine_config,
        native_inside_probability=native_inside_probability,
        diagnostics=diagnostics,
    )


def _source_native_contraction_evidence(
    source: Polygon,
    candidate: Polygon,
    *,
    native_inside_probability: np.ndarray | None,
    image_shape: tuple[int, int],
    config: EdgeGraphConfig,
) -> dict[str, object]:
    """Require independent evidence that a chord removes non-overlay pixels.

    A line across a real branch can have excellent edge contrast and preserve
    polygon topology. Those signals only show that a chord is visually
    plausible; they do not establish that the lobe on the other side is
    outside the service area. The proposal lane's source-native appearance
    posterior answers that semantic question independently of the shortcut.
    """

    if native_inside_probability is None:
        return {
            "accepted": False,
            "reason": "native-evidence-unavailable",
            "removed_pixel_count": 0,
            "evidence_coverage": 0.0,
        }
    height, width = image_shape
    native = np.asarray(native_inside_probability, dtype=np.float32)
    if native.shape != (height, width):
        return {
            "accepted": False,
            "reason": "native-evidence-shape",
            "removed_pixel_count": 0,
            "evidence_coverage": 0.0,
        }
    source_mask = rasterize_geometry_mask(source, width=width, height=height)
    candidate_mask = rasterize_geometry_mask(candidate, width=width, height=height)
    removed = source_mask & ~candidate_mask
    removed_count = int(np.count_nonzero(removed))
    if removed_count < 8:
        return {
            "accepted": False,
            "reason": "insufficient-removed-region",
            "removed_pixel_count": removed_count,
            "evidence_coverage": 0.0,
        }
    available = removed & np.isfinite(native)
    available_count = int(np.count_nonzero(available))
    coverage = available_count / removed_count
    if available_count == 0:
        mean_inside = 1.0
        p90_inside = 1.0
    else:
        values = native[available]
        mean_inside = float(np.mean(values))
        p90_inside = float(np.quantile(values, 0.90))
    accepted = bool(
        coverage >= config.shortcut_minimum_native_evidence_coverage
        and mean_inside <= config.shortcut_maximum_removed_inside_mean
        and p90_inside <= config.shortcut_maximum_removed_inside_p90
    )
    if coverage < config.shortcut_minimum_native_evidence_coverage:
        reason = "insufficient-native-coverage"
    elif mean_inside > config.shortcut_maximum_removed_inside_mean:
        reason = "removed-region-inside-mean"
    elif p90_inside > config.shortcut_maximum_removed_inside_p90:
        reason = "removed-region-inside-tail"
    else:
        reason = "accepted"
    return {
        "accepted": accepted,
        "reason": reason,
        "removed_pixel_count": removed_count,
        "evidence_coverage": round(coverage, 6),
        "removed_inside_mean": round(mean_inside, 6),
        "removed_inside_p90": round(p90_inside, 6),
    }


def _summarize_shortcut_native_guard(
    rings: list[_RefinedRing],
    *,
    enabled: bool,
    evidence_available: bool,
) -> dict[str, object]:
    records = [
        item.shortcut_native_diagnostics
        for item in rings
        if item.shortcut_native_diagnostics
    ]
    accepted = [
        record["accepted_evidence"]
        for record in records
        if isinstance(record.get("accepted_evidence"), dict)
    ]
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason, count in dict(record.get("reason_counts", {})).items():
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(count)
    return {
        "enabled": bool(enabled),
        "evidence_available": bool(evidence_available),
        "evaluated_candidate_count": sum(
            int(record.get("evaluated_candidate_count", 0))
            for record in records
        ),
        "rejected_candidate_count": sum(
            int(record.get("rejected_candidate_count", 0))
            for record in records
        ),
        "accepted_candidate_count": sum(
            int(record.get("accepted_candidate_count", 0))
            for record in records
        ),
        "reason_counts": reason_counts,
        "accepted_removed_inside_mean": round(
            max(
                (float(item["removed_inside_mean"]) for item in accepted),
                default=0.0,
            ),
            6,
        ),
        "accepted_removed_inside_p90": round(
            max(
                (float(item["removed_inside_p90"]) for item in accepted),
                default=0.0,
            ),
            6,
        ),
    }


def _shortcut_supported_protrusion_at_scale(
    rgb_float: np.ndarray,
    points: np.ndarray,
    *,
    hints: Any,
    config: EdgeGraphConfig,
    native_inside_probability: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> tuple[np.ndarray, int, float]:
    """Search one anchor scale for a native-supported contraction.

    A low-resolution selector can merge adjacent similarly colored map regions,
    producing a large outward lobe whose true edge is not normal-reachable from
    every point on the lobe.  The contour graph therefore also considers
    straight chords between coarse anchors.  A chord is accepted only when it
    contracts the selector polygon, preserves the guided seed, and the native
    pixels show a persistent target-to-nontarget transition along most of it.
    """

    target_rgb = (
        hints.get("target_rgb")
        if isinstance(hints, dict)
        else getattr(hints, "target_rgb", None)
    )
    if not config.shortcut_enabled or target_rgb is None or len(points) < 12:
        return points, 0, 0.0
    height, width = rgb_float.shape[:2]
    # A contour clipped by the viewport has no observable closed exterior.
    # Contracting it with an interior chord can erase a legitimate lobe while
    # still satisfying polygon validity and seed checks, so fail closed.
    border_margin = 1.0
    if bool(
        np.any(points[:, 0] <= border_margin)
        or np.any(points[:, 1] <= border_margin)
        or np.any(points[:, 0] >= width - 1.0 - border_margin)
        or np.any(points[:, 1] >= height - 1.0 - border_margin)
    ):
        return points, 0, 0.0
    source = Polygon(points)
    if source.is_empty or not source.is_valid or source.area <= 0.0:
        return points, 0, 0.0
    approximation = cv2.approxPolyDP(
        points[:, np.newaxis, :],
        epsilon=float(config.shortcut_anchor_tolerance_px),
        closed=True,
    )[:, 0, :]
    anchor_indices = sorted(
        {
            int(np.argmin(np.linalg.norm(points - anchor, axis=1)))
            for anchor in approximation
        }
    )
    if len(anchor_indices) < 4:
        return points, 0, 0.0

    seed_value = (
        hints.get("seed_point")
        if isinstance(hints, dict)
        else getattr(hints, "seed_point", None)
    )
    seed = Point(float(seed_value[0]), float(seed_value[1])) if seed_value is not None else None
    best_score = -float("inf")
    best_points: np.ndarray | None = None
    best_depth = 0.0
    best_native_evidence: dict[str, object] | None = None
    anchor_count = len(anchor_indices)
    maximum_span = min(config.shortcut_maximum_anchor_span, anchor_count - 2)
    for anchor_position, start in enumerate(anchor_indices):
        for span in range(2, maximum_span + 1):
            end = anchor_indices[(anchor_position + span) % anchor_count]
            detour = _cyclic_points(points, start, end)
            chord = detour[-1] - detour[0]
            chord_length = float(np.linalg.norm(chord))
            if chord_length < config.shortcut_minimum_chord_length_px:
                continue
            detour_length = float(np.linalg.norm(np.diff(detour, axis=0), axis=1).sum())
            if detour_length < chord_length * config.shortcut_minimum_detour_ratio:
                continue
            depth = _maximum_distance_to_segment(detour, detour[0], detour[-1])
            if depth < config.shortcut_minimum_depth_px:
                continue
            if depth > min(height, width) * config.shortcut_maximum_depth_fraction:
                continue
            remaining = _cyclic_points(points, end, start)
            candidate = Polygon(remaining)
            if candidate.is_empty or not candidate.is_valid or candidate.area <= 0.0:
                continue
            area_ratio = candidate.area / source.area
            if not 0.70 <= area_ratio <= 0.998:
                continue
            # This graph move is deliberately contraction-only. It cannot fill
            # a real concavity or invent coverage beyond the semantic selector.
            if candidate.difference(source).area > max(2.0, source.area * 0.00002):
                continue
            if seed is not None and not candidate.buffer(0.5).covers(seed):
                continue
            screen_support, screen_separation, screen_contrast, _screen_offset = _chord_native_support(
                rgb_float,
                detour[0],
                detour[-1],
                candidate=candidate,
                target_rgb=target_rgb,
                config=config,
                alignment_search=False,
            )
            if screen_support < 0.35:
                continue
            if screen_separation < config.shortcut_minimum_target_separation * 0.45:
                continue
            if screen_contrast < config.shortcut_minimum_edge_contrast * 0.45:
                continue
            support_fraction, median_separation, median_contrast, alignment_offset = _chord_native_support(
                rgb_float,
                detour[0],
                detour[-1],
                candidate=candidate,
                target_rgb=target_rgb,
                config=config,
                alignment_search=True,
            )
            if support_fraction < config.shortcut_minimum_support_fraction:
                continue
            if median_separation < config.shortcut_minimum_target_separation:
                continue
            if median_contrast < config.shortcut_minimum_edge_contrast:
                continue
            aligned_remaining = remaining
            native_candidate = candidate
            if abs(alignment_offset) >= config.shortcut_alignment_step_px * 0.5:
                chord_direction = (detour[-1] - detour[0]) / chord_length
                chord_normal = np.asarray(
                    [-chord_direction[1], chord_direction[0]],
                    dtype=np.float32,
                )
                aligned_start = detour[0] + chord_normal * alignment_offset
                aligned_end = detour[-1] + chord_normal * alignment_offset
                aligned_remaining = np.concatenate(
                    [remaining, aligned_start[None, :], aligned_end[None, :]],
                    axis=0,
                )
                aligned_candidate = Polygon(aligned_remaining)
                if aligned_candidate.is_empty or not aligned_candidate.is_valid:
                    continue
                aligned_area_ratio = aligned_candidate.area / source.area
                if not 0.70 <= aligned_area_ratio <= 1.0:
                    continue
                if aligned_candidate.difference(source).area > max(2.0, source.area * 0.00002):
                    continue
                if seed is not None and not aligned_candidate.buffer(0.5).covers(seed):
                    continue
                area_ratio = aligned_area_ratio
                native_candidate = aligned_candidate
            native_evidence = _source_native_contraction_evidence(
                source,
                native_candidate,
                native_inside_probability=native_inside_probability,
                image_shape=(height, width),
                config=config,
            )
            if diagnostics is not None:
                diagnostics["evaluated_candidate_count"] = (
                    int(diagnostics.get("evaluated_candidate_count", 0)) + 1
                )
            if not bool(native_evidence["accepted"]):
                if diagnostics is not None:
                    diagnostics["rejected_candidate_count"] = (
                        int(diagnostics.get("rejected_candidate_count", 0)) + 1
                    )
                    reason = str(native_evidence["reason"])
                    reasons = dict(diagnostics.get("reason_counts", {}))
                    reasons[reason] = int(reasons.get(reason, 0)) + 1
                    diagnostics["reason_counts"] = reasons
                continue
            score = (
                support_fraction
                + 2.0 * median_separation
                + median_contrast
                + min(0.50, depth / 100.0)
                + min(0.25, 2.0 * (1.0 - area_ratio))
            )
            if score <= best_score:
                continue
            best_score = score
            best_points = aligned_remaining
            best_depth = depth
            best_native_evidence = native_evidence
    if best_points is None:
        return points, 0, 0.0
    if diagnostics is not None and best_native_evidence is not None:
        diagnostics["accepted_candidate_count"] = 1
        diagnostics["accepted_evidence"] = best_native_evidence
    return (
        resample_closed_contour(best_points, step_px=config.contour_sample_step_px),
        1,
        float(best_depth),
    )


def _maximum_distance_to_segment(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = np.asarray(end - start, dtype=np.float32)
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-9:
        return float(np.linalg.norm(points - start, axis=1).max(initial=0.0))
    projection = np.clip(((points - start) @ segment) / denominator, 0.0, 1.0)
    closest = start + projection[:, None] * segment
    return float(np.linalg.norm(points - closest, axis=1).max(initial=0.0))


def _chord_native_support(
    rgb_float: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    candidate: Polygon,
    target_rgb: Any,
    config: EdgeGraphConfig,
    alignment_search: bool = True,
) -> tuple[float, float, float, float]:
    chord = np.asarray(end - start, dtype=np.float32)
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    tangent = chord / chord_length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    sample_count = max(12, min(256, int(math.ceil(chord_length / 2.0))))
    fractions = np.linspace(0.04, 0.96, sample_count, dtype=np.float32)
    samples = start[None, :] + fractions[:, None] * chord[None, :]
    midpoint = samples[len(samples) // 2]
    side_probe = 7.0
    plus_inside = candidate.buffer(0.5).covers(
        Point(*(midpoint + normal * side_probe))
    )
    minus_inside = candidate.buffer(0.5).covers(
        Point(*(midpoint - normal * side_probe))
    )
    if plus_inside == minus_inside:
        return 0.0, 0.0, 0.0, 0.0
    inside_sign = 1.0 if plus_inside else -1.0
    target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 3) / 255.0
    best_metrics = (0.0, 0.0, 0.0, 0.0)
    best_score = -float("inf")
    offsets = (
        np.arange(
            -config.shortcut_alignment_radius_px,
            config.shortcut_alignment_radius_px + config.shortcut_alignment_step_px * 0.5,
            config.shortcut_alignment_step_px,
            dtype=np.float32,
        )
        if alignment_search
        else np.asarray([0.0], dtype=np.float32)
    )
    for alignment_offset in offsets:
        shifted_samples = samples + normal[None, :] * float(alignment_offset)
        inside_distances: list[np.ndarray] = []
        outside_distances: list[np.ndarray] = []
        contrasts: list[np.ndarray] = []
        for distance in (4.0, 7.0, 10.0):
            inside = _sample_rgb_coordinates(
                rgb_float,
                shifted_samples + normal[None, :] * (inside_sign * distance),
            )
            outside = _sample_rgb_coordinates(
                rgb_float,
                shifted_samples - normal[None, :] * (inside_sign * distance),
            )
            inside_distances.append(
                np.linalg.norm(inside - target, axis=1) / math.sqrt(3.0)
            )
            outside_distances.append(
                np.linalg.norm(outside - target, axis=1) / math.sqrt(3.0)
            )
            contrasts.append(np.linalg.norm(inside - outside, axis=1) / math.sqrt(3.0))
        inside_distance = np.median(inside_distances, axis=0)
        outside_distance = np.median(outside_distances, axis=0)
        separation = outside_distance - inside_distance
        contrast = np.median(contrasts, axis=0)
        supported = (
            (separation >= config.shortcut_minimum_target_separation)
            & (contrast >= config.shortcut_minimum_edge_contrast)
        )
        support_fraction = float(np.mean(supported))
        median_separation = float(np.median(np.clip(separation, 0.0, 1.0)))
        median_contrast = float(np.median(contrast))
        score = (
            support_fraction
            + 3.0 * median_separation
            + median_contrast
            - 0.002 * abs(float(alignment_offset))
        )
        if score > best_score:
            best_score = score
            best_metrics = (
                support_fraction,
                median_separation,
                median_contrast,
                float(alignment_offset),
            )
    return best_metrics


def _sample_rgb_coordinates(rgb: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    sampled = cv2.remap(
        rgb,
        coordinates[:, 0:1].astype(np.float32),
        coordinates[:, 1:2].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1, 3)


def _adaptive_wide_recenter(
    rgb: np.ndarray,
    coarse_probabilities: np.ndarray,
    points: np.ndarray,
    *,
    hints: Any,
    config: EdgeGraphConfig,
    native_fields: _NativeImageFields,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover coherent native edges that lie outside the learned fine strip.

    The selector is intentionally low resolution and can be tens of source
    pixels wrong around thin notches or when similarly colored map content
    touches the service fill.  The learned 49-bin head cannot recover evidence
    outside +/-12 px, so a cheap, model-free pass first scans a wider band.
    Only long, coherent runs whose native edge evidence materially beats the
    selector-centered evidence are moved.  Isolated road and label crossings
    therefore remain rejected.
    """

    zero = np.zeros(len(points), dtype=np.float32)
    target_rgb = (
        hints.get("target_rgb")
        if isinstance(hints, dict)
        else getattr(hints, "target_rgb", None)
    )
    if (
        not config.wide_recenter_enabled
        or target_rgb is None
        or len(points) < config.wide_recenter_minimum_run_samples
    ):
        return points, zero
    selector_scale = max(rgb.shape[:2]) / 320.0
    radius = min(
        float(config.wide_recenter_max_radius_px),
        max(
            float(config.strip_radius_px),
            float(config.wide_recenter_radius_per_selector_scale) * selector_scale,
        ),
    )
    radius = math.floor(radius / config.wide_recenter_step_px) * config.wide_recenter_step_px
    if radius <= config.strip_radius_px + config.wide_recenter_step_px:
        return points, zero

    wide_config = EdgeGraphConfig(
        **{
            **config.__dict__,
            "strip_radius_px": radius,
            "strip_step_px": config.wide_recenter_step_px,
            "wide_recenter_enabled": False,
        }
    )
    strip = sample_normal_strip(
        rgb,
        coarse_probabilities,
        points,
        config=wide_config,
        _native_fields=native_fields,
    )
    evidence, target_separation = _wide_native_edge_evidence(
        strip.features,
        strip.offsets,
        target_rgb=target_rgb,
    )
    aggregate = _circular_profile_average(evidence, radius=4)
    coarse_profile = strip.features[3]
    side_bins = max(
        1,
        int(round(4.0 / max(1e-6, float(strip.offsets[1] - strip.offsets[0])))),
    )
    strip_center = len(strip.offsets) // 2
    inside_sign = (
        -1.0
        if float(np.mean(coarse_profile[:, max(0, strip_center - side_bins)]))
        >= float(np.mean(coarse_profile[:, min(len(strip.offsets) - 1, strip_center + side_bins)]))
        else 1.0
    )
    center_prior = np.exp(-0.5 * (strip.offsets / max(8.0, radius * 0.55)) ** 2)[None, :]
    decoder_logits = np.asarray(4.0 * aggregate + 0.20 * center_prior, dtype=np.float32)
    # The failure this pass exists to repair is a selector blob: native detail
    # was swallowed by an over-expanded low-resolution mask. Do not let an
    # unrelated exterior road win the wide search; the fine strip still
    # retains symmetric freedom for ordinary subpixel expansion/contraction.
    exterior_wide = strip.offsets * inside_sign < -config.strip_radius_px
    decoder_logits[:, exterior_wide] -= 12.0
    curvature = _contour_corner_probability(points)
    decoded = decode_cyclic_offsets(
        decoder_logits,
        strip.offsets,
        corner_probability=curvature,
        reliability=np.ones(len(points), dtype=np.float32),
        smoothness_weight=config.wide_recenter_smoothness_weight,
        maximum_bin_jump=config.wide_recenter_maximum_bin_jump,
        start_candidate_count=5,
    )

    chosen_indices = np.clip(
        np.rint((decoded - strip.offsets[0]) / config.wide_recenter_step_px).astype(np.int32),
        0,
        len(strip.offsets) - 1,
    )
    center_band = np.abs(strip.offsets) <= min(4.0, config.strip_radius_px * 0.40)
    center_evidence = np.max(aggregate[:, center_band], axis=1)
    chosen_evidence = aggregate[np.arange(len(points)), chosen_indices]
    gain = chosen_evidence - center_evidence
    candidates = (
        (decoded * inside_sign >= config.wide_recenter_minimum_move_px)
        & (gain >= config.wide_recenter_minimum_evidence_gain)
        & (center_evidence <= config.wide_recenter_maximum_center_evidence)
        & (chosen_evidence >= 0.20)
        & (target_separation[np.arange(len(points)), chosen_indices] >= 0.025)
    )
    coherent = _keep_cyclic_true_runs(candidates, config.wide_recenter_minimum_run_samples)
    # Include the smooth shoulders selected by the cyclic decoder so the
    # recentered contour reaches a real corner instead of creating a new kink.
    expanded = coherent.copy()
    for distance in range(1, 3):
        expanded |= np.roll(coherent, distance) | np.roll(coherent, -distance)
    accepted = expanded & (gain >= -0.025) & (decoded * inside_sign >= 1.0)
    recenter_offsets = np.where(accepted, decoded, 0.0).astype(np.float32)
    recentered = points + strip.normals * recenter_offsets[:, None]
    return np.ascontiguousarray(recentered, dtype=np.float32), recenter_offsets


def _wide_native_edge_evidence(
    features: np.ndarray,
    offsets: np.ndarray,
    *,
    target_rgb: Any,
) -> tuple[np.ndarray, np.ndarray]:
    luminance = features[4]
    magnitude = features[5]
    signed_gradient = features[6]
    step_bins = max(1, int(round(3.0 / max(1e-6, float(offsets[1] - offsets[0])))))
    indices = np.arange(len(offsets))
    left_indices = np.maximum(0, indices - step_bins)
    right_indices = np.minimum(len(offsets) - 1, indices + step_bins)
    luminance_step = np.abs(luminance[:, right_indices] - luminance[:, left_indices])
    rgb = features[:3].transpose(1, 2, 0)
    rgb_step = np.linalg.norm(rgb[:, right_indices] - rgb[:, left_indices], axis=2) / math.sqrt(3.0)
    target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    coarse_profile = features[3]
    side_bins = max(1, int(round(4.0 / max(1e-6, float(offsets[1] - offsets[0])))))
    center = len(offsets) // 2
    negative_inside_score = float(np.mean(coarse_profile[:, max(0, center - side_bins)]))
    positive_inside_score = float(np.mean(coarse_profile[:, min(len(offsets) - 1, center + side_bins)]))
    inside_sign = -1 if negative_inside_score >= positive_inside_score else 1
    # Compare regions, not the nearest pixel pair. Thin roads and labels can
    # produce a stronger local gradient than the service edge, but they do not
    # create a persistent target-colored half-plane over the next 4-10 px.
    side_distances_px = (4.0, 6.0, 8.0, 10.0, 12.0)
    inside_distances: list[np.ndarray] = []
    outside_distances: list[np.ndarray] = []
    offset_step = max(1e-6, float(offsets[1] - offsets[0]))
    for side_distance_px in side_distances_px:
        distance_bins = max(1, int(round(side_distance_px / offset_step)))
        inside_indices = np.clip(indices + inside_sign * distance_bins, 0, len(offsets) - 1)
        outside_indices = np.clip(indices - inside_sign * distance_bins, 0, len(offsets) - 1)
        inside_distances.append(
            np.linalg.norm(rgb[:, inside_indices] - target, axis=2) / math.sqrt(3.0)
        )
        outside_distances.append(
            np.linalg.norm(rgb[:, outside_indices] - target, axis=2) / math.sqrt(3.0)
        )
    inside_target_distance = np.median(inside_distances, axis=0)
    outside_target_distance = np.median(outside_distances, axis=0)
    target_support = np.clip((0.32 - inside_target_distance) / 0.32, 0.0, 1.0)
    target_separation = (
        np.clip(outside_target_distance - inside_target_distance, 0.0, 1.0)
        * target_support
    )
    native_edge = (
        0.65 * magnitude
        + 0.85 * np.abs(signed_gradient)
        + 1.10 * luminance_step
        + 0.90 * rgb_step
    )
    # Centered opposing gradients are especially strong evidence for the
    # actual vector path when a browser renders a 1-3 px boundary stroke.
    paired = np.zeros_like(native_edge)
    maximum_half_width = min(6, (len(offsets) - 1) // 2)
    for half_width in range(1, maximum_half_width + 1):
        left = native_edge[:, : -2 * half_width]
        right = native_edge[:, 2 * half_width :]
        opposite = (
            signed_gradient[:, : -2 * half_width]
            * signed_gradient[:, 2 * half_width :]
        ) < 0.0
        support = np.minimum(left, right) * np.where(opposite, 1.0, 0.25)
        paired[:, half_width:-half_width] = np.maximum(
            paired[:, half_width:-half_width],
            support,
        )
    # Wide relocation is a semantic decision, not merely edge detection. A
    # road can be the strongest gradient in the band, so target-colored region
    # separation dominates while native edge strength acts as support.
    evidence = 0.25 * np.clip(native_edge + 0.80 * paired, 0.0, 5.0) + 3.0 * target_separation
    return (
        np.asarray(np.clip(evidence, 0.0, 4.25), dtype=np.float32),
        np.asarray(target_separation, dtype=np.float32),
    )


def _circular_profile_average(values: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(values, dtype=np.float32)
    weights = np.asarray([radius + 1 - abs(index) for index in range(-radius, radius + 1)], dtype=np.float32)
    total = np.zeros_like(values, dtype=np.float32)
    for index, weight in zip(range(-radius, radius + 1), weights, strict=True):
        total += float(weight) * np.roll(values, index, axis=0)
    return total / float(weights.sum())


def _contour_corner_probability(points: np.ndarray) -> np.ndarray:
    before = points - np.roll(points, 3, axis=0)
    after = np.roll(points, -3, axis=0) - points
    before /= np.maximum(np.linalg.norm(before, axis=1, keepdims=True), 1e-6)
    after /= np.maximum(np.linalg.norm(after, axis=1, keepdims=True), 1e-6)
    turn = np.arccos(np.clip(np.sum(before * after, axis=1), -1.0, 1.0))
    return np.asarray(np.clip(turn / (math.pi / 5.0), 0.0, 1.0), dtype=np.float32)


def _keep_cyclic_true_runs(values: np.ndarray, minimum_length: int) -> np.ndarray:
    mask = np.asarray(values, dtype=bool).reshape(-1)
    kept = np.zeros_like(mask)
    if not mask.any():
        return kept
    if mask.all():
        kept[:] = len(mask) >= minimum_length
        return kept
    start = int(np.flatnonzero(~mask)[0])
    run: list[int] = []
    for step in range(1, len(mask) + 1):
        index = (start + step) % len(mask)
        if mask[index]:
            run.append(index)
            continue
        if len(run) >= minimum_length:
            kept[run] = True
        run = []
    if len(run) >= minimum_length:
        kept[run] = True
    return kept


def decode_cyclic_offsets(
    offset_logits: np.ndarray,
    offsets: np.ndarray,
    *,
    corner_probability: np.ndarray | None = None,
    reliability: np.ndarray | None = None,
    smoothness_weight: float = 0.18,
    maximum_bin_jump: int = 6,
    start_candidate_count: int = 1,
) -> np.ndarray:
    logits = np.asarray(offset_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != len(offsets):
        raise ValueError("offset logits must have shape [contour_length, strip_bins]")
    length, bins = logits.shape
    if length < 3:
        raise ValueError("cyclic decoding requires at least three contour samples")
    if start_candidate_count < 1:
        raise ValueError("start_candidate_count must be positive")
    probability = _softmax(logits, axis=1)
    unary = -np.log(np.clip(probability, 1e-9, 1.0))
    corner = np.zeros(length, dtype=np.float32) if corner_probability is None else np.asarray(corner_probability, dtype=np.float32)
    reliable = np.ones(length, dtype=np.float32) if reliability is None else np.asarray(reliability, dtype=np.float32)
    unary *= (0.35 + 0.65 * np.clip(reliable, 0.0, 1.0))[:, None]
    # The contour start is arbitrary but its strongest mode is highly stable;
    # fixing that mode keeps the cyclic solve linear and avoids multiplying
    # latency by the number of strip bins.
    start_candidates = np.argsort(unary[0])[: min(bins, start_candidate_count)]
    best_cost = float("inf")
    best_path: np.ndarray | None = None
    state_indices = np.arange(bins, dtype=np.int32)
    deltas = np.arange(-maximum_bin_jump, maximum_bin_jump + 1, dtype=np.int32)
    source_indices = state_indices[:, None] + deltas[None, :]
    valid_sources = (source_indices >= 0) & (source_indices < bins)
    clipped_sources = np.clip(source_indices, 0, bins - 1)
    delta_squared = (deltas.astype(np.float32) ** 2)[None, :]
    for start in start_candidates:
        previous = np.full(bins, np.inf, dtype=np.float32)
        previous[int(start)] = unary[0, int(start)]
        back = np.zeros((length, bins), dtype=np.int16)
        for row in range(1, length):
            relaxation = 1.0 - 0.88 * float(np.clip(corner[row], 0.0, 1.0))
            transition = previous[clipped_sources] + (
                smoothness_weight * relaxation * delta_squared
            )
            transition = np.where(valid_sources, transition, np.inf)
            chosen_delta = np.argmin(transition, axis=1)
            current = unary[row] + transition[state_indices, chosen_delta]
            back[row] = source_indices[state_indices, chosen_delta]
            previous = current
        closure_delta = state_indices.astype(np.float32) - float(start)
        final_costs = previous + smoothness_weight * closure_delta * closure_delta
        end = int(np.argmin(final_costs))
        total = float(final_costs[end])
        if total >= best_cost:
            continue
        path = np.empty(length, dtype=np.int32)
        path[-1] = end
        for row in range(length - 1, 0, -1):
            path[row - 1] = int(back[row, path[row]])
        best_cost = total
        best_path = path
    assert best_path is not None

    # Recover sub-bin phase from the local posterior while retaining the DP's
    # globally consistent mode choice.
    decoded = np.empty(length, dtype=np.float32)
    for row, state in enumerate(best_path):
        lo = max(0, int(state) - 1)
        hi = min(bins, int(state) + 2)
        local = probability[row, lo:hi]
        local_mean = float(
            np.sum(local * offsets[lo:hi]) / max(1e-9, float(local.sum()))
        )
        decoded[row] = local_mean
    return decoded


def _align_rectilinear_candidate_to_native_phase(
    rgb_float: np.ndarray,
    candidate: Polygon,
    *,
    target_rgb: Any,
    image_shape: tuple[int, int],
    config: EdgeGraphConfig,
) -> _RectilinearNativeAlignment:
    """Move a rectilinear ring only when a centered outline is observable.

    The learned strip localizer can coherently choose one rendered side of a
    centered outline.  Once the rectilinear fitter has recovered the sharp
    run structure, this lane estimates one renderer-phase correction from
    native pixels and applies it to every parallel run before re-intersecting
    adjacent lines.  A proposal/validation split prevents the same pixels from
    both choosing and approving the correction.

    A single visible fill transition is deliberately insufficient.  The
    correction requires two-sided, width-consistent stroke evidence over a
    material fraction of the perimeter.  Solid fills therefore remain owned
    by the learned localizer rather than an unobservable stroke-width prior.
    """

    diagnostics: dict[str, object] = {
        "evaluated": True,
        "accepted": False,
        "reason": "insufficient-centered-stroke-evidence",
    }
    if (
        not isinstance(candidate, Polygon)
        or candidate.is_empty
        or not candidate.is_valid
        or len(candidate.interiors) != 0
        or candidate.area <= 0.0
    ):
        diagnostics["reason"] = "invalid-current-candidate"
        return _RectilinearNativeAlignment(None, diagnostics)
    points = np.asarray(candidate.exterior.coords[:-1], dtype=np.float32)
    if len(points) < 4:
        diagnostics["reason"] = "insufficient-runs"
        return _RectilinearNativeAlignment(None, diagnostics)
    deltas = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(deltas, axis=1)
    minimum_length = float(np.min(lengths, initial=float("inf")))
    diagnostics.update(
        {
            "run_count": len(points),
            "minimum_run_length_px": round(minimum_length, 6),
        }
    )
    if (
        not np.isfinite(lengths).all()
        or minimum_length
        < config.rectilinear_native_phase_minimum_line_length_px
    ):
        diagnostics["reason"] = "unsupported-short-run"
        return _RectilinearNativeAlignment(None, diagnostics)

    proposal_evidence: list[_RectilinearPhaseEvidence] = []
    validation_evidence: list[_RectilinearPhaseEvidence] = []
    line_models: list[tuple[np.ndarray, np.ndarray]] = []
    for start, delta, length in zip(points, deltas, lengths, strict=True):
        direction = np.asarray(delta / max(1e-9, float(length)), dtype=np.float32)
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        proposal, validation = _rectilinear_line_phase_evidence(
            rgb_float,
            start,
            direction,
            normal,
            float(length),
            target_rgb=target_rgb,
            config=config,
        )
        proposal_evidence.append(proposal)
        validation_evidence.append(validation)
        line_models.append((start, direction))

    proposal_shift, proposal_coverage, proposal_width, proposal_width_mad = (
        _rectilinear_ring_phase_consensus(
            proposal_evidence,
            lengths,
            config=config,
        )
    )
    diagnostics.update(
        {
            "proposal_shift_px": round(proposal_shift, 6),
            "proposal_paired_coverage": round(proposal_coverage, 6),
            "proposal_stroke_width_px": round(proposal_width, 6),
            "proposal_stroke_width_mad_px": round(proposal_width_mad, 6),
        }
    )
    if proposal_coverage < config.rectilinear_native_phase_minimum_proposal_coverage:
        diagnostics["reason"] = "proposal-coverage-gate"
        return _RectilinearNativeAlignment(None, diagnostics)
    if not (
        config.rectilinear_native_phase_minimum_shift_px
        <= abs(proposal_shift)
        <= config.rectilinear_native_phase_maximum_shift_px
    ):
        diagnostics["reason"] = "phase-displacement-gate"
        return _RectilinearNativeAlignment(None, diagnostics)
    if (
        proposal_width < config.rectilinear_native_phase_minimum_stroke_width_px
        or proposal_width_mad
        > config.rectilinear_native_phase_maximum_stroke_width_mad_px
    ):
        diagnostics["reason"] = "stroke-width-gate"
        return _RectilinearNativeAlignment(None, diagnostics)

    validation_coverage, validation_gain = _rectilinear_phase_validation(
        validation_evidence,
        lengths,
        proposal_shift=proposal_shift,
        config=config,
    )
    diagnostics.update(
        {
            "validation_paired_coverage": round(validation_coverage, 6),
            "validation_score_gain": round(validation_gain, 6),
        }
    )
    if (
        validation_coverage
        < config.rectilinear_native_phase_minimum_validation_coverage
    ):
        diagnostics["reason"] = "validation-coverage-gate"
        return _RectilinearNativeAlignment(None, diagnostics)
    if validation_gain < config.rectilinear_native_phase_minimum_validation_gain:
        diagnostics["reason"] = "validation-score-gate"
        return _RectilinearNativeAlignment(None, diagnostics)

    shifted_models: list[tuple[np.ndarray, np.ndarray]] = []
    for center, direction in line_models:
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        shifted_models.append(
            (
                np.asarray(
                    center + normal * proposal_shift,
                    dtype=np.float32,
                ),
                direction,
            )
        )
    vertices: list[np.ndarray] = []
    for index, model in enumerate(shifted_models):
        intersection = _line_intersection(shifted_models[index - 1], model)
        if intersection is None:
            diagnostics["reason"] = "parallel-adjacent-runs"
            return _RectilinearNativeAlignment(None, diagnostics)
        vertices.append(intersection)
    aligned = Polygon(np.asarray(vertices, dtype=np.float32))
    if (
        aligned.is_empty
        or not aligned.is_valid
        or aligned.area <= 0.0
        or len(aligned.interiors) != 0
    ):
        diagnostics["reason"] = "invalid-aligned-candidate"
        return _RectilinearNativeAlignment(None, diagnostics)

    height, width = image_shape
    current_mask = rasterize_geometry_mask(candidate, width=width, height=height)
    aligned_mask = rasterize_geometry_mask(aligned, width=width, height=height)
    current_area = int(np.count_nonzero(current_mask))
    aligned_area = int(np.count_nonzero(aligned_mask))
    intersection = int(np.count_nonzero(current_mask & aligned_mask))
    union = int(np.count_nonzero(current_mask | aligned_mask))
    overlap = intersection / max(1, union)
    area_ratio = aligned_area / max(1, current_area)
    current_topology = _digital_mask_topology(current_mask)
    aligned_topology = _digital_mask_topology(aligned_mask)
    diagnostics.update(
        {
            "candidate_current_iou": round(overlap, 6),
            "candidate_current_area_ratio": round(area_ratio, 6),
            "current_topology": {
                "components": current_topology[0],
                "holes": current_topology[1],
            },
            "candidate_topology": {
                "components": aligned_topology[0],
                "holes": aligned_topology[1],
            },
        }
    )
    if current_topology != aligned_topology:
        diagnostics["reason"] = "topology-gate"
        return _RectilinearNativeAlignment(None, diagnostics)
    if overlap < config.rectilinear_native_phase_minimum_current_iou:
        diagnostics["reason"] = "near-identity-overlap-gate"
        return _RectilinearNativeAlignment(None, diagnostics)
    if not (
        config.rectilinear_native_phase_minimum_area_ratio
        <= area_ratio
        <= config.rectilinear_native_phase_maximum_area_ratio
    ):
        diagnostics["reason"] = "near-identity-area-gate"
        return _RectilinearNativeAlignment(None, diagnostics)
    diagnostics["accepted"] = True
    diagnostics["reason"] = "accepted-centered-stroke-phase"
    return _RectilinearNativeAlignment(aligned, diagnostics)


def _rectilinear_line_phase_evidence(
    rgb_float: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    length: float,
    *,
    target_rgb: Any,
    config: EdgeGraphConfig,
) -> tuple[_RectilinearPhaseEvidence, _RectilinearPhaseEvidence]:
    margin = min(8.0, max(0.75, 0.12 * length))
    usable_length = max(0.0, length - 2.0 * margin)
    sample_count = max(
        8,
        min(
            160,
            int(
                math.ceil(
                    usable_length
                    / config.rectilinear_native_phase_sample_step_px
                )
            ),
        ),
    )
    distances = np.linspace(
        margin,
        max(margin, length - margin),
        sample_count,
        dtype=np.float32,
    )
    base = start[None, :] + distances[:, None] * direction[None, :]
    offsets = np.arange(
        -config.rectilinear_native_phase_radius_px,
        config.rectilinear_native_phase_radius_px
        + 0.5 * config.rectilinear_native_phase_step_px,
        config.rectilinear_native_phase_step_px,
        dtype=np.float32,
    )
    coordinates = (
        base[:, None, :]
        + normal[None, None, :] * offsets[None, :, None]
    )
    rgb_profiles = _sample_rgb_coordinates(
        rgb_float,
        coordinates.reshape(-1, 2),
    ).reshape(sample_count, len(offsets), 3)
    luminance = np.asarray(
        0.2126 * rgb_profiles[:, :, 0]
        + 0.7152 * rgb_profiles[:, :, 1]
        + 0.0722 * rgb_profiles[:, :, 2],
        dtype=np.float32,
    )
    normal_gradient = np.asarray(
        np.gradient(
            luminance,
            config.rectilinear_native_phase_step_px,
            axis=1,
        ),
        dtype=np.float32,
    )
    proposal_indices = np.arange(0, sample_count, 2, dtype=np.int32)
    validation_indices = np.arange(1, sample_count, 2, dtype=np.int32)
    if len(validation_indices) < 3:
        validation_indices = proposal_indices
    return (
        _summarize_rectilinear_phase(
            rgb_profiles[proposal_indices],
            luminance[proposal_indices],
            normal_gradient[proposal_indices],
            offsets,
            target_rgb=target_rgb,
            config=config,
        ),
        _summarize_rectilinear_phase(
            rgb_profiles[validation_indices],
            luminance[validation_indices],
            normal_gradient[validation_indices],
            offsets,
            target_rgb=target_rgb,
            config=config,
        ),
    )


def _summarize_rectilinear_phase(
    rgb_profiles: np.ndarray,
    luminance: np.ndarray,
    normal_gradient: np.ndarray,
    offsets: np.ndarray,
    *,
    target_rgb: Any,
    config: EdgeGraphConfig,
) -> _RectilinearPhaseEvidence:
    count = len(rgb_profiles)
    if count == 0:
        return _RectilinearPhaseEvidence(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    phase = estimate_edge_phase(
        rgb_profiles,
        luminance,
        normal_gradient,
        offsets,
        target_rgb=target_rgb,
        inside_sign=np.ones(count, dtype=np.int8),
    )
    paired = phase.paired & (phase.reliability >= 0.55)
    if not bool(np.any(paired)):
        return _RectilinearPhaseEvidence(count, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    centers = phase.centers_px[paired]
    weights = phase.reliability[paired]
    shift = _weighted_median_float(centers, weights)
    coherent = np.abs(centers - shift) <= (
        config.rectilinear_native_phase_maximum_edge_disagreement_px
    )
    if not bool(np.any(coherent)):
        return _RectilinearPhaseEvidence(count, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    coherent_centers = centers[coherent]
    coherent_weights = weights[coherent]
    shift = _weighted_median_float(coherent_centers, coherent_weights)
    coherent_widths = phase.stroke_widths_px[paired][coherent]
    phase_sigma = 0.45
    candidate_score = np.exp(
        -0.5 * ((coherent_centers - shift) / phase_sigma) ** 2
    )
    current_score = np.exp(
        -0.5 * (coherent_centers / phase_sigma) ** 2
    )
    score_gain = float(
        np.average(
            candidate_score - current_score,
            weights=coherent_weights,
        )
    )
    return _RectilinearPhaseEvidence(
        sample_count=count,
        paired_count=int(np.count_nonzero(coherent)),
        shift_px=shift,
        coverage=float(np.count_nonzero(coherent) / count),
        reliability=float(np.median(coherent_weights)),
        stroke_width_px=_weighted_median_float(
            coherent_widths,
            coherent_weights,
        ),
        score_gain=score_gain,
    )


def _rectilinear_ring_phase_consensus(
    evidence: list[_RectilinearPhaseEvidence],
    lengths: np.ndarray,
    *,
    config: EdgeGraphConfig,
) -> tuple[float, float, float, float]:
    eligible = [
        (item, float(length))
        for item, length in zip(evidence, lengths, strict=True)
        if (
            item.paired_count > 0
            and item.coverage >= 0.35
            and item.reliability >= 0.60
        )
    ]
    if not eligible:
        return 0.0, 0.0, 0.0, float("inf")
    weights = np.asarray(
        [length * item.coverage for item, length in eligible],
        dtype=np.float64,
    )
    shifts = np.asarray(
        [item.shift_px for item, _length in eligible],
        dtype=np.float64,
    )
    shift = _weighted_median_float(shifts, weights)
    coherent = np.abs(shifts - shift) <= (
        config.rectilinear_native_phase_maximum_edge_disagreement_px
    )
    if not bool(np.any(coherent)):
        return 0.0, 0.0, 0.0, float("inf")
    coherent_weights = weights[coherent]
    coherent_items = [
        item
        for (item, _length), keep in zip(eligible, coherent, strict=True)
        if bool(keep)
    ]
    perimeter = max(1e-9, float(np.sum(lengths)))
    coverage = float(np.sum(coherent_weights) / perimeter)
    widths = np.asarray(
        [item.stroke_width_px for item in coherent_items],
        dtype=np.float64,
    )
    width = _weighted_median_float(widths, coherent_weights)
    width_mad = _weighted_median_float(
        np.abs(widths - width),
        coherent_weights,
    )
    return shift, coverage, width, width_mad


def _rectilinear_phase_validation(
    evidence: list[_RectilinearPhaseEvidence],
    lengths: np.ndarray,
    *,
    proposal_shift: float,
    config: EdgeGraphConfig,
) -> tuple[float, float]:
    weights: list[float] = []
    gains: list[float] = []
    for item, length in zip(evidence, lengths, strict=True):
        if (
            item.paired_count == 0
            or item.coverage < 0.35
            or item.reliability < 0.55
            or abs(item.shift_px - proposal_shift)
            > config.rectilinear_native_phase_maximum_edge_disagreement_px
        ):
            continue
        weight = float(length) * item.coverage
        sigma = 0.45
        candidate_score = math.exp(
            -0.5 * ((item.shift_px - proposal_shift) / sigma) ** 2
        )
        current_score = math.exp(
            -0.5 * (item.shift_px / sigma) ** 2
        )
        weights.append(weight)
        gains.append(item.reliability * (candidate_score - current_score))
    perimeter = max(1e-9, float(np.sum(lengths)))
    coverage = float(sum(weights) / perimeter)
    gain = (
        float(np.average(gains, weights=weights))
        if weights
        else 0.0
    )
    return coverage, gain


def _weighted_median_float(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    importance = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(numeric) == 0 or len(numeric) != len(importance):
        return 0.0
    order = np.argsort(numeric)
    sorted_values = numeric[order]
    sorted_weights = np.maximum(0.0, importance[order])
    total = float(np.sum(sorted_weights))
    if total <= 0.0:
        return float(np.median(sorted_values))
    index = int(
        np.searchsorted(
            np.cumsum(sorted_weights),
            0.5 * total,
            side="left",
        )
    )
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def fit_corner_protected_ring(
    points: np.ndarray,
    *,
    corner_probability: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    target_rgb: Any = None,
    config: EdgeGraphConfig | None = None,
    _rgb_float: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], int, int]:
    cfg = config or EdgeGraphConfig()
    dense = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(dense) < 8:
        return _closed_coordinates(dense), 0, 0
    approximation = cv2.approxPolyDP(
        dense[:, np.newaxis, :],
        epsilon=float(cfg.line_anchor_tolerance_px),
        closed=True,
    )[:, 0, :]
    anchor_indices = {int(np.argmin(np.linalg.norm(dense - anchor, axis=1))) for anchor in approximation}
    protected = _corner_peak_indices(corner_probability, cfg.corner_probability_threshold)
    anchor_indices.update(protected)
    anchors = sorted(anchor_indices)
    if len(anchors) < 3:
        simplified = cv2.approxPolyDP(
            dense[:, np.newaxis, :],
            epsilon=float(cfg.curve_tolerance_px),
            closed=True,
        )[:, 0, :]
        return _closed_coordinates(simplified), 0, len(protected)

    runs: list[np.ndarray] = []
    line_models: list[tuple[np.ndarray, np.ndarray] | None] = []
    run_lengths: list[float] = []
    for run_index, start in enumerate(anchors):
        end = anchors[(run_index + 1) % len(anchors)]
        run = _cyclic_points(dense, start, end)
        runs.append(run)
        line_models.append(_fit_line_run(run, config=cfg))
        run_lengths.append(float(np.linalg.norm(run[-1] - run[0])))

    if rgb is not None and target_rgb is not None and cfg.line_alignment_radius_px > 0.0:
        rgb_float = (
            _rgb_float
            if _rgb_float is not None
            else np.asarray(rgb, dtype=np.float32) / 255.0
        )
        line_models = [
            _align_line_model_to_native_target(
                rgb_float,
                run,
                model,
                target_rgb=target_rgb,
                config=cfg,
            )
            for run, model in zip(runs, line_models, strict=True)
        ]

    corner_response: np.ndarray | None = None
    if rgb is not None and cfg.rectilinear_connector_enabled:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        corner_response = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
        response_scale = max(1e-9, float(np.quantile(corner_response, 0.995)))
        corner_response = np.clip(corner_response / response_scale, 0.0, 1.0)

    collapsed_connectors = _short_corner_connectors(
        runs,
        line_models,
        run_lengths,
        corner_response=corner_response,
        config=cfg,
    )

    output: list[np.ndarray] = []
    line_count = sum(
        model is not None and index not in collapsed_connectors
        for index, model in enumerate(line_models)
    )
    for run_index, run in enumerate(runs):
        if run_index in collapsed_connectors:
            continue
        model = line_models[run_index]
        previous_model = line_models[run_index - 1]
        if model is not None:
            previous_index = (run_index - 1) % len(runs)
            if previous_index in collapsed_connectors:
                start_point = collapsed_connectors[previous_index]
            else:
                start_point = _line_join(
                    previous_model,
                    model,
                    run[0],
                    maximum_displacement=cfg.maximum_intersection_displacement_px,
                )
            next_model = line_models[(run_index + 1) % len(runs)]
            next_index = (run_index + 1) % len(runs)
            if next_index in collapsed_connectors:
                end_point = collapsed_connectors[next_index]
            else:
                end_point = _line_join(
                    model,
                    next_model,
                    run[-1],
                    maximum_displacement=cfg.maximum_intersection_displacement_px,
                )
            if not output or np.linalg.norm(output[-1] - start_point) > 1e-4:
                output.append(start_point)
            output.append(end_point)
        else:
            simplified = cv2.approxPolyDP(
                run[:, np.newaxis, :],
                epsilon=float(cfg.curve_tolerance_px),
                closed=False,
            )[:, 0, :]
            for point in simplified:
                if not output or np.linalg.norm(output[-1] - point) > 1e-4:
                    output.append(point)
    coordinates = _closed_coordinates(np.asarray(output, dtype=np.float32))
    candidate = Polygon(coordinates)
    if not candidate.is_valid or candidate.is_empty:
        fallback = cv2.approxPolyDP(
            dense[:, np.newaxis, :],
            epsilon=float(cfg.curve_tolerance_px),
            closed=True,
        )[:, 0, :]
        return _closed_coordinates(fallback), 0, len(protected)
    return coordinates, line_count, len(protected)


def rasterize_geometry_mask(
    geometry: Polygon | MultiPolygon,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    for polygon in polygons:
        exterior = _cv_ring(polygon.exterior.coords, width=width, height=height)
        if len(exterior) >= 3:
            cv2.fillPoly(mask, [exterior], 1)
        for interior in polygon.interiors:
            hole = _cv_ring(interior.coords, width=width, height=height)
            if len(hole) >= 3:
                cv2.fillPoly(mask, [hole], 0)
    return mask.astype(bool)


def _run_edgegraph_session(
    features: np.ndarray,
    session: EdgeGraphSessionLike,
    *,
    config: EdgeGraphConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = features.shape[1]
    chunk = max(64, int(config.inference_chunk_length))
    context = max(0, min(int(config.inference_context), chunk // 3))
    input_name = session.get_inputs()[0].name
    offset_parts: list[np.ndarray] = []
    corner_parts: list[np.ndarray] = []
    reliability_parts: list[np.ndarray] = []
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        indices = np.arange(start - context, end + context) % length
        batch = np.ascontiguousarray(features[:, indices][np.newaxis], dtype=np.float32)
        outputs = session.run(None, {input_name: batch})
        if not outputs:
            raise ValueError("EdgeGraph refiner returned no outputs")
        offset = np.asarray(outputs[0], dtype=np.float32)
        if offset.ndim == 4 and offset.shape[1] == 1:
            offset = offset[:, 0]
        if offset.ndim != 3 or offset.shape[0] != 1 or offset.shape[2] != features.shape[2]:
            raise ValueError(f"EdgeGraph refiner returned unexpected offset shape {offset.shape}")
        core = slice(context, context + (end - start))
        offset_parts.append(offset[0, core])
        corner = (
            np.asarray(outputs[1], dtype=np.float32).reshape(1, -1)[0, core]
            if len(outputs) > 1
            else np.full(end - start, -4.0, dtype=np.float32)
        )
        reliability = (
            np.asarray(outputs[2], dtype=np.float32).reshape(1, -1)[0, core]
            if len(outputs) > 2
            else _deterministic_reliability_logits(offset[0, core])
        )
        corner_parts.append(corner)
        reliability_parts.append(reliability)
    return (
        np.concatenate(offset_parts, axis=0),
        np.concatenate(corner_parts, axis=0),
        np.concatenate(reliability_parts, axis=0),
    )


def _deterministic_reliability_logits(offset_logits: np.ndarray) -> np.ndarray:
    probability = _softmax(offset_logits, axis=1)
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-9, 1.0)), axis=1)
    normalized = entropy / max(1e-9, math.log(probability.shape[1]))
    return ((0.65 - normalized) * 8.0).astype(np.float32)


def _fit_line_run(run: np.ndarray, *, config: EdgeGraphConfig) -> tuple[np.ndarray, np.ndarray] | None:
    if len(run) < 3 or float(np.linalg.norm(run[-1] - run[0])) < config.minimum_line_length_px:
        return None
    center = run.mean(axis=0)
    _u, _s, vh = np.linalg.svd(run - center, full_matrices=False)
    direction = vh[0].astype(np.float32)
    direction /= max(1e-9, float(np.linalg.norm(direction)))
    residual = np.abs((run[:, 0] - center[0]) * direction[1] - (run[:, 1] - center[1]) * direction[0])
    # Map labels frequently cross an otherwise exact service-area edge.  A
    # localized crossing must not turn an entire long straight run back into a
    # rough curve, so acceptance is deliberately based on the inlier majority.
    if float(np.quantile(residual, config.line_fit_inlier_quantile)) > config.line_fit_tolerance_px:
        return None
    return center.astype(np.float32), direction


def _align_line_model_to_native_target(
    rgb_float: np.ndarray,
    run: np.ndarray,
    model: tuple[np.ndarray, np.ndarray] | None,
    *,
    target_rgb: Any,
    config: EdgeGraphConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    if model is None:
        return None
    center, direction = model
    projections = (run - center) @ direction
    low = float(np.quantile(projections, 0.06))
    high = float(np.quantile(projections, 0.94))
    if high - low < config.minimum_line_length_px:
        return model
    sample_count = max(12, min(160, int(math.ceil((high - low) / 3.0))))
    positions = np.linspace(low, high, sample_count, dtype=np.float32)
    base = center[None, :] + positions[:, None] * direction[None, :]
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 3) / 255.0

    plus = _sample_rgb_coordinates(rgb_float, base + normal[None, :] * 4.0)
    minus = _sample_rgb_coordinates(rgb_float, base - normal[None, :] * 4.0)
    plus_distance = float(np.median(np.linalg.norm(plus - target, axis=1)))
    minus_distance = float(np.median(np.linalg.norm(minus - target, axis=1)))
    inside_sign = 1.0 if plus_distance <= minus_distance else -1.0

    best_score = -float("inf")
    best_support = 0.0
    best_offset = 0.0
    zero_score = -float("inf")
    for offset in np.arange(
        -config.line_alignment_radius_px,
        config.line_alignment_radius_px + config.line_alignment_step_px * 0.5,
        config.line_alignment_step_px,
        dtype=np.float32,
    ):
        shifted = base + normal[None, :] * float(offset)
        inside_distances: list[np.ndarray] = []
        outside_distances: list[np.ndarray] = []
        contrasts: list[np.ndarray] = []
        for distance in (2.0, 4.0, 6.0):
            inside = _sample_rgb_coordinates(
                rgb_float,
                shifted + normal[None, :] * (inside_sign * distance),
            )
            outside = _sample_rgb_coordinates(
                rgb_float,
                shifted - normal[None, :] * (inside_sign * distance),
            )
            inside_distances.append(
                np.linalg.norm(inside - target, axis=1) / math.sqrt(3.0)
            )
            outside_distances.append(
                np.linalg.norm(outside - target, axis=1) / math.sqrt(3.0)
            )
            contrasts.append(np.linalg.norm(inside - outside, axis=1) / math.sqrt(3.0))
        inside_distance = np.median(inside_distances, axis=0)
        outside_distance = np.median(outside_distances, axis=0)
        separation = outside_distance - inside_distance
        contrast = np.median(contrasts, axis=0)
        supported = (separation >= 0.02) & (contrast >= 0.02)
        support = float(np.mean(supported))
        score = (
            support
            + 3.0 * float(np.median(np.clip(separation, 0.0, 1.0)))
            + float(np.median(contrast))
            - 0.004 * abs(float(offset))
        )
        if abs(float(offset)) < config.line_alignment_step_px * 0.25:
            zero_score = score
        if score > best_score:
            best_score = score
            best_support = support
            best_offset = float(offset)
    if best_support < config.line_alignment_minimum_support_fraction:
        return model
    if best_score < zero_score + config.line_alignment_minimum_score_gain:
        return model
    return (center + normal * best_offset).astype(np.float32), direction


def _short_corner_connectors(
    runs: list[np.ndarray],
    line_models: list[tuple[np.ndarray, np.ndarray] | None],
    run_lengths: list[float],
    *,
    corner_response: np.ndarray | None = None,
    config: EdgeGraphConfig,
) -> dict[int, np.ndarray]:
    """Find tiny bevels caused by a rounded selector corner.

    A connector is removed only when it is much shorter than both neighboring
    line runs, points in a genuinely different direction, and the outer-line
    intersection remains within the observed connector.  Long chamfers,
    parallel steps, and unsupported intersections are preserved.
    """

    collapsed: dict[int, np.ndarray] = {}
    count = len(runs)
    dominant_axes = (
        _dominant_rectilinear_axes(
            line_models,
            run_lengths,
            config=config,
        )
        if config.rectilinear_connector_enabled
        else None
    )
    for index, model in enumerate(line_models):
        if model is None:
            continue
        previous_index = (index - 1) % count
        next_index = (index + 1) % count
        previous_model = line_models[previous_index]
        next_model = line_models[next_index]
        if previous_model is None or next_model is None:
            continue
        neighboring_length = min(run_lengths[previous_index], run_lengths[next_index])
        strict_size = (
            run_lengths[index] <= config.maximum_corner_connector_length_px
            and run_lengths[index]
            <= config.maximum_corner_connector_neighbor_ratio * neighboring_length
        )
        relaxed_rectilinear_size = False
        if dominant_axes is not None:
            connector_axis_error = _axis_alignment_error_degrees(
                model[1],
                dominant_axes,
            )
            previous_axis_error = _axis_alignment_error_degrees(
                previous_model[1],
                dominant_axes,
            )
            next_axis_error = _axis_alignment_error_degrees(
                next_model[1],
                dominant_axes,
            )
            relaxed_rectilinear_size = (
                run_lengths[index] <= config.rectilinear_connector_maximum_length_px
                and run_lengths[index]
                <= config.rectilinear_connector_neighbor_ratio * neighboring_length
                and connector_axis_error
                >= config.rectilinear_connector_minimum_axis_error_degrees
                and previous_axis_error <= config.rectilinear_alignment_tolerance_degrees
                and next_axis_error <= config.rectilinear_alignment_tolerance_degrees
                and _undirected_line_angle_degrees(previous_model[1], next_model[1])
                >= config.rectilinear_connector_minimum_corner_angle_degrees
            )
        if not strict_size and not relaxed_rectilinear_size:
            continue
        connector_direction = model[1]
        if (
            _undirected_line_angle_degrees(connector_direction, previous_model[1])
            < config.minimum_corner_connector_angle_degrees
            or _undirected_line_angle_degrees(connector_direction, next_model[1])
            < config.minimum_corner_connector_angle_degrees
            or _undirected_line_angle_degrees(previous_model[1], next_model[1])
            < config.minimum_reconstructed_corner_angle_degrees
        ):
            continue
        intersection = _line_intersection(previous_model, next_model)
        if intersection is None:
            continue
        connector_distance = _point_segment_distance(
            intersection,
            runs[index][0],
            runs[index][-1],
        )
        maximum_distance = config.maximum_intersection_displacement_px
        if relaxed_rectilinear_size:
            maximum_distance = max(
                maximum_distance,
                min(14.0, run_lengths[index] * 0.75),
            )
        if connector_distance > maximum_distance:
            continue
        if (
            relaxed_rectilinear_size
            and not strict_size
            and _corner_response_near(corner_response, intersection)
            < config.rectilinear_connector_minimum_corner_response
        ):
            continue
        collapsed[index] = intersection
    return collapsed


def _corner_response_near(response: np.ndarray | None, point: np.ndarray) -> float:
    if response is None:
        return 0.0
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    x0 = max(0, x - 2)
    x1 = min(response.shape[1], x + 3)
    y0 = max(0, y - 2)
    y1 = min(response.shape[0], y + 3)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(np.max(response[y0:y1, x0:x1], initial=0.0))


def _dominant_rectilinear_axes(
    line_models: list[tuple[np.ndarray, np.ndarray] | None],
    run_lengths: list[float],
    *,
    config: EdgeGraphConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    weighted_cosine = 0.0
    weighted_sine = 0.0
    total_length = 0.0
    for model, length in zip(line_models, run_lengths, strict=True):
        if model is None or length <= 0.0:
            continue
        direction = model[1]
        angle = math.atan2(float(direction[1]), float(direction[0]))
        weighted_cosine += length * math.cos(4.0 * angle)
        weighted_sine += length * math.sin(4.0 * angle)
        total_length += length
    if total_length <= 0.0 or math.hypot(weighted_cosine, weighted_sine) < total_length * 0.20:
        return None
    axis_angle = 0.25 * math.atan2(weighted_sine, weighted_cosine)
    first = np.asarray([math.cos(axis_angle), math.sin(axis_angle)], dtype=np.float32)
    second = np.asarray([-first[1], first[0]], dtype=np.float32)
    axes = (first, second)
    aligned_length = sum(
        length
        for model, length in zip(line_models, run_lengths, strict=True)
        if model is not None
        and _axis_alignment_error_degrees(model[1], axes)
        <= config.rectilinear_alignment_tolerance_degrees
    )
    if aligned_length / total_length < config.rectilinear_minimum_aligned_length_fraction:
        return None
    return axes


def _axis_alignment_error_degrees(
    direction: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray],
) -> float:
    return min(
        _undirected_line_angle_degrees(direction, axis)
        for axis in axes
    )


def _undirected_line_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _line_intersection(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    first_center, first_direction = first
    second_center, second_direction = second
    matrix = np.stack([first_direction, -second_direction], axis=1)
    if abs(float(np.linalg.det(matrix))) < 1e-4:
        return None
    parameters = np.linalg.solve(matrix, second_center - first_center)
    return (first_center + first_direction * float(parameters[0])).astype(np.float32)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator <= 1e-9:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
    nearest = start + vector * fraction
    return float(np.linalg.norm(point - nearest))


def _line_join(
    first: tuple[np.ndarray, np.ndarray] | None,
    second: tuple[np.ndarray, np.ndarray] | None,
    fallback: np.ndarray,
    *,
    maximum_displacement: float,
) -> np.ndarray:
    if first is None or second is None:
        model = second if first is None else first
        if model is None:
            return fallback.astype(np.float32)
        center, direction = model
        return (center + direction * float(np.dot(fallback - center, direction))).astype(np.float32)
    intersection = _line_intersection(first, second)
    if intersection is None:
        return fallback.astype(np.float32)
    if float(np.linalg.norm(intersection - fallback)) > maximum_displacement:
        return fallback.astype(np.float32)
    return intersection.astype(np.float32)


def _corner_peak_indices(probability: np.ndarray | None, threshold: float) -> set[int]:
    if probability is None:
        return set()
    values = np.asarray(probability, dtype=np.float32).reshape(-1)
    peaks = (values >= threshold) & (values >= np.roll(values, 1)) & (values > np.roll(values, -1))
    indices = np.flatnonzero(peaks)
    if len(indices) <= 64:
        return {int(index) for index in indices}
    strongest = indices[np.argsort(values[indices])[-64:]]
    return {int(index) for index in strongest}


def _reliable_corner_anchor_probability(
    corner_probability: np.ndarray,
    reliability: np.ndarray,
    *,
    minimum_reliability: float,
) -> np.ndarray:
    """Suppress discrete corner anchors on profiles the model marks unreliable."""

    corners = np.asarray(corner_probability, dtype=np.float32).reshape(-1)
    reliable = np.asarray(reliability, dtype=np.float32).reshape(-1)
    if corners.shape != reliable.shape:
        raise ValueError("corner probabilities and reliability must have matching shapes")
    return np.ascontiguousarray(
        np.where(reliable >= float(minimum_reliability), corners, 0.0),
        dtype=np.float32,
    )


def _vector_corner_anchor_probability(
    corner_probability: np.ndarray,
    reliability: np.ndarray,
    *,
    learned_vector_owner: bool,
    config: EdgeGraphConfig,
) -> np.ndarray:
    anchored = _reliable_corner_anchor_probability(
        corner_probability,
        reliability,
        minimum_reliability=config.corner_anchor_minimum_reliability,
    )
    if learned_vector_owner and not config.learned_corner_anchors_enabled:
        # The current corner head remains a continuous cyclic-smoothing cue,
        # but its local peaks are not calibrated well enough to become
        # immutable vector vertices. Robust line-run intersections own sharp
        # reconstruction; freezing noisy learned peaks creates bevels and
        # micro-segments that materially worsen corner fidelity.
        return np.zeros_like(anchored, dtype=np.float32)
    return anchored


def _coarse_polygon_for_contour(contours: list[np.ndarray], hierarchy: np.ndarray, index: int) -> Polygon:
    exterior = [(float(point[0][0]), float(point[0][1])) for point in contours[index]]
    holes: list[list[tuple[float, float]]] = []
    child = int(hierarchy[index][2])
    while child != -1:
        holes.append([(float(point[0][0]), float(point[0][1])) for point in contours[child]])
        child = int(hierarchy[child][0])
    polygon = Polygon(exterior, holes)
    return polygon if polygon.is_valid else polygon.buffer(0)


def _topology_compatible(candidate: Polygon | MultiPolygon, coarse: Polygon | MultiPolygon) -> bool:
    if candidate.is_empty or not candidate.is_valid:
        return False
    # ``buffer(0)`` can split a self-touching raster contour into one real
    # component plus a sub-pixel/handful-of-pixels repair sliver.  Such a sliver
    # is not semantic topology and must not force an otherwise supported sharp
    # vector back to the dense coarse contour.  Keep this deliberately tiny:
    # anything larger than 8 px² or 0.005% of the coarse area remains material.
    repair_artifact_area = max(8.0, float(coarse.area) * 0.00005)
    candidate_parts = [candidate] if isinstance(candidate, Polygon) else [
        part for part in candidate.geoms if part.area > repair_artifact_area
    ]
    coarse_parts = [coarse] if isinstance(coarse, Polygon) else [
        part for part in coarse.geoms if part.area > repair_artifact_area
    ]
    if not candidate_parts or not coarse_parts:
        return False
    if len(candidate_parts) != len(coarse_parts):
        return False
    if sum(len(part.interiors) for part in candidate_parts) != sum(len(part.interiors) for part in coarse_parts):
        return False
    if coarse.area <= 0.0:
        return False
    ratio = candidate.area / coarse.area
    return 0.70 <= ratio <= 1.35


def _recover_source_native_target_fill_geometry(
    rgb: np.ndarray,
    *,
    current_mask: np.ndarray,
    target_rgb: Any,
    seed_point: tuple[float, float] | None,
    proposal_diagnostics: dict[str, object],
    config: EdgeGraphConfig,
    native_fields: _NativeImageFields,
) -> _TargetFillReconstruction:
    """Recover a sharp vector only from an independently supported fill mask.

    This is a narrow post-localization escape hatch for a structural proposal
    whose tiny terminal corner is visible in the source fill but is rounded
    away by normal-profile localization.  It is not a general color
    segmentation path: the source-native component must be almost identical
    to the localized result, preserve its simple topology, and have materially
    stronger native edge alignment.  Otherwise the learned vector remains the
    sole output.
    """

    diagnostics: dict[str, object] = {
        "evaluated": False,
        "accepted": False,
        "reason": "no-accepted-structural-proposal",
    }
    accepted_proposals = int(proposal_diagnostics.get("accepted_count", 0))
    if accepted_proposals < 1:
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["evaluated"] = True
    if target_rgb is None or seed_point is None:
        diagnostics["reason"] = "missing-guidance"
        return _TargetFillReconstruction(None, None, diagnostics)

    target = np.asarray(target_rgb, dtype=np.float32).reshape(-1)
    if target.size != 3 or not np.isfinite(target).all():
        diagnostics["reason"] = "invalid-target-color"
        return _TargetFillReconstruction(None, None, diagnostics)
    if float(np.max(target, initial=0.0)) <= 1.5:
        target = target * 255.0
    target = np.clip(target, 0.0, 255.0)
    image = np.asarray(rgb, dtype=np.float32)
    target_distance = np.linalg.norm(image - target.reshape(1, 1, 3), axis=2)
    target_pixels = target_distance <= 20.0
    count, labels = cv2.connectedComponents(
        target_pixels.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        diagnostics["reason"] = "missing-target-component"
        return _TargetFillReconstruction(None, None, diagnostics)

    seed_x = int(round(float(seed_point[0])))
    seed_y = int(round(float(seed_point[1])))
    label = 0
    if 0 <= seed_x < labels.shape[1] and 0 <= seed_y < labels.shape[0]:
        label = int(labels[seed_y, seed_x])
    nearest_distance = 0.0
    if label == 0:
        ys, xs = np.where(target_pixels)
        if not len(xs):
            diagnostics["reason"] = "missing-target-component"
            return _TargetFillReconstruction(None, None, diagnostics)
        squared_distance = (xs - seed_x) ** 2 + (ys - seed_y) ** 2
        nearest = int(np.argmin(squared_distance))
        nearest_distance = float(math.sqrt(float(squared_distance[nearest])))
        label = int(labels[ys[nearest], xs[nearest]])
    component = labels == label
    component = cv2.dilate(
        component.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)

    current = np.asarray(current_mask, dtype=bool)
    current_area = int(np.count_nonzero(current))
    component_area = int(np.count_nonzero(component))
    if current_area == 0 or component_area == 0:
        diagnostics["reason"] = "empty-candidate"
        return _TargetFillReconstruction(None, None, diagnostics)
    intersection = int(np.count_nonzero(component & current))
    union = int(np.count_nonzero(component | current))
    overlap = intersection / max(1, union)
    area_ratio = component_area / current_area
    component_topology = _digital_mask_topology(component)
    current_topology = _digital_mask_topology(current)
    component_energy_median, component_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        component,
    )
    current_energy_median, current_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        current,
    )
    diagnostics.update(
        {
            "nearest_target_distance_px": round(nearest_distance, 6),
            "candidate_current_iou": round(overlap, 6),
            "candidate_current_area_ratio": round(area_ratio, 6),
            "candidate_topology": {
                "components": component_topology[0],
                "holes": component_topology[1],
            },
            "current_topology": {
                "components": current_topology[0],
                "holes": current_topology[1],
            },
            "candidate_boundary_energy_median": round(
                component_energy_median,
                6,
            ),
            "current_boundary_energy_median": round(
                current_energy_median,
                6,
            ),
            "candidate_boundary_energy_mean": round(
                component_energy_mean,
                6,
            ),
            "current_boundary_energy_mean": round(
                current_energy_mean,
                6,
            ),
        }
    )
    if not _target_fill_candidate_gate(
        proposal_accepted_count=accepted_proposals,
        nearest_target_distance_px=nearest_distance,
        candidate_current_iou=overlap,
        candidate_current_area_ratio=area_ratio,
        candidate_topology=component_topology,
        current_topology=current_topology,
        candidate_boundary_energy_median=component_energy_median,
        current_boundary_energy_median=current_energy_median,
        candidate_boundary_energy_mean=component_energy_mean,
        current_boundary_energy_mean=current_energy_mean,
    ):
        diagnostics["reason"] = "candidate-gate"
        return _TargetFillReconstruction(None, None, diagnostics)

    contours, _hierarchy = cv2.findContours(
        component.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        diagnostics["reason"] = "missing-candidate-contour"
        return _TargetFillReconstruction(None, None, diagnostics)
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    candidate_config = replace(
        config,
        line_anchor_tolerance_px=min(2.0, config.line_anchor_tolerance_px),
    )
    coordinates, line_count, _protected = fit_corner_protected_ring(
        contour,
        config=candidate_config,
    )
    candidate: Polygon | MultiPolygon = Polygon(coordinates)
    if candidate.is_empty or not candidate.is_valid or candidate.area <= 0.0:
        diagnostics["reason"] = "invalid-vector-candidate"
        return _TargetFillReconstruction(None, None, diagnostics)
    # OpenCV contours describe pixel centers.  The small mitred inset converts
    # that cell-centered outline to the vector phase consumed by our rasterizer
    # without rounding the independently observed sharp corners.
    candidate = candidate.buffer(-0.25, join_style=2)
    if (
        not isinstance(candidate, Polygon)
        or candidate.is_empty
        or not candidate.is_valid
        or candidate.area <= 0.0
        or len(candidate.interiors) != 0
    ):
        diagnostics["reason"] = "invalid-inset-candidate"
        return _TargetFillReconstruction(None, None, diagnostics)
    candidate = orient_pixel_polygon(candidate)
    candidate_mask = rasterize_geometry_mask(
        candidate,
        width=current.shape[1],
        height=current.shape[0],
    )
    final_topology = _digital_mask_topology(candidate_mask)
    final_area_ratio = int(np.count_nonzero(candidate_mask)) / current_area
    final_intersection = int(np.count_nonzero(candidate_mask & current))
    final_union = int(np.count_nonzero(candidate_mask | current))
    final_overlap = final_intersection / max(1, final_union)
    diagnostics.update(
        {
            "vector_line_count": line_count,
            "vector_vertex_count": len(candidate.exterior.coords) - 1,
            "vector_current_iou": round(final_overlap, 6),
            "vector_current_area_ratio": round(final_area_ratio, 6),
        }
    )
    if (
        final_topology != current_topology
        or final_overlap < 0.985
        or not 0.98 <= final_area_ratio <= 1.02
    ):
        diagnostics["reason"] = "vector-gate"
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["accepted"] = True
    diagnostics["reason"] = "accepted"
    return _TargetFillReconstruction(candidate, candidate_mask, diagnostics)


def _recover_source_native_flat_fill_geometry(
    rgb: np.ndarray,
    *,
    current_mask: np.ndarray,
    target_rgb: Any,
    seed_point: tuple[float, float] | None,
    proposal_diagnostics: dict[str, object],
    shortcut_count: int,
    mean_reliability: float,
    config: EdgeGraphConfig,
    native_fields: _NativeImageFields,
) -> _TargetFillReconstruction:
    """Recover an occluded sharp ring from a high-purity flat fill.

    This lane is intentionally separate from structural proposals.  It only
    runs when a source-supported shortcut was needed, the learned localization
    remained uncertain, and no proposal already owns geometry recovery.  The
    target radius is learned from pixels well inside and outside the localized
    ring; a fixed RGB radius is unsafe for pale overlays close to a white map.
    A large close may bridge map roads or labels, but its result is exportable
    only when topology, area, overlap, color separation, and native boundary
    energy independently agree with the learned vector.
    """

    diagnostics: dict[str, object] = {
        "evaluated": False,
        "accepted": False,
        "reason": "insufficient-structural-evidence",
        "mode": "adaptive-flat-fill",
    }
    accepted_proposals = int(proposal_diagnostics.get("accepted_count", 0))
    if (
        not config.flat_fill_recovery_enabled
        or accepted_proposals != 0
        or shortcut_count < 1
        or not np.isfinite(mean_reliability)
        or mean_reliability >= config.flat_fill_maximum_reliability
    ):
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["evaluated"] = True
    if target_rgb is None or seed_point is None:
        diagnostics["reason"] = "missing-guidance"
        return _TargetFillReconstruction(None, None, diagnostics)

    target = np.asarray(target_rgb, dtype=np.float32).reshape(-1)
    if target.size != 3 or not np.isfinite(target).all():
        diagnostics["reason"] = "invalid-target-color"
        return _TargetFillReconstruction(None, None, diagnostics)
    if float(np.max(target, initial=0.0)) <= 1.5:
        target = target * 255.0
    target = np.clip(target, 0.0, 255.0)

    current = np.asarray(current_mask, dtype=bool)
    current_area = int(np.count_nonzero(current))
    if current_area == 0 or current.all():
        diagnostics["reason"] = "empty-or-full-current-mask"
        return _TargetFillReconstruction(None, None, diagnostics)
    distance_inside = cv2.distanceTransform(
        current.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    distance_outside = cv2.distanceTransform(
        (~current).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    inside_training = distance_inside > config.flat_fill_training_margin_px
    outside_training = distance_outside > config.flat_fill_training_margin_px
    if (
        int(np.count_nonzero(inside_training)) < 64
        or int(np.count_nonzero(outside_training)) < 64
    ):
        diagnostics["reason"] = "insufficient-color-training"
        return _TargetFillReconstruction(None, None, diagnostics)

    image = np.asarray(rgb, dtype=np.float32)
    target_distance = np.linalg.norm(
        image - target.reshape(1, 1, 3),
        axis=2,
    )
    inside_ceiling = float(np.quantile(target_distance[inside_training], 0.95))
    outside_floor = float(np.quantile(target_distance[outside_training], 0.05))
    target_separation = outside_floor - inside_ceiling
    target_radius = min(
        config.flat_fill_maximum_target_radius,
        max(1.0, 0.5 * (inside_ceiling + outside_floor)),
    )
    diagnostics.update(
        {
            "proposal_accepted_count": accepted_proposals,
            "shortcut_count": int(shortcut_count),
            "mean_reliability": round(float(mean_reliability), 6),
            "inside_target_distance_p95": round(inside_ceiling, 6),
            "outside_target_distance_p05": round(outside_floor, 6),
            "target_separation": round(target_separation, 6),
            "target_radius": round(target_radius, 6),
        }
    )
    if target_separation < config.flat_fill_minimum_target_separation:
        diagnostics["reason"] = "weak-target-separation"
        return _TargetFillReconstruction(None, None, diagnostics)

    target_pixels = target_distance <= target_radius
    kernel_size = int(
        math.ceil(
            min(current.shape)
            * config.flat_fill_closing_fraction
        )
    )
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(
        3,
        min(config.flat_fill_maximum_closing_kernel_px, kernel_size),
    )
    closed = cv2.morphologyEx(
        target_pixels.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        ),
    ).astype(bool)
    count, labels = cv2.connectedComponents(
        closed.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        diagnostics["reason"] = "missing-target-component"
        return _TargetFillReconstruction(None, None, diagnostics)

    seed_x = int(round(float(seed_point[0])))
    seed_y = int(round(float(seed_point[1])))
    label = 0
    if 0 <= seed_x < labels.shape[1] and 0 <= seed_y < labels.shape[0]:
        label = int(labels[seed_y, seed_x])
    nearest_distance = 0.0
    if label == 0:
        ys, xs = np.where(closed)
        if not len(xs):
            diagnostics["reason"] = "missing-target-component"
            return _TargetFillReconstruction(None, None, diagnostics)
        squared_distance = (xs - seed_x) ** 2 + (ys - seed_y) ** 2
        nearest = int(np.argmin(squared_distance))
        nearest_distance = float(math.sqrt(float(squared_distance[nearest])))
        label = int(labels[ys[nearest], xs[nearest]])
    component = labels == label
    component_area = int(np.count_nonzero(component))
    if component_area == 0:
        diagnostics["reason"] = "empty-candidate"
        return _TargetFillReconstruction(None, None, diagnostics)

    component_topology = _digital_mask_topology(component)
    current_topology = _digital_mask_topology(current)
    intersection = int(np.count_nonzero(component & current))
    union = int(np.count_nonzero(component | current))
    overlap = intersection / max(1, union)
    area_ratio = component_area / current_area
    closed_addition_fraction = (
        int(np.count_nonzero(component & ~target_pixels))
        / current_area
    )
    component_energy_median, component_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        component,
    )
    current_energy_median, current_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        current,
    )
    diagnostics.update(
        {
            "closing_kernel_px": kernel_size,
            "nearest_target_distance_px": round(nearest_distance, 6),
            "closed_addition_fraction": round(closed_addition_fraction, 6),
            "candidate_current_iou": round(overlap, 6),
            "candidate_current_area_ratio": round(area_ratio, 6),
            "candidate_topology": {
                "components": component_topology[0],
                "holes": component_topology[1],
            },
            "current_topology": {
                "components": current_topology[0],
                "holes": current_topology[1],
            },
            "candidate_boundary_energy_median": round(
                component_energy_median,
                6,
            ),
            "current_boundary_energy_median": round(
                current_energy_median,
                6,
            ),
            "candidate_boundary_energy_mean": round(
                component_energy_mean,
                6,
            ),
            "current_boundary_energy_mean": round(
                current_energy_mean,
                6,
            ),
        }
    )
    strict_candidate = _flat_fill_candidate_gate(
        proposal_accepted_count=accepted_proposals,
        shortcut_count=shortcut_count,
        mean_reliability=mean_reliability,
        target_separation=target_separation,
        nearest_target_distance_px=nearest_distance,
        closed_addition_fraction=closed_addition_fraction,
        candidate_current_iou=overlap,
        candidate_current_area_ratio=area_ratio,
        candidate_topology=component_topology,
        current_topology=current_topology,
        candidate_boundary_energy_median=component_energy_median,
        current_boundary_energy_median=current_energy_median,
        candidate_boundary_energy_mean=component_energy_mean,
        current_boundary_energy_mean=current_energy_mean,
        config=config,
    )
    structural_candidate = _flat_fill_structural_candidate_gate(
        proposal_accepted_count=accepted_proposals,
        shortcut_count=shortcut_count,
        mean_reliability=mean_reliability,
        target_separation=target_separation,
        nearest_target_distance_px=nearest_distance,
        closed_addition_fraction=closed_addition_fraction,
        candidate_current_iou=overlap,
        candidate_current_area_ratio=area_ratio,
        candidate_topology=component_topology,
        current_topology=current_topology,
        candidate_boundary_energy_median=component_energy_median,
        current_boundary_energy_median=current_energy_median,
        candidate_boundary_energy_mean=component_energy_mean,
        current_boundary_energy_mean=current_energy_mean,
        config=config,
    )
    if strict_candidate:
        candidate_gate_mode = "near-identity"
    elif structural_candidate:
        candidate_gate_mode = "structural-recovery"
    else:
        diagnostics["reason"] = "candidate-gate"
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["candidate_gate_mode"] = candidate_gate_mode

    contours, _hierarchy = cv2.findContours(
        component.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        diagnostics["reason"] = "missing-candidate-contour"
        return _TargetFillReconstruction(None, None, diagnostics)
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    candidate_config = replace(
        config,
        line_anchor_tolerance_px=min(2.0, config.line_anchor_tolerance_px),
    )
    coordinates, line_count, _protected = fit_corner_protected_ring(
        contour,
        config=candidate_config,
    )
    candidate: Polygon | MultiPolygon = Polygon(coordinates)
    if candidate.is_empty or not candidate.is_valid or candidate.area <= 0.0:
        diagnostics["reason"] = "invalid-vector-candidate"
        return _TargetFillReconstruction(None, None, diagnostics)
    candidate = candidate.buffer(
        config.flat_fill_vector_phase_px,
        join_style=2,
    )
    if (
        not isinstance(candidate, Polygon)
        or candidate.is_empty
        or not candidate.is_valid
        or candidate.area <= 0.0
        or len(candidate.interiors) != 0
    ):
        diagnostics["reason"] = "invalid-phased-candidate"
        return _TargetFillReconstruction(None, None, diagnostics)
    candidate = orient_pixel_polygon(candidate)
    candidate_mask = rasterize_geometry_mask(
        candidate,
        width=current.shape[1],
        height=current.shape[0],
    )
    final_topology = _digital_mask_topology(candidate_mask)
    final_area_ratio = int(np.count_nonzero(candidate_mask)) / current_area
    final_intersection = int(np.count_nonzero(candidate_mask & current))
    final_union = int(np.count_nonzero(candidate_mask | current))
    final_overlap = final_intersection / max(1, final_union)
    diagnostics.update(
        {
            "vector_line_count": line_count,
            "vector_vertex_count": len(candidate.exterior.coords) - 1,
            "vector_phase_px": config.flat_fill_vector_phase_px,
            "vector_current_iou": round(final_overlap, 6),
            "vector_current_area_ratio": round(final_area_ratio, 6),
        }
    )
    if candidate_gate_mode == "structural-recovery":
        minimum_vector_iou = config.flat_fill_structural_minimum_vector_iou
        minimum_vector_area_ratio = (
            config.flat_fill_structural_minimum_vector_area_ratio
        )
        maximum_vector_area_ratio = (
            config.flat_fill_structural_maximum_vector_area_ratio
        )
    else:
        minimum_vector_iou = config.flat_fill_minimum_vector_iou
        minimum_vector_area_ratio = config.flat_fill_minimum_vector_area_ratio
        maximum_vector_area_ratio = config.flat_fill_maximum_vector_area_ratio
    if (
        final_topology != current_topology
        or final_overlap < minimum_vector_iou
        or not (
            minimum_vector_area_ratio
            <= final_area_ratio
            <= maximum_vector_area_ratio
        )
    ):
        diagnostics["reason"] = "vector-gate"
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["accepted"] = True
    diagnostics["reason"] = "accepted"
    return _TargetFillReconstruction(candidate, candidate_mask, diagnostics)


def _recover_source_native_graphcut_geometry(
    rgb: np.ndarray,
    *,
    current_mask: np.ndarray,
    seed_point: tuple[float, float] | None,
    proposal_diagnostics: dict[str, object],
    maximum_displacement_px: float,
    config: EdgeGraphConfig,
    native_fields: _NativeImageFields,
    current_geometry: Polygon | MultiPolygon | None = None,
    target_rgb: tuple[int, int, int] | None = None,
    mean_reliability: float = 1.0,
) -> _TargetFillReconstruction:
    """Retry one bounded native GraphCut under a fail-closed necessity route.

    Normal-strip localization cannot reach source structure beyond its 12 px
    support, so the original strip-saturation route retains priority and its
    existing behavior.  A centered-stroke route may also run when either the
    current localization is observably unreliable or a separate phase probe on
    high-reliability current geometry observes a coherent inward correction.
    Those two centered necessity routes are mutually exclusive.  The
    selector-derived sure foreground/background seeds keep every route a
    correction of the selected region rather than unconstrained segmentation.
    """

    diagnostics: dict[str, object] = {
        "evaluated": False,
        "accepted": False,
        "reason": "insufficient-strip-saturation",
        "mode": "strip-saturation-graphcut",
        "maximum_displacement_px": round(float(maximum_displacement_px), 6),
        "mean_reliability": round(float(mean_reliability), 6),
    }
    accepted_proposals = int(proposal_diagnostics.get("accepted_count", 0))
    strip_saturation_eligible = bool(
        np.isfinite(maximum_displacement_px)
        and maximum_displacement_px
        >= config.graphcut_minimum_maximum_displacement_px
    )
    diagnostics["necessity_routes"] = {
        "strip_saturation": {
            "eligible": strip_saturation_eligible,
            "minimum_maximum_displacement_px": (
                config.graphcut_minimum_maximum_displacement_px
            ),
        },
        "low_current_reliability": {
            "eligible": bool(
                np.isfinite(mean_reliability)
                and mean_reliability
                <= config.graphcut_centered_maximum_current_reliability
            ),
            "maximum_current_reliability": (
                config.graphcut_centered_maximum_current_reliability
            ),
        },
        "current_phase_positive": {
            "evaluated": False,
            "eligible": False,
            "reason": "not-evaluated",
        },
    }
    if not config.graphcut_recovery_enabled or accepted_proposals != 0:
        diagnostics["reason"] = (
            "disabled"
            if not config.graphcut_recovery_enabled
            else "structural-owner-conflict"
        )
        return _TargetFillReconstruction(None, None, diagnostics)
    if not np.isfinite(maximum_displacement_px) or not np.isfinite(
        mean_reliability
    ):
        diagnostics["reason"] = "non-finite-current-diagnostics"
        return _TargetFillReconstruction(None, None, diagnostics)

    diagnostics["evaluated"] = True
    if seed_point is None:
        diagnostics["reason"] = "missing-seed"
        return _TargetFillReconstruction(None, None, diagnostics)

    current = np.asarray(current_mask, dtype=bool)
    current_area = int(np.count_nonzero(current))
    current_topology = _digital_mask_topology(current)
    if current_area == 0 or current.all() or current_topology != (1, 0):
        diagnostics["reason"] = "unsupported-current-topology"
        return _TargetFillReconstruction(None, None, diagnostics)

    necessity_route: str | None = None
    if strip_saturation_eligible:
        necessity_route = "strip-saturation"
    elif config.graphcut_centered_recovery_enabled:
        low_reliability_eligible = bool(
            mean_reliability
            <= config.graphcut_centered_maximum_current_reliability
        )
        if low_reliability_eligible:
            necessity_route = "low-current-reliability-structural-residual"
            diagnostics["mode"] = "centered-stroke-graphcut"
            diagnostics["necessity_routes"]["current_phase_positive"] = {
                "evaluated": False,
                "eligible": False,
                "reason": "mutually-exclusive-low-reliability-route",
            }
        elif (
            isinstance(current_geometry, Polygon)
            and target_rgb is not None
        ):
            current_phase = _graphcut_phase_evidence(
                rgb,
                current,
                current_geometry,
                target_rgb=target_rgb,
                config=config,
                native_fields=native_fields,
                sample_step_px=(
                    config.graphcut_current_phase_sample_step_px
                ),
            )
            current_phase_eligible = _graphcut_current_phase_positive_gate(
                current_phase,
                config=config,
            )
            diagnostics["necessity_routes"]["current_phase_positive"] = {
                "evaluated": True,
                "eligible": current_phase_eligible,
                "reason": (
                    "accepted"
                    if current_phase_eligible
                    else "phase-evidence-gate"
                ),
                "phase": current_phase,
            }
            if current_phase_eligible:
                necessity_route = "current-geometry-paired-phase-positive"
                diagnostics["mode"] = "centered-stroke-graphcut"
        else:
            diagnostics["necessity_routes"]["current_phase_positive"] = {
                "evaluated": False,
                "eligible": False,
                "reason": (
                    "missing-current-polygon"
                    if not isinstance(current_geometry, Polygon)
                    else "missing-target-color"
                ),
            }
    if necessity_route is None:
        diagnostics["reason"] = "necessity-gate"
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["necessity_route"] = necessity_route

    distance_inside = cv2.distanceTransform(
        current.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    distance_outside = cv2.distanceTransform(
        (~current).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    sure_foreground = (
        distance_inside >= config.graphcut_sure_foreground_margin_px
    )
    sure_background = (
        distance_outside >= config.graphcut_sure_background_margin_px
    )
    if (
        int(np.count_nonzero(sure_foreground)) < 64
        or int(np.count_nonzero(sure_background)) < 64
    ):
        diagnostics["reason"] = "insufficient-graphcut-seeds"
        return _TargetFillReconstruction(None, None, diagnostics)

    graphcut_mask = np.where(
        current,
        cv2.GC_PR_FGD,
        cv2.GC_PR_BGD,
    ).astype(np.uint8)
    graphcut_mask[sure_foreground] = cv2.GC_FGD
    graphcut_mask[sure_background] = cv2.GC_BGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        # OpenCV's GMM initialization otherwise depends on process-global RNG
        # state, which can make a promotion-gated candidate nondeterministic.
        cv2.setRNGSeed(0)
        cv2.grabCut(
            np.ascontiguousarray(rgb, dtype=np.uint8),
            graphcut_mask,
            None,
            background_model,
            foreground_model,
            1,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        diagnostics["reason"] = "graphcut-failed"
        return _TargetFillReconstruction(None, None, diagnostics)
    candidate = np.isin(
        graphcut_mask,
        (cv2.GC_FGD, cv2.GC_PR_FGD),
    )
    candidate = _select_guided_mask(
        candidate,
        ExtractionHints(seed_point=seed_point),
    )
    candidate_area = int(np.count_nonzero(candidate))
    candidate_topology = _digital_mask_topology(candidate)
    intersection = int(np.count_nonzero(candidate & current))
    union = int(np.count_nonzero(candidate | current))
    overlap = intersection / max(1, union)
    area_ratio = candidate_area / current_area
    candidate_energy_median, candidate_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        candidate,
    )
    current_energy_median, current_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        current,
    )
    diagnostics.update(
        {
            "proposal_accepted_count": accepted_proposals,
            "sure_foreground_pixel_count": int(np.count_nonzero(sure_foreground)),
            "sure_background_pixel_count": int(np.count_nonzero(sure_background)),
            "candidate_current_iou": round(overlap, 6),
            "candidate_current_area_ratio": round(area_ratio, 6),
            "candidate_topology": {
                "components": candidate_topology[0],
                "holes": candidate_topology[1],
            },
            "current_topology": {
                "components": current_topology[0],
                "holes": current_topology[1],
            },
            "candidate_boundary_energy_median": round(
                candidate_energy_median,
                6,
            ),
            "current_boundary_energy_median": round(
                current_energy_median,
                6,
            ),
            "candidate_boundary_energy_mean": round(
                candidate_energy_mean,
                6,
            ),
            "current_boundary_energy_mean": round(
                current_energy_mean,
                6,
            ),
        }
    )
    candidate_gate_evidence = {
        "proposal_accepted_count": accepted_proposals,
        "candidate_current_iou": overlap,
        "candidate_current_area_ratio": area_ratio,
        "candidate_topology": candidate_topology,
        "current_topology": current_topology,
        "candidate_boundary_energy_median": candidate_energy_median,
        "current_boundary_energy_median": current_energy_median,
        "candidate_boundary_energy_mean": candidate_energy_mean,
        "current_boundary_energy_mean": current_energy_mean,
        "config": config,
    }
    candidate_accepted = (
        _graphcut_candidate_gate(
            maximum_displacement_px=maximum_displacement_px,
            **candidate_gate_evidence,
        )
        if necessity_route == "strip-saturation"
        else _centered_graphcut_candidate_gate(**candidate_gate_evidence)
    )
    if not candidate_accepted:
        diagnostics["reason"] = "candidate-gate"
        return _TargetFillReconstruction(None, None, diagnostics)

    contours, _hierarchy = cv2.findContours(
        candidate.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        diagnostics["reason"] = "missing-candidate-contour"
        return _TargetFillReconstruction(None, None, diagnostics)
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    coordinates, line_count, _protected = fit_corner_protected_ring(
        contour,
        config=replace(
            config,
            line_anchor_tolerance_px=min(
                2.0,
                config.line_anchor_tolerance_px,
            ),
        ),
    )
    geometry: Polygon | MultiPolygon = Polygon(coordinates)
    if (
        not isinstance(geometry, Polygon)
        or geometry.is_empty
        or not geometry.is_valid
        or geometry.area <= 0.0
        or len(geometry.interiors) != 0
    ):
        diagnostics["reason"] = "invalid-vector-candidate"
        return _TargetFillReconstruction(None, None, diagnostics)
    geometry = orient_pixel_polygon(geometry)
    vector_mask = rasterize_geometry_mask(
        geometry,
        width=current.shape[1],
        height=current.shape[0],
    )
    vector_topology = _digital_mask_topology(vector_mask)
    vector_intersection = int(np.count_nonzero(vector_mask & candidate))
    vector_union = int(np.count_nonzero(vector_mask | candidate))
    vector_overlap = vector_intersection / max(1, vector_union)
    vector_area_ratio = int(np.count_nonzero(vector_mask)) / max(1, candidate_area)
    diagnostics.update(
        {
            "vector_line_count": line_count,
            "vector_vertex_count": len(geometry.exterior.coords) - 1,
            "vector_candidate_iou": round(vector_overlap, 6),
            "vector_candidate_area_ratio": round(vector_area_ratio, 6),
        }
    )
    if necessity_route == "strip-saturation":
        if (
            vector_topology != candidate_topology
            or vector_overlap < config.graphcut_minimum_vector_iou
            or not (
                config.graphcut_minimum_vector_area_ratio
                <= vector_area_ratio
                <= config.graphcut_maximum_vector_area_ratio
            )
        ):
            diagnostics["reason"] = "vector-gate"
            return _TargetFillReconstruction(None, None, diagnostics)
        diagnostics["accepted"] = True
        diagnostics["reason"] = "accepted"
        return _TargetFillReconstruction(geometry, vector_mask, diagnostics)

    if target_rgb is None:
        diagnostics["reason"] = "missing-target-color"
        return _TargetFillReconstruction(None, None, diagnostics)
    if not isinstance(current_geometry, Polygon):
        diagnostics["reason"] = "missing-current-polygon"
        return _TargetFillReconstruction(None, None, diagnostics)

    candidate_phase = _graphcut_phase_evidence(
        rgb,
        candidate,
        geometry,
        target_rgb=target_rgb,
        config=config,
        native_fields=native_fields,
    )
    diagnostics["candidate_phase"] = candidate_phase
    if not _graphcut_centered_phase_gate(candidate_phase, config=config):
        diagnostics["reason"] = "candidate-phase-gate"
        return _TargetFillReconstruction(None, None, diagnostics)

    phase_distance = float(candidate_phase["center_px"])
    phased_geometry = geometry.buffer(-phase_distance, join_style=2)
    if (
        not isinstance(phased_geometry, Polygon)
        or phased_geometry.is_empty
        or not phased_geometry.is_valid
        or phased_geometry.area <= 0.0
        or len(phased_geometry.interiors) != 0
    ):
        diagnostics["reason"] = "invalid-centered-phase"
        return _TargetFillReconstruction(None, None, diagnostics)
    phased_geometry = orient_pixel_polygon(phased_geometry)
    phased_mask = rasterize_geometry_mask(
        phased_geometry,
        width=current.shape[1],
        height=current.shape[0],
    )
    phased_topology = _digital_mask_topology(phased_mask)
    phased_intersection = int(np.count_nonzero(phased_mask & current))
    phased_union = int(np.count_nonzero(phased_mask | current))
    phased_current_iou = phased_intersection / max(1, phased_union)
    phased_current_area_ratio = (
        int(np.count_nonzero(phased_mask)) / max(1, current_area)
    )
    current_vertex_count = len(current_geometry.exterior.coords) - 1
    vector_vertex_count = len(geometry.exterior.coords) - 1
    phased_vertex_count = len(phased_geometry.exterior.coords) - 1
    vector_current_vertex_ratio = (
        vector_vertex_count / max(1, current_vertex_count)
    )
    phased_current_vertex_ratio = (
        phased_vertex_count / max(1, current_vertex_count)
    )
    vector_vertex_density = vector_vertex_count / max(
        1.0,
        float(geometry.length),
    )
    structural_distance = _symmetric_mask_boundary_distance_summary(
        phased_mask,
        current,
    )
    diagnostics.update(
        {
            "phase_distance_px": round(phase_distance, 6),
            "phased_current_iou": round(phased_current_iou, 6),
            "phased_current_area_ratio": round(
                phased_current_area_ratio,
                6,
            ),
            "phased_topology": {
                "components": phased_topology[0],
                "holes": phased_topology[1],
            },
            "current_vertex_count": current_vertex_count,
            "phased_vertex_count": phased_vertex_count,
            "vector_current_vertex_ratio": round(
                vector_current_vertex_ratio,
                6,
            ),
            "phased_current_vertex_ratio": round(
                phased_current_vertex_ratio,
                6,
            ),
            "vector_vertex_density": round(vector_vertex_density, 8),
            "phased_current_boundary_distance": {
                key: round(float(value), 6)
                for key, value in structural_distance.items()
            },
        }
    )
    if not _graphcut_centered_vector_gate(
        vector_candidate_iou=vector_overlap,
        vector_candidate_area_ratio=vector_area_ratio,
        phased_current_iou=phased_current_iou,
        phased_current_area_ratio=phased_current_area_ratio,
        current_topology=current_topology,
        candidate_topology=candidate_topology,
        vector_topology=vector_topology,
        phased_topology=phased_topology,
        vector_current_vertex_ratio=vector_current_vertex_ratio,
        phased_current_vertex_ratio=phased_current_vertex_ratio,
        vector_vertex_count=vector_vertex_count,
        vector_vertex_density=vector_vertex_density,
        phased_current_boundary_p99_px=structural_distance["p99_px"],
        phased_current_boundary_max_px=structural_distance["max_px"],
        config=config,
    ):
        diagnostics["reason"] = "centered-vector-gate"
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["accepted"] = True
    diagnostics["reason"] = "accepted-centered-stroke"
    return _TargetFillReconstruction(
        phased_geometry,
        phased_mask,
        diagnostics,
    )


def _recover_angle_lattice_geometry(
    current_geometry: Polygon | MultiPolygon,
    *,
    current_mask: np.ndarray,
    config: EdgeGraphConfig,
    native_fields: _NativeImageFields,
) -> _TargetFillReconstruction:
    """Sharpen long runs with a bounded, source-validated angle lattice."""

    diagnostics: dict[str, object] = {
        "evaluated": False,
        "accepted": False,
        "reason": "disabled",
        "candidates": [],
    }
    if not config.angle_lattice_enabled:
        return _TargetFillReconstruction(None, None, diagnostics)
    if (
        not isinstance(current_geometry, (Polygon, MultiPolygon))
        or current_geometry.is_empty
        or not current_geometry.is_valid
    ):
        diagnostics["reason"] = "unsupported-current-geometry"
        return _TargetFillReconstruction(None, None, diagnostics)
    current = np.asarray(current_mask, dtype=bool)
    current_area = int(np.count_nonzero(current))
    current_topology = _digital_mask_topology(current)
    if current_area == 0:
        diagnostics["reason"] = "empty-current-mask"
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["evaluated"] = True
    current_energy_median, current_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        current,
    )
    candidate_records: list[dict[str, object]] = []
    for spacing in (90.0, 45.0, 22.5):
        result = snap_geometry_to_angle_lattice(
            current_geometry,
            spacing_degrees=spacing,
            simplification_tolerance_px=(
                config.angle_lattice_simplification_tolerance_px
            ),
            minimum_segment_length_px=(
                config.angle_lattice_minimum_segment_length_px
            ),
            maximum_segment_residual_degrees=(
                config.angle_lattice_maximum_segment_residual_degrees
            ),
            maximum_vertex_displacement_px=(
                config.angle_lattice_maximum_vertex_displacement_px
            ),
        )
        record: dict[str, object] = {
            "spacing_degrees": spacing,
            "phase_degrees": round(result.phase_degrees, 6),
            "p90_residual_degrees": (
                round(result.p90_residual_degrees, 6)
                if np.isfinite(result.p90_residual_degrees)
                else None
            ),
            "mean_residual_degrees": (
                round(result.mean_residual_degrees, 6)
                if np.isfinite(result.mean_residual_degrees)
                else None
            ),
            "maximum_vertex_displacement_px": (
                round(result.maximum_vertex_displacement_px, 6)
                if np.isfinite(result.maximum_vertex_displacement_px)
                else None
            ),
            "snapped_segment_count": result.snapped_segment_count,
            "inferred_connector_count": result.inferred_connector_count,
            "reason": result.reason,
            "accepted": False,
        }
        candidate_records.append(record)
        candidate_geometry = result.geometry
        if candidate_geometry is None:
            continue
        if (
            result.p90_residual_degrees
            > config.angle_lattice_maximum_p90_residual_degrees
            or result.mean_residual_degrees
            > config.angle_lattice_maximum_mean_residual_degrees
        ):
            record["reason"] = "orientation-residual-gate"
            continue
        if (
            result.p90_residual_degrees
            < config.angle_lattice_minimum_material_p90_residual_degrees
            and result.maximum_vertex_displacement_px
            < config.angle_lattice_minimum_material_displacement_px
        ):
            record["reason"] = "near-identity"
            continue
        candidate_mask = rasterize_geometry_mask(
            candidate_geometry,
            width=current.shape[1],
            height=current.shape[0],
        )
        candidate_topology = _digital_mask_topology(candidate_mask)
        intersection = int(np.count_nonzero(candidate_mask & current))
        union = int(np.count_nonzero(candidate_mask | current))
        overlap = intersection / max(1, union)
        candidate_energy_median, candidate_energy_mean = (
            _mask_boundary_native_energy(
                native_fields.magnitude,
                candidate_mask,
            )
        )
        record.update(
            {
                "current_iou": round(overlap, 6),
                "topology": {
                    "components": candidate_topology[0],
                    "holes": candidate_topology[1],
                },
                "boundary_energy_median_gain": round(
                    candidate_energy_median - current_energy_median,
                    6,
                ),
                "boundary_energy_mean_gain": round(
                    candidate_energy_mean - current_energy_mean,
                    6,
                ),
            }
        )
        if candidate_topology != current_topology:
            record["reason"] = "topology-gate"
            continue
        if overlap < config.angle_lattice_minimum_current_iou:
            record["reason"] = "overlap-gate"
            continue
        if (
            candidate_energy_median
            < (
                current_energy_median
                - config.angle_lattice_maximum_boundary_median_drop
            )
            or candidate_energy_mean
            < (
                current_energy_mean
                - config.angle_lattice_maximum_boundary_mean_drop
            )
        ):
            record["reason"] = "native-energy-gate"
            continue
        if isinstance(candidate_geometry, Polygon):
            candidate_geometry = orient_pixel_polygon(candidate_geometry)
        else:
            candidate_geometry = MultiPolygon(
                [
                    orient_pixel_polygon(part)
                    for part in candidate_geometry.geoms
                ]
            )
        record["accepted"] = True
        record["reason"] = "accepted"
        diagnostics.update(
            {
                "accepted": True,
                "reason": "accepted",
                "selected_spacing_degrees": spacing,
                "selected_phase_degrees": round(
                    result.phase_degrees,
                    6,
                ),
                "selected_current_iou": round(overlap, 6),
            }
        )
        diagnostics["candidates"] = candidate_records
        return _TargetFillReconstruction(
            candidate_geometry,
            candidate_mask,
            diagnostics,
        )
    diagnostics["reason"] = "candidate-gate"
    diagnostics["candidates"] = candidate_records
    return _TargetFillReconstruction(None, None, diagnostics)


def _recover_multiscale_line_graph_geometry(
    rgb: np.ndarray,
    *,
    current_mask: np.ndarray,
    current_geometry: Polygon | MultiPolygon,
    target_rgb: tuple[int, int, int] | None,
    seed_point: tuple[float, float] | None,
    direct_lattice_accepted: bool,
    direct_lattice_diagnostics: dict[str, object],
    config: EdgeGraphConfig,
    native_fields: _NativeImageFields,
) -> _TargetFillReconstruction:
    """Recover source structure through a stable multiscale line graph.

    The selector/refiner result remains the semantic owner.  GraphCut supplies
    one bounded source-native raster proposal, never final geometry.  Multiple
    line fits then compete under raster fidelity, native-energy, corner
    agreement, topology, and minimum-description-length gates.  No metric in
    this selector depends on fixture labels or reference geometry.
    """

    diagnostics: dict[str, object] = {
        "evaluated": False,
        "accepted": False,
        "reason": "disabled",
        "candidate_count": 0,
        "eligible_candidate_count": 0,
    }
    if not config.multiscale_line_graph_enabled:
        return _TargetFillReconstruction(None, None, diagnostics)
    if (
        target_rgb is None
        or seed_point is None
        or not isinstance(current_geometry, Polygon)
        or current_geometry.is_empty
        or not current_geometry.is_valid
    ):
        diagnostics["reason"] = "missing-guidance-or-current-polygon"
        return _TargetFillReconstruction(None, None, diagnostics)

    current = np.asarray(current_mask, dtype=bool)
    current_area = int(np.count_nonzero(current))
    current_topology = _digital_mask_topology(current)
    current_holes = [
        np.asarray(interior.coords, dtype=np.float64)
        for interior in current_geometry.interiors
    ]
    if (
        current_area == 0
        or current.all()
        or current_topology[0] != 1
        or len(current_holes) > 4
    ):
        diagnostics["reason"] = "unsupported-current-topology"
        return _TargetFillReconstruction(None, None, diagnostics)
    diagnostics["evaluated"] = True
    lattice_direction_signal = any(
        isinstance(candidate, dict)
        and isinstance(candidate.get("p90_residual_degrees"), (int, float))
        and isinstance(candidate.get("mean_residual_degrees"), (int, float))
        and float(candidate["p90_residual_degrees"]) <= 5.0
        and float(candidate["mean_residual_degrees"]) <= 2.0
        for candidate in direct_lattice_diagnostics.get("candidates", [])
    )
    diagnostics["lattice_direction_signal"] = lattice_direction_signal

    distance_inside = cv2.distanceTransform(
        current.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    distance_outside = cv2.distanceTransform(
        (~current).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    sure_foreground = (
        distance_inside >= config.graphcut_sure_foreground_margin_px
    )
    sure_background = (
        distance_outside >= config.graphcut_sure_background_margin_px
    )
    if (
        int(np.count_nonzero(sure_foreground)) < 64
        or int(np.count_nonzero(sure_background)) < 64
    ):
        diagnostics["reason"] = "insufficient-graphcut-seeds"
        return _TargetFillReconstruction(None, None, diagnostics)

    graphcut_mask = np.where(
        current,
        cv2.GC_PR_FGD,
        cv2.GC_PR_BGD,
    ).astype(np.uint8)
    graphcut_mask[sure_foreground] = cv2.GC_FGD
    graphcut_mask[sure_background] = cv2.GC_BGD
    try:
        cv2.setRNGSeed(0)
        cv2.grabCut(
            np.ascontiguousarray(rgb, dtype=np.uint8),
            graphcut_mask,
            None,
            np.zeros((1, 65), dtype=np.float64),
            np.zeros((1, 65), dtype=np.float64),
            1,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        diagnostics["reason"] = "graphcut-failed"
        return _TargetFillReconstruction(None, None, diagnostics)
    source_candidate = _select_guided_mask(
        np.isin(
            graphcut_mask,
            (cv2.GC_FGD, cv2.GC_PR_FGD),
        ),
        ExtractionHints(seed_point=seed_point),
    )
    source_area = int(np.count_nonzero(source_candidate))
    source_topology = _digital_mask_topology(source_candidate)
    intersection = int(np.count_nonzero(source_candidate & current))
    union = int(np.count_nonzero(source_candidate | current))
    source_current_iou = intersection / max(1, union)
    source_current_area_ratio = source_area / max(1, current_area)
    source_energy_median, source_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        source_candidate,
    )
    current_energy_median, current_energy_mean = _mask_boundary_native_energy(
        native_fields.magnitude,
        current,
    )
    structural_distance = _symmetric_mask_boundary_distance_summary(
        source_candidate,
        current,
    )
    hole_area, hole_fraction = _mask_hole_area_summary(source_candidate)
    tiny_hole_route = bool(
        source_topology[0] == 1
        and source_topology[1] > 0
        and hole_area
        <= config.multiscale_line_graph_maximum_tiny_hole_area_px
        and hole_fraction
        <= config.multiscale_line_graph_maximum_tiny_hole_area_fraction
    )
    diagnostics.update(
        {
            "source_current_iou": round(source_current_iou, 6),
            "source_current_area_ratio": round(
                source_current_area_ratio,
                6,
            ),
            "source_topology": {
                "components": source_topology[0],
                "holes": source_topology[1],
            },
            "source_hole_area_px": round(hole_area, 6),
            "source_hole_area_fraction": round(hole_fraction, 8),
            "tiny_hole_route": tiny_hole_route,
            "source_boundary_energy_median_gain": round(
                source_energy_median - current_energy_median,
                6,
            ),
            "source_boundary_energy_mean_gain": round(
                source_energy_mean - current_energy_mean,
                6,
            ),
            "source_current_boundary_distance": {
                key: round(float(value), 6)
                for key, value in structural_distance.items()
            },
        }
    )
    topology_eligible = bool(
        source_topology == current_topology
        or tiny_hole_route
    )
    source_energy_gain = source_energy_mean - current_energy_mean
    source_energy_median_gain = (
        source_energy_median - current_energy_median
    )
    strong_tail_route = bool(
        source_current_iou >= 0.995
        and source_energy_gain >= 0.0
        and structural_distance["p99_px"] >= 4.0
        and structural_distance["max_px"] >= 8.0
    )
    material_source_route = bool(
        source_current_iou >= 0.989
        and source_energy_gain >= 0.04
        and structural_distance["p99_px"] >= 4.0
        and structural_distance["max_px"] >= 7.0
    )
    high_energy_source_route = bool(
        source_current_iou >= 0.9845
        and source_energy_gain >= 0.05
        and (
            source_energy_median_gain >= 0.05
            or source_energy_gain >= 0.10
        )
        and structural_distance["p95_px"] <= 4.0
        and structural_distance["max_px"] <= 8.0
    )
    lattice_tail_route = bool(
        direct_lattice_accepted
        and source_current_iou >= 0.995
        and source_energy_gain >= 0.005
        and structural_distance["p99_px"] >= 2.0
        and structural_distance["max_px"] >= 5.0
    )
    energy_supported_route = bool(
        source_current_iou
        >= config.multiscale_line_graph_minimum_candidate_iou
        and source_energy_gain
        >= config.multiscale_line_graph_minimum_boundary_mean_gain
    )
    diagnostics["source_necessity_routes"] = {
        "energy_supported": energy_supported_route,
        "strong_tail": strong_tail_route,
        "material_source": material_source_route,
        "high_energy_source": high_energy_source_route,
        "lattice_tail": lattice_tail_route,
    }
    if (
        not topology_eligible
        or not (
            energy_supported_route
            or strong_tail_route
            or material_source_route
            or high_energy_source_route
            or lattice_tail_route
        )
        or not (
            config.multiscale_line_graph_minimum_candidate_area_ratio
            <= source_current_area_ratio
            <= config.multiscale_line_graph_maximum_candidate_area_ratio
        )
        or source_energy_median
        < (
            current_energy_median
            - config.multiscale_line_graph_maximum_boundary_median_drop
        )
    ):
        diagnostics["reason"] = "source-candidate-gate"
        return _TargetFillReconstruction(None, None, diagnostics)
    if (
        direct_lattice_accepted
        and not lattice_tail_route
        and structural_distance["p99_px"]
        < config.multiscale_line_graph_lattice_skip_p99_px
        and structural_distance["max_px"]
        < config.multiscale_line_graph_lattice_skip_max_px
    ):
        diagnostics["reason"] = "near-identity-lattice-owner"
        return _TargetFillReconstruction(None, None, diagnostics)

    contours, _hierarchy = cv2.findContours(
        source_candidate.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        diagnostics["reason"] = "missing-source-contour"
        return _TargetFillReconstruction(None, None, diagnostics)
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    screen_coordinates, _screen_line_count, _screen_protected = (
        fit_corner_protected_ring(
            contour,
            config=replace(
                config,
                line_anchor_tolerance_px=min(
                    2.0,
                    config.line_anchor_tolerance_px,
                ),
            ),
        )
    )
    screen_geometry = Polygon(screen_coordinates)
    if (
        screen_geometry.is_empty
        or not screen_geometry.is_valid
        or screen_geometry.area <= 0.0
    ):
        diagnostics["reason"] = "invalid-screen-vector"
        return _TargetFillReconstruction(None, None, diagnostics)
    current_vertex_count = len(current_geometry.exterior.coords) - 1
    screen_vertex_count = len(screen_geometry.exterior.coords) - 1
    screen_vertex_ratio = screen_vertex_count / max(1, current_vertex_count)
    diagnostics.update(
        {
            "current_vertex_count": current_vertex_count,
            "screen_vertex_count": screen_vertex_count,
            "screen_vertex_ratio": round(screen_vertex_ratio, 6),
        }
    )
    if (
        not tiny_hole_route
        and not material_source_route
        and not high_energy_source_route
        and not lattice_tail_route
        and screen_vertex_ratio
        > config.multiscale_line_graph_maximum_default_vertex_ratio
    ):
        diagnostics["reason"] = "unstable-screen-vector"
        return _TargetFillReconstruction(None, None, diagnostics)

    phase_evidence = _graphcut_phase_evidence(
        rgb,
        source_candidate,
        screen_geometry,
        target_rgb=target_rgb,
        config=config,
        native_fields=native_fields,
    )
    use_phase = _multiscale_line_graph_phase_gate(
        phase_evidence,
        config=config,
    )
    phase_distance = (
        float(phase_evidence["center_px"])
        if use_phase
        else 0.0
    )
    diagnostics.update(
        {
            "phase": phase_evidence,
            "phase_applied": use_phase,
            "phase_distance_px": round(phase_distance, 6),
        }
    )
    structural_need = bool(
        structural_distance["p99_px"]
        >= config.multiscale_line_graph_minimum_structural_p99_px
        or structural_distance["max_px"]
        >= config.multiscale_line_graph_minimum_structural_max_px
        or use_phase
    )
    diagnostics["structural_need"] = structural_need
    if not structural_need:
        diagnostics["reason"] = "structural-necessity-gate"
        return _TargetFillReconstruction(None, None, diagnostics)

    candidates: list[_LineGraphCandidate] = []
    for anchor_tolerance in (
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
    ):
        for residual_tolerance in (1.0, 1.5, 2.0):
            coordinates, line_count, _protected = fit_corner_protected_ring(
                contour,
                config=replace(
                    config,
                    line_anchor_tolerance_px=anchor_tolerance,
                    line_fit_tolerance_px=residual_tolerance,
                    rectilinear_fit_enabled=False,
                    graphcut_recovery_enabled=False,
                ),
            )
            base_geometry = Polygon(coordinates, current_holes)
            minimum_base_mean_gain = (
                -0.005
                if strong_tail_route
                else (
                    0.0
                    if high_energy_source_route
                    else (
                        0.005
                        if lattice_tail_route
                        else (
                            config
                            .multiscale_line_graph_minimum_base_boundary_mean_gain
                        )
                    )
                )
            )
            if (
                base_geometry.is_empty
                or not base_geometry.is_valid
                or base_geometry.area <= 0.0
            ):
                continue
            base_geometry = orient_pixel_polygon(base_geometry)
            base_mask = rasterize_geometry_mask(
                base_geometry,
                width=current.shape[1],
                height=current.shape[0],
            )
            if _digital_mask_topology(base_mask) != current_topology:
                continue
            base_source_intersection = int(
                np.count_nonzero(base_mask & source_candidate)
            )
            base_source_union = int(
                np.count_nonzero(base_mask | source_candidate)
            )
            source_iou = base_source_intersection / max(
                1,
                base_source_union,
            )
            source_area_ratio = int(np.count_nonzero(base_mask)) / max(
                1,
                source_area,
            )
            base_energy_median, base_energy_mean = (
                _mask_boundary_native_energy(
                    native_fields.magnitude,
                    base_mask,
                )
            )
            if (
                source_iou
                < config.multiscale_line_graph_minimum_source_iou
                or not (
                    config.multiscale_line_graph_minimum_source_area_ratio
                    <= source_area_ratio
                    <= config.multiscale_line_graph_maximum_source_area_ratio
                )
                or base_energy_median
                < (
                    current_energy_median
                    - config.multiscale_line_graph_maximum_boundary_median_drop
                )
                or base_energy_mean
                < (
                    current_energy_mean
                    + minimum_base_mean_gain
                )
            ):
                continue

            output_geometry = base_geometry
            if use_phase:
                output_geometry = base_geometry.buffer(
                    -phase_distance,
                    join_style=2,
                )
                if (
                    not isinstance(output_geometry, Polygon)
                    or output_geometry.is_empty
                    or not output_geometry.is_valid
                    or output_geometry.area <= 0.0
                    or len(output_geometry.interiors)
                    != len(current_holes)
                ):
                    continue
                output_geometry = orient_pixel_polygon(output_geometry)
            output_mask = rasterize_geometry_mask(
                output_geometry,
                width=current.shape[1],
                height=current.shape[0],
            )
            if _digital_mask_topology(output_mask) != current_topology:
                continue
            output_intersection = int(
                np.count_nonzero(output_mask & current)
            )
            output_union = int(np.count_nonzero(output_mask | current))
            output_current_iou = output_intersection / max(
                1,
                output_union,
            )
            output_current_area_ratio = int(
                np.count_nonzero(output_mask)
            ) / max(1, current_area)
            if (
                output_current_iou
                < config.multiscale_line_graph_minimum_output_current_iou
                or not (
                    config.multiscale_line_graph_minimum_output_area_ratio
                    <= output_current_area_ratio
                    <= config.multiscale_line_graph_maximum_output_area_ratio
                )
            ):
                continue
            vertex_count = len(output_geometry.exterior.coords) - 1
            if (
                vertex_count
                > config.multiscale_line_graph_maximum_vertex_count
                or vertex_count
                > max(
                    current_vertex_count
                    * config.multiscale_line_graph_maximum_vertex_ratio,
                    current_vertex_count
                    + config.multiscale_line_graph_maximum_added_vertices,
                )
                or (
                    tiny_hole_route
                    and vertex_count > current_vertex_count
                )
            ):
                continue
            corner_agreement = geometry_corner_f1(
                output_geometry,
                current_geometry,
                tolerance_px=2.0,
            )
            minimum_corner_f1 = (
                0.50
                if tiny_hole_route
                else (
                    config
                    .multiscale_line_graph_minimum_corner_agreement_f1
                )
            )
            minimum_corner_recall = (
                0.50
                if tiny_hole_route
                else (
                    config
                    .multiscale_line_graph_minimum_corner_agreement_recall
                )
            )
            if lattice_direction_signal and not tiny_hole_route:
                minimum_corner_recall = min(
                    minimum_corner_recall,
                    0.70,
                )
            maximum_corner_p95_degrees = (
                90.0
                if lattice_direction_signal
                else (
                    config
                    .multiscale_line_graph_maximum_corner_agreement_p95_degrees
                )
            )
            if (
                float(corner_agreement["f1"])
                < minimum_corner_f1
                or float(corner_agreement["recall"])
                < minimum_corner_recall
                or float(
                    corner_agreement[
                        "p95_angular_error_degrees"
                    ]
                )
                > maximum_corner_p95_degrees
            ):
                continue
            candidates.append(
                _LineGraphCandidate(
                    key=(
                        f"e{anchor_tolerance:g}"
                        f"-r{residual_tolerance:g}"
                        f"-{'center' if use_phase else 'zero'}"
                    ),
                    geometry=output_geometry,
                    mask=output_mask,
                    source_iou=source_iou,
                    source_area_ratio=source_area_ratio,
                    current_iou=output_current_iou,
                    current_area_ratio=output_current_area_ratio,
                    boundary_energy_median=base_energy_median,
                    boundary_energy_mean=base_energy_mean,
                    vertex_count=vertex_count,
                    corner_agreement=corner_agreement,
                    line_count=line_count,
                )
            )
    diagnostics["candidate_count"] = 24
    diagnostics["eligible_candidate_count"] = len(candidates)
    if not candidates:
        diagnostics["reason"] = "no-stable-line-graph"
        return _TargetFillReconstruction(None, None, diagnostics)

    best_source = max(
        candidates,
        key=lambda item: (
            item.source_iou,
            -item.vertex_count,
            item.key,
        ),
    )
    supported: list[_LineGraphCandidate] = []
    for candidate in candidates:
        source_loss = best_source.source_iou - candidate.source_iou
        removed_vertices = max(
            0,
            best_source.vertex_count - candidate.vertex_count,
        )
        if (
            source_loss
            > config.multiscale_line_graph_maximum_source_loss
        ):
            continue
        if (
            removed_vertices > 0
            and source_loss / removed_vertices
            > (
                config
                .multiscale_line_graph_maximum_source_loss_per_removed_vertex
            )
        ):
            continue
        supported.append(candidate)
    if not supported:
        diagnostics["reason"] = "description-length-gate"
        return _TargetFillReconstruction(None, None, diagnostics)
    if lattice_direction_signal:
        selected = min(
            supported,
            key=lambda item: (
                item.vertex_count,
                -item.source_iou,
                -float(item.corner_agreement["f1"]),
                item.key,
            ),
        )
    else:
        selected = min(
            supported,
            key=lambda item: (
                -float(item.corner_agreement["f1"]),
                float(
                    item.corner_agreement[
                        "p95_angular_error_degrees"
                    ]
                ),
                item.vertex_count,
                -item.source_iou,
                item.key,
            ),
        )
    diagnostics.update(
        {
            "accepted": True,
            "reason": "accepted",
            "selected_key": selected.key,
            "selected_source_iou": round(selected.source_iou, 6),
            "selected_source_area_ratio": round(
                selected.source_area_ratio,
                6,
            ),
            "selected_current_iou": round(
                selected.current_iou,
                6,
            ),
            "selected_current_area_ratio": round(
                selected.current_area_ratio,
                6,
            ),
            "selected_vertex_count": selected.vertex_count,
            "selected_line_count": selected.line_count,
            "selected_corner_agreement": {
                key: (
                    round(float(value), 6)
                    if isinstance(value, (int, float))
                    else value
                )
                for key, value in selected.corner_agreement.items()
            },
            "best_source_iou": round(best_source.source_iou, 6),
            "best_source_vertex_count": best_source.vertex_count,
        }
    )
    return _TargetFillReconstruction(
        selected.geometry,
        selected.mask,
        diagnostics,
    )


def _multiscale_line_graph_phase_gate(
    evidence: dict[str, object],
    *,
    config: EdgeGraphConfig,
) -> bool:
    center_p90 = evidence.get("center_p90_deviation_px")
    return bool(
        float(evidence["paired_fraction"])
        >= config.multiscale_line_graph_phase_minimum_paired_fraction
        and float(evidence["reliable_paired_fraction"])
        >= config.multiscale_line_graph_phase_minimum_reliable_fraction
        and float(evidence["coherent_paired_fraction"])
        >= config.multiscale_line_graph_phase_minimum_coherent_fraction
        and float(evidence["positive_center_fraction"])
        >= config.multiscale_line_graph_phase_minimum_positive_fraction
        and (
            config.multiscale_line_graph_phase_minimum_center_px
            <= float(evidence["center_px"])
            <= config.multiscale_line_graph_phase_maximum_center_px
        )
        and center_p90 is not None
        and float(center_p90)
        <= config.multiscale_line_graph_phase_maximum_p90_deviation_px
        and 0.0 < float(evidence["stroke_width_px"])
        <= config.multiscale_line_graph_phase_maximum_stroke_width_px
    )


def _mask_hole_area_summary(mask: np.ndarray) -> tuple[float, float]:
    contours, hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    if hierarchy is None:
        return 0.0, 0.0
    hole_area = float(
        sum(
            cv2.contourArea(contour)
            for index, contour in enumerate(contours)
            if int(hierarchy[0][index][3]) >= 0
        )
    )
    return (
        hole_area,
        hole_area / max(1, int(np.count_nonzero(mask))),
    )


def _flat_fill_candidate_gate(
    *,
    proposal_accepted_count: int,
    shortcut_count: int,
    mean_reliability: float,
    target_separation: float,
    nearest_target_distance_px: float,
    closed_addition_fraction: float,
    candidate_current_iou: float,
    candidate_current_area_ratio: float,
    candidate_topology: tuple[int, int],
    current_topology: tuple[int, int],
    candidate_boundary_energy_median: float,
    current_boundary_energy_median: float,
    candidate_boundary_energy_mean: float,
    current_boundary_energy_mean: float,
    config: EdgeGraphConfig,
) -> bool:
    """Require independent evidence before a closed flat fill may own geometry."""

    return bool(
        proposal_accepted_count == 0
        and shortcut_count > 0
        and mean_reliability < config.flat_fill_maximum_reliability
        and target_separation >= config.flat_fill_minimum_target_separation
        and nearest_target_distance_px <= 4.0
        and closed_addition_fraction
        <= config.flat_fill_maximum_closed_addition_fraction
        and candidate_topology == current_topology == (1, 0)
        and candidate_current_iou >= config.flat_fill_minimum_candidate_iou
        and (
            config.flat_fill_minimum_candidate_area_ratio
            <= candidate_current_area_ratio
            <= config.flat_fill_maximum_candidate_area_ratio
        )
        and candidate_boundary_energy_median
        >= (
            current_boundary_energy_median
            - config.flat_fill_maximum_boundary_median_drop
        )
        and candidate_boundary_energy_mean
        >= (
            current_boundary_energy_mean
            + config.flat_fill_minimum_boundary_mean_gain
        )
    )


def _flat_fill_structural_candidate_gate(
    *,
    proposal_accepted_count: int,
    shortcut_count: int,
    mean_reliability: float,
    target_separation: float,
    nearest_target_distance_px: float,
    closed_addition_fraction: float,
    candidate_current_iou: float,
    candidate_current_area_ratio: float,
    candidate_topology: tuple[int, int],
    current_topology: tuple[int, int],
    candidate_boundary_energy_median: float,
    current_boundary_energy_median: float,
    candidate_boundary_energy_mean: float,
    current_boundary_energy_mean: float,
    config: EdgeGraphConfig,
) -> bool:
    """Allow a material fill correction only under stronger native evidence.

    This lane is intentionally distinct from the near-identity cleanup gate.
    A candidate may differ by several percent only when localization is
    explicitly unreliable, target color is strongly separated, the large
    close remains bounded, topology is unchanged, and both median and mean
    source-boundary energy improve materially.
    """

    return bool(
        proposal_accepted_count == 0
        and shortcut_count > 0
        and mean_reliability
        <= config.flat_fill_structural_maximum_reliability
        and target_separation
        >= config.flat_fill_structural_minimum_target_separation
        and nearest_target_distance_px <= 4.0
        and closed_addition_fraction
        <= config.flat_fill_structural_maximum_closed_addition_fraction
        and candidate_topology == current_topology == (1, 0)
        and candidate_current_iou
        >= config.flat_fill_structural_minimum_candidate_iou
        and (
            config.flat_fill_structural_minimum_candidate_area_ratio
            <= candidate_current_area_ratio
            <= config.flat_fill_structural_maximum_candidate_area_ratio
        )
        and candidate_boundary_energy_median
        >= (
            current_boundary_energy_median
            + config.flat_fill_structural_minimum_boundary_median_gain
        )
        and candidate_boundary_energy_mean
        >= (
            current_boundary_energy_mean
            + config.flat_fill_structural_minimum_boundary_mean_gain
        )
    )


def _graphcut_candidate_gate(
    *,
    proposal_accepted_count: int,
    maximum_displacement_px: float,
    candidate_current_iou: float,
    candidate_current_area_ratio: float,
    candidate_topology: tuple[int, int],
    current_topology: tuple[int, int],
    candidate_boundary_energy_median: float,
    current_boundary_energy_median: float,
    candidate_boundary_energy_mean: float,
    current_boundary_energy_mean: float,
    config: EdgeGraphConfig,
) -> bool:
    """Require strip saturation plus topology and native-energy agreement."""

    return bool(
        proposal_accepted_count == 0
        and maximum_displacement_px
        >= config.graphcut_minimum_maximum_displacement_px
        and candidate_topology == current_topology == (1, 0)
        and candidate_current_iou >= config.graphcut_minimum_candidate_iou
        and (
            config.graphcut_minimum_candidate_area_ratio
            <= candidate_current_area_ratio
            <= config.graphcut_maximum_candidate_area_ratio
        )
        and candidate_boundary_energy_median
        >= (
            current_boundary_energy_median
            - config.graphcut_maximum_boundary_median_drop
        )
        and candidate_boundary_energy_mean
        >= (
            current_boundary_energy_mean
            + config.graphcut_minimum_boundary_mean_gain
        )
    )


def _centered_graphcut_candidate_gate(
    *,
    proposal_accepted_count: int,
    candidate_current_iou: float,
    candidate_current_area_ratio: float,
    candidate_topology: tuple[int, int],
    current_topology: tuple[int, int],
    candidate_boundary_energy_median: float,
    current_boundary_energy_median: float,
    candidate_boundary_energy_mean: float,
    current_boundary_energy_mean: float,
    config: EdgeGraphConfig,
) -> bool:
    """Gate a bounded GraphCut before vectorization or phase estimation."""

    return bool(
        proposal_accepted_count == 0
        and candidate_topology == current_topology == (1, 0)
        and candidate_current_iou
        >= config.graphcut_centered_minimum_candidate_iou
        and (
            config.graphcut_centered_minimum_candidate_area_ratio
            <= candidate_current_area_ratio
            <= config.graphcut_centered_maximum_candidate_area_ratio
        )
        and candidate_boundary_energy_median
        >= (
            current_boundary_energy_median
            - config.graphcut_centered_maximum_boundary_median_drop
        )
        and candidate_boundary_energy_mean
        >= (
            current_boundary_energy_mean
            + config.graphcut_centered_minimum_boundary_mean_gain
        )
    )


def _graphcut_phase_evidence(
    rgb: np.ndarray,
    mask: np.ndarray,
    geometry: Polygon,
    *,
    target_rgb: tuple[int, int, int],
    config: EdgeGraphConfig,
    native_fields: _NativeImageFields,
    sample_step_px: float | None = None,
) -> dict[str, object]:
    """Summarize paired native-edge phase around one polygonal contour."""

    resolved_sample_step_px = (
        config.contour_sample_step_px
        if sample_step_px is None
        else float(sample_step_px)
    )
    points = resample_closed_contour(
        np.asarray(geometry.exterior.coords, dtype=np.float32),
        step_px=resolved_sample_step_px,
    )
    strip = sample_normal_strip(
        rgb,
        np.asarray(mask, dtype=np.float32),
        points,
        config=config,
        _native_fields=native_fields,
    )
    inside_sign = _inside_sign_from_coarse_profiles(
        strip.features[3],
        strip.offsets,
    )
    phase = estimate_edge_phase(
        strip.features[:3].transpose(1, 2, 0),
        strip.features[4],
        strip.features[6],
        strip.offsets,
        target_rgb=target_rgb,
        inside_sign=inside_sign,
    )
    paired = np.asarray(phase.paired, dtype=bool)
    reliability_values = np.asarray(phase.reliability, dtype=np.float32)
    reliable = paired & (
        reliability_values
        >= config.graphcut_centered_minimum_phase_reliability_p10
    )
    oriented_centers = (
        np.asarray(phase.centers_px, dtype=np.float32)
        * np.asarray(inside_sign, dtype=np.float32)
    )
    centers = oriented_centers[reliable]
    reliability = reliability_values[reliable]
    widths = np.asarray(phase.stroke_widths_px, dtype=np.float32)[reliable]
    if len(centers):
        center = _weighted_median_float(centers, reliability)
        center_deviation = np.abs(centers - center)
        center_mad = _weighted_median_float(
            center_deviation,
            reliability,
        )
        center_p90_deviation = float(
            np.quantile(center_deviation, 0.90)
        )
        coherent = (
            center_deviation
            <= config.graphcut_centered_coherent_center_tolerance_px
        )
        coherent_fraction = float(
            np.count_nonzero(coherent) / max(1, len(paired))
        )
        positive_fraction = float(
            np.mean(
                centers
                >= config.graphcut_centered_positive_center_threshold_px
            )
        )
        negative_fraction = float(
            np.mean(
                centers
                <= -config.graphcut_centered_positive_center_threshold_px
            )
        )
        reliability_median = float(np.median(reliability))
        reliability_p10 = float(np.quantile(reliability, 0.10))
        width = _weighted_median_float(widths, reliability)
        width_deviation = np.abs(widths - width)
        width_mad = _weighted_median_float(
            width_deviation,
            reliability,
        )
        width_p90_deviation = float(
            np.quantile(width_deviation, 0.90)
        )
    else:
        center = 0.0
        center_mad = float("inf")
        center_p90_deviation = float("inf")
        coherent_fraction = 0.0
        positive_fraction = 0.0
        negative_fraction = 0.0
        reliability_median = 0.0
        reliability_p10 = 0.0
        width = 0.0
        width_mad = float("inf")
        width_p90_deviation = float("inf")
    return {
        "sample_step_px": round(resolved_sample_step_px, 6),
        "profile_count": len(paired),
        "paired_count": int(np.count_nonzero(paired)),
        "paired_fraction": round(float(np.mean(paired)), 6),
        "reliable_paired_count": int(np.count_nonzero(reliable)),
        "reliable_paired_fraction": round(
            float(np.count_nonzero(reliable) / max(1, len(paired))),
            6,
        ),
        "coherent_paired_fraction": round(coherent_fraction, 6),
        "positive_center_fraction": round(positive_fraction, 6),
        "negative_center_fraction": round(negative_fraction, 6),
        "reliability_median": round(reliability_median, 6),
        "reliability_p10": round(reliability_p10, 6),
        "center_px": round(center, 6),
        "center_mad_px": (
            round(center_mad, 6)
            if np.isfinite(center_mad)
            else None
        ),
        "center_p90_deviation_px": (
            round(center_p90_deviation, 6)
            if np.isfinite(center_p90_deviation)
            else None
        ),
        "stroke_width_px": round(width, 6),
        "stroke_width_mad_px": (
            round(width_mad, 6)
            if np.isfinite(width_mad)
            else None
        ),
        "stroke_width_p90_deviation_px": (
            round(width_p90_deviation, 6)
            if np.isfinite(width_p90_deviation)
            else None
        ),
        "center_to_half_width_ratio": (
            round(center / max(1e-6, 0.5 * width), 6)
            if width > 0.0
            else 0.0
        ),
    }


def _graphcut_centered_phase_gate(
    evidence: dict[str, object],
    *,
    config: EdgeGraphConfig,
) -> bool:
    """Require a reliable, coherent inward center for the GraphCut stroke."""

    center_mad = evidence.get("center_mad_px")
    center_p90 = evidence.get("center_p90_deviation_px")
    width_mad = evidence.get("stroke_width_mad_px")
    width_p90 = evidence.get("stroke_width_p90_deviation_px")
    return bool(
        float(evidence["paired_fraction"])
        >= config.graphcut_centered_minimum_paired_fraction
        and float(evidence["reliable_paired_fraction"])
        >= config.graphcut_centered_minimum_reliable_paired_fraction
        and float(evidence["coherent_paired_fraction"])
        >= config.graphcut_centered_minimum_coherent_paired_fraction
        and float(evidence["reliability_p10"])
        >= config.graphcut_centered_minimum_phase_reliability_p10
        and float(evidence["positive_center_fraction"])
        >= config.graphcut_centered_minimum_positive_center_fraction
        and (
            config.graphcut_centered_minimum_center_px
            <= float(evidence["center_px"])
            <= config.graphcut_centered_maximum_center_px
        )
        and center_mad is not None
        and float(center_mad)
        <= config.graphcut_centered_maximum_center_mad_px
        and center_p90 is not None
        and float(center_p90)
        <= config.graphcut_centered_maximum_center_p90_deviation_px
        and (
            config.graphcut_centered_minimum_stroke_width_px
            <= float(evidence["stroke_width_px"])
            <= config.graphcut_centered_maximum_stroke_width_px
        )
        and width_mad is not None
        and float(width_mad)
        <= config.graphcut_centered_maximum_stroke_width_mad_px
        and width_p90 is not None
        and float(width_p90)
        <= (
            config
            .graphcut_centered_maximum_stroke_width_p90_deviation_px
        )
        and (
            config.graphcut_centered_minimum_center_to_half_width_ratio
            <= float(evidence["center_to_half_width_ratio"])
            <= config.graphcut_centered_maximum_center_to_half_width_ratio
        )
    )


def _graphcut_current_phase_positive_gate(
    evidence: dict[str, object],
    *,
    config: EdgeGraphConfig,
) -> bool:
    """Admit high-reliability geometry only under an independent phase need."""

    center_mad = evidence.get("center_mad_px")
    center_p90 = evidence.get("center_p90_deviation_px")
    width_mad = evidence.get("stroke_width_mad_px")
    width_p90 = evidence.get("stroke_width_p90_deviation_px")
    return bool(
        float(evidence["paired_fraction"])
        >= config.graphcut_current_phase_minimum_paired_fraction
        and float(evidence["reliable_paired_fraction"])
        >= config.graphcut_current_phase_minimum_reliable_paired_fraction
        and float(evidence["coherent_paired_fraction"])
        >= config.graphcut_current_phase_minimum_coherent_paired_fraction
        and float(evidence["reliability_p10"])
        >= config.graphcut_centered_minimum_phase_reliability_p10
        and float(evidence["positive_center_fraction"])
        >= config.graphcut_current_phase_minimum_positive_center_fraction
        and (
            config.graphcut_current_phase_minimum_center_px
            <= float(evidence["center_px"])
            <= config.graphcut_current_phase_maximum_center_px
        )
        and center_mad is not None
        and float(center_mad)
        <= config.graphcut_centered_maximum_center_mad_px
        and center_p90 is not None
        and float(center_p90)
        <= config.graphcut_current_phase_maximum_center_p90_deviation_px
        and (
            config.graphcut_centered_minimum_stroke_width_px
            <= float(evidence["stroke_width_px"])
            <= config.graphcut_centered_maximum_stroke_width_px
        )
        and width_mad is not None
        and float(width_mad)
        <= config.graphcut_centered_maximum_stroke_width_mad_px
        and width_p90 is not None
        and float(width_p90)
        <= (
            config
            .graphcut_centered_maximum_stroke_width_p90_deviation_px
        )
        and (
            config.graphcut_centered_minimum_center_to_half_width_ratio
            <= float(evidence["center_to_half_width_ratio"])
            <= config.graphcut_centered_maximum_center_to_half_width_ratio
        )
    )


def _graphcut_centered_vector_gate(
    *,
    vector_candidate_iou: float,
    vector_candidate_area_ratio: float,
    phased_current_iou: float,
    phased_current_area_ratio: float,
    current_topology: tuple[int, int],
    candidate_topology: tuple[int, int],
    vector_topology: tuple[int, int],
    phased_topology: tuple[int, int],
    vector_current_vertex_ratio: float,
    phased_current_vertex_ratio: float,
    vector_vertex_count: int,
    vector_vertex_density: float,
    phased_current_boundary_p99_px: float,
    phased_current_boundary_max_px: float,
    config: EdgeGraphConfig,
) -> bool:
    """Require topology, overlap, complexity, and observable structural need."""

    return bool(
        current_topology
        == candidate_topology
        == vector_topology
        == phased_topology
        == (1, 0)
        and vector_candidate_iou
        >= config.graphcut_centered_minimum_vector_iou
        and (
            config.graphcut_centered_minimum_vector_area_ratio
            <= vector_candidate_area_ratio
            <= config.graphcut_centered_maximum_vector_area_ratio
        )
        and phased_current_iou
        >= config.graphcut_centered_minimum_phased_current_iou
        and (
            config.graphcut_centered_minimum_phased_current_area_ratio
            <= phased_current_area_ratio
            <= config.graphcut_centered_maximum_phased_current_area_ratio
        )
        and vector_current_vertex_ratio
        <= config.graphcut_centered_maximum_vector_current_vertex_ratio
        and phased_current_vertex_ratio
        <= config.graphcut_centered_maximum_phased_current_vertex_ratio
        and vector_vertex_count
        <= config.graphcut_centered_maximum_vector_vertex_count
        and vector_vertex_density
        <= config.graphcut_centered_maximum_vector_vertex_density
        and phased_current_boundary_p99_px
        >= config.graphcut_centered_minimum_structural_p99_px
        and phased_current_boundary_max_px
        >= config.graphcut_centered_minimum_structural_max_px
    )


def _symmetric_mask_boundary_distance_summary(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float]:
    """Return symmetric boundary distances without depending on truth labels."""

    kernel = np.ones((3, 3), dtype=np.uint8)

    def interior_boundary(mask: np.ndarray) -> np.ndarray:
        binary = np.asarray(mask, dtype=bool)
        eroded = cv2.erode(
            binary.astype(np.uint8),
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return binary & (eroded == 0)

    left_boundary = interior_boundary(left)
    right_boundary = interior_boundary(right)
    if not left_boundary.any() and not right_boundary.any():
        return {
            "mean_px": 0.0,
            "p95_px": 0.0,
            "p99_px": 0.0,
            "max_px": 0.0,
        }
    if not left_boundary.any() or not right_boundary.any():
        return {
            "mean_px": float("inf"),
            "p95_px": float("inf"),
            "p99_px": float("inf"),
            "max_px": float("inf"),
        }
    distance_to_right = cv2.distanceTransform(
        (~right_boundary).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    distance_to_left = cv2.distanceTransform(
        (~left_boundary).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    distances = np.concatenate(
        [
            distance_to_right[left_boundary],
            distance_to_left[right_boundary],
        ]
    ).astype(np.float64, copy=False)
    return {
        "mean_px": float(np.mean(distances)),
        "p95_px": float(np.percentile(distances, 95.0)),
        "p99_px": float(np.percentile(distances, 99.0)),
        "max_px": float(np.max(distances)),
    }


def _target_fill_candidate_gate(
    *,
    proposal_accepted_count: int,
    nearest_target_distance_px: float,
    candidate_current_iou: float,
    candidate_current_area_ratio: float,
    candidate_topology: tuple[int, int],
    current_topology: tuple[int, int],
    candidate_boundary_energy_median: float,
    current_boundary_energy_median: float,
    candidate_boundary_energy_mean: float,
    current_boundary_energy_mean: float,
) -> bool:
    """Require independent semantic, topology, overlap, and edge evidence."""

    return bool(
        proposal_accepted_count > 0
        and nearest_target_distance_px <= 4.0
        and candidate_topology == current_topology == (1, 0)
        and candidate_current_iou >= 0.99
        and 0.985 <= candidate_current_area_ratio <= 1.015
        and candidate_boundary_energy_median
        >= current_boundary_energy_median + 0.05
        and candidate_boundary_energy_mean
        >= current_boundary_energy_mean - 0.01
    )


def _digital_mask_topology(mask: np.ndarray) -> tuple[int, int]:
    binary = np.asarray(mask, dtype=np.uint8)
    component_count, _labels = cv2.connectedComponents(
        binary,
        connectivity=8,
    )
    _contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    holes = (
        0
        if hierarchy is None
        else sum(1 for item in hierarchy[0] if int(item[3]) >= 0)
    )
    return max(0, int(component_count) - 1), int(holes)


def _mask_boundary_native_energy(
    edge_magnitude: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    boundary = cv2.morphologyEx(
        np.asarray(mask, dtype=np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    values = np.asarray(edge_magnitude, dtype=np.float32)[boundary > 0]
    if not len(values):
        return 0.0, 0.0
    return float(np.median(values)), float(np.mean(values))


def _clean_selector_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    height, width = binary.shape
    minimum_component_area = max(48, int(round(binary.size * 0.00015)))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros(binary.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_component_area:
            cleaned[labels == label] = 1
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        maximum_noise_hole_area = max(32.0, height * width * 0.00045)
        for index, contour in enumerate(contours):
            if int(hierarchy[0][index][3]) >= 0 and cv2.contourArea(contour) < maximum_noise_hole_area:
                cv2.drawContours(cleaned, [contour], -1, 1, thickness=-1)
    return cleaned.astype(bool)


def _select_guided_mask(mask: np.ndarray, hints: Any) -> np.ndarray:
    seed = hints.get("seed_point") if isinstance(hints, dict) else getattr(hints, "seed_point", None)
    if seed is None:
        return mask
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    x = int(round(float(np.clip(seed[0], 0, mask.shape[1] - 1))))
    y = int(round(float(np.clip(seed[1], 0, mask.shape[0] - 1))))
    label = int(labels[y, x])
    if label == 0:
        ys, xs = np.where(mask)
        if not len(xs):
            return mask
        nearest = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        label = int(labels[ys[nearest], xs[nearest]])
    return labels == label


def _strip_offsets(config: EdgeGraphConfig) -> np.ndarray:
    count = int(round((2.0 * config.strip_radius_px) / config.strip_step_px)) + 1
    return np.linspace(
        -config.strip_radius_px,
        config.strip_radius_px,
        count,
        dtype=np.float32,
    )


def _remap_profiles(
    values: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    map_x = points[:, 0:1] + normals[:, 0:1] * offsets[None, :]
    map_y = points[:, 1:2] + normals[:, 1:2] * offsets[None, :]
    return cv2.remap(
        values,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _cyclic_points(points: np.ndarray, start: int, end: int) -> np.ndarray:
    if start <= end:
        return points[start : end + 1]
    return np.concatenate([points[start:], points[: end + 1]], axis=0)


def _closed_coordinates(points: np.ndarray) -> list[tuple[float, float]]:
    coordinates = [(float(point[0]), float(point[1])) for point in np.asarray(points).reshape(-1, 2)]
    if coordinates and coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    return coordinates


def _cv_ring(coordinates: Any, *, width: int, height: int) -> np.ndarray:
    points = np.asarray(list(coordinates), dtype=np.float64)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    points[:, 0] = np.clip(np.rint(points[:, 0]), 0, max(0, width - 1))
    points[:, 1] = np.clip(np.rint(points[:, 1]), 0, max(0, height - 1))
    return points.astype(np.int32)


def _softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(np.clip(shifted, -60.0, 0.0))
    return exponent / np.maximum(exponent.sum(axis=axis, keepdims=True), 1e-9)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


__all__ = [
    "EdgeGraphConfig",
    "EdgeGraphResult",
    "EdgeGraphSessionLike",
    "NormalStrip",
    "automatic_edgegraph_hints",
    "decode_cyclic_offsets",
    "deterministic_edge_logits",
    "fit_corner_protected_ring",
    "rasterize_geometry_mask",
    "refine_boundary_with_edgegraph",
    "resample_closed_contour",
    "sample_normal_strip",
]
