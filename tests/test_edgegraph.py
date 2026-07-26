import cv2
import numpy as np
from shapely.geometry import MultiPolygon, box

import map_boundary_builder.edgegraph as edgegraph
from map_boundary_builder.edgegraph import (
    EdgeGraphConfig,
    _fit_line_run,
    _gated_phase_logits,
    _reliable_corner_anchor_probability,
    _shortcut_supported_protrusion,
    _topology_compatible,
    _vector_corner_anchor_probability,
    automatic_edgegraph_hints,
    decode_cyclic_offsets,
    fit_corner_protected_ring,
    refine_boundary_with_edgegraph,
    resample_closed_contour,
    sample_normal_strip,
)
from map_boundary_builder.edge_phase import EdgePhaseResult
from map_boundary_builder.edgegraph_proposal import SourceNativeProposalResult
from map_boundary_builder.evaluation import boundary_distance_summary_px, boundary_f1, corner_f1


def test_automatic_edgegraph_hints_use_strongest_component_and_deep_interior() -> None:
    rgb = np.full((100, 140, 3), (230, 230, 230), dtype=np.uint8)
    coarse = np.zeros((100, 140), dtype=np.float32)
    coarse[15:85, 12:82] = 0.92
    coarse[30:60, 105:130] = 0.98
    rgb[15:85, 12:82] = (48, 126, 202)
    rgb[30:60, 105:130] = (214, 80, 70)

    hints, diagnostics = automatic_edgegraph_hints(rgb, coarse, threshold=0.45)

    assert hints.seed_point is not None
    seed_x, seed_y = hints.seed_point
    assert 12 <= seed_x < 82
    assert 15 <= seed_y < 85
    assert hints.target_rgb == (48, 126, 202)
    assert diagnostics["route"] == "selector-bootstrap-v1"
    assert diagnostics["selector_component_count"] == 2
    assert diagnostics["selected_component_coverage"] == round((70 * 70) / (100 * 140), 6)
    assert diagnostics["seed_depth_px"] >= 34.0
    assert set(diagnostics["threshold_stability_iou"]) == {"0.40", "0.50"}


def test_automatic_edgegraph_hints_fail_closed_for_empty_selector() -> None:
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    coarse = np.zeros((40, 60), dtype=np.float32)

    try:
        automatic_edgegraph_hints(rgb, coarse)
    except ValueError as exc:
        assert "did not identify" in str(exc)
    else:
        raise AssertionError("empty selector unexpectedly produced automatic hints")


def test_normal_strip_uses_native_subpixel_profiles_and_documented_channels() -> None:
    rgb = np.full((80, 120, 3), 220, dtype=np.uint8)
    coarse = np.zeros((80, 120), dtype=np.float32)
    coarse[20:60, 25:95] = 1.0
    contour = np.asarray([[25, 20], [25, 60], [95, 60], [95, 20]], dtype=np.float32)
    points = resample_closed_contour(contour, step_px=2.0)

    strip = sample_normal_strip(rgb, coarse, points)

    assert strip.features.shape == (7, len(points), 49)
    assert np.isclose(strip.offsets[0], -12.0)
    assert np.isclose(strip.offsets[-1], 12.0)
    assert np.isclose(np.diff(strip.offsets), 0.5).all()
    assert np.isfinite(strip.features).all()


def test_cyclic_decoder_recovers_subpixel_mode_without_discontinuous_jumps() -> None:
    offsets = np.linspace(-12.0, 12.0, 49, dtype=np.float32)
    logits = np.full((96, 49), -6.0, dtype=np.float32)
    logits[:, 27] = 6.0
    logits[:, 28] = 5.0
    logits[40:44, 34] = 6.5  # a locally tempting but discontinuous distractor

    decoded = decode_cyclic_offsets(logits, offsets)

    assert np.max(np.abs(np.diff(np.r_[decoded, decoded[:1]]))) <= 0.51
    assert 1.45 <= float(np.median(decoded)) <= 1.75


def test_phase_fusion_requires_reliability_and_learned_agreement_for_single_edges() -> None:
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    phase = EdgePhaseResult(
        logits=np.ones((5, 3), dtype=np.float32),
        centers_px=np.asarray([0.0, 0.0, 1.0, 3.0, 3.0], dtype=np.float32),
        stroke_widths_px=np.asarray([2.0, 2.0, 0.0, 0.0, 2.0], dtype=np.float32),
        reliability=np.asarray([0.64, 0.65, 0.90, 0.90, 0.90], dtype=np.float32),
        paired=np.asarray([True, True, False, False, True]),
    )
    learned = np.full((5, 3), -2.0, dtype=np.float32)
    learned[:, 1] = 2.0
    learned[2, 2] = 3.0

    logits, applied = _gated_phase_logits(
        phase,
        offsets,
        learned_logits=learned,
        config=EdgeGraphConfig(),
    )

    assert applied.tolist() == [False, True, True, False, False]
    assert np.all(logits[~applied] == 0.0)
    assert np.all(logits[applied] == 0.1)


def test_phase_fusion_uses_reliable_native_phase_without_learned_refiner() -> None:
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    phase = EdgePhaseResult(
        logits=np.ones((2, 3), dtype=np.float32),
        centers_px=np.zeros(2, dtype=np.float32),
        stroke_widths_px=np.zeros(2, dtype=np.float32),
        reliability=np.asarray([0.64, 0.80], dtype=np.float32),
        paired=np.asarray([False, False]),
    )

    _logits, applied = _gated_phase_logits(
        phase,
        offsets,
        learned_logits=None,
        config=EdgeGraphConfig(),
    )

    assert applied.tolist() == [False, True]


def test_rectilinear_native_phase_requires_held_out_centered_stroke_evidence(
    monkeypatch,
) -> None:
    candidate = box(100.0, 100.0, 1100.0, 1100.0)
    proposal = edgegraph._RectilinearPhaseEvidence(
        sample_count=20,
        paired_count=18,
        shift_px=-1.0,
        coverage=0.90,
        reliability=0.82,
        stroke_width_px=3.0,
        score_gain=0.60,
    )
    validation = edgegraph._RectilinearPhaseEvidence(
        sample_count=20,
        paired_count=17,
        shift_px=-0.95,
        coverage=0.85,
        reliability=0.80,
        stroke_width_px=3.0,
        score_gain=0.58,
    )
    monkeypatch.setattr(
        edgegraph,
        "_rectilinear_line_phase_evidence",
        lambda *_args, **_kwargs: (proposal, validation),
    )

    aligned = edgegraph._align_rectilinear_candidate_to_native_phase(
        np.zeros((1200, 1200, 3), dtype=np.float32),
        candidate,
        target_rgb=(80, 120, 180),
        image_shape=(1200, 1200),
        config=EdgeGraphConfig(),
    )

    assert aligned.candidate is not None
    assert aligned.diagnostics["accepted"] is True
    assert aligned.diagnostics["reason"] == "accepted-centered-stroke-phase"
    assert aligned.diagnostics["proposal_shift_px"] == -1.0
    assert aligned.diagnostics["candidate_current_iou"] >= 0.99


def test_rectilinear_native_phase_rejects_single_visible_transition(
    monkeypatch,
) -> None:
    candidate = box(100.0, 100.0, 1100.0, 1100.0)
    single_edge = edgegraph._RectilinearPhaseEvidence(
        sample_count=20,
        paired_count=0,
        shift_px=1.5,
        coverage=0.95,
        reliability=0.99,
        stroke_width_px=0.0,
        score_gain=0.90,
    )
    monkeypatch.setattr(
        edgegraph,
        "_rectilinear_line_phase_evidence",
        lambda *_args, **_kwargs: (single_edge, single_edge),
    )

    aligned = edgegraph._align_rectilinear_candidate_to_native_phase(
        np.zeros((1200, 1200, 3), dtype=np.float32),
        candidate,
        target_rgb=(80, 120, 180),
        image_shape=(1200, 1200),
        config=EdgeGraphConfig(),
    )

    assert aligned.candidate is None
    assert aligned.diagnostics["accepted"] is False
    assert aligned.diagnostics["reason"] == "proposal-coverage-gate"


def test_discrete_corner_anchors_require_reliable_profiles() -> None:
    corners = np.asarray([0.92, 0.81, 0.79, 0.20], dtype=np.float32)
    reliability = np.asarray([0.95, 0.74, 0.75, 0.99], dtype=np.float32)

    anchored = _reliable_corner_anchor_probability(
        corners,
        reliability,
        minimum_reliability=EdgeGraphConfig().corner_anchor_minimum_reliability,
    )

    assert np.allclose(anchored, [0.92, 0.0, 0.79, 0.20])


def test_corner_anchor_reliability_is_validated() -> None:
    for value in (-0.01, 1.01):
        try:
            EdgeGraphConfig(corner_anchor_minimum_reliability=value)
        except ValueError as error:
            assert "corner anchor reliability" in str(error)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("invalid corner reliability must fail closed")


def test_learned_vector_corner_peaks_remain_soft_unless_explicitly_enabled() -> None:
    corners = np.asarray([0.92, 0.81, 0.79, 0.20], dtype=np.float32)
    reliability = np.asarray([0.95, 0.74, 0.75, 0.99], dtype=np.float32)

    automatic = _vector_corner_anchor_probability(
        corners,
        reliability,
        learned_vector_owner=True,
        config=EdgeGraphConfig(),
    )
    explicit = _vector_corner_anchor_probability(
        corners,
        reliability,
        learned_vector_owner=True,
        config=EdgeGraphConfig(learned_corner_anchors_enabled=True),
    )
    deterministic = _vector_corner_anchor_probability(
        corners,
        reliability,
        learned_vector_owner=False,
        config=EdgeGraphConfig(),
    )

    assert np.count_nonzero(automatic) == 0
    assert np.allclose(explicit, [0.92, 0.0, 0.79, 0.20])
    assert np.array_equal(deterministic, explicit)


def test_topology_guard_ignores_only_tiny_buffer_repair_components() -> None:
    candidate = box(0.0, 0.0, 100.0, 100.0)
    tiny_repair = box(110.0, 0.0, 112.0, 4.0)  # Exactly 8 px² is ignored.
    material_component = box(110.0, 0.0, 112.67, 3.0)  # 8.01 px².

    assert _topology_compatible(candidate, MultiPolygon([candidate, tiny_repair]))
    assert not _topology_compatible(
        candidate,
        MultiPolygon([candidate, material_component]),
    )


def test_chord_contraction_requires_removed_region_to_be_source_native_outside() -> None:
    source = box(10.0, 10.0, 90.0, 90.0)
    candidate = box(10.0, 10.0, 60.0, 90.0)
    native = np.full((100, 100), 0.05, dtype=np.float32)

    supported = edgegraph._source_native_contraction_evidence(
        source,
        candidate,
        native_inside_probability=native,
        image_shape=native.shape,
        config=EdgeGraphConfig(),
    )
    native[:, 61:] = 0.95
    real_branch = edgegraph._source_native_contraction_evidence(
        source,
        candidate,
        native_inside_probability=native,
        image_shape=native.shape,
        config=EdgeGraphConfig(),
    )
    unavailable = edgegraph._source_native_contraction_evidence(
        source,
        candidate,
        native_inside_probability=None,
        image_shape=native.shape,
        config=EdgeGraphConfig(),
    )

    assert supported["accepted"] is True
    assert supported["removed_inside_mean"] == 0.05
    assert real_branch["accepted"] is False
    assert real_branch["reason"] == "removed-region-inside-mean"
    assert unavailable["accepted"] is False
    assert unavailable["reason"] == "native-evidence-unavailable"


def test_chord_contraction_rejects_unevaluated_native_region() -> None:
    source = box(10.0, 10.0, 90.0, 90.0)
    candidate = box(10.0, 10.0, 60.0, 90.0)
    native = np.full((100, 100), 0.05, dtype=np.float32)
    native[:, 70:] = np.nan

    evidence = edgegraph._source_native_contraction_evidence(
        source,
        candidate,
        native_inside_probability=native,
        image_shape=native.shape,
        config=EdgeGraphConfig(),
    )

    assert evidence["accepted"] is False
    assert evidence["reason"] == "insufficient-native-coverage"


def test_chord_shortcut_fails_closed_for_viewport_clipped_ring() -> None:
    rgb = np.full((100, 140, 3), 0.7, dtype=np.float32)
    points = np.asarray(
        [
            [0.0, 15.0],
            [20.0, 15.0],
            [40.0, 5.0],
            [70.0, 20.0],
            [100.0, 5.0],
            [130.0, 15.0],
            [139.0, 15.0],
            [139.0, 85.0],
            [115.0, 85.0],
            [90.0, 95.0],
            [45.0, 85.0],
            [0.0, 85.0],
        ],
        dtype=np.float32,
    )

    refined, count, depth = _shortcut_supported_protrusion(
        rgb,
        points,
        hints={"target_rgb": (40, 120, 240), "seed_point": (70.0, 50.0)},
        config=EdgeGraphConfig(),
    )

    assert count == 0
    assert depth == 0.0
    assert np.array_equal(refined, points)


def test_chord_shortcut_retries_fine_anchors_with_a_shallow_depth_cap(
    monkeypatch,
) -> None:
    points = np.asarray(
        [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]],
        dtype=np.float32,
    )
    fine = points + np.asarray([0.25, 0.0], dtype=np.float32)
    calls: list[tuple[float, float]] = []

    def fake_search(
        _rgb_float,
        _points,
        *,
        hints,
        config,
        native_inside_probability,
        diagnostics,
    ):
        del native_inside_probability, diagnostics
        assert hints["target_rgb"] == (40, 120, 240)
        calls.append(
            (
                config.shortcut_anchor_tolerance_px,
                config.shortcut_maximum_depth_fraction,
            )
        )
        if len(calls) == 1:
            return points, 0, 0.0
        return fine, 1, 15.0

    monkeypatch.setattr(
        edgegraph,
        "_shortcut_supported_protrusion_at_scale",
        fake_search,
    )

    refined, count, depth = _shortcut_supported_protrusion(
        np.zeros((100, 100, 3), dtype=np.float32),
        points,
        hints={"target_rgb": (40, 120, 240)},
        config=EdgeGraphConfig(),
    )

    assert calls == [(6.0, 0.16), (3.0, 0.05)]
    assert count == 1
    assert depth == 15.0
    assert np.array_equal(refined, fine)


def test_chord_shortcut_keeps_a_supported_broad_deep_candidate(
    monkeypatch,
) -> None:
    points = np.asarray(
        [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]],
        dtype=np.float32,
    )
    calls: list[tuple[float, float]] = []

    def fake_search(
        _rgb_float,
        _points,
        *,
        hints,
        config,
        native_inside_probability,
        diagnostics,
    ):
        del native_inside_probability, diagnostics
        calls.append(
            (
                config.shortcut_anchor_tolerance_px,
                config.shortcut_maximum_depth_fraction,
            )
        )
        return points, 1, 47.0

    monkeypatch.setattr(
        edgegraph,
        "_shortcut_supported_protrusion_at_scale",
        fake_search,
    )

    refined, count, depth = _shortcut_supported_protrusion(
        np.zeros((100, 100, 3), dtype=np.float32),
        points,
        hints={"target_rgb": (40, 120, 240)},
        config=EdgeGraphConfig(),
    )

    assert calls == [(6.0, 0.16)]
    assert count == 1
    assert depth == 47.0
    assert np.array_equal(refined, points)


def test_target_fill_candidate_gate_requires_every_independent_evidence() -> None:
    supported = {
        "proposal_accepted_count": 1,
        "nearest_target_distance_px": 0.0,
        "candidate_current_iou": 0.995586,
        "candidate_current_area_ratio": 1.003257,
        "candidate_topology": (1, 0),
        "current_topology": (1, 0),
        "candidate_boundary_energy_median": 0.487164,
        "current_boundary_energy_median": 0.267221,
        "candidate_boundary_energy_mean": 0.538542,
        "current_boundary_energy_mean": 0.460756,
    }

    assert edgegraph._target_fill_candidate_gate(**supported)
    rejected_values = {
        "proposal_accepted_count": 0,
        "nearest_target_distance_px": 4.01,
        "candidate_current_iou": 0.989,
        "candidate_current_area_ratio": 0.984,
        "candidate_topology": (1, 1),
        "current_topology": (2, 0),
        "candidate_boundary_energy_median": 0.317,
        "candidate_boundary_energy_mean": 0.449,
    }
    for name, value in rejected_values.items():
        evidence = {**supported, name: value}
        assert not edgegraph._target_fill_candidate_gate(**evidence), name
    excessive_area = {
        **supported,
        "candidate_current_area_ratio": 1.016,
    }
    assert not edgegraph._target_fill_candidate_gate(**excessive_area)


def test_flat_fill_candidate_gate_requires_every_independent_evidence() -> None:
    config = EdgeGraphConfig()
    supported = {
        "proposal_accepted_count": 0,
        "shortcut_count": 1,
        "mean_reliability": 0.581633,
        "target_separation": 14.0,
        "nearest_target_distance_px": 0.0,
        "closed_addition_fraction": 0.046908,
        "candidate_current_iou": 0.990705,
        "candidate_current_area_ratio": 0.993114,
        "candidate_topology": (1, 0),
        "current_topology": (1, 0),
        "candidate_boundary_energy_median": 0.151515,
        "current_boundary_energy_median": 0.151515,
        "candidate_boundary_energy_mean": 0.137720,
        "current_boundary_energy_mean": 0.113518,
        "config": config,
    }

    assert edgegraph._flat_fill_candidate_gate(**supported)
    rejected_values = {
        "proposal_accepted_count": 1,
        "shortcut_count": 0,
        "mean_reliability": config.flat_fill_maximum_reliability,
        "target_separation": config.flat_fill_minimum_target_separation - 0.001,
        "nearest_target_distance_px": 4.01,
        "closed_addition_fraction": (
            config.flat_fill_maximum_closed_addition_fraction + 0.001
        ),
        "candidate_current_iou": config.flat_fill_minimum_candidate_iou - 0.001,
        "candidate_current_area_ratio": (
            config.flat_fill_minimum_candidate_area_ratio - 0.001
        ),
        "candidate_topology": (1, 1),
        "current_topology": (2, 0),
        "candidate_boundary_energy_median": (
            supported["current_boundary_energy_median"]
            - config.flat_fill_maximum_boundary_median_drop
            - 0.001
        ),
        "candidate_boundary_energy_mean": (
            supported["current_boundary_energy_mean"]
            + config.flat_fill_minimum_boundary_mean_gain
            - 0.001
        ),
    }
    for name, value in rejected_values.items():
        evidence = {**supported, name: value}
        assert not edgegraph._flat_fill_candidate_gate(**evidence), name
    excessive_area = {
        **supported,
        "candidate_current_area_ratio": (
            config.flat_fill_maximum_candidate_area_ratio + 0.001
        ),
    }
    assert not edgegraph._flat_fill_candidate_gate(**excessive_area)


def test_structural_flat_fill_gate_requires_stronger_independent_evidence() -> None:
    config = EdgeGraphConfig()
    supported = {
        "proposal_accepted_count": 0,
        "shortcut_count": 1,
        "mean_reliability": 0.416,
        "target_separation": 14.0,
        "nearest_target_distance_px": 0.0,
        "closed_addition_fraction": 0.048,
        "candidate_current_iou": 0.957,
        "candidate_current_area_ratio": 1.021,
        "candidate_topology": (1, 0),
        "current_topology": (1, 0),
        "candidate_boundary_energy_median": 0.152,
        "current_boundary_energy_median": 0.126,
        "candidate_boundary_energy_mean": 0.138,
        "current_boundary_energy_mean": 0.094,
        "config": config,
    }

    assert edgegraph._flat_fill_structural_candidate_gate(**supported)
    rejected_values = {
        "shortcut_count": 0,
        "mean_reliability": (
            config.flat_fill_structural_maximum_reliability + 0.001
        ),
        "target_separation": (
            config.flat_fill_structural_minimum_target_separation - 0.001
        ),
        "closed_addition_fraction": (
            config.flat_fill_structural_maximum_closed_addition_fraction + 0.001
        ),
        "candidate_current_iou": (
            config.flat_fill_structural_minimum_candidate_iou - 0.001
        ),
        "candidate_topology": (1, 1),
        "candidate_boundary_energy_median": (
            supported["current_boundary_energy_median"]
            + config.flat_fill_structural_minimum_boundary_median_gain
            - 0.001
        ),
        "candidate_boundary_energy_mean": (
            supported["current_boundary_energy_mean"]
            + config.flat_fill_structural_minimum_boundary_mean_gain
            - 0.001
        ),
    }
    for name, value in rejected_values.items():
        evidence = {**supported, name: value}
        assert not edgegraph._flat_fill_structural_candidate_gate(**evidence), name


def test_graphcut_gate_requires_strip_saturation_and_native_energy_gain() -> None:
    config = EdgeGraphConfig()
    supported = {
        "proposal_accepted_count": 0,
        "maximum_displacement_px": 11.99,
        "candidate_current_iou": 0.976,
        "candidate_current_area_ratio": 1.023,
        "candidate_topology": (1, 0),
        "current_topology": (1, 0),
        "candidate_boundary_energy_median": 0.464,
        "current_boundary_energy_median": 0.464,
        "candidate_boundary_energy_mean": 0.449,
        "current_boundary_energy_mean": 0.383,
        "config": config,
    }

    assert edgegraph._graphcut_candidate_gate(**supported)
    rejected_values = {
        "proposal_accepted_count": 1,
        "maximum_displacement_px": (
            config.graphcut_minimum_maximum_displacement_px - 0.001
        ),
        "candidate_current_iou": config.graphcut_minimum_candidate_iou - 0.001,
        "candidate_current_area_ratio": (
            config.graphcut_maximum_candidate_area_ratio + 0.001
        ),
        "candidate_topology": (1, 1),
        "candidate_boundary_energy_median": (
            supported["current_boundary_energy_median"]
            - config.graphcut_maximum_boundary_median_drop
            - 0.001
        ),
        "candidate_boundary_energy_mean": (
            supported["current_boundary_energy_mean"]
            + config.graphcut_minimum_boundary_mean_gain
            - 0.001
        ),
    }
    for name, value in rejected_values.items():
        evidence = {**supported, name: value}
        assert not edgegraph._graphcut_candidate_gate(**evidence), name


def test_graphcut_retry_recovers_source_supported_missing_lobe() -> None:
    rgb = np.full((400, 500, 3), 255, dtype=np.uint8)
    truth = np.zeros((400, 500), dtype=bool)
    truth[100:300, 150:350] = True
    truth[170:230, 100:150] = True
    rgb[truth] = (60, 120, 190)
    current = truth.copy()
    current[170:230, 100:150] = False
    before = current.copy()

    result = edgegraph._recover_source_native_graphcut_geometry(
        rgb,
        current_mask=current,
        seed_point=(200.0, 200.0),
        proposal_diagnostics={"accepted_count": 0},
        maximum_displacement_px=12.0,
        config=EdgeGraphConfig(
            graphcut_minimum_candidate_iou=0.90,
            graphcut_maximum_candidate_area_ratio=1.10,
            graphcut_minimum_boundary_mean_gain=0.0,
            graphcut_minimum_vector_iou=0.98,
            graphcut_minimum_vector_area_ratio=0.95,
            graphcut_maximum_vector_area_ratio=1.05,
        ),
        native_fields=edgegraph._prepare_native_image_fields(rgb),
    )

    assert np.array_equal(current, before), "the learned mask must remain immutable"
    assert result.diagnostics["accepted"] is True
    assert result.diagnostics["mode"] == "strip-saturation-graphcut"
    assert result.mask is not None
    assert result.mask[200, 120]
    assert np.array_equal(result.mask, truth)


def _supported_centered_graphcut_phase(
    *,
    center_px: float = 0.9,
    positive_center_fraction: float = 0.96,
) -> dict[str, object]:
    return {
        "profile_count": 100,
        "paired_count": 90,
        "paired_fraction": 0.90,
        "reliable_paired_count": 82,
        "reliable_paired_fraction": 0.82,
        "coherent_paired_fraction": 0.76,
        "positive_center_fraction": positive_center_fraction,
        "reliability_median": 0.74,
        "reliability_p10": 0.62,
        "center_px": center_px,
        "center_mad_px": 0.12,
        "center_p90_deviation_px": 0.48,
        "stroke_width_px": 2.8,
        "stroke_width_mad_px": 0.14,
        "stroke_width_p90_deviation_px": 0.56,
        "center_to_half_width_ratio": center_px / 1.4,
    }


def test_centered_graphcut_phase_routes_are_strict_and_mutually_exclusive() -> None:
    config = EdgeGraphConfig()
    candidate_phase = _supported_centered_graphcut_phase()
    current_phase = _supported_centered_graphcut_phase(
        center_px=0.48,
        positive_center_fraction=0.79,
    )

    assert edgegraph._graphcut_centered_phase_gate(
        candidate_phase,
        config=config,
    )
    assert edgegraph._graphcut_current_phase_positive_gate(
        current_phase,
        config=config,
    )

    negative_current_phase = {
        **current_phase,
        "center_px": -0.48,
        "positive_center_fraction": 0.02,
        "center_to_half_width_ratio": -0.34,
    }
    assert not edgegraph._graphcut_current_phase_positive_gate(
        negative_current_phase,
        config=config,
    )
    assert not edgegraph._graphcut_centered_phase_gate(
        {**candidate_phase, "center_p90_deviation_px": 0.651},
        config=config,
    )


def test_centered_graphcut_gates_reject_complexity_and_near_identity() -> None:
    config = EdgeGraphConfig()
    candidate = {
        "proposal_accepted_count": 0,
        "candidate_current_iou": 0.985,
        "candidate_current_area_ratio": 1.015,
        "candidate_topology": (1, 0),
        "current_topology": (1, 0),
        "candidate_boundary_energy_median": 0.60,
        "current_boundary_energy_median": 0.60,
        "candidate_boundary_energy_mean": 0.62,
        "current_boundary_energy_mean": 0.50,
        "config": config,
    }
    assert edgegraph._centered_graphcut_candidate_gate(**candidate)
    assert not edgegraph._centered_graphcut_candidate_gate(
        **{
            **candidate,
            "candidate_boundary_energy_mean": 0.599,
        }
    )

    vector = {
        "vector_candidate_iou": 0.998,
        "vector_candidate_area_ratio": 1.002,
        "phased_current_iou": 0.994,
        "phased_current_area_ratio": 1.006,
        "current_topology": (1, 0),
        "candidate_topology": (1, 0),
        "vector_topology": (1, 0),
        "phased_topology": (1, 0),
        "vector_current_vertex_ratio": 1.0,
        "phased_current_vertex_ratio": 1.0,
        "vector_vertex_count": 12,
        "vector_vertex_density": 0.006,
        "phased_current_boundary_p99_px": 1.5,
        "phased_current_boundary_max_px": 2.25,
        "config": config,
    }
    assert edgegraph._graphcut_centered_vector_gate(**vector)
    assert not edgegraph._graphcut_centered_vector_gate(
        **{**vector, "vector_current_vertex_ratio": 1.001}
    )
    assert not edgegraph._graphcut_centered_vector_gate(
        **{
            **vector,
            "phased_current_boundary_p99_px": 1.0,
            "phased_current_boundary_max_px": 1.414,
        }
    )


def test_centered_graphcut_routes_phase_the_outer_contour(
    monkeypatch,
) -> None:
    height, width = 400, 500
    current_geometry = box(150.0, 100.0, 350.0, 300.0)
    candidate_geometry = current_geometry.buffer(1.0, join_style=2)
    current = edgegraph.rasterize_geometry_mask(
        current_geometry,
        width=width,
        height=height,
    )
    candidate = edgegraph.rasterize_geometry_mask(
        candidate_geometry,
        width=width,
        height=height,
    )
    rgb = np.full((height, width, 3), 220, dtype=np.uint8)

    def fake_grabcut(
        _rgb,
        graphcut_mask,
        _rect,
        _background,
        _foreground,
        _iterations,
        _mode,
    ):
        graphcut_mask[:] = cv2.GC_BGD
        graphcut_mask[candidate] = cv2.GC_FGD

    monkeypatch.setattr(cv2, "grabCut", fake_grabcut)
    phase_calls: list[tuple[str, float | None]] = []

    def fake_phase(_rgb, _mask, geometry, **_kwargs):
        current_probe = geometry.equals_exact(
            current_geometry,
            tolerance=1e-8,
        )
        phase = (
            _supported_centered_graphcut_phase(
                center_px=0.48,
                positive_center_fraction=0.79,
            )
            if current_probe
            else _supported_centered_graphcut_phase(center_px=1.0)
        )
        phase_calls.append(
            (
                "current" if current_probe else "candidate",
                _kwargs.get("sample_step_px"),
            )
        )
        return phase

    monkeypatch.setattr(edgegraph, "_graphcut_phase_evidence", fake_phase)
    config = EdgeGraphConfig(
        graphcut_centered_minimum_candidate_iou=0.90,
        graphcut_centered_minimum_candidate_area_ratio=0.90,
        graphcut_centered_maximum_candidate_area_ratio=1.10,
        graphcut_centered_minimum_boundary_mean_gain=0.0,
        graphcut_centered_minimum_vector_iou=0.90,
        graphcut_centered_minimum_vector_area_ratio=0.90,
        graphcut_centered_maximum_vector_area_ratio=1.10,
        graphcut_centered_minimum_phased_current_iou=0.90,
        graphcut_centered_minimum_phased_current_area_ratio=0.90,
        graphcut_centered_maximum_phased_current_area_ratio=1.10,
        graphcut_centered_maximum_vector_current_vertex_ratio=1.10,
        graphcut_centered_maximum_phased_current_vertex_ratio=1.10,
        graphcut_centered_minimum_structural_p99_px=0.0,
        graphcut_centered_minimum_structural_max_px=0.0,
    )
    shared = {
        "rgb": rgb,
        "current_mask": current,
        "current_geometry": current_geometry,
        "target_rgb": (80, 120, 180),
        "seed_point": (250.0, 200.0),
        "proposal_diagnostics": {"accepted_count": 0},
        "maximum_displacement_px": 4.0,
        "config": config,
        "native_fields": edgegraph._prepare_native_image_fields(rgb),
    }

    low_reliability = edgegraph._recover_source_native_graphcut_geometry(
        **shared,
        mean_reliability=0.80,
    )

    assert low_reliability.diagnostics["accepted"] is True
    assert (
        low_reliability.diagnostics["necessity_route"]
        == "low-current-reliability-structural-residual"
    )
    assert phase_calls == [("candidate", None)]
    assert low_reliability.diagnostics["phase_distance_px"] == 1.0

    phase_calls.clear()
    high_reliability = edgegraph._recover_source_native_graphcut_geometry(
        **shared,
        mean_reliability=0.99,
    )

    assert high_reliability.diagnostics["accepted"] is True
    assert (
        high_reliability.diagnostics["necessity_route"]
        == "current-geometry-paired-phase-positive"
    )
    assert phase_calls == [
        ("current", config.graphcut_current_phase_sample_step_px),
        ("candidate", None),
    ]
    assert high_reliability.diagnostics["phase_distance_px"] == 1.0
    assert (
        high_reliability.diagnostics["necessity_routes"]
        ["low_current_reliability"]["eligible"]
        is False
    )


def test_supported_target_fill_component_reconstructs_a_sharp_vector(
    monkeypatch,
) -> None:
    rgb = np.full((256, 320, 3), 255, dtype=np.uint8)
    rgb[40:220, 50:270] = (50, 100, 150)
    target = np.all(
        rgb == np.asarray((50, 100, 150), dtype=np.uint8),
        axis=2,
    )
    current = cv2.dilate(
        target.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    monkeypatch.setattr(
        edgegraph,
        "_target_fill_candidate_gate",
        lambda **_evidence: True,
    )

    result = edgegraph._recover_source_native_target_fill_geometry(
        rgb,
        current_mask=current,
        target_rgb=(50, 100, 150),
        seed_point=(100.0, 100.0),
        proposal_diagnostics={"accepted_count": 1},
        config=EdgeGraphConfig(),
        native_fields=edgegraph._prepare_native_image_fields(rgb),
    )

    assert result.geometry is not None, result.diagnostics
    assert result.mask is not None
    assert result.diagnostics["accepted"] is True
    assert result.diagnostics["reason"] == "accepted"
    assert result.diagnostics["vector_line_count"] == 4
    assert result.diagnostics["vector_vertex_count"] == 4
    assert result.diagnostics["vector_current_iou"] > 0.999


def test_adaptive_flat_fill_recovers_near_white_occluded_corner() -> None:
    rgb = np.full((512, 640, 3), 255, dtype=np.uint8)
    target_rgb = (238, 255, 255)
    rgb[70:440, 60:580] = target_rgb
    current_uint8 = np.zeros((512, 640), dtype=np.uint8)
    current_uint8[70:440, 60:580] = 1
    cv2.fillPoly(
        current_uint8,
        [
            np.asarray(
                [[60, 410], [60, 440], [90, 440]],
                dtype=np.int32,
            )
        ],
        0,
    )
    current = current_uint8.astype(bool)
    before = current.copy()

    result = edgegraph._recover_source_native_flat_fill_geometry(
        rgb,
        current_mask=current,
        target_rgb=target_rgb,
        seed_point=(300.0, 250.0),
        proposal_diagnostics={"accepted_count": 0},
        shortcut_count=1,
        mean_reliability=0.58,
        config=EdgeGraphConfig(),
        native_fields=edgegraph._prepare_native_image_fields(rgb),
    )

    assert np.array_equal(current, before), "the localized mask must remain immutable"
    assert result.geometry is not None, result.diagnostics
    assert result.mask is not None
    assert result.diagnostics["accepted"] is True
    assert result.diagnostics["reason"] == "accepted"
    assert result.diagnostics["target_radius"] < 17.0
    assert result.diagnostics["candidate_current_iou"] > 0.99
    assert result.diagnostics["vector_current_iou"] > 0.985
    assert result.diagnostics["vector_vertex_count"] == 4
    assert not current[438, 62]
    assert result.mask[438, 62]


def test_adaptive_flat_fill_rejects_real_narrow_notch() -> None:
    rgb = np.full((512, 640, 3), 255, dtype=np.uint8)
    target_rgb = (238, 255, 255)
    current = np.zeros((512, 640), dtype=bool)
    current[70:440, 60:580] = True
    current[420:440, 300:320] = False
    rgb[current] = target_rgb

    result = edgegraph._recover_source_native_flat_fill_geometry(
        rgb,
        current_mask=current,
        target_rgb=target_rgb,
        seed_point=(200.0, 250.0),
        proposal_diagnostics={"accepted_count": 0},
        shortcut_count=1,
        mean_reliability=0.58,
        config=EdgeGraphConfig(),
        native_fields=edgegraph._prepare_native_image_fields(rgb),
    )

    assert result.geometry is None
    assert result.mask is None
    assert result.diagnostics["evaluated"] is True
    assert result.diagnostics["reason"] == "candidate-gate"
    assert (
        result.diagnostics["candidate_boundary_energy_mean"]
        < result.diagnostics["current_boundary_energy_mean"]
    )


def test_adaptive_flat_fill_requires_shortcut_and_low_reliability() -> None:
    rgb = np.full((256, 320, 3), 255, dtype=np.uint8)
    current = np.zeros((256, 320), dtype=bool)
    current[40:220, 50:270] = True
    rgb[current] = (238, 255, 255)
    shared = {
        "rgb": rgb,
        "current_mask": current,
        "target_rgb": (238, 255, 255),
        "seed_point": (100.0, 100.0),
        "proposal_diagnostics": {"accepted_count": 0},
        "config": EdgeGraphConfig(),
        "native_fields": edgegraph._prepare_native_image_fields(rgb),
    }

    no_shortcut = edgegraph._recover_source_native_flat_fill_geometry(
        **shared,
        shortcut_count=0,
        mean_reliability=0.58,
    )
    reliable = edgegraph._recover_source_native_flat_fill_geometry(
        **shared,
        shortcut_count=1,
        mean_reliability=0.90,
    )

    assert no_shortcut.geometry is None
    assert no_shortcut.diagnostics["evaluated"] is False
    assert no_shortcut.diagnostics["reason"] == "insufficient-structural-evidence"
    assert reliable.geometry is None
    assert reliable.diagnostics["evaluated"] is False
    assert reliable.diagnostics["reason"] == "insufficient-structural-evidence"


def test_adaptive_flat_fill_rejects_unseparated_target_color() -> None:
    rgb = np.full((256, 320, 3), (242, 255, 255), dtype=np.uint8)
    current = np.zeros((256, 320), dtype=bool)
    current[40:220, 50:270] = True
    rgb[current] = (238, 255, 255)

    result = edgegraph._recover_source_native_flat_fill_geometry(
        rgb,
        current_mask=current,
        target_rgb=(238, 255, 255),
        seed_point=(100.0, 100.0),
        proposal_diagnostics={"accepted_count": 0},
        shortcut_count=1,
        mean_reliability=0.58,
        config=EdgeGraphConfig(),
        native_fields=edgegraph._prepare_native_image_fields(rgb),
    )

    assert result.geometry is None
    assert result.mask is None
    assert result.diagnostics["evaluated"] is True
    assert result.diagnostics["reason"] == "weak-target-separation"


def test_line_fit_uses_inlier_majority_for_local_map_crossing() -> None:
    x = np.linspace(0.0, 200.0, 101, dtype=np.float32)
    run = np.stack([x, np.zeros_like(x)], axis=1)
    run[45:53, 1] = 3.25
    config = EdgeGraphConfig()

    center = run.mean(axis=0)
    _u, _s, vh = np.linalg.svd(run - center, full_matrices=False)
    direction = vh[0]
    residual = np.abs(
        (run[:, 0] - center[0]) * direction[1]
        - (run[:, 1] - center[1]) * direction[0]
    )

    assert np.quantile(residual, 0.95) > config.line_fit_tolerance_px
    assert np.quantile(residual, config.line_fit_inlier_quantile) < config.line_fit_tolerance_px
    assert _fit_line_run(run, config=config) is not None


def test_vector_fit_collapses_only_tiny_supported_corner_bevel() -> None:
    vertices = np.asarray(
        [[0.0, 0.0], [192.0, 0.0], [200.0, 8.0], [200.0, 200.0], [0.0, 200.0]],
        dtype=np.float32,
    )
    dense_parts: list[np.ndarray] = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        count = max(2, int(np.ceil(np.linalg.norm(end - start))))
        dense_parts.append(np.linspace(start, end, count, endpoint=False, dtype=np.float32))
    dense = np.concatenate(dense_parts, axis=0)

    coordinates, line_count, _protected = fit_corner_protected_ring(dense)
    fitted = np.asarray(coordinates[:-1], dtype=np.float32)

    assert line_count == 4
    assert len(fitted) == 4
    assert np.min(np.linalg.norm(fitted - np.asarray([200.0, 0.0]), axis=1)) < 0.25


def test_vector_fit_preserves_short_parallel_step() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [200.0, 0.0],
            [200.0, 100.0],
            [110.0, 100.0],
            [110.0, 106.0],
            [0.0, 106.0],
        ],
        dtype=np.float32,
    )
    dense_parts: list[np.ndarray] = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        count = max(2, int(np.ceil(np.linalg.norm(end - start))))
        dense_parts.append(np.linspace(start, end, count, endpoint=False, dtype=np.float32))
    dense = np.concatenate(dense_parts, axis=0)

    coordinates, _line_count, _protected = fit_corner_protected_ring(dense)
    fitted = np.asarray(coordinates[:-1], dtype=np.float32)

    assert np.min(np.linalg.norm(fitted - np.asarray([110.0, 100.0]), axis=1)) < 0.25
    assert np.min(np.linalg.norm(fitted - np.asarray([110.0, 106.0]), axis=1)) < 0.25


def test_edgegraph_recovers_sharp_native_edges_from_coarse_selector() -> None:
    height, width = 256, 320
    truth = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(
        [[31, 24], [263, 24], [263, 78], [287, 78], [287, 212], [181, 212], [181, 232], [31, 232]],
        dtype=np.int32,
    )
    cv2.fillPoly(truth, [polygon], 1)
    rgb = np.full((height, width, 3), 220, dtype=np.uint8)
    rgb[truth > 0] = (35, 120, 240)
    low = cv2.resize(truth.astype(np.float32), (40, 32), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(low, (width, height), interpolation=cv2.INTER_LINEAR)
    coarse_mask = coarse >= 0.45

    result = refine_boundary_with_edgegraph(rgb, coarse, config=EdgeGraphConfig())

    coarse_distance = boundary_distance_summary_px(coarse_mask, truth)
    refined_distance = boundary_distance_summary_px(result.mask, truth)
    assert result.pixel_geometry.is_valid
    assert result.diagnostics["source_native"] is True
    assert result.diagnostics["selector_geometry_exported"] is False
    assert refined_distance["max_px"] < coarse_distance["max_px"]
    assert boundary_f1(result.mask, truth, tolerance_px=1.0)["f1"] > 0.99
    assert corner_f1(result.mask, truth, tolerance_px=2.0)["f1"] > corner_f1(
        coarse_mask, truth, tolerance_px=2.0
    )["f1"]


def _stub_refined_ring(
    coordinates: list[tuple[float, float]],
) -> edgegraph._RefinedRing:
    return edgegraph._RefinedRing(
        coordinates=coordinates,
        sample_count=len(coordinates) - 1,
        corner_count=4,
        line_run_count=4,
        mean_reliability=1.0,
        mean_entropy=0.0,
        mean_displacement_px=0.0,
        maximum_displacement_px=0.0,
        wide_recenter_count=0,
        wide_recenter_maximum_px=0.0,
        shortcut_count=0,
        shortcut_maximum_depth_px=0.0,
        phase_reliability=1.0,
        phase_paired_fraction=1.0,
        phase_applied_fraction=0.0,
        phase_mean_stroke_width_px=0.0,
        rectilinear_fit_accepted=False,
        rectilinear_fit_reason="disabled",
        rectilinear_fit_run_count=0,
        rectilinear_fit_support_p95_px=None,
    )


def _disable_source_native_fill_reconstruction(monkeypatch) -> None:
    def no_reconstruction(*_args, **_kwargs):
        return edgegraph._TargetFillReconstruction(
            geometry=None,
            mask=None,
            diagnostics={
                "evaluated": False,
                "accepted": False,
                "reason": "test-disabled",
            },
        )

    monkeypatch.setattr(
        edgegraph,
        "_recover_source_native_target_fill_geometry",
        no_reconstruction,
    )
    monkeypatch.setattr(
        edgegraph,
        "_recover_source_native_flat_fill_geometry",
        no_reconstruction,
    )


def test_source_native_proposal_topology_fallback_retries_original_selector_once(
    monkeypatch,
) -> None:
    height, width = 96, 128
    selector = np.zeros((height, width), dtype=np.float32)
    selector[18:78, 18:88] = 1.0
    proposal_mask = selector >= 0.45
    proposal_mask[42:54, 87:114] = True
    proposal_probabilities = selector.copy()
    proposal_probabilities[proposal_mask] = 1.0
    proposal_diagnostics = {
        "source_native_proposals": True,
        "reason": "accepted",
        "accepted_count": 1,
        "shortcut_suppression_recommended": False,
    }
    proposal_calls = 0
    refinement_routes: list[bool] = []

    def fake_proposal(*_args, **_kwargs):
        nonlocal proposal_calls
        proposal_calls += 1
        return SourceNativeProposalResult(
            mask=proposal_mask,
            probabilities=proposal_probabilities,
            diagnostics=proposal_diagnostics,
        )

    def fake_refine_ring(
        _rgb,
        _coarse,
        contour,
        *,
        session,
        hints,
        config,
        native_fields,
    ):
        del session, hints, native_fields
        refinement_routes.append(config.source_native_proposals_enabled)
        if config.source_native_proposals_enabled:
            return _stub_refined_ring(
                [
                    (18.0, 18.0),
                    (113.0, 77.0),
                    (18.0, 77.0),
                    (113.0, 18.0),
                    (18.0, 18.0),
                ]
            )
        coordinates = [
            (float(point[0]), float(point[1]))
            for point in np.asarray(contour)
        ]
        coordinates.append(coordinates[0])
        return _stub_refined_ring(coordinates)

    monkeypatch.setattr(
        edgegraph,
        "recover_source_native_rectilinear_proposals",
        fake_proposal,
    )
    monkeypatch.setattr(edgegraph, "_refine_ring", fake_refine_ring)
    _disable_source_native_fill_reconstruction(monkeypatch)

    result = refine_boundary_with_edgegraph(
        np.full((height, width, 3), 220, dtype=np.uint8),
        selector,
        hints={"target_rgb": (40, 120, 240), "seed_point": (50.0, 48.0)},
        config=EdgeGraphConfig(rectilinear_fit_enabled=False),
    )

    assert proposal_calls == 1
    assert refinement_routes == [True, False]
    assert result.mask[48, 105] == 0
    assert result.mask[48, 50] == 1
    assert result.diagnostics["topology_fallback_count"] == 0
    retry = result.diagnostics["source_native_proposal_retry"]
    assert retry["attempted"] is True
    assert retry["accepted"] is True
    assert retry["reason"] == "proposal-topology-fallback"
    assert retry["rejected_proposal"] == proposal_diagnostics
    assert result.diagnostics["source_native_proposal"]["accepted_count"] == 0


def test_valid_source_native_proposal_remains_without_retry(
    monkeypatch,
) -> None:
    height, width = 96, 128
    selector = np.zeros((height, width), dtype=np.float32)
    selector[18:78, 18:88] = 1.0
    proposal_mask = selector >= 0.45
    proposal_mask[42:54, 87:114] = True
    proposal_probabilities = selector.copy()
    proposal_probabilities[proposal_mask] = 1.0
    proposal_diagnostics = {
        "source_native_proposals": True,
        "reason": "accepted",
        "accepted_count": 1,
        "shortcut_suppression_recommended": False,
    }
    proposal_calls = 0
    refinement_routes: list[bool] = []

    def fake_proposal(*_args, **_kwargs):
        nonlocal proposal_calls
        proposal_calls += 1
        return SourceNativeProposalResult(
            mask=proposal_mask,
            probabilities=proposal_probabilities,
            diagnostics=proposal_diagnostics,
        )

    def fake_refine_ring(
        _rgb,
        _coarse,
        contour,
        *,
        session,
        hints,
        config,
        native_fields,
    ):
        del session, hints, native_fields
        refinement_routes.append(config.source_native_proposals_enabled)
        coordinates = [
            (float(point[0]), float(point[1]))
            for point in np.asarray(contour)
        ]
        coordinates.append(coordinates[0])
        return _stub_refined_ring(coordinates)

    monkeypatch.setattr(
        edgegraph,
        "recover_source_native_rectilinear_proposals",
        fake_proposal,
    )
    monkeypatch.setattr(edgegraph, "_refine_ring", fake_refine_ring)
    _disable_source_native_fill_reconstruction(monkeypatch)

    result = refine_boundary_with_edgegraph(
        np.full((height, width, 3), 220, dtype=np.uint8),
        selector,
        hints={"target_rgb": (40, 120, 240), "seed_point": (50.0, 48.0)},
        config=EdgeGraphConfig(rectilinear_fit_enabled=False),
    )

    assert proposal_calls == 1
    assert refinement_routes == [True]
    assert result.mask[48, 105] == 1
    assert result.diagnostics["topology_fallback_count"] == 0
    assert result.diagnostics["source_native_proposal"] == proposal_diagnostics
    assert "source_native_proposal_retry" not in result.diagnostics


class _FakeInput:
    name = "edge_strip"


class _CenterSession:
    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, feed):
        values = feed["edge_strip"]
        length, width = values.shape[2:]
        logits = np.full((1, length, width), -8.0, dtype=np.float32)
        logits[:, :, width // 2] = 8.0
        corners = np.full((1, length), -8.0, dtype=np.float32)
        reliability = np.full((1, length), 8.0, dtype=np.float32)
        return [logits, corners, reliability]


def _integration_config(*, rectilinear_fit_enabled: bool) -> EdgeGraphConfig:
    return EdgeGraphConfig(
        source_native_proposals_enabled=False,
        shortcut_enabled=False,
        edge_phase_logit_weight=0.0,
        rectilinear_fit_enabled=rectilinear_fit_enabled,
    )


def test_rectilinear_integration_turns_noisy_dense_ring_into_sharp_polygon() -> None:
    height, width = 220, 300
    amplitude = 1.5
    frequency = 0.20
    noisy: list[tuple[int, int]] = []
    for x in range(35, 266):
        noisy.append((x, 30 + round(amplitude * np.sin(x * frequency))))
    for y in range(30, 191):
        noisy.append((265 + round(amplitude * np.sin(y * frequency)), y))
    for x in range(265, 34, -1):
        noisy.append((x, 190 + round(amplitude * np.sin(x * frequency))))
    for y in range(190, 29, -1):
        noisy.append((35 + round(amplitude * np.sin(y * frequency)), y))
    coarse = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(coarse, [np.asarray(noisy, dtype=np.int32)], 1.0)
    rgb = np.full((height, width, 3), 220, dtype=np.uint8)

    result = refine_boundary_with_edgegraph(
        rgb,
        coarse,
        _CenterSession(),
        config=_integration_config(rectilinear_fit_enabled=True),
    )

    diagnostics = result.diagnostics["rectilinear_fit"]
    assert diagnostics["result"] == "accepted"
    assert diagnostics["accepted_count"] == 1
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["rings"] == [
        {
            "accepted": True,
                "reason": "accepted",
                "run_count": 4,
                "support_p95_px": diagnostics["rings"][0]["support_p95_px"],
                "native_phase_alignment": {
                    "evaluated": False,
                    "accepted": False,
                    "reason": "target-color-unavailable",
                },
            }
        ]
    assert 0.0 < diagnostics["rings"][0]["support_p95_px"] < 2.0
    vertices = np.asarray(result.pixel_geometry.exterior.coords[:-1], dtype=np.float64)
    assert len(vertices) == 4
    edges = np.roll(vertices, -1, axis=0) - vertices
    unit = edges / np.linalg.norm(edges, axis=1, keepdims=True)
    assert np.max(np.abs(np.sum(unit * np.roll(unit, -1, axis=0), axis=1))) < 1e-5


def _assert_non_rectilinear_integration_falls_back(coarse: np.ndarray) -> None:
    rgb = np.full((*coarse.shape, 3), 220, dtype=np.uint8)
    enabled = refine_boundary_with_edgegraph(
        rgb,
        coarse,
        _CenterSession(),
        config=_integration_config(rectilinear_fit_enabled=True),
    )
    disabled = refine_boundary_with_edgegraph(
        rgb,
        coarse,
        _CenterSession(),
        config=_integration_config(rectilinear_fit_enabled=False),
    )

    diagnostics = enabled.diagnostics["rectilinear_fit"]
    assert diagnostics["result"] == "fallback"
    assert diagnostics["accepted_count"] == 0
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["rings"][0]["accepted"] is False
    assert diagnostics["rings"][0]["reason"] == "weak-axis-concentration"
    assert np.array_equal(enabled.mask, disabled.mask)
    assert enabled.pixel_geometry.equals_exact(disabled.pixel_geometry, tolerance=1e-8)


def test_angular_integration_uses_unchanged_generic_fallback() -> None:
    coarse = np.zeros((240, 300), dtype=np.float32)
    angular = np.asarray(
        [(30, 30), (200, 30), (260, 100), (185, 205), (45, 180)],
        dtype=np.int32,
    )
    cv2.fillPoly(coarse, [angular], 1.0)

    _assert_non_rectilinear_integration_falls_back(coarse)


def test_radial_integration_uses_unchanged_generic_fallback() -> None:
    coarse = np.zeros((240, 300), dtype=np.float32)
    cv2.circle(coarse, (150, 120), 80, 1.0, -1)

    _assert_non_rectilinear_integration_falls_back(coarse)


class _NineChannelInput:
    name = "edge_strip"
    shape = ["batch", 9, "contour_length", 49]


class _NineChannelCenterSession:
    def __init__(self) -> None:
        self.observed_shapes: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [_NineChannelInput()]

    def run(self, _outputs, feed):
        values = feed["edge_strip"]
        self.observed_shapes.append(tuple(values.shape))
        assert values.shape[1] == 9
        assert np.all((values[:, 7] >= 0.0) & (values[:, 7] <= 1.0))
        assert np.all((values[:, 8] == 0.0) | (values[:, 8] == 1.0))
        length, width = values.shape[2:]
        logits = np.full((1, length, width), -8.0, dtype=np.float32)
        logits[:, :, width // 2] = 8.0
        corners = np.full((1, length), -8.0, dtype=np.float32)
        reliability = np.full((1, length), 8.0, dtype=np.float32)
        return [logits, corners, reliability]


class _TenChannelInput:
    name = "edge_strip_vector_v3"
    shape = ["batch", 10, "contour_length", 49]


class _TenChannelCenterSession:
    def __init__(self) -> None:
        self.observed_shapes: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [_TenChannelInput()]

    def run(self, _outputs, feed):
        values = feed["edge_strip_vector_v3"]
        self.observed_shapes.append(tuple(values.shape))
        assert values.shape[1] == 10
        assert np.all((values[:, 6] >= 0.0) & (values[:, 6] <= 1.0))
        assert np.all((values[:, 7] == 0.0) | (values[:, 7] == 1.0))
        expected_offsets = np.linspace(-1.0, 1.0, values.shape[-1], dtype=np.float32)
        assert np.allclose(values[:, 8], expected_offsets[None, None, :])
        assert np.all((values[:, 9] == -1.0) | (values[:, 9] == 1.0))
        assert np.all(values[:, 9] == values[:, 9, :, :1])
        length, width = values.shape[2:]
        logits = np.full((1, length, width), -8.0, dtype=np.float32)
        logits[:, :, width // 2] = 8.0
        corners = np.full((1, length), -8.0, dtype=np.float32)
        reliability = np.full((1, length), 8.0, dtype=np.float32)
        return [logits, corners, reliability]


def test_edgegraph_onnx_contract_accepts_dynamic_contour_length() -> None:
    rgb = np.full((128, 160, 3), 220, dtype=np.uint8)
    coarse = np.zeros((128, 160), dtype=np.float32)
    coarse[16:112, 20:140] = 1.0

    result = refine_boundary_with_edgegraph(rgb, coarse, _CenterSession())

    assert result.pixel_geometry.is_valid
    assert result.diagnostics["refiner_enabled"] is True
    assert result.diagnostics["sample_count"] > 128
    assert result.diagnostics["topology_fallback_count"] == 0


def test_edgegraph_runtime_adapts_vector_v2_nine_channel_contract() -> None:
    rgb = np.full((96, 128, 3), 220, dtype=np.uint8)
    coarse = np.zeros((96, 128), dtype=np.float32)
    coarse[12:84, 16:112] = 1.0
    session = _NineChannelCenterSession()

    result = refine_boundary_with_edgegraph(
        rgb,
        coarse,
        session,
        hints={"target_rgb": (40, 120, 240), "seed_point": (64.0, 48.0)},
    )

    assert result.pixel_geometry.is_valid
    assert session.observed_shapes
    assert all(shape[1] == 9 for shape in session.observed_shapes)
    assert result.diagnostics["localization_fusion"] == "learned-vector-v2"
    assert result.diagnostics["refiner_input_channels"] == 9


def test_edgegraph_runtime_adapts_vector_v3_ten_channel_contract() -> None:
    rgb = np.full((96, 128, 3), 220, dtype=np.uint8)
    coarse = np.zeros((96, 128), dtype=np.float32)
    coarse[12:84, 16:112] = 1.0
    session = _TenChannelCenterSession()

    result = refine_boundary_with_edgegraph(
        rgb,
        coarse,
        session,
        hints={"target_rgb": (40, 120, 240), "seed_point": (64.0, 48.0)},
    )

    assert result.pixel_geometry.is_valid
    assert session.observed_shapes
    assert all(shape[1] == 10 for shape in session.observed_shapes)
    assert result.diagnostics["localization_fusion"] == "learned-vector-v3"
    assert result.diagnostics["refiner_input_channels"] == 10
