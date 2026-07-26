from __future__ import annotations

import numpy as np

import map_boundary_builder.edgegraph_proposal as edgegraph_proposal
from map_boundary_builder.edgegraph_proposal import (
    SourceNativeProposalConfig,
    _StructuralProposal,
    _deep_thin_expansion_energy_exception_evidence,
    recover_source_native_rectilinear_proposals,
)
from map_boundary_builder.evaluation import boundary_distance_summary_px, topology_signature


BACKGROUND = np.asarray((28, 34, 40), dtype=np.uint8)
TARGET = np.asarray((188, 76, 132), dtype=np.uint8)


def _scene(mask: np.ndarray) -> np.ndarray:
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    rgb[:] = BACKGROUND
    rgb[mask] = TARGET
    return rgb


def _selector(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 0.93, 0.07).astype(np.float32)


def test_structural_proposal_classifier_covers_mixed_authored_geometry() -> None:
    # Long supported rectilinear branches can belong to an otherwise curved or
    # road-following service area.  Downstream native-support/topology gates,
    # rather than an overly strict global shape classifier, own acceptance.
    assert SourceNativeProposalConfig().minimum_rectilinear_fraction == 0.50


def _structural_expansion(
    *,
    confidence: float = 0.86,
    terminal_contrast: float = 0.40,
    terminal_support: float = 1.0,
    depth: float = 58.0,
    viewport: bool = False,
) -> _StructuralProposal:
    return _StructuralProposal(
        kind="expansion",
        change_mask=np.ones((1, 1), dtype=bool),
        core_area=120,
        maximum_depth=depth,
        support_fraction=0.18,
        mean_confidence=confidence,
        terminal_contrast=terminal_contrast,
        terminal_support_fraction=terminal_support,
        viewport_terminal=viewport,
    )


def test_deep_thin_expansion_energy_exception_accepts_scale_normalized_evidence() -> None:
    proposal = _structural_expansion()
    evidence = _deep_thin_expansion_energy_exception_evidence(
        [proposal],
        [proposal],
        selector_area_px=180_000,
        changed_pixel_count=260,
        rectilinear_fraction=0.95,
        config=SourceNativeProposalConfig(),
    )

    assert evidence["eligible"] is True
    assert evidence["reason"] == "strong-native-deep-thin-expansion"
    assert float(evidence["changed_selector_area_fraction"]) < 0.0025
    assert float(evidence["normalized_depth"]) >= 0.10
    assert float(evidence["slenderness_ratio"]) >= 6.0


def test_deep_thin_expansion_energy_exception_rejects_each_ambiguous_evidence_axis() -> None:
    proposal = _structural_expansion()
    config = SourceNativeProposalConfig()
    cases = (
        (
            "ambiguous-candidate-count",
            [proposal, proposal, proposal],
            [proposal, proposal, proposal],
            260,
            0.95,
        ),
        (
            "viewport-expansion",
            [_structural_expansion(viewport=True)],
            [_structural_expansion(viewport=True)],
            260,
            0.95,
        ),
        (
            "insufficient-native-confidence",
            [_structural_expansion(confidence=0.56)],
            [_structural_expansion(confidence=0.56)],
            260,
            0.95,
        ),
        (
            "insufficient-terminal-contrast",
            [_structural_expansion(terminal_contrast=0.20)],
            [_structural_expansion(terminal_contrast=0.20)],
            260,
            0.95,
        ),
        (
            "insufficient-terminal-support",
            [_structural_expansion(terminal_support=0.80)],
            [_structural_expansion(terminal_support=0.80)],
            260,
            0.95,
        ),
        (
            "insufficient-rectilinearity",
            [proposal],
            [proposal],
            260,
            0.80,
        ),
        (
            "insufficient-normalized-depth",
            [_structural_expansion(depth=20.0)],
            [_structural_expansion(depth=20.0)],
            260,
            0.95,
        ),
        (
            "insufficient-slenderness",
            [_structural_expansion(depth=45.0)],
            [_structural_expansion(depth=45.0)],
            400,
            0.95,
        ),
        (
            "excessive-selector-area-fraction",
            [_structural_expansion(depth=80.0)],
            [_structural_expansion(depth=80.0)],
            900,
            0.95,
        ),
    )

    for expected_reason, candidates, accepted, changed, rectilinear in cases:
        evidence = _deep_thin_expansion_energy_exception_evidence(
            candidates,
            accepted,
            selector_area_px=180_000,
            changed_pixel_count=changed,
            rectilinear_fraction=rectilinear,
            config=config,
        )
        assert evidence["eligible"] is False
        assert evidence["reason"] == expected_reason


def test_deep_thin_expansion_energy_exception_controls_only_global_veto(
    monkeypatch,
) -> None:
    truth = np.zeros((512, 512), dtype=bool)
    truth[40:430, 40:470] = True
    truth[430:488, 250:254] = True
    coarse = truth.copy()
    coarse[430:488, 250:254] = False

    def forced_boundary_energy(_edge_magnitude, candidate: np.ndarray) -> float:
        return 0.80 if np.array_equal(candidate, coarse) else 0.20

    monkeypatch.setattr(
        edgegraph_proposal,
        "_boundary_edge_energy",
        forced_boundary_energy,
    )
    shared = {
        "rgb": _scene(truth),
        "coarse_probabilities": _selector(coarse),
        "coarse_mask": coarse,
        "target_rgb": tuple(int(value) for value in TARGET),
        "seed_point": (100.0, 100.0),
    }

    accepted = recover_source_native_rectilinear_proposals(**shared)
    accepted_exception = accepted.diagnostics[
        "deep_thin_expansion_energy_exception"
    ]
    assert accepted_exception["global_energy_veto_triggered"] is True
    assert accepted_exception["eligible"] is True
    assert accepted_exception["applied"] is True
    assert accepted.diagnostics["accepted_expansion_count"] == 1
    assert accepted.diagnostics["rejected_energy_count"] == 0
    assert accepted.mask[480, 252]

    rejected = recover_source_native_rectilinear_proposals(
        **shared,
        config=SourceNativeProposalConfig(
            deep_thin_expansion_energy_exception_enabled=False,
        ),
    )
    rejected_exception = rejected.diagnostics[
        "deep_thin_expansion_energy_exception"
    ]
    assert rejected_exception["global_energy_veto_triggered"] is True
    assert rejected_exception["eligible"] is False
    assert rejected_exception["applied"] is False
    assert rejected_exception["reason"] == "disabled"
    assert rejected.diagnostics["accepted_count"] == 0
    assert rejected.diagnostics["rejected_energy_count"] == 1
    assert np.array_equal(rejected.mask, coarse)


def test_source_native_proposal_recovers_long_thin_expansion() -> None:
    truth = np.zeros((176, 192), dtype=bool)
    truth[24:132, 36:156] = True
    truth[132:168, 82:90] = True
    coarse = truth.copy()
    coarse[132:168, 82:90] = False
    probabilities = _selector(coarse)
    before = probabilities.copy()

    result = recover_source_native_rectilinear_proposals(
        _scene(truth),
        probabilities,
        coarse_mask=coarse,
        target_rgb=tuple(int(value) for value in TARGET),
        seed_point=(72.0, 72.0),
    )

    assert np.array_equal(probabilities, before), "the input selector must remain immutable"
    assert result.diagnostics["accepted_expansion_count"] >= 1
    assert result.diagnostics["accepted_contraction_count"] == 0
    assert result.diagnostics["topology_preserved"] is True
    assert result.mask[160, 86]
    assert int(np.count_nonzero(result.mask & ~coarse)) >= 250
    assert topology_signature(result.mask) == topology_signature(coarse)
    before_tail = boundary_distance_summary_px(coarse, truth)["max_px"]
    after_tail = boundary_distance_summary_px(result.mask, truth)["max_px"]
    assert before_tail >= 30.0
    assert after_tail <= 3.0
    assert float(result.probabilities[160, 86]) > 0.45


def test_source_native_proposal_recovers_supported_viewport_continuation() -> None:
    truth = np.zeros((256, 320), dtype=bool)
    truth[24:232, 100:300] = True
    truth[112:132, 0:100] = True
    coarse = truth.copy()
    coarse[112:132, 0:100] = False

    result = recover_source_native_rectilinear_proposals(
        _scene(truth),
        _selector(coarse),
        coarse_mask=coarse,
        target_rgb=tuple(int(value) for value in TARGET),
        seed_point=(180.0, 100.0),
    )

    assert result.diagnostics["accepted_viewport_expansion_count"] == 1
    assert result.diagnostics["accepted_expansion_count"] == 1
    assert result.diagnostics["topology_preserved"] is True
    assert result.mask[120, 5]
    assert topology_signature(result.mask) == topology_signature(coarse)
    before_tail = boundary_distance_summary_px(coarse, truth)["max_px"]
    after_tail = boundary_distance_summary_px(result.mask, truth)["max_px"]
    assert before_tail >= 95.0
    assert after_tail <= 3.0


def test_source_native_proposal_recovers_deep_narrow_notch() -> None:
    coarse = np.zeros((176, 192), dtype=bool)
    coarse[20:156, 28:164] = True
    truth = coarse.copy()
    truth[20:112, 86:94] = False

    result = recover_source_native_rectilinear_proposals(
        _scene(truth),
        _selector(coarse),
        coarse_mask=coarse,
        target_rgb=tuple(int(value) for value in TARGET),
        seed_point=(58.0, 132.0),
    )

    assert result.diagnostics["accepted_contraction_count"] >= 1
    assert result.diagnostics["accepted_expansion_count"] == 0
    assert result.diagnostics["topology_preserved"] is True
    assert not result.mask[72, 90]
    assert topology_signature(result.mask) == topology_signature(coarse)
    before_tail = boundary_distance_summary_px(coarse, truth)["max_px"]
    after_tail = boundary_distance_summary_px(result.mask, truth)["max_px"]
    assert before_tail >= 60.0
    assert after_tail <= 3.0
    assert float(result.probabilities[72, 90]) < 0.45


def test_source_native_proposal_rejects_enclosed_map_line_as_a_hole() -> None:
    coarse = np.zeros((176, 192), dtype=bool)
    coarse[20:156, 28:164] = True
    rgb = _scene(coarse)
    rgb[76:82, 52:142] = BACKGROUND

    result = recover_source_native_rectilinear_proposals(
        rgb,
        _selector(coarse),
        coarse_mask=coarse,
        target_rgb=tuple(int(value) for value in TARGET),
        seed_point=(58.0, 132.0),
    )

    assert np.array_equal(result.mask, coarse)
    assert result.diagnostics["accepted_count"] == 0
    assert result.diagnostics["topology_preserved"] is True


def test_source_native_proposal_rejects_road_continuation_without_terminal_cap() -> None:
    coarse = np.zeros((176, 192), dtype=bool)
    coarse[24:152, 34:154] = True
    rgb = _scene(coarse)
    # A target-like basemap road happens to cross the selected boundary.  It
    # has parallel side edges but no terminal cap, so extending it would create
    # exactly the long false spur this stage is designed to avoid.
    rgb[78:84, 0:192] = TARGET

    result = recover_source_native_rectilinear_proposals(
        rgb,
        _selector(coarse),
        coarse_mask=coarse,
        target_rgb=tuple(int(value) for value in TARGET),
        seed_point=(70.0, 120.0),
    )

    assert np.array_equal(result.mask, coarse)
    assert result.diagnostics["accepted_expansion_count"] == 0
    assert result.diagnostics["topology_preserved"] is True


def test_source_native_proposal_rejects_low_energy_boundary() -> None:
    coarse = np.zeros((176, 192), dtype=bool)
    coarse[24:72, 72:120] = True
    truth = coarse.copy()
    truth[72:142, 92:100] = True
    rgb = _scene(coarse)
    # The target-colored branch is embedded in a nearly target-colored band.
    # Its appearance and terminal cap are sufficient for a structural
    # proposal, but its side boundary has materially less native edge energy
    # than the crisp coarse rectangle.
    near_target = np.clip(TARGET.astype(np.int16) - 20, 0, 255).astype(np.uint8)
    rgb[72:146, 84:108] = near_target
    rgb[truth] = TARGET
    shared = {
        "maximum_proposal_area_fraction": 0.80,
        "maximum_total_area_change_fraction": 0.80,
        "minimum_terminal_contrast": 0.001,
        "minimum_terminal_support_fraction": 0.10,
    }

    permissive = recover_source_native_rectilinear_proposals(
        rgb,
        _selector(coarse),
        coarse_mask=coarse,
        target_rgb=tuple(int(value) for value in TARGET),
        seed_point=(90.0, 50.0),
        config=SourceNativeProposalConfig(
            **shared,
            minimum_boundary_energy_ratio=0.01,
            maximum_boundary_energy_drop=1.0,
        ),
    )
    guarded = recover_source_native_rectilinear_proposals(
        rgb,
        _selector(coarse),
        coarse_mask=coarse,
        target_rgb=tuple(int(value) for value in TARGET),
        seed_point=(90.0, 50.0),
        config=SourceNativeProposalConfig(
            **shared,
            minimum_boundary_energy_ratio=0.99,
            maximum_boundary_energy_drop=0.001,
        ),
    )

    assert permissive.diagnostics["accepted_count"] == 1
    assert permissive.diagnostics["shortcut_suppression_recommended"] is True
    assert float(permissive.diagnostics["minimum_proposal_confidence"]) > 0.0
    assert float(permissive.diagnostics["minimum_terminal_contrast"]) > 0.0
    assert float(permissive.diagnostics["minimum_terminal_support_fraction"]) > 0.0
    assert np.array_equal(guarded.mask, coarse)
    assert guarded.diagnostics["accepted_count"] == 0
    assert guarded.diagnostics["rejected_energy_count"] == 1
    assert float(guarded.diagnostics["evaluated_boundary_energy_ratio"]) < 0.99
    assert guarded.diagnostics["shortcut_suppression_recommended"] is False


def test_source_native_proposal_fails_closed_without_training_support() -> None:
    coarse = np.zeros((9, 9), dtype=bool)
    coarse[3:6, 3:6] = True
    result = recover_source_native_rectilinear_proposals(
        _scene(coarse),
        _selector(coarse),
        coarse_mask=coarse,
        target_rgb=tuple(int(value) for value in TARGET),
        seed_point=(4.0, 4.0),
        config=SourceNativeProposalConfig(training_margin_px=2),
    )

    assert np.array_equal(result.mask, coarse)
    assert result.diagnostics["reason"] in {
        "insufficient-training-support",
        "non-rectilinear-selector",
    }


def test_source_native_proposal_validates_source_shapes() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    probabilities = np.zeros((31, 32), dtype=np.float32)
    try:
        recover_source_native_rectilinear_proposals(
            rgb,
            probabilities,
            target_rgb=(1, 2, 3),
        )
    except ValueError as exc:
        assert "must match" in str(exc)
    else:
        raise AssertionError("mismatched selector dimensions must fail")
