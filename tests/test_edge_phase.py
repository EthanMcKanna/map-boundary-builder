from __future__ import annotations

import numpy as np
import pytest

from map_boundary_builder.edge_phase import (
    EdgePhaseConfig,
    _transition_candidates,
    centered_edge_phase_logits,
    estimate_edge_phase,
)


OFFSETS = np.arange(-12.0, 12.01, 0.5, dtype=np.float32)
OUTSIDE = np.asarray([0.80, 0.84, 0.78], dtype=np.float32)
FILL = np.asarray([0.18, 0.52, 0.91], dtype=np.float32)
STROKE = np.asarray([0.035, 0.08, 0.22], dtype=np.float32)


def _smooth_step(values: np.ndarray, center: float, softness: float = 0.13) -> np.ndarray:
    exponent = np.clip(-(values - center) / softness, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(exponent))


def _profile(
    *,
    center: float,
    stroke_width: float,
    distractor_center: float | None = None,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    values = OFFSETS
    if stroke_width <= 0.0:
        inside = _smooth_step(values, center)
        rgb = OUTSIDE[None, :] * (1.0 - inside[:, None]) + FILL[None, :] * inside[:, None]
    else:
        outer = _smooth_step(values, center - stroke_width / 2.0)
        inner = _smooth_step(values, center + stroke_width / 2.0)
        rgb = (
            OUTSIDE[None, :] * (1.0 - outer[:, None])
            + STROKE[None, :] * (outer - inner)[:, None]
            + FILL[None, :] * inner[:, None]
        )
    if distractor_center is not None:
        road_width = 1.25
        enter = _smooth_step(values, distractor_center - road_width / 2.0, softness=0.10)
        leave = _smooth_step(values, distractor_center + road_width / 2.0, softness=0.10)
        road_alpha = np.clip(enter - leave, 0.0, 1.0)
        road = np.asarray([0.99, 0.98, 0.96], dtype=np.float32)
        rgb = rgb * (1.0 - road_alpha[:, None]) + road[None, :] * road_alpha[:, None]

    luminance = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gradient = np.gradient(luminance, OFFSETS).astype(np.float32)
    if reverse:
        return rgb[::-1], luminance[::-1], -gradient[::-1], -1
    return rgb, luminance, gradient, 1


@pytest.mark.parametrize("stroke_width", [0.0, 1.0, 2.0, 3.0])
def test_centered_stroke_width_does_not_move_vector_phase(stroke_width: float) -> None:
    expected_center = 0.25
    rgb, luminance, gradient, sign = _profile(
        center=expected_center,
        stroke_width=stroke_width,
    )

    result = estimate_edge_phase(
        rgb,
        luminance,
        gradient,
        OFFSETS,
        target_rgb=FILL,
        inside_sign=sign,
    )

    assert abs(float(result.centers_px[0]) - expected_center) <= 0.5
    decoded = float(OFFSETS[int(np.argmax(result.logits[0]))])
    assert abs(decoded - expected_center) <= 0.5
    assert float(result.reliability[0]) >= 0.35
    if stroke_width >= 2.0:
        assert bool(result.paired[0])
        assert abs(float(result.stroke_widths_px[0]) - stroke_width) <= 0.75


def test_target_semantics_reject_stronger_distractor_line_without_moving_phase() -> None:
    expected_center = 2.25
    distractor_center = -5.0
    rgb, luminance, gradient, sign = _profile(
        center=expected_center,
        stroke_width=2.0,
        distractor_center=distractor_center,
    )

    logits = centered_edge_phase_logits(
        rgb,
        luminance,
        gradient,
        OFFSETS,
        target_rgb=FILL * 255.0,
        inside_sign=sign,
    )

    decoded = float(OFFSETS[int(np.argmax(logits[0]))])
    distractor_index = int(np.argmin(np.abs(OFFSETS - distractor_center)))
    expected_index = int(np.argmin(np.abs(OFFSETS - expected_center)))
    assert abs(decoded - expected_center) <= 0.5
    assert float(logits[0, expected_index]) > float(logits[0, distractor_index]) + 1.0


def test_fill_boundary_and_nearby_interior_line_are_not_false_stroke_pair() -> None:
    expected_center = -0.25
    rgb, luminance, gradient, sign = _profile(
        center=expected_center,
        stroke_width=0.0,
        distractor_center=3.5,
    )

    result = estimate_edge_phase(
        rgb,
        luminance,
        gradient,
        OFFSETS,
        target_rgb=FILL,
        inside_sign=sign,
    )

    assert not bool(result.paired[0])
    assert abs(float(result.centers_px[0]) - expected_center) <= 0.5


def test_low_contrast_inner_stroke_transition_still_centers_pair() -> None:
    expected_center = 0.25
    stroke_width = 3.0
    values = OFFSETS
    outer = _smooth_step(values, expected_center - stroke_width / 2.0)
    inner = _smooth_step(values, expected_center + stroke_width / 2.0)
    low_contrast_stroke = FILL + np.asarray([0.018, -0.022, 0.015], dtype=np.float32)
    rgb = (
        OUTSIDE[None, :] * (1.0 - outer[:, None])
        + low_contrast_stroke[None, :] * (outer - inner)[:, None]
        + FILL[None, :] * inner[:, None]
    )
    luminance = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gradient = np.gradient(luminance, OFFSETS).astype(np.float32)

    result = estimate_edge_phase(
        rgb,
        luminance,
        gradient,
        OFFSETS,
        target_rgb=FILL,
        inside_sign=1,
    )

    assert bool(result.paired[0])
    assert abs(float(result.centers_px[0]) - expected_center) <= 0.5
    assert abs(float(result.stroke_widths_px[0]) - stroke_width) <= 0.75


def test_contour_consensus_prefers_one_persistent_stroke_width() -> None:
    profiles = [
        _profile(center=-0.3 + index * 0.02, stroke_width=2.0)
        for index in range(32)
    ]
    rgb = np.stack([item[0] for item in profiles])
    luminance = np.stack([item[1] for item in profiles])
    gradient = np.stack([item[2] for item in profiles])

    result = estimate_edge_phase(
        rgb,
        luminance,
        gradient,
        OFFSETS,
        target_rgb=FILL,
        inside_sign=np.ones(len(profiles), dtype=np.int8),
    )

    assert float(np.mean(result.paired)) >= 0.95
    assert float(np.quantile(np.abs(result.stroke_widths_px - 2.0), 0.95)) <= 0.75


def test_target_color_only_selects_fixed_observed_phase_candidates() -> None:
    rgb, luminance, gradient, sign = _profile(
        center=0.25,
        stroke_width=2.0,
        distractor_center=-5.0,
    )
    config = EdgePhaseConfig()
    transitions = _transition_candidates(
        rgb,
        luminance,
        gradient,
        OFFSETS,
        config=config,
    )
    fixed_phases = [transition.position for transition in transitions]
    fixed_phases.extend(
        0.5 * (left.position + right.position)
        for index, left in enumerate(transitions)
        for right in transitions[index + 1 :]
        if right.position - left.position <= config.maximum_stroke_width_px
    )

    for target in (FILL, np.clip(FILL + 0.03, 0.0, 1.0), OUTSIDE):
        result = estimate_edge_phase(
            rgb,
            luminance,
            gradient,
            OFFSETS,
            target_rgb=target,
            inside_sign=sign,
        )
        assert min(abs(float(result.centers_px[0]) - phase) for phase in fixed_phases) <= 1e-5


def test_negative_inside_orientation_preserves_native_phase() -> None:
    expected_center = -1.25
    rgb, luminance, gradient, sign = _profile(
        center=-expected_center,
        stroke_width=3.0,
        reverse=True,
    )

    result = estimate_edge_phase(
        rgb,
        luminance,
        gradient,
        OFFSETS,
        target_rgb=FILL,
        inside_sign=sign,
    )

    assert abs(float(result.centers_px[0]) - expected_center) <= 0.5


def test_target_evidence_corrects_decisively_reversed_coarse_orientation() -> None:
    expected_center = 1.25
    rgb, luminance, gradient, _sign = _profile(
        center=expected_center,
        stroke_width=2.0,
    )

    result = estimate_edge_phase(
        rgb,
        luminance,
        gradient,
        OFFSETS,
        target_rgb=FILL,
        inside_sign=-1,
    )

    assert abs(float(result.centers_px[0]) - expected_center) <= 0.5


def test_input_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="luminance"):
        centered_edge_phase_logits(
            np.zeros((2, len(OFFSETS), 3), dtype=np.float32),
            np.zeros((1, len(OFFSETS)), dtype=np.float32),
            np.zeros((2, len(OFFSETS)), dtype=np.float32),
            OFFSETS,
            target_rgb=FILL,
            inside_sign=1,
        )
