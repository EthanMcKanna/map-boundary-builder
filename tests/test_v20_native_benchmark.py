from __future__ import annotations

from pathlib import Path

from tools import benchmark_v20_native


def _active_score(
    slug: str,
    *,
    iou: float,
    distance_p95: float,
    distance_p99: float,
    duration: float,
) -> dict[str, object]:
    return {
        "slug": slug,
        "status": "active",
        "duration_s": duration,
        "metrics": {
            "iou": iou,
            "boundary_f1_1px": 0.95,
            "boundary_distance_p95_px": distance_p95,
            "boundary_distance_p99_px": distance_p99,
            "boundary_distance_max_px": distance_p99 + 0.5,
            "corner_f1": 0.94,
            "topology_matches": True,
            "raster_topology_matches": True,
            "geometry_valid": True,
            "geometry_empty": False,
            "reference_geometry_valid": True,
        },
    }


def test_native_raster_conditions_cover_resolution_and_codec_pressure() -> None:
    conditions = benchmark_v20_native.raster_conditions(1176, 947)

    assert len(conditions) == 12
    assert len({condition.slug for condition in conditions}) == len(conditions)
    assert {condition.encoding for condition in conditions} == {"png", "jpeg", "webp"}
    assert {condition.width for condition in conditions} == {1176, 960, 768, 640}
    assert all(condition.width > 0 and condition.height > 0 for condition in conditions)


def test_native_report_summary_is_fail_closed_and_uses_native_distance_quantiles() -> None:
    scores = [
        _active_score(
            f"fixture-{index}",
            iou=0.98 + (index * 0.001),
            distance_p95=0.4 + (index * 0.1),
            distance_p99=0.7 + (index * 0.1),
            duration=0.1 + (index * 0.01),
        )
        for index in range(9)
    ]
    scores.append(
        {
            "slug": "broken",
            "status": "failed",
            "duration_s": 0.2,
            "error": "fixture failed",
        }
    )

    summary = benchmark_v20_native.summarize_scores(scores)

    assert summary["fixture_count"] == 10
    assert summary["scored_fixtures"] == 9
    assert summary["failed_fixtures"] == 1
    assert summary["complete_metric_rows"] is True
    assert summary["native_pixel_references"] is True
    assert summary["topology_error_count"] == 0
    assert summary["raster_topology_error_count"] == 0
    assert summary["geometry_invalid_count"] == 0
    assert summary["p95_boundary_distance_px"] > 1.1
    assert summary["p99_boundary_distance_px"] > 1.4


def test_native_report_summary_flags_missing_metrics() -> None:
    score = _active_score(
        "incomplete",
        iou=0.98,
        distance_p95=0.5,
        distance_p99=0.8,
        duration=0.2,
    )
    del score["metrics"]["corner_f1"]  # type: ignore[index]

    summary = benchmark_v20_native.summarize_scores([score])

    assert summary["complete_metric_rows"] is False
    assert summary["mean_corner_f1"] is None


def test_default_baseline_path_preserves_json_suffix() -> None:
    output = Path("out/v20-native.json")

    assert benchmark_v20_native.default_baseline_path(output) == Path(
        "out/v20-native-v12-baseline.json"
    )
