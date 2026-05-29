import json
from pathlib import Path
from types import SimpleNamespace

from map_boundary_builder.benchmark import (
    BenchmarkFixture,
    compare_report_regressions,
    load_fixture_config,
    run_benchmark,
    score_full_fixture_in_process,
)


KNOWN_REFERENCE_MISMATCH_FIXTURES = {
    "bay-area-tesla",
    "bay-area-waymo",
    "bay-area-zoox",
    "houston-tesla",
    "houston-waymo",
    "las-vegas-zoox",
    "miami-waymo",
}


def test_known_stale_reference_fixtures_are_reference_mismatches() -> None:
    config = load_fixture_config(Path("benchmarks/service-area-fixtures.json"))
    fixtures = config["fixtures"]

    assert KNOWN_REFERENCE_MISMATCH_FIXTURES <= set(fixtures)
    for slug in KNOWN_REFERENCE_MISMATCH_FIXTURES:
        assert fixtures[slug]["status"] == "reference_mismatch"
        assert "changed" in fixtures[slug]["note"] or "different shapes" in fixtures[slug]["note"]

def test_reference_mismatch_fixtures_are_reported_but_not_scored(tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    config_path = tmp_path / "fixtures.json"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "houston-waymo.json").write_text("{}\n")
    (image_dir / "Waymo Houston.png").write_bytes(b"not an image")
    config_path.write_text(
        json.dumps(
            {
                "fixtures": {
                    "houston-waymo": {
                        "status": "reference_mismatch",
                        "note": "changed live service area",
                    }
                }
            }
        )
        + "\n"
    )

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="extraction",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=config_path,
    )

    assert report["summary"]["passed"] is False
    assert report["summary"]["scored_fixtures"] == 0
    assert report["summary"]["skipped_fixtures"] == 1
    assert report["summary"]["skipped_by_status"] == {"reference_mismatch": 1}
    assert report["scores"] == [
        {
            "slug": "houston-waymo",
            "image": "Waymo Houston.png",
            "mode": "extraction",
            "passed": False,
            "iou": None,
            "area_ratio": None,
            "centroid_distance_m": None,
            "vertices": None,
            "style": None,
            "duration_s": None,
            "georeference_source": None,
            "combined_confidence": None,
            "catalog_slug": None,
            "stage_elapsed_s": None,
            "error": None,
            "status": "reference_mismatch",
            "note": "changed live service area",
        }
    ]


def test_report_regression_check_flags_fixture_iou_drop() -> None:
    baseline = {
        "summary": {"average_iou": 0.95},
        "scores": [
            {"slug": "orlando-waymo", "status": "active", "iou": 0.931476},
            {"slug": "miami-waymo", "status": "reference_mismatch", "iou": None},
        ],
    }
    candidate = {
        "summary": {"average_iou": 0.94},
        "scores": [
            {"slug": "orlando-waymo", "status": "active", "iou": 0.781303},
            {"slug": "miami-waymo", "status": "reference_mismatch", "iou": None},
        ],
    }

    check = compare_report_regressions(candidate, baseline)

    assert check["passed"] is False
    assert check["compared_fixtures"] == 1
    assert check["issues"] == [
        {
            "slug": "orlando-waymo",
            "kind": "iou_drop",
            "baseline_iou": 0.931476,
            "candidate_iou": 0.781303,
            "drop": 0.150173,
        },
        {
            "kind": "average_iou_drop",
            "baseline_average_iou": 0.95,
            "candidate_average_iou": 0.94,
            "drop": 0.01,
        },
    ]


def test_report_regression_check_allows_configured_tolerance() -> None:
    baseline = {
        "summary": {"average_iou": 0.95},
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": 0.98332}],
    }
    candidate = {
        "summary": {"average_iou": 0.949},
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": 0.982806}],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_iou_drop=0.001,
        max_mean_iou_drop=0.002,
    )

    assert check["passed"] is True
    assert check["issues"] == []


def test_report_regression_check_can_flag_duration_increase() -> None:
    baseline = {
        "summary": {"average_iou": 0.95, "total_duration_s": 4.0},
        "scores": [
            {"slug": "phoenix-waymo", "status": "active", "iou": 0.98332, "duration_s": 1.0},
            {"slug": "nashville-waymo", "status": "active", "iou": 0.986282, "duration_s": 0.8},
        ],
    }
    candidate = {
        "summary": {"average_iou": 0.95, "total_duration_s": 4.8},
        "scores": [
            {"slug": "phoenix-waymo", "status": "active", "iou": 0.98332, "duration_s": 1.35},
            {"slug": "nashville-waymo", "status": "active", "iou": 0.986282, "duration_s": 0.82},
        ],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_duration_increase_ratio=0.1,
        max_total_duration_increase_ratio=0.1,
    )

    assert check["passed"] is False
    assert check["issues"] == [
        {
            "slug": "phoenix-waymo",
            "kind": "duration_increase",
            "baseline_duration_s": 1.0,
            "candidate_duration_s": 1.35,
            "increase_s": 0.35,
            "increase_ratio": 0.35,
        },
        {
            "kind": "total_duration_increase",
            "baseline_total_duration_s": 4.0,
            "candidate_total_duration_s": 4.8,
            "increase_s": 0.8,
            "increase_ratio": 0.2,
        },
    ]


def test_report_regression_check_duration_ratio_can_ignore_small_absolute_noise() -> None:
    baseline = {
        "summary": {"average_iou": 0.95, "total_duration_s": 4.0},
        "scores": [{"slug": "austin-tesla", "status": "active", "iou": 0.973925, "duration_s": 0.1}],
    }
    candidate = {
        "summary": {"average_iou": 0.95, "total_duration_s": 4.2},
        "scores": [{"slug": "austin-tesla", "status": "active", "iou": 0.973925, "duration_s": 0.19}],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_duration_increase_ratio=0.25,
        max_duration_increase_s=0.1,
        max_total_duration_increase_ratio=0.01,
        max_total_duration_increase_s=0.25,
    )

    assert check["passed"] is True
    assert check["issues"] == []


def test_in_process_full_fixture_scores_without_debug_artifacts(tmp_path: Path, monkeypatch) -> None:
    polygon = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-112.0, 33.0],
                            [-111.0, 33.0],
                            [-111.0, 34.0],
                            [-112.0, 33.0],
                        ]
                    ],
                },
            }
        ],
    }
    image_path = tmp_path / "Waymo Phoenix.png"
    reference_path = tmp_path / "phoenix-waymo.json"
    image_path.write_bytes(b"unused by patched runner")
    reference_path.write_text(json.dumps(polygon) + "\n")
    fixture = BenchmarkFixture(
        slug="phoenix-waymo",
        provider="waymo",
        area="Phoenix",
        image_path=image_path,
        reference_path=reference_path,
    )
    calls = []

    def fake_build_boundary(image, city, output_path, *, debug_dir, options, progress):
        calls.append(
            {
                "image": image,
                "city": city,
                "debug_dir": debug_dir,
                "allow_catalog": options.allow_catalog,
                "write_mask_artifact": options.write_mask_artifact,
            }
        )
        progress(
            {"stage": "inspect", "message": "Reading image metadata", "percent": 5, "status": "running"}
        )
        progress(
            {"stage": "complete", "message": "Boundary export ready", "percent": 100, "status": "complete"}
        )
        return SimpleNamespace(
            geojson=polygon,
            summary={
                "style": "bright-blue",
                "georeference_source": "ocr-georeference:nominatim-label-fit",
                "combined_confidence": 0.96,
                "catalog_slug": None,
            },
        )

    monkeypatch.setattr("map_boundary_builder.runner.build_boundary", fake_build_boundary)

    score = score_full_fixture_in_process(
        fixture,
        output_path=tmp_path / "out" / "boundary.geojson",
        debug_dir=None,
        min_iou=0.99,
        city_overrides=True,
        no_catalog=True,
        debug_artifacts=False,
    )

    assert score.passed is True
    assert score.iou == 1.0
    assert score.georeference_source == "ocr-georeference:nominatim-label-fit"
    assert calls == [
        {
            "image": image_path,
            "city": "Phoenix",
            "debug_dir": None,
            "allow_catalog": False,
            "write_mask_artifact": False,
        }
    ]
