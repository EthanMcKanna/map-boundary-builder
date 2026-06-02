import json
import os
from pathlib import Path
from types import SimpleNamespace

from shapely.geometry import Polygon

import map_boundary_builder.benchmark as benchmark_module
from map_boundary_builder.benchmark import (
    BenchmarkFixture,
    BenchmarkScore,
    check_report_source_requirements,
    check_report_latency_budgets,
    compare_report_regressions,
    discover_fixtures,
    load_fixture_config,
    parse_georeference_source_requirements,
    parse_ocr_engine_duration_budgets,
    parse_ocr_engine_count_budgets,
    parse_stage_duration_budgets,
    parse_image_name,
    run_benchmark,
    score_full_fixture_in_process,
    summarize_georeference_sources,
    summarize_ocr_label_events,
    summarize_stage_max_rows,
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

KNOWN_CHANGED_REFERENCE_MISMATCH_FIXTURES = {
    "bay-area-tesla",
    "bay-area-waymo",
    "bay-area-zoox",
    "houston-tesla",
    "houston-waymo",
    "miami-waymo",
}


def test_known_stale_reference_fixtures_are_reference_mismatches() -> None:
    config = load_fixture_config(Path("benchmarks/service-area-fixtures.json"))
    fixtures = config["fixtures"]
    changed_areas = config["changed_areas"]

    assert {"bay-area", "houston", "miami"} <= set(changed_areas)
    for area_slug in ("bay-area", "houston", "miami"):
        assert changed_areas[area_slug]["status"] == "reference_mismatch"
        assert "changed" in changed_areas[area_slug]["note"]
    assert KNOWN_REFERENCE_MISMATCH_FIXTURES <= set(fixtures)
    for slug in KNOWN_REFERENCE_MISMATCH_FIXTURES:
        assert fixtures[slug]["status"] == "reference_mismatch"
        assert "changed" in fixtures[slug]["note"] or "different shapes" in fixtures[slug]["note"]


def test_parse_image_name_supports_avride_provider() -> None:
    provider, area_slug, area_name = parse_image_name(Path("Avride Dallas.png"))

    assert provider == "avride"
    assert area_slug == "dallas"
    assert area_name == "Dallas"


def test_parse_stage_duration_budgets_accepts_repeated_and_comma_values() -> None:
    budgets = parse_stage_duration_budgets(["ocr=4.0, extract=1.5", "georeference=0.75"])

    assert budgets == {"extract": 1.5, "georeference": 0.75, "ocr": 4.0}


def test_parse_stage_duration_budgets_rejects_missing_separator() -> None:
    try:
        parse_stage_duration_budgets(["ocr:4.0"])
    except ValueError as exc:
        assert "STAGE=SECONDS" in str(exc)
    else:
        raise AssertionError("Expected invalid stage budget to raise ValueError")


def test_parse_ocr_engine_duration_budgets_accepts_aliases() -> None:
    budgets = parse_ocr_engine_duration_budgets(
        ["det_elapsed_s=0.3,total_elapsed_s=0.7", "rec_elapsed_s=0.4"]
    )

    assert budgets == {
        "det_elapsed_s": 0.3,
        "rec_elapsed_s": 0.4,
        "total_s": 0.7,
    }


def test_parse_ocr_engine_duration_budgets_rejects_unknown_metrics() -> None:
    try:
        parse_ocr_engine_duration_budgets(["ocr=1"])
    except ValueError as exc:
        assert "Unknown OCR engine duration metric" in str(exc)
    else:
        raise AssertionError("Expected invalid OCR engine duration budget to raise ValueError")


def test_parse_ocr_engine_count_budgets_accepts_known_counts() -> None:
    budgets = parse_ocr_engine_count_budgets(
        ["selected_box_count=30, raw_box_count=50", "label_count=24"]
    )

    assert budgets == {
        "label_count": 24.0,
        "raw_box_count": 50.0,
        "selected_box_count": 30.0,
    }


def test_parse_ocr_engine_count_budgets_rejects_unknown_counts() -> None:
    try:
        parse_ocr_engine_count_budgets(["det_elapsed_s=1"])
    except ValueError as exc:
        assert "Unknown OCR engine count metric" in str(exc)
    else:
        raise AssertionError("Expected invalid OCR engine count budget to raise ValueError")


def test_parse_georeference_source_requirements_accepts_repeated_and_comma_values() -> None:
    sources = parse_georeference_source_requirements(
        ["catalog-shape-match, ocr-georeference:nominatim-label-fit", "catalog-shape-match"]
    )

    assert sources == [
        "catalog-shape-match",
        "ocr-georeference:nominatim-label-fit",
    ]


def test_summarize_georeference_sources_counts_missing_sources() -> None:
    scores = [
        BenchmarkScore(
            slug="dallas-avride",
            image="Avride Dallas.png",
            mode="full",
            passed=True,
            iou=0.99,
            area_ratio=1.0,
            centroid_distance_m=1.0,
            vertices=12,
            style="blue-fill",
            georeference_source="catalog-shape-match",
        ),
        BenchmarkScore(
            slug="dallas-waymo",
            image="Waymo Dallas.png",
            mode="full",
            passed=True,
            iou=0.98,
            area_ratio=1.0,
            centroid_distance_m=2.0,
            vertices=10,
            style="blue-fill",
            georeference_source=" catalog-shape-match ",
        ),
        BenchmarkScore(
            slug="phoenix-waymo",
            image="Waymo Phoenix.png",
            mode="full",
            passed=False,
            iou=None,
            area_ratio=None,
            centroid_distance_m=None,
            vertices=None,
            style=None,
        ),
    ]

    assert summarize_georeference_sources(scores) == {
        "<missing>": 1,
        "catalog-shape-match": 2,
    }


def test_summarize_stage_max_rows_records_stage_tails() -> None:
    scores = [
        BenchmarkScore(
            slug="phoenix-waymo",
            image="Waymo Phoenix.png",
            mode="full",
            passed=True,
            iou=0.98,
            area_ratio=1.0,
            centroid_distance_m=2.0,
            vertices=42,
            style="bright-blue",
            stage_elapsed_s={"ocr": 1.2345678, "extract": 0.2},
        ),
        BenchmarkScore(
            slug="dallas-waymo",
            image="Waymo Dallas.png",
            mode="full",
            passed=True,
            iou=0.96,
            area_ratio=1.0,
            centroid_distance_m=1.0,
            vertices=20,
            style="bright-blue",
            stage_elapsed_s={"ocr": 0.8, "extract": 0.3456789},
        ),
    ]

    assert summarize_stage_max_rows(scores) == {
        "extract": {"slug": "dallas-waymo", "duration_s": 0.345679},
        "ocr": {"slug": "phoenix-waymo", "duration_s": 1.234568},
    }


def test_summarize_ocr_label_events_counts_retries() -> None:
    scores = [
        BenchmarkScore(
            slug="phoenix-waymo",
            image="Waymo Phoenix.png",
            mode="full",
            passed=True,
            iou=0.98,
            area_ratio=1.0,
            centroid_distance_m=2.0,
            vertices=42,
            style="bright-blue",
            ocr_label_events=[
                {"message": "Map labels read", "label_count": 74},
                {"message": "Full-detail map labels read", "label_count": 116},
            ],
            ocr_full_detail_retry=True,
        ),
        BenchmarkScore(
            slug="dallas-waymo",
            image="Waymo Dallas.png",
            mode="full",
            passed=True,
            iou=0.96,
            area_ratio=1.0,
            centroid_distance_m=1.0,
            vertices=20,
            style="bright-blue",
            ocr_label_event="Map labels read",
            ocr_full_detail_retry=False,
        ),
    ]

    assert summarize_ocr_label_events(scores) == {
        "event_counts": {
            "Full-detail map labels read": 1,
            "Map labels read": 2,
        },
        "full_detail_retry_count": 1,
        "full_detail_retry_rows": ["phoenix-waymo"],
    }


def test_run_benchmark_report_includes_runtime_config(monkeypatch, tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    polygon_dir.mkdir()
    image_dir.mkdir()

    monkeypatch.setattr(benchmark_module, "get_pipeline_version", lambda: "pipeline-test")
    monkeypatch.setattr(
        benchmark_module,
        "ocr_runtime_config",
        lambda: {"rapidocr_max_dimension": 1600},
    )
    monkeypatch.delenv("MAP_BOUNDARY_BLOCK_NETWORK", raising=False)
    monkeypatch.setenv("MAP_BOUNDARY_PRECOMPUTE_ROAD_FEATURES", "0")

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
        fixture_config=tmp_path / "fixtures.json",
        block_network=True,
    )

    runtime_config = report["runtime_config"]
    generation_env = runtime_config["generation_env"]
    assert runtime_config["pipeline_version"] == "pipeline-test"
    assert runtime_config["ocr"] == {"rapidocr_max_dimension": 1600}
    assert generation_env["MAP_BOUNDARY_BLOCK_NETWORK"] == "1"
    assert generation_env["MAP_BOUNDARY_PRECOMPUTE_ROAD_FEATURES"] == "0"
    assert generation_env["MAP_BOUNDARY_RUNNER_OCR_CACHE"] == "1"
    assert generation_env["MAP_BOUNDARY_EXTRACTION_TRIMMED_CACHE_MAX_PIXELS"] == "3000000"
    assert generation_env["MAP_BOUNDARY_SCALED_EXTRACTION_MEMORY_CACHE_MAX"] == "24"
    assert os.environ.get("MAP_BOUNDARY_BLOCK_NETWORK") is None


def test_run_benchmark_summary_includes_georeference_source_counts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "dallas-waymo.json").write_text("{}\n")
    (image_dir / "Waymo Dallas.png").write_bytes(b"not an image")

    def fake_score_full_fixture(fixture: BenchmarkFixture, **_kwargs) -> BenchmarkScore:
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True,
            iou=0.99,
            area_ratio=1.0,
            centroid_distance_m=5.0,
            vertices=24,
            style="blue-fill",
            duration_s=0.12,
            georeference_source="catalog-shape-match",
            stage_elapsed_s={"ocr": 0.08, "extract": 0.03},
            ocr_label_events=[
                {"message": "Map labels read", "label_count": 8},
                {"message": "Full-detail map labels read", "label_count": 12},
            ],
            ocr_full_detail_retry=True,
        )

    monkeypatch.setattr(benchmark_module, "score_full_fixture", fake_score_full_fixture)

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="full",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=tmp_path / "fixtures.json",
        execution="in-process",
    )

    assert report["summary"]["active_georeference_sources"] == {"catalog-shape-match": 1}
    assert report["summary"]["active_stage_duration_s"] == {"extract": 0.03, "ocr": 0.08}
    assert report["summary"]["active_stage_max_rows"] == {
        "extract": {"slug": "dallas-waymo", "duration_s": 0.03},
        "ocr": {"slug": "dallas-waymo", "duration_s": 0.08},
    }
    assert report["summary"]["smoked_skipped_stage_max_rows"] == {}
    assert report["summary"]["evaluated_stage_max_rows"] == {
        "extract": {"slug": "dallas-waymo", "duration_s": 0.03},
        "ocr": {"slug": "dallas-waymo", "duration_s": 0.08},
    }
    assert report["summary"]["smoked_skipped_georeference_sources"] == {}
    assert report["summary"]["evaluated_georeference_sources"] == {"catalog-shape-match": 1}
    assert report["summary"]["active_ocr_label_event_counts"] == {
        "Full-detail map labels read": 1,
        "Map labels read": 1,
    }
    assert report["summary"]["active_ocr_full_detail_retry_count"] == 1
    assert report["summary"]["active_ocr_full_detail_retry_rows"] == ["dallas-waymo"]
    assert report["summary"]["smoked_skipped_ocr_label_event_counts"] == {}
    assert report["summary"]["smoked_skipped_ocr_full_detail_retry_count"] == 0
    assert report["summary"]["smoked_skipped_ocr_full_detail_retry_rows"] == []
    assert report["summary"]["evaluated_ocr_label_event_counts"] == {
        "Full-detail map labels read": 1,
        "Map labels read": 1,
    }
    assert report["summary"]["evaluated_ocr_full_detail_retry_count"] == 1
    assert report["summary"]["evaluated_ocr_full_detail_retry_rows"] == ["dallas-waymo"]


def test_run_benchmark_repeat_profile_records_warm_samples(monkeypatch, tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    config_path = tmp_path / "fixtures.json"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "phoenix-waymo.json").write_text("{}\n")
    (image_dir / "Waymo Phoenix.png").write_bytes(b"not an image")
    durations = iter([2.0, 1.2, 0.8])
    calls = []

    def fake_score_full_fixture(fixture: BenchmarkFixture, **kwargs) -> BenchmarkScore:
        calls.append((fixture.slug, kwargs["execution"], kwargs["score_reference"]))
        duration = next(durations)
        selected_box_count = int(duration * 10)
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True,
            iou=0.99,
            area_ratio=1.0,
            centroid_distance_m=0.0,
            vertices=42,
            style="bright-blue",
            duration_s=duration,
            georeference_source="ocr-georeference:nominatim-label-fit",
            combined_confidence=0.86,
            catalog_slug=None,
            road_match_score=0.7,
            road_match_elapsed_s=0.05,
            stage_elapsed_s={"ocr": duration / 2},
            ocr_engine_profile={
                "calls": 1,
                "det_elapsed_s": duration / 4,
                "rec_elapsed_s": duration / 5,
                "total_s": duration / 2,
                "selected_box_count": selected_box_count,
                "raw_box_count": selected_box_count + 2,
                "calls_detail": [
                    {
                        "det_elapsed_s": duration / 4,
                        "rec_elapsed_s": duration / 5,
                        "total_s": duration / 2,
                        "selected_box_count": selected_box_count,
                        "raw_box_count": selected_box_count + 2,
                    }
                ],
            },
            status=fixture.status,
            note=fixture.note,
        )

    monkeypatch.setattr(benchmark_module, "score_full_fixture", fake_score_full_fixture)

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="full",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=config_path,
        execution="in-process",
        repeat_profile_runs=2,
        repeat_profile_warmups=1,
    )

    assert calls == [
        ("phoenix-waymo", "in-process", True),
        ("phoenix-waymo", "in-process", True),
        ("phoenix-waymo", "in-process", True),
    ]
    assert report["summary"]["total_duration_s"] == 2.0
    assert report["thresholds"]["repeat_profile_runs"] == 2
    assert report["thresholds"]["repeat_profile_warmups"] == 1
    repeat_profile = report["repeat_profile"]
    assert repeat_profile["runs_per_fixture"] == 2
    assert repeat_profile["warmup_runs_per_fixture"] == 1
    assert repeat_profile["summary"]["samples"] == 2
    assert repeat_profile["summary"]["analyzed_samples"] == 1
    assert repeat_profile["summary"]["passed_samples"] == 1
    assert repeat_profile["summary"]["subsecond_samples"] == 1
    assert repeat_profile["summary"]["subsecond_fixture_min_duration_count"] == 1
    assert repeat_profile["summary"]["stable_signature_fixtures"] == 1
    assert repeat_profile["summary"]["unstable_signature_fixtures"] == []
    assert repeat_profile["summary"]["min_duration_s"] == 0.8
    assert repeat_profile["summary"]["stage_duration_s"] == {
        "ocr": {
            "samples": 1,
            "min_duration_s": 0.4,
            "median_duration_s": 0.4,
            "average_duration_s": 0.4,
            "p90_duration_s": 0.4,
            "p95_duration_s": 0.4,
            "max_duration_s": 0.4,
        }
    }
    assert repeat_profile["summary"]["ocr_engine_profile"] == {
        "fixtures": 1,
        "calls": 1,
        "det_elapsed_s": 0.2,
        "rec_elapsed_s": 0.16,
        "total_s": 0.4,
        "raw_box_count": 10,
        "selected_box_count": 8,
    }
    assert repeat_profile["summary"]["ocr_engine_stage_duration_s"] == {
        "det_elapsed_s": {
            "samples": 1,
            "min_duration_s": 0.2,
            "median_duration_s": 0.2,
            "average_duration_s": 0.2,
            "p90_duration_s": 0.2,
            "p95_duration_s": 0.2,
            "max_duration_s": 0.2,
        },
        "rec_elapsed_s": {
            "samples": 1,
            "min_duration_s": 0.16,
            "median_duration_s": 0.16,
            "average_duration_s": 0.16,
            "p90_duration_s": 0.16,
            "p95_duration_s": 0.16,
            "max_duration_s": 0.16,
        },
        "total_s": {
            "samples": 1,
            "min_duration_s": 0.4,
            "median_duration_s": 0.4,
            "average_duration_s": 0.4,
            "p90_duration_s": 0.4,
            "p95_duration_s": 0.4,
            "max_duration_s": 0.4,
        },
    }
    assert repeat_profile["summary"]["ocr_engine_count_metric"] == {
        "raw_box_count": {
            "samples": 1,
            "min_count": 10,
            "median_count": 10,
            "average_count": 10,
            "p90_count": 10,
            "p95_count": 10,
            "max_count": 10,
        },
        "selected_box_count": {
            "samples": 1,
            "min_count": 8,
            "median_count": 8,
            "average_count": 8,
            "p90_count": 8,
            "p95_count": 8,
            "max_count": 8,
        },
    }
    assert repeat_profile["fixtures"]["phoenix-waymo"]["min_iou"] == 0.99
    assert repeat_profile["fixtures"]["phoenix-waymo"]["signature_stability"]["stable"] is True
    assert repeat_profile["fixtures"]["phoenix-waymo"]["signature_stability"]["unique_signatures"] == 1
    assert repeat_profile["fixtures"]["phoenix-waymo"]["stage_duration_s"]["ocr"]["max_duration_s"] == 0.4
    assert (
        repeat_profile["fixtures"]["phoenix-waymo"]["ocr_engine_stage_duration_s"]["total_s"]["p95_duration_s"]
        == 0.4
    )
    assert repeat_profile["fixtures"]["phoenix-waymo"]["ocr_engine_count_metric"]["selected_box_count"][
        "p95_count"
    ] == 8
    assert repeat_profile["samples"][0]["warmup"] is True
    assert repeat_profile["samples"][1]["repeat_index"] == 2


def test_repeat_profile_flags_output_signature_drift() -> None:
    repeat_profile = benchmark_module.summarize_repeat_profile_samples(
        [
            {
                "slug": "phoenix-waymo",
                "repeat_index": 1,
                "warmup": False,
                "passed": True,
                "status": "active",
                "iou": 0.91,
                "area_ratio": 1.0,
                "centroid_distance_m": 12.0,
                "vertices": 42,
                "style": "bright-blue",
                "duration_s": 0.52,
                "georeference_source": "ocr-georeference:nominatim-label-fit",
                "combined_confidence": 0.88,
                "catalog_slug": None,
                "road_match_score": 0.7,
                "ocr_label_count": 56,
                "ocr_label_event": "Map labels read",
                "ocr_full_detail_retry": False,
                "ocr_top_labels": ["Phoenix", "Waymo"],
            },
            {
                "slug": "phoenix-waymo",
                "repeat_index": 2,
                "warmup": False,
                "passed": True,
                "status": "active",
                "iou": 0.91,
                "area_ratio": 1.0,
                "centroid_distance_m": 12.0,
                "vertices": 42,
                "style": "bright-blue",
                "duration_s": 0.48,
                "georeference_source": "catalog-shape-match",
                "combined_confidence": 0.94,
                "catalog_slug": "phoenix-waymo",
                "road_match_score": None,
                "ocr_label_count": 20,
                "ocr_label_event": "Map labels read",
                "ocr_full_detail_retry": False,
                "ocr_top_labels": ["Phoenix"],
            },
        ],
        runs_per_fixture=2,
        warmup_runs_per_fixture=0,
    )

    assert repeat_profile["summary"]["stable_signature_fixtures"] == 0
    assert repeat_profile["summary"]["unstable_signature_fixtures"] == ["phoenix-waymo"]
    stability = repeat_profile["fixtures"]["phoenix-waymo"]["signature_stability"]
    assert stability["stable"] is False
    assert stability["unique_signatures"] == 2
    assert [signature["count"] for signature in stability["signatures"]] == [1, 1]


def test_repeat_profile_duration_stats_record_tail_percentiles() -> None:
    stats = benchmark_module.duration_distribution_stats([1.0, 2.0, 4.0])

    assert stats == {
        "min_duration_s": 1.0,
        "median_duration_s": 2.0,
        "average_duration_s": 2.333333,
        "p90_duration_s": 3.6,
        "p95_duration_s": 3.8,
        "max_duration_s": 4.0,
    }


def test_benchmark_score_preserves_sub_millisecond_duration_precision() -> None:
    row = BenchmarkScore(
        slug="phoenix-waymo",
        image="Waymo Phoenix.png",
        mode="full",
        passed=True,
        iou=0.98,
        area_ratio=1.0,
        centroid_distance_m=0.0,
        vertices=42,
        style="bright-blue",
        image_width=2400,
        image_height=2400,
        road_match_score=0.681518,
        road_match_elapsed_s=0.195375,
        duration_s=1.00049,
    ).as_dict()

    assert row["image_width"] == 2400
    assert row["image_height"] == 2400
    assert row["duration_s"] == 1.00049
    assert row["road_match_score"] == 0.681518
    assert row["road_match_elapsed_s"] == 0.195375


def test_benchmark_score_reports_ocr_label_summary() -> None:
    row = BenchmarkScore(
        slug="miami-waymo",
        image="Waymo Miami.png",
        mode="full",
        passed=True,
        iou=0.91,
        area_ratio=1.0,
        centroid_distance_m=0.0,
        vertices=42,
        style="bright-blue",
        ocr_label_count=23,
        ocr_top_labels=["Miami", "Brickell", "Coral Gables"],
        ocr_label_event="Full-detail map labels read",
        ocr_label_events=[
            {"message": "Map labels read", "label_count": 9, "top_labels": ["Miami"]},
            {
                "message": "Full-detail map labels read",
                "label_count": 23,
                "top_labels": ["Miami", "Brickell", "Coral Gables"],
            },
        ],
        ocr_full_detail_retry=True,
    ).as_dict()

    assert row["ocr_label_count"] == 23
    assert row["ocr_top_labels"] == ["Miami", "Brickell", "Coral Gables"]
    assert row["ocr_label_event"] == "Full-detail map labels read"
    assert row["ocr_label_events"] == [
        {"message": "Map labels read", "label_count": 9, "top_labels": ["Miami"]},
        {
            "message": "Full-detail map labels read",
            "label_count": 23,
            "top_labels": ["Miami", "Brickell", "Coral Gables"],
        },
    ]
    assert row["ocr_full_detail_retry"] is True


def test_benchmark_score_reports_ocr_engine_profile() -> None:
    profile = {
        "calls": 1,
        "det_elapsed_s": 0.42,
        "rec_elapsed_s": 0.18,
        "raw_box_count": 26,
        "selected_box_count": 14,
    }
    row = BenchmarkScore(
        slug="nashville-waymo",
        image="Waymo Nashville.png",
        mode="full",
        passed=True,
        iou=0.95,
        area_ratio=1.0,
        centroid_distance_m=0.0,
        vertices=42,
        style="bright-blue",
        ocr_engine_profile=profile,
    ).as_dict()

    assert row["ocr_engine_profile"] == profile


def test_ocr_label_summary_from_events_uses_last_ocr_label_read() -> None:
    summary = benchmark_module.ocr_label_summary_from_events(
        [
            {
                "stage": "ocr",
                "message": "Map labels read",
                "details": {"label_count": 4, "top_labels": ["Phoenix"]},
            },
            {
                "stage": "extract",
                "message": "Map boundary extracted",
                "details": {"label_count": 99, "top_labels": ["ignored"]},
            },
            {
                "stage": "ocr",
                "message": "Full-detail map labels read",
                "details": {"label_count": 12, "top_labels": ["Miami", 42, "Aventura"]},
            },
        ]
    )

    assert summary == {
        "ocr_label_count": 12,
        "ocr_top_labels": ["Miami", "Aventura"],
        "ocr_label_event": "Full-detail map labels read",
        "ocr_label_events": [
            {"message": "Map labels read", "label_count": 4, "top_labels": ["Phoenix"]},
            {
                "message": "Full-detail map labels read",
                "label_count": 12,
                "top_labels": ["Miami", "Aventura"],
            },
        ],
        "ocr_full_detail_retry": True,
    }


def test_summarize_ocr_engine_profiles_totals_fixture_calls() -> None:
    scores = [
        BenchmarkScore(
            slug="dallas-waymo",
            image="Waymo Dallas.png",
            mode="full",
            passed=True,
            iou=0.96,
            area_ratio=1.0,
            centroid_distance_m=0.0,
            vertices=42,
            style="bright-blue",
            ocr_engine_profile={
                "calls": 1,
                "det_elapsed_s": 0.63,
                "rec_elapsed_s": 0.09,
                "raw_box_count": 17,
                "selected_box_count": 9,
            },
        ),
        BenchmarkScore(
            slug="phoenix-waymo",
            image="Waymo Phoenix.png",
            mode="full",
            passed=True,
            iou=0.98,
            area_ratio=1.0,
            centroid_distance_m=0.0,
            vertices=42,
            style="bright-blue",
            ocr_engine_profile={
                "calls": 1,
                "det_elapsed_s": 0.33,
                "rec_elapsed_s": 0.36,
                "raw_box_count": 85,
                "selected_box_count": 44,
            },
        ),
    ]

    assert benchmark_module.summarize_ocr_engine_profiles(scores) == {
        "fixtures": 2,
        "calls": 2,
        "det_elapsed_s": 0.96,
        "rec_elapsed_s": 0.45,
        "raw_box_count": 102,
        "selected_box_count": 53,
    }


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
            "image_width": None,
            "image_height": None,
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
            "catalog_shape_iou": None,
            "catalog_area_ratio": None,
            "road_match_score": None,
            "road_match_elapsed_s": None,
            "stage_elapsed_s": None,
            "ocr_label_count": None,
            "ocr_top_labels": None,
            "ocr_label_event": None,
            "ocr_label_events": None,
            "ocr_full_detail_retry": None,
            "ocr_engine_profile": None,
            "error": None,
            "status": "reference_mismatch",
            "note": "changed live service area",
        }
    ]


def test_changed_area_config_marks_new_provider_fixture_reference_mismatch(tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    config_path = tmp_path / "fixtures.json"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "houston-avride.json").write_text("{}\n")
    (image_dir / "Avride Houston.png").write_bytes(b"not an image")
    config_path.write_text(
        json.dumps(
            {
                "changed_areas": {
                    "houston": {
                        "status": "reference_mismatch",
                        "note": "changed live service area",
                    }
                },
                "fixtures": {},
            }
        )
        + "\n"
    )

    config = load_fixture_config(config_path)
    fixtures, inventory = discover_fixtures(polygon_dir, image_dir, config)

    assert inventory["matched_images"] == 1
    assert fixtures == [
        BenchmarkFixture(
            slug="houston-avride",
            provider="avride",
            area="Houston",
            image_path=image_dir / "Avride Houston.png",
            reference_path=polygon_dir / "houston-avride.json",
            status="reference_mismatch",
            note="changed live service area",
        )
    ]


def test_discover_fixtures_accepts_gif_images(tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "dallas-tesla.json").write_text("{}\n")
    gif_image = image_dir / "Tesla Dallas.gif"
    gif_image.write_bytes(b"GIF89a")

    fixtures, inventory = discover_fixtures(polygon_dir, image_dir, {"path": None, "fixtures": {}})

    assert inventory["matched_images"] == 1
    assert fixtures == [
        BenchmarkFixture(
            slug="dallas-tesla",
            provider="tesla",
            area="Dallas",
            image_path=gif_image,
            reference_path=polygon_dir / "dallas-tesla.json",
        )
    ]


def test_discover_fixtures_accepts_bmp_images(tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "dallas-tesla.json").write_text("{}\n")
    bmp_image = image_dir / "Tesla Dallas.bmp"
    bmp_image.write_bytes(b"BM")

    fixtures, inventory = discover_fixtures(polygon_dir, image_dir, {"path": None, "fixtures": {}})

    assert inventory["matched_images"] == 1
    assert fixtures == [
        BenchmarkFixture(
            slug="dallas-tesla",
            provider="tesla",
            area="Dallas",
            image_path=bmp_image,
            reference_path=polygon_dir / "dallas-tesla.json",
        )
    ]


def test_discover_fixtures_accepts_tiff_images(tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "dallas-tesla.json").write_text("{}\n")
    tiff_image = image_dir / "Tesla Dallas.tiff"
    tiff_image.write_bytes(b"II*\x00")

    fixtures, inventory = discover_fixtures(polygon_dir, image_dir, {"path": None, "fixtures": {}})

    assert inventory["matched_images"] == 1
    assert fixtures == [
        BenchmarkFixture(
            slug="dallas-tesla",
            provider="tesla",
            area="Dallas",
            image_path=tiff_image,
            reference_path=polygon_dir / "dallas-tesla.json",
        )
    ]


def test_fixture_config_can_use_current_image_for_drifted_reference(tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    current_dir = tmp_path / "current"
    config_path = tmp_path / "fixtures.json"
    polygon_dir.mkdir()
    image_dir.mkdir()
    current_dir.mkdir()

    (polygon_dir / "miami-waymo.json").write_text("{}\n")
    stale_image = image_dir / "Waymo Miami.png"
    current_image = current_dir / "Miami Current.png"
    stale_image.write_bytes(b"stale image")
    current_image.write_bytes(b"current image")
    config_path.write_text(
        json.dumps(
            {
                "fixtures": {
                    "miami-waymo": {
                        "status": "reference_mismatch",
                        "note": "changed live service area",
                        "current_image": "../current/Miami Current.png",
                    }
                }
            }
        )
        + "\n"
    )

    fixtures, inventory = discover_fixtures(polygon_dir, image_dir, load_fixture_config(config_path))

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.slug == "miami-waymo"
    assert fixture.image_path == current_image.resolve()
    assert fixture.reference_path == polygon_dir / "miami-waymo.json"
    assert fixture.status == "reference_mismatch"
    assert inventory["configured_image_overrides"] == {"miami-waymo": str(current_image.resolve())}
    assert inventory["missing_configured_images"] == {}


def test_missing_current_image_override_falls_back_to_discovered_image(tmp_path: Path) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    config_path = tmp_path / "fixtures.json"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "houston-waymo.json").write_text("{}\n")
    stale_image = image_dir / "Waymo Houston.png"
    stale_image.write_bytes(b"stale image")
    config_path.write_text(
        json.dumps(
            {
                "fixtures": {
                    "houston-waymo": {
                        "status": "reference_mismatch",
                        "note": "changed live service area",
                        "current_image": "../missing/Houston Current.png",
                    }
                }
            }
        )
        + "\n"
    )

    fixtures, inventory = discover_fixtures(polygon_dir, image_dir, load_fixture_config(config_path))

    assert len(fixtures) == 1
    assert fixtures[0].image_path == stale_image
    assert inventory["configured_image_overrides"] == {}
    assert inventory["missing_configured_images"] == {
        "houston-waymo": str((image_dir / "../missing/Houston Current.png").resolve())
    }


def test_smoke_skipped_full_fixtures_runs_without_scoring_stale_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    calls = []

    def fake_score_full_fixture(fixture: BenchmarkFixture, **kwargs) -> BenchmarkScore:
        calls.append((fixture.slug, kwargs["score_reference"], kwargs["catalog_probe_missed"]))
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True,
            iou=None,
            area_ratio=None,
            centroid_distance_m=None,
            vertices=42,
            style="bright-blue",
            duration_s=0.12,
            georeference_source="ocr-georeference:nominatim-label-fit",
            combined_confidence=0.86,
            catalog_slug=None,
            road_match_score=0.706233,
            road_match_elapsed_s=0.04,
            stage_elapsed_s={"ocr": 0.08},
            status=fixture.status,
            note=fixture.note,
        )

    monkeypatch.setattr(benchmark_module, "score_full_fixture", fake_score_full_fixture)

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="full",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=config_path,
        smoke_skipped=True,
        catalog_probe_missed=True,
    )

    assert calls == [("houston-waymo", False, True)]
    assert report["summary"]["passed"] is True
    assert report["thresholds"]["catalog_probe_missed"] is True
    assert report["summary"]["scored_fixtures"] == 0
    assert report["summary"]["skipped_fixtures"] == 1
    assert report["summary"]["smoked_skipped_fixtures"] == 1
    assert report["summary"]["failed_smoked_skipped_fixtures"] == 0
    assert report["summary"]["active_total_duration_s"] == 0
    assert report["summary"]["smoked_skipped_duration_s"] == 0.12
    assert report["summary"]["evaluated_duration_s"] == 0.12
    assert report["summary"]["active_stage_duration_s"] == {}
    assert report["summary"]["smoked_skipped_stage_duration_s"] == {"ocr": 0.08}
    assert report["summary"]["evaluated_stage_duration_s"] == {"ocr": 0.08}
    assert report["summary"]["active_road_match_elapsed_s"] == 0
    assert report["summary"]["smoked_skipped_road_match_elapsed_s"] == 0.04
    assert report["summary"]["evaluated_road_match_elapsed_s"] == 0.04
    assert report["scores"][0]["status"] == "reference_mismatch"
    assert report["scores"][0]["iou"] is None
    assert report["scores"][0]["georeference_source"] == "ocr-georeference:nominatim-label-fit"
    assert report["scores"][0]["road_match_score"] == 0.706233
    assert report["scores"][0]["road_match_elapsed_s"] == 0.04


def test_catalog_reference_lookup_covers_changed_reference_mismatches() -> None:
    for slug in KNOWN_CHANGED_REFERENCE_MISMATCH_FIXTURES:
        area_slug, provider = slug.rsplit("-", 1)
        fixture = BenchmarkFixture(
            slug=slug,
            provider=provider,
            area=area_slug.replace("-", " ").title(),
            image_path=Path(f"{slug}.png"),
            reference_path=Path(f"{slug}.json"),
            status="reference_mismatch",
        )

        geometry = benchmark_module.catalog_reference_geometry_for_fixture(fixture)

        assert geometry is not None
        assert not geometry.is_empty


def test_score_skipped_catalog_references_scores_against_current_catalog_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    catalog_geometry = Polygon(
        [
            (-95.5, 29.6),
            (-95.2, 29.6),
            (-95.2, 29.9),
            (-95.5, 29.9),
            (-95.5, 29.6),
        ]
    )
    calls = []

    def fake_catalog_reference(fixture: BenchmarkFixture):
        assert fixture.slug == "houston-waymo"
        return catalog_geometry

    def fake_score_full_fixture(fixture: BenchmarkFixture, **kwargs) -> BenchmarkScore:
        calls.append((fixture, kwargs))
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True,
            iou=1.0,
            area_ratio=1.0,
            centroid_distance_m=0.0,
            vertices=42,
            style="bright-blue",
            duration_s=0.12,
            georeference_source="catalog-shape-match",
            combined_confidence=0.98,
            catalog_slug="houston-waymo",
            catalog_shape_iou=0.92,
            catalog_area_ratio=1.03,
            stage_elapsed_s={"match_catalog": 0.01},
            status=fixture.status,
            note=fixture.note,
        )

    monkeypatch.setattr(benchmark_module, "catalog_reference_geometry_for_fixture", fake_catalog_reference)
    monkeypatch.setattr(benchmark_module, "score_full_fixture", fake_score_full_fixture)

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="full",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=config_path,
        score_skipped_catalog_references=True,
    )

    assert len(calls) == 1
    scored_fixture, score_kwargs = calls[0]
    assert scored_fixture.status == "active"
    assert scored_fixture.note == (
        "changed live service area. "
        "Scored against current catalog geometry instead of the stale saved reference."
    )
    assert score_kwargs["score_reference"] is True
    assert score_kwargs["reference_geometry"] is catalog_geometry
    assert report["thresholds"]["score_skipped_catalog_references"] is True
    assert report["thresholds"]["require_scored_catalog_evidence"] is True
    assert report["summary"]["passed"] is True
    assert report["summary"]["scored_fixtures"] == 1
    assert report["summary"]["skipped_fixtures"] == 0
    assert report["summary"]["active_stage_duration_s"] == {"match_catalog": 0.01}
    assert report["summary"]["smoked_skipped_stage_duration_s"] == {}
    assert report["summary"]["evaluated_stage_duration_s"] == {"match_catalog": 0.01}
    assert report["scores"][0]["status"] == "active"
    assert report["scores"][0]["iou"] == 1.0
    assert report["scores"][0]["catalog_shape_iou"] == 0.92
    assert report["scores"][0]["catalog_area_ratio"] == 1.03


def test_score_skipped_catalog_references_require_source_image_catalog_evidence_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    monkeypatch.setattr(
        benchmark_module,
        "catalog_reference_geometry_for_fixture",
        lambda _fixture: Polygon([(-95.5, 29.6), (-95.2, 29.6), (-95.2, 29.9), (-95.5, 29.9)]),
    )

    def fake_score_full_fixture(fixture: BenchmarkFixture, **_kwargs) -> BenchmarkScore:
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True,
            iou=1.0,
            area_ratio=1.0,
            centroid_distance_m=0.0,
            vertices=42,
            style="bright-blue",
            duration_s=0.12,
            georeference_source="catalog-shape-match:georef-contained",
            combined_confidence=0.84,
            catalog_slug="houston-waymo",
            catalog_shape_iou=0.411686,
            catalog_area_ratio=0.513975,
            status=fixture.status,
            note=fixture.note,
        )

    monkeypatch.setattr(benchmark_module, "score_full_fixture", fake_score_full_fixture)

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="full",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=config_path,
        score_skipped_catalog_references=True,
    )

    score = report["scores"][0]
    assert report["summary"]["passed"] is False
    assert report["summary"]["failed_fixtures"] == 1
    assert score["iou"] == 1.0
    assert score["catalog_shape_iou"] == 0.411686
    assert "source-image catalog evidence is weak" in score["error"]
    assert "catalog_shape_iou 0.411686" in score["error"]


def test_score_output_geometry_accepts_direct_reference_geometry() -> None:
    geometry = Polygon(
        [
            (-118.5, 34.0),
            (-118.2, 34.0),
            (-118.2, 34.2),
            (-118.5, 34.2),
            (-118.5, 34.0),
        ]
    )

    metrics = benchmark_module.score_output_geometry(
        geometry,
        None,
        0.99,
        reference_geometry=geometry,
    )

    assert metrics["passed"] is True
    assert metrics["iou"] == 1.0


def test_smoke_skipped_catalog_miss_requirement_fails_catalog_hits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    config_path = tmp_path / "fixtures.json"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "miami-waymo.json").write_text("{}\n")
    (image_dir / "Waymo Miami.png").write_bytes(b"not an image")
    config_path.write_text(
        json.dumps(
            {
                "fixtures": {
                    "miami-waymo": {
                        "status": "reference_mismatch",
                        "note": "changed live service area",
                    }
                }
            }
        )
        + "\n"
    )

    def fake_score_full_fixture(fixture: BenchmarkFixture, **kwargs) -> BenchmarkScore:
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True,
            iou=None,
            area_ratio=None,
            centroid_distance_m=None,
            vertices=42,
            style="bright-blue",
            duration_s=0.12,
            georeference_source="catalog-shape-match",
            combined_confidence=0.98,
            catalog_slug="miami-waymo",
            status=fixture.status,
            note=fixture.note,
        )

    monkeypatch.setattr(benchmark_module, "score_full_fixture", fake_score_full_fixture)

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="full",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=config_path,
        smoke_skipped=True,
        require_smoked_catalog_miss=True,
    )

    assert report["summary"]["passed"] is False
    assert report["summary"]["smoked_skipped_fixtures"] == 1
    assert report["summary"]["failed_smoked_skipped_fixtures"] == 1
    assert report["summary"]["active_total_duration_s"] == 0
    assert report["summary"]["smoked_skipped_duration_s"] == 0.12
    assert report["summary"]["evaluated_duration_s"] == 0.12
    assert report["scores"][0]["catalog_slug"] == "miami-waymo"
    assert "expected OCR/georeference catalog miss" in report["scores"][0]["error"]


def test_catalog_miss_requirement_implies_smoke_skipped_fixtures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    config_path = tmp_path / "fixtures.json"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "bay-area-waymo.json").write_text("{}\n")
    (image_dir / "Waymo Bay Area.png").write_bytes(b"not an image")
    config_path.write_text(
        json.dumps(
            {
                "fixtures": {
                    "bay-area-waymo": {
                        "status": "reference_mismatch",
                        "note": "changed live service area",
                    }
                }
            }
        )
        + "\n"
    )
    calls = []

    def fake_score_full_fixture(fixture: BenchmarkFixture, **kwargs) -> BenchmarkScore:
        calls.append((fixture.slug, kwargs["score_reference"], kwargs["catalog_probe_missed"]))
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True,
            iou=None,
            area_ratio=None,
            centroid_distance_m=None,
            vertices=42,
            style="bright-blue",
            duration_s=0.12,
            georeference_source="ocr-georeference:nominatim-label-fit",
            combined_confidence=0.86,
            catalog_slug=None,
            status=fixture.status,
            note=fixture.note,
        )

    monkeypatch.setattr(benchmark_module, "score_full_fixture", fake_score_full_fixture)

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="full",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=config_path,
        catalog_probe_missed=True,
        require_smoked_catalog_miss=True,
    )

    assert calls == [("bay-area-waymo", False, True)]
    assert report["thresholds"]["smoke_skipped"] is True
    assert report["thresholds"]["require_smoked_catalog_miss"] is True
    assert report["summary"]["smoked_skipped_fixtures"] == 1
    assert report["summary"]["failed_smoked_skipped_fixtures"] == 0
    assert report["summary"]["active_total_duration_s"] == 0
    assert report["summary"]["smoked_skipped_duration_s"] == 0.12
    assert report["summary"]["evaluated_duration_s"] == 0.12
    assert report["summary"]["passed"] is True


def test_block_network_sets_policy_during_full_generation(tmp_path: Path, monkeypatch) -> None:
    polygon_dir = tmp_path / "polygons"
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    polygon_dir.mkdir()
    image_dir.mkdir()

    (polygon_dir / "phoenix-waymo.json").write_text("{}\n")
    (image_dir / "Waymo Phoenix.png").write_bytes(b"not an image")
    seen_policy = []

    def fake_score_full_fixture(fixture: BenchmarkFixture, **kwargs) -> BenchmarkScore:
        seen_policy.append(os.environ.get("MAP_BOUNDARY_BLOCK_NETWORK"))
        return BenchmarkScore(
            slug=fixture.slug,
            image=fixture.image_path.name,
            mode="full",
            passed=True,
            iou=1.0,
            area_ratio=1.0,
            centroid_distance_m=0.0,
            vertices=42,
            style="bright-blue",
            duration_s=0.12,
            georeference_source="ocr-georeference:nominatim-label-fit",
            combined_confidence=0.96,
            catalog_slug=None,
            status=fixture.status,
            note=fixture.note,
        )

    monkeypatch.delenv("MAP_BOUNDARY_BLOCK_NETWORK", raising=False)
    monkeypatch.setattr(benchmark_module, "score_full_fixture", fake_score_full_fixture)

    report = run_benchmark(
        polygon_dir=polygon_dir,
        image_dir=image_dir,
        out_dir=out_dir,
        mode="full",
        min_iou=0.78,
        mean_iou=0.90,
        timeout_seconds=1,
        city_overrides=False,
        only_filters=[],
        fixture_config=tmp_path / "missing-config.json",
        block_network=True,
    )

    assert seen_policy == ["1"]
    assert os.environ.get("MAP_BOUNDARY_BLOCK_NETWORK") is None
    assert report["thresholds"]["block_network"] is True
    assert report["summary"]["passed"] is True


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
    assert check["compared_iou_fixtures"] == 1
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
            "baseline_average_iou": 0.931476,
            "candidate_average_iou": 0.781303,
            "drop": 0.150173,
            "average_iou_scope": "compared_fixtures",
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


def test_report_regression_check_ignores_newly_scored_fixture_in_mean() -> None:
    baseline = {
        "summary": {"average_iou": 0.90},
        "scores": [
            {"slug": "orlando-waymo", "status": "active", "iou": 0.90},
            {"slug": "bay-area-tesla", "status": "reference_mismatch", "iou": None},
        ],
    }
    candidate = {
        "summary": {"average_iou": 0.85},
        "scores": [
            {"slug": "orlando-waymo", "status": "active", "iou": 0.90},
            {"slug": "bay-area-tesla", "status": "active", "iou": 0.80},
        ],
    }

    check = compare_report_regressions(candidate, baseline, max_mean_iou_drop=0.0)

    assert check["passed"] is True
    assert check["compared_fixtures"] == 1
    assert check["compared_iou_fixtures"] == 1
    assert check["baseline_average_iou"] == 0.90
    assert check["candidate_average_iou"] == 0.90
    assert check["average_iou_scope"] == "compared_fixtures"
    assert check["issues"] == []


def test_report_regression_check_skips_unselected_filtered_fixtures() -> None:
    baseline = {
        "summary": {"average_iou": 0.96},
        "scores": [
            {"slug": "nashville-waymo", "status": "active", "iou": 0.986282},
            {"slug": "phoenix-waymo", "status": "active", "iou": 0.98332},
        ],
    }
    candidate = {
        "summary": {"average_iou": 0.98332},
        "inventory": {"filtered_from": 15, "only_filters": ["phoenix"]},
        "scores": [
            {"slug": "phoenix-waymo", "status": "active", "iou": 0.98332},
        ],
    }

    check = compare_report_regressions(candidate, baseline, max_mean_iou_drop=0.0)

    assert check["passed"] is True
    assert check["comparison_scope"] == "filtered_candidate"
    assert check["candidate_only_filters"] == ["phoenix"]
    assert check["compared_fixtures"] == 1
    assert check["compared_iou_fixtures"] == 1
    assert check["omitted_baseline_fixtures"] == 1
    assert check["omitted_baseline_slugs"] == ["nashville-waymo"]
    assert check["baseline_average_iou"] == 0.98332
    assert check["candidate_average_iou"] == 0.98332
    assert check["issues"] == []


def test_report_regression_check_flags_selected_filtered_fixture_without_iou() -> None:
    baseline = {
        "summary": {"average_iou": 0.98},
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": 0.98332}],
    }
    candidate = {
        "summary": {"average_iou": 0.0},
        "inventory": {"filtered_from": 15, "only_filters": ["phoenix"]},
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": None}],
    }

    check = compare_report_regressions(candidate, baseline)

    assert check["passed"] is False
    assert check["comparison_scope"] == "filtered_candidate"
    assert check["compared_fixtures"] == 1
    assert check["compared_iou_fixtures"] == 0
    assert check["omitted_baseline_fixtures"] == 0
    assert check["issues"] == [
        {
            "slug": "phoenix-waymo",
            "kind": "missing_candidate_score",
            "baseline_iou": 0.98332,
        }
    ]


def test_report_regression_check_flags_unfiltered_missing_candidate_score() -> None:
    baseline = {
        "summary": {"average_iou": 0.96},
        "scores": [
            {"slug": "nashville-waymo", "status": "active", "iou": 0.986282},
            {"slug": "phoenix-waymo", "status": "active", "iou": 0.98332},
        ],
    }
    candidate = {
        "summary": {"average_iou": 0.98332},
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": 0.98332}],
    }

    check = compare_report_regressions(candidate, baseline)

    assert check["passed"] is False
    assert check["comparison_scope"] == "full_candidate"
    assert check["compared_fixtures"] == 2
    assert check["compared_iou_fixtures"] == 1
    assert check["omitted_baseline_fixtures"] == 0
    assert check["issues"] == [
        {
            "slug": "nashville-waymo",
            "kind": "missing_candidate_score",
            "baseline_iou": 0.986282,
        }
    ]


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


def test_report_regression_check_can_flag_ocr_label_loss() -> None:
    baseline = {
        "summary": {"average_iou": 0.95},
        "scores": [
            {
                "slug": "phoenix-waymo",
                "status": "active",
                "iou": 0.98332,
                "ocr_label_count": 74,
                "ocr_top_labels": [
                    "Scottsdale",
                    "International",
                    "Fashion Park",
                    "School Park",
                ],
            }
        ],
    }
    candidate = {
        "summary": {"average_iou": 0.95},
        "scores": [
            {
                "slug": "phoenix-waymo",
                "status": "active",
                "iou": 0.98332,
                "ocr_label_count": 50,
                "ocr_top_labels": ["Scottsdale", "Tempe"],
            }
        ],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_ocr_label_count_drop=10,
        min_ocr_top_label_retention=0.75,
    )

    assert check["passed"] is False
    assert check["max_ocr_label_count_drop"] == 10
    assert check["min_ocr_top_label_retention"] == 0.75
    assert check["compared_ocr_label_counts"] == 1
    assert check["compared_ocr_top_label_sets"] == 1
    assert check["issues"] == [
        {
            "slug": "phoenix-waymo",
            "kind": "ocr_label_count_drop",
            "baseline_ocr_label_count": 74,
            "candidate_ocr_label_count": 50,
            "drop": 24,
            "max_drop": 10,
        },
        {
            "slug": "phoenix-waymo",
            "kind": "ocr_top_label_retention_drop",
            "baseline_ocr_top_label_count": 4,
            "candidate_ocr_top_label_count": 2,
            "retained_ocr_top_label_count": 1,
            "retention": 0.25,
            "min_retention": 0.75,
            "missing_ocr_top_labels": [
                "fashion park",
                "international",
                "school park",
            ],
        },
    ]


def test_report_regression_check_allows_configured_ocr_label_tolerance() -> None:
    baseline = {
        "summary": {"average_iou": 0.95},
        "scores": [
            {
                "slug": "nashville-waymo",
                "status": "active",
                "iou": 0.986282,
                "ocr_label_count": 19,
                "ocr_top_labels": ["Nashville", "Edgefield", "Inglewood", "South Nashville"],
            }
        ],
    }
    candidate = {
        "summary": {"average_iou": 0.95},
        "scores": [
            {
                "slug": "nashville-waymo",
                "status": "active",
                "iou": 0.986282,
                "ocr_label_count": 17,
                "ocr_top_labels": ["Nashville", "Edgefield", "Inglewood"],
            }
        ],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_ocr_label_count_drop=2,
        min_ocr_top_label_retention=0.75,
    )

    assert check["passed"] is True
    assert check["compared_ocr_label_counts"] == 1
    assert check["compared_ocr_top_label_sets"] == 1
    assert check["issues"] == []


def test_report_regression_check_flags_missing_candidate_ocr_evidence() -> None:
    baseline = {
        "summary": {"average_iou": 0.95},
        "scores": [
            {
                "slug": "dallas-waymo",
                "status": "active",
                "iou": 0.95701,
                "ocr_label_count": 11,
                "ocr_top_labels": ["Deep Ellum", "Highland Park", "Dallas"],
            }
        ],
    }
    candidate = {
        "summary": {"average_iou": 0.95},
        "scores": [{"slug": "dallas-waymo", "status": "active", "iou": 0.95701}],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_ocr_label_count_drop=0,
        min_ocr_top_label_retention=1.0,
    )

    assert check["passed"] is False
    assert check["compared_ocr_label_counts"] == 0
    assert check["compared_ocr_top_label_sets"] == 0
    assert check["issues"] == [
        {
            "slug": "dallas-waymo",
            "kind": "missing_candidate_ocr_label_count",
            "baseline_ocr_label_count": 11,
        },
        {
            "slug": "dallas-waymo",
            "kind": "missing_candidate_ocr_top_labels",
            "baseline_ocr_top_label_count": 3,
        },
    ]


def test_report_regression_check_can_flag_evaluated_duration_increase() -> None:
    baseline = {
        "summary": {
            "average_iou": 0.95,
            "total_duration_s": 2.5,
            "smoked_skipped_duration_s": 0.5,
        },
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": 0.98332, "duration_s": 0.5}],
    }
    candidate = {
        "summary": {
            "average_iou": 0.95,
            "total_duration_s": 2.6,
            "evaluated_duration_s": 4.2,
        },
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": 0.98332, "duration_s": 0.52}],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_evaluated_duration_increase_ratio=0.1,
        max_evaluated_duration_increase_s=0.1,
    )

    assert check["passed"] is False
    assert check["max_evaluated_duration_increase_ratio"] == 0.1
    assert check["max_evaluated_duration_increase_s"] == 0.1
    assert check["issues"] == [
        {
            "kind": "evaluated_duration_increase",
            "baseline_evaluated_duration_s": 3.0,
            "candidate_evaluated_duration_s": 4.2,
            "increase_s": 1.2,
            "increase_ratio": 0.4,
        }
    ]


def test_report_regression_check_can_flag_evaluated_road_match_increase() -> None:
    baseline = {
        "summary": {
            "average_iou": 0.95,
            "evaluated_road_match_elapsed_s": 0.4,
        },
        "scores": [{"slug": "nashville-waymo", "status": "active", "iou": 0.986282}],
    }
    candidate = {
        "summary": {
            "average_iou": 0.95,
            "evaluated_road_match_elapsed_s": 0.7,
        },
        "scores": [{"slug": "nashville-waymo", "status": "active", "iou": 0.986282}],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_evaluated_road_match_increase_ratio=0.2,
        max_evaluated_road_match_increase_s=0.05,
    )

    assert check["passed"] is False
    assert check["max_evaluated_road_match_increase_ratio"] == 0.2
    assert check["max_evaluated_road_match_increase_s"] == 0.05
    assert check["issues"] == [
        {
            "kind": "evaluated_road_match_elapsed_increase",
            "baseline_evaluated_road_match_elapsed_s": 0.4,
            "candidate_evaluated_road_match_elapsed_s": 0.7,
            "increase_s": 0.3,
            "increase_ratio": 0.75,
        }
    ]


def test_print_table_reports_aggregate_road_match_regression(capsys, tmp_path: Path) -> None:
    report = {
        "mode": "full",
        "summary": {
            "passed": False,
            "passed_fixtures": 1,
            "scored_fixtures": 1,
            "skipped_fixtures": 0,
            "average_iou": 0.95,
            "min_iou": 0.95,
            "total_duration_s": 1.0,
        },
        "scores": [
            {
                "slug": "phoenix-waymo",
                "status": "active",
                "passed": True,
                "iou": 0.95,
                "area_ratio": 1.0,
                "duration_s": 1.0,
                "vertices": 42,
                "style": "bright-blue",
                "georeference_source": "ocr-georeference:nominatim-label-fit+osm-road-refine",
                "error": None,
                "note": None,
            }
        ],
        "inventory": {"references_without_images": []},
        "regression_check": {
            "passed": False,
            "baseline_report": "baseline.json",
            "issues": [
                {
                    "kind": "evaluated_road_match_elapsed_increase",
                    "baseline_evaluated_road_match_elapsed_s": 0.4,
                    "candidate_evaluated_road_match_elapsed_s": 0.7,
                    "increase_s": 0.3,
                    "increase_ratio": 0.75,
                },
                {
                    "kind": "missing_candidate_score",
                    "slug": "nashville-waymo",
                    "baseline_iou": 0.986282,
                },
                {
                    "slug": "phoenix-waymo",
                    "kind": "ocr_label_count_drop",
                    "baseline_ocr_label_count": 74,
                    "candidate_ocr_label_count": 50,
                    "drop": 24,
                    "max_drop": 10,
                },
                {
                    "slug": "phoenix-waymo",
                    "kind": "ocr_top_label_retention_drop",
                    "retention": 0.25,
                    "min_retention": 0.75,
                },
            ],
        },
    }

    benchmark_module.print_table(report, tmp_path / "report.json")

    output = capsys.readouterr().out
    assert "evaluated road-match duration 0.400s -> 0.700s" in output
    assert "nashville-waymo: missing candidate score" in output
    assert "phoenix-waymo: OCR labels 74 -> 50" in output
    assert "phoenix-waymo: OCR top-label retention 0.250 < 0.750" in output


def test_report_regression_check_can_flag_evaluated_stage_duration_increase() -> None:
    baseline = {
        "summary": {
            "average_iou": 0.95,
            "evaluated_stage_duration_s": {"extract": 1.0, "ocr": 2.0},
        },
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": 0.98332}],
    }
    candidate = {
        "summary": {
            "average_iou": 0.95,
            "evaluated_stage_duration_s": {"extract": 1.05, "ocr": 2.5},
        },
        "scores": [{"slug": "phoenix-waymo", "status": "active", "iou": 0.98332}],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_evaluated_stage_duration_increase_ratio=0.1,
        max_evaluated_stage_duration_increase_s=0.1,
    )

    assert check["passed"] is False
    assert check["max_evaluated_stage_duration_increase_ratio"] == 0.1
    assert check["max_evaluated_stage_duration_increase_s"] == 0.1
    assert check["compared_evaluated_stage_durations"] == 2
    assert check["issues"] == [
        {
            "stage": "ocr",
            "kind": "evaluated_stage_duration_increase",
            "baseline_stage_duration_s": 2.0,
            "candidate_stage_duration_s": 2.5,
            "increase_s": 0.5,
            "increase_ratio": 0.25,
        }
    ]


def test_report_regression_check_duration_ratio_can_ignore_small_absolute_noise() -> None:
    baseline = {
        "summary": {
            "average_iou": 0.95,
            "total_duration_s": 4.0,
            "evaluated_duration_s": 5.0,
            "evaluated_stage_duration_s": {"ocr": 5.0},
            "evaluated_road_match_elapsed_s": 0.2,
        },
        "scores": [{"slug": "austin-tesla", "status": "active", "iou": 0.973925, "duration_s": 0.1}],
    }
    candidate = {
        "summary": {
            "average_iou": 0.95,
            "total_duration_s": 4.2,
            "evaluated_duration_s": 5.2,
            "evaluated_stage_duration_s": {"ocr": 5.2},
            "evaluated_road_match_elapsed_s": 0.23,
        },
        "scores": [{"slug": "austin-tesla", "status": "active", "iou": 0.973925, "duration_s": 0.19}],
    }

    check = compare_report_regressions(
        candidate,
        baseline,
        max_duration_increase_ratio=0.25,
        max_duration_increase_s=0.1,
        max_total_duration_increase_ratio=0.01,
        max_total_duration_increase_s=0.25,
        max_evaluated_duration_increase_ratio=0.01,
        max_evaluated_duration_increase_s=0.25,
        max_evaluated_stage_duration_increase_ratio=0.01,
        max_evaluated_stage_duration_increase_s=0.25,
        max_evaluated_road_match_increase_ratio=0.01,
        max_evaluated_road_match_increase_s=0.05,
    )

    assert check["passed"] is True
    assert check["issues"] == []


def test_report_latency_budget_check_flags_absolute_duration_excess() -> None:
    report = {
        "summary": {
            "total_duration_s": 4.2,
            "smoked_skipped_duration_s": 0.9,
            "evaluated_duration_s": 5.1,
        },
        "scores": [
            {"slug": "phoenix-waymo", "status": "active", "duration_s": 1.21},
            {"slug": "miami-waymo", "status": "reference_mismatch", "duration_s": 6.0},
            {"slug": "dallas-tesla", "status": "active", "duration_s": 0.19},
        ],
    }

    check = check_report_latency_budgets(
        report,
        max_duration_s=1.0,
        max_total_duration_s=4.0,
        max_evaluated_duration_s=5.0,
    )

    assert check["passed"] is False
    assert check["max_evaluated_duration_s"] == 5.0
    assert check["issues"] == [
        {
            "slug": "phoenix-waymo",
            "kind": "duration_budget_exceeded",
            "duration_s": 1.21,
            "max_duration_s": 1.0,
            "excess_s": 0.21,
        },
        {
            "kind": "total_duration_budget_exceeded",
            "total_duration_s": 4.2,
            "max_total_duration_s": 4.0,
            "excess_s": 0.2,
        },
        {
            "kind": "evaluated_duration_budget_exceeded",
            "evaluated_duration_s": 5.1,
            "max_evaluated_duration_s": 5.0,
            "excess_s": 0.1,
        },
    ]


def test_report_source_requirement_check_passes_when_sources_match() -> None:
    report = {
        "summary": {},
        "scores": [
            {
                "slug": "dallas-waymo",
                "status": "active",
                "duration_s": 0.08,
                "georeference_source": "catalog-shape-match",
            },
            {
                "slug": "phoenix-waymo",
                "status": "reference_mismatch",
                "duration_s": 0.11,
                "georeference_source": "catalog-shape-match",
            },
        ],
    }

    check = check_report_source_requirements(
        report,
        required_active_georeference_sources=["catalog-shape-match"],
        required_evaluated_georeference_sources=["catalog-shape-match"],
    )

    assert check["passed"] is True
    assert check["active_georeference_sources"] == {"catalog-shape-match": 1}
    assert check["evaluated_georeference_sources"] == {"catalog-shape-match": 2}
    assert check["issues"] == []


def test_report_source_requirement_check_flags_active_and_evaluated_mismatches() -> None:
    report = {
        "summary": {},
        "scores": [
            {
                "slug": "dallas-waymo",
                "status": "active",
                "duration_s": 0.08,
                "georeference_source": "ocr-georeference:nominatim-label-fit",
            },
            {
                "slug": "phoenix-waymo",
                "status": "active",
                "duration_s": 0.11,
                "georeference_source": "catalog-shape-match",
            },
            {
                "slug": "miami-waymo",
                "status": "reference_mismatch",
                "duration_s": 0.22,
                "georeference_source": None,
            },
            {
                "slug": "bay-area-waymo",
                "status": "reference_mismatch",
                "georeference_source": "ocr-georeference:nominatim-label-fit",
            },
        ],
    }

    check = check_report_source_requirements(
        report,
        required_active_georeference_sources=["catalog-shape-match"],
        required_evaluated_georeference_sources=["catalog-shape-match"],
    )

    assert check["passed"] is False
    assert check["active_georeference_sources"] == {
        "catalog-shape-match": 1,
        "ocr-georeference:nominatim-label-fit": 1,
    }
    assert check["evaluated_georeference_sources"] == {
        "<missing>": 1,
        "catalog-shape-match": 1,
        "ocr-georeference:nominatim-label-fit": 1,
    }
    assert check["issues"] == [
        {
            "slug": "dallas-waymo",
            "kind": "active_georeference_source_mismatch",
            "georeference_source": "ocr-georeference:nominatim-label-fit",
            "required_georeference_sources": ["catalog-shape-match"],
        },
        {
            "slug": "dallas-waymo",
            "kind": "evaluated_georeference_source_mismatch",
            "georeference_source": "ocr-georeference:nominatim-label-fit",
            "required_georeference_sources": ["catalog-shape-match"],
        },
        {
            "slug": "miami-waymo",
            "kind": "evaluated_georeference_source_mismatch",
            "georeference_source": "<missing>",
            "required_georeference_sources": ["catalog-shape-match"],
        },
    ]


def test_report_latency_budget_check_passes_when_within_budget() -> None:
    report = {
        "summary": {
            "total_duration_s": 2.9,
            "smoked_skipped_duration_s": 0.4,
            "evaluated_duration_s": 3.3,
        },
        "scores": [{"slug": "dallas-tesla", "status": "active", "duration_s": 0.19}],
    }

    check = check_report_latency_budgets(
        report,
        max_duration_s=1.0,
        max_total_duration_s=3.0,
        max_evaluated_duration_s=3.5,
    )

    assert check["passed"] is True
    assert check["max_evaluated_duration_s"] == 3.5
    assert check["active_total_duration_s"] == 2.9
    assert check["smoked_skipped_duration_s"] == 0.4
    assert check["evaluated_duration_s"] == 3.3
    assert check["issues"] == []


def test_report_latency_budget_check_computes_evaluated_duration_when_missing() -> None:
    report = {
        "summary": {
            "total_duration_s": 2.5,
            "smoked_skipped_duration_s": 0.8,
        },
        "scores": [{"slug": "orlando-waymo", "status": "active", "duration_s": 0.24}],
    }

    check = check_report_latency_budgets(report, max_evaluated_duration_s=3.0)

    assert check["passed"] is False
    assert check["evaluated_duration_s"] == 3.3
    assert check["issues"] == [
        {
            "kind": "evaluated_duration_budget_exceeded",
            "evaluated_duration_s": 3.3,
            "max_evaluated_duration_s": 3.0,
            "excess_s": 0.3,
        }
    ]


def test_report_latency_budget_check_flags_evaluated_stage_excess_and_missing() -> None:
    report = {
        "summary": {
            "total_duration_s": 2.5,
            "evaluated_stage_duration_s": {
                "extract": 0.8,
                "ocr": 3.25,
            },
        },
        "scores": [{"slug": "phoenix-waymo", "status": "active", "duration_s": 0.7}],
    }

    check = check_report_latency_budgets(
        report,
        max_evaluated_stage_duration_s={
            "extract": 1.0,
            "georeference": 0.5,
            "ocr": 3.0,
        },
    )

    assert check["passed"] is False
    assert check["max_evaluated_stage_duration_s"] == {
        "extract": 1.0,
        "georeference": 0.5,
        "ocr": 3.0,
    }
    assert check["evaluated_stage_duration_s"] == {"extract": 0.8, "ocr": 3.25}
    assert check["issues"] == [
        {
            "stage": "georeference",
            "kind": "evaluated_stage_duration_missing",
            "max_evaluated_stage_duration_s": 0.5,
        },
        {
            "stage": "ocr",
            "kind": "evaluated_stage_duration_budget_exceeded",
            "evaluated_stage_duration_s": 3.25,
            "max_evaluated_stage_duration_s": 3.0,
            "excess_s": 0.25,
        },
    ]


def test_report_latency_budget_check_flags_road_match_excess() -> None:
    report = {
        "summary": {
            "total_duration_s": 2.5,
            "active_road_match_elapsed_s": 0.35,
            "smoked_skipped_road_match_elapsed_s": 0.25,
        },
        "scores": [{"slug": "nashville-waymo", "status": "active", "duration_s": 0.7}],
    }

    check = check_report_latency_budgets(report, max_evaluated_road_match_s=0.5)

    assert check["passed"] is False
    assert check["max_evaluated_road_match_s"] == 0.5
    assert check["evaluated_road_match_elapsed_s"] == 0.6
    assert check["issues"] == [
        {
            "kind": "evaluated_road_match_budget_exceeded",
            "evaluated_road_match_elapsed_s": 0.6,
            "max_evaluated_road_match_s": 0.5,
            "excess_s": 0.1,
        }
    ]


def test_report_latency_budget_check_passes_ocr_engine_profile_budgets() -> None:
    report = {
        "summary": {
            "total_duration_s": 5.5,
            "evaluated_ocr_engine_profile": {
                "det_elapsed_s": 2.76,
                "rec_elapsed_s": 1.24,
                "profiled_total_s": 4.05,
            },
        },
        "scores": [{"slug": "phoenix-waymo", "status": "active", "duration_s": 0.9}],
    }

    check = check_report_latency_budgets(
        report,
        max_evaluated_ocr_engine_duration_s={
            "det_elapsed_s": 3.0,
            "rec_elapsed_s": 1.5,
        },
    )

    assert check["passed"] is True
    assert check["max_evaluated_ocr_engine_duration_s"] == {
        "det_elapsed_s": 3.0,
        "rec_elapsed_s": 1.5,
    }
    assert check["evaluated_ocr_engine_duration_s"] == {
        "det_elapsed_s": 2.76,
        "rec_elapsed_s": 1.24,
        "profiled_total_s": 4.05,
    }
    assert check["issues"] == []


def test_report_latency_budget_check_flags_ocr_engine_excess_and_missing_metric() -> None:
    report = {
        "summary": {
            "total_duration_s": 5.5,
            "evaluated_ocr_engine_profile": {
                "det_elapsed_s": 3.25,
            },
        },
        "scores": [{"slug": "dallas-waymo", "status": "active", "duration_s": 1.1}],
    }

    check = check_report_latency_budgets(
        report,
        max_evaluated_ocr_engine_duration_s={
            "det_elapsed_s": 3.0,
            "rec_elapsed_s": 1.5,
        },
    )

    assert check["passed"] is False
    assert check["evaluated_ocr_engine_duration_s"] == {"det_elapsed_s": 3.25}
    assert check["issues"] == [
        {
            "metric": "det_elapsed_s",
            "kind": "evaluated_ocr_engine_duration_budget_exceeded",
            "evaluated_ocr_engine_duration_s": 3.25,
            "max_evaluated_ocr_engine_duration_s": 3.0,
            "excess_s": 0.25,
        },
        {
            "metric": "rec_elapsed_s",
            "kind": "evaluated_ocr_engine_duration_missing",
            "max_evaluated_ocr_engine_duration_s": 1.5,
        },
    ]


def test_report_latency_budget_check_flags_missing_ocr_engine_profile() -> None:
    report = {
        "summary": {"total_duration_s": 5.5},
        "scores": [{"slug": "orlando-waymo", "status": "active", "duration_s": 0.6}],
    }

    check = check_report_latency_budgets(
        report,
        max_evaluated_ocr_engine_duration_s={"det_elapsed_s": 3.0},
    )

    assert check["passed"] is False
    assert check["evaluated_ocr_engine_duration_s"] == {}
    assert check["issues"] == [
        {
            "metric": "det_elapsed_s",
            "kind": "evaluated_ocr_engine_profile_missing",
            "max_evaluated_ocr_engine_duration_s": 3.0,
        }
    ]


def test_report_latency_budget_check_passes_ocr_engine_count_budgets() -> None:
    report = {
        "summary": {
            "total_duration_s": 5.5,
            "evaluated_ocr_engine_profile": {
                "raw_box_count": 42,
                "selected_box_count": 28,
                "label_count": 19,
            },
        },
        "scores": [{"slug": "phoenix-waymo", "status": "active", "duration_s": 0.9}],
    }

    check = check_report_latency_budgets(
        report,
        max_evaluated_ocr_engine_count={
            "raw_box_count": 50,
            "selected_box_count": 30,
        },
    )

    assert check["passed"] is True
    assert check["max_evaluated_ocr_engine_count"] == {
        "raw_box_count": 50.0,
        "selected_box_count": 30.0,
    }
    assert check["evaluated_ocr_engine_count"] == {
        "raw_box_count": 42,
        "selected_box_count": 28,
        "label_count": 19,
    }
    assert check["issues"] == []


def test_report_latency_budget_check_flags_ocr_engine_count_excess_and_missing_metric() -> None:
    report = {
        "summary": {
            "total_duration_s": 5.5,
            "evaluated_ocr_engine_profile": {
                "selected_box_count": 31,
            },
        },
        "scores": [{"slug": "dallas-waymo", "status": "active", "duration_s": 1.1}],
    }

    check = check_report_latency_budgets(
        report,
        max_evaluated_ocr_engine_count={
            "raw_box_count": 50,
            "selected_box_count": 30,
        },
    )

    assert check["passed"] is False
    assert check["evaluated_ocr_engine_count"] == {"selected_box_count": 31}
    assert check["issues"] == [
        {
            "metric": "raw_box_count",
            "kind": "evaluated_ocr_engine_count_missing",
            "max_evaluated_ocr_engine_count": 50.0,
        },
        {
            "metric": "selected_box_count",
            "kind": "evaluated_ocr_engine_count_budget_exceeded",
            "evaluated_ocr_engine_count": 31,
            "max_evaluated_ocr_engine_count": 30.0,
            "excess_count": 1.0,
        },
    ]


def test_report_latency_budget_check_flags_missing_ocr_engine_profile_for_count_budget() -> None:
    report = {
        "summary": {"total_duration_s": 5.5},
        "scores": [{"slug": "orlando-waymo", "status": "active", "duration_s": 0.6}],
    }

    check = check_report_latency_budgets(
        report,
        max_evaluated_ocr_engine_count={"selected_box_count": 30},
    )

    assert check["passed"] is False
    assert check["evaluated_ocr_engine_count"] == {}
    assert check["issues"] == [
        {
            "metric": "selected_box_count",
            "kind": "evaluated_ocr_engine_profile_missing",
            "max_evaluated_ocr_engine_count": 30.0,
        }
    ]


def test_print_table_reports_ocr_engine_latency_budget_failure(capsys, tmp_path: Path) -> None:
    report = {
        "mode": "full",
        "summary": {
            "passed": False,
            "passed_fixtures": 1,
            "scored_fixtures": 1,
            "skipped_fixtures": 0,
            "average_iou": 0.95,
            "min_iou": 0.95,
            "total_duration_s": 5.5,
        },
        "scores": [
            {
                "slug": "dallas-waymo",
                "status": "active",
                "passed": True,
                "iou": 0.95,
                "area_ratio": 1.0,
                "duration_s": 1.1,
                "vertices": 42,
                "style": "bright-blue",
                "georeference_source": "ocr-georeference:nominatim-label-fit",
                "error": None,
                "note": None,
            }
        ],
        "inventory": {"references_without_images": []},
        "latency_budget_check": {
            "passed": False,
            "issues": [
                {
                    "metric": "det_elapsed_s",
                    "kind": "evaluated_ocr_engine_duration_budget_exceeded",
                    "evaluated_ocr_engine_duration_s": 3.25,
                    "max_evaluated_ocr_engine_duration_s": 3.0,
                    "excess_s": 0.25,
                },
                {
                    "metric": "rec_elapsed_s",
                    "kind": "evaluated_ocr_engine_duration_missing",
                    "max_evaluated_ocr_engine_duration_s": 1.5,
                },
                {
                    "metric": "selected_box_count",
                    "kind": "evaluated_ocr_engine_count_budget_exceeded",
                    "evaluated_ocr_engine_count": 31,
                    "max_evaluated_ocr_engine_count": 30.0,
                    "excess_count": 1.0,
                },
                {
                    "metric": "raw_box_count",
                    "kind": "evaluated_ocr_engine_count_missing",
                    "max_evaluated_ocr_engine_count": 50.0,
                },
                {
                    "metric": "total_s",
                    "kind": "repeat_profile_ocr_engine_p95_duration_budget_exceeded",
                    "repeat_profile_ocr_engine_p95_duration_s": 0.71,
                    "max_repeat_profile_ocr_engine_p95_duration_s": 0.65,
                    "excess_s": 0.06,
                },
                {
                    "metric": "det_elapsed_s",
                    "kind": "repeat_profile_ocr_engine_p95_duration_missing",
                    "max_repeat_profile_ocr_engine_p95_duration_s": 0.3,
                },
                {
                    "metric": "selected_box_count",
                    "kind": "repeat_profile_ocr_engine_p95_count_budget_exceeded",
                    "repeat_profile_ocr_engine_p95_count": 31.0,
                    "max_repeat_profile_ocr_engine_p95_count": 30.0,
                    "excess_count": 1.0,
                },
                {
                    "metric": "raw_box_count",
                    "kind": "repeat_profile_ocr_engine_p95_count_missing",
                    "max_repeat_profile_ocr_engine_p95_count": 50.0,
                },
            ],
        },
    }

    benchmark_module.print_table(report, tmp_path / "report.json")

    output = capsys.readouterr().out
    assert "det_elapsed_s: evaluated OCR engine duration 3.250s > budget 3.000s" in output
    assert "rec_elapsed_s: missing evaluated OCR engine duration" in output
    assert "selected_box_count: evaluated OCR engine count 31.0 > budget 30.0" in output
    assert "raw_box_count: missing evaluated OCR engine count" in output
    assert "total_s: repeat OCR engine p95 duration 0.710s > budget 0.650s" in output
    assert "det_elapsed_s: missing repeat OCR engine p95 duration" in output
    assert "selected_box_count: repeat OCR engine p95 count 31.0 > budget 30.0" in output
    assert "raw_box_count: missing repeat OCR engine p95 count" in output


def test_print_table_reports_georeference_source_requirement_failure(
    capsys,
    tmp_path: Path,
) -> None:
    report = {
        "mode": "full",
        "summary": {
            "passed": False,
            "passed_fixtures": 1,
            "scored_fixtures": 1,
            "skipped_fixtures": 0,
            "average_iou": 0.95,
            "min_iou": 0.95,
            "total_duration_s": 0.6,
            "active_georeference_sources": {
                "ocr-georeference:nominatim-label-fit": 1,
            },
        },
        "scores": [
            {
                "slug": "dallas-waymo",
                "status": "active",
                "passed": True,
                "iou": 0.95,
                "area_ratio": 1.0,
                "duration_s": 0.6,
                "vertices": 42,
                "style": "bright-blue",
                "georeference_source": "ocr-georeference:nominatim-label-fit",
                "error": None,
                "note": None,
            }
        ],
        "inventory": {"references_without_images": []},
        "source_requirement_check": {
            "passed": False,
            "issues": [
                {
                    "slug": "dallas-waymo",
                    "kind": "active_georeference_source_mismatch",
                    "georeference_source": "ocr-georeference:nominatim-label-fit",
                    "required_georeference_sources": ["catalog-shape-match"],
                },
            ],
        },
    }

    benchmark_module.print_table(report, tmp_path / "report.json")

    output = capsys.readouterr().out
    assert "active sources: ocr-georeference:nominatim-label-fit=1" in output
    assert "FAIL source requirement: 1 issues" in output
    assert (
        "dallas-waymo: active source ocr-georeference:nominatim-label-fit "
        "not in [catalog-shape-match]"
    ) in output


def test_print_table_reports_stage_and_ocr_tail_summaries(
    capsys,
    tmp_path: Path,
) -> None:
    report = {
        "mode": "full",
        "summary": {
            "passed": True,
            "passed_fixtures": 1,
            "scored_fixtures": 1,
            "skipped_fixtures": 1,
            "smoked_skipped_fixtures": 1,
            "failed_smoked_skipped_fixtures": 0,
            "average_iou": 0.95,
            "min_iou": 0.95,
            "total_duration_s": 0.95,
            "smoked_skipped_duration_s": 1.2,
            "evaluated_duration_s": 2.15,
            "active_stage_max_rows": {
                "extract": {"slug": "dallas-waymo", "duration_s": 0.282593},
                "ocr": {"slug": "phoenix-waymo", "duration_s": 0.690724},
            },
            "smoked_skipped_stage_max_rows": {
                "ocr": {"slug": "bay-area-waymo", "duration_s": 1.2},
            },
            "evaluated_stage_max_rows": {
                "extract": {"slug": "dallas-waymo", "duration_s": 0.282593},
                "ocr": {"slug": "bay-area-waymo", "duration_s": 1.2},
            },
            "active_ocr_label_event_counts": {
                "Full-detail map labels read": 1,
                "Map labels read": 3,
            },
            "active_ocr_full_detail_retry_count": 1,
            "active_ocr_full_detail_retry_rows": ["phoenix-waymo"],
            "smoked_skipped_ocr_label_event_counts": {"Map labels read": 1},
            "smoked_skipped_ocr_full_detail_retry_count": 0,
            "smoked_skipped_ocr_full_detail_retry_rows": [],
            "evaluated_ocr_label_event_counts": {
                "Full-detail map labels read": 1,
                "Map labels read": 4,
            },
            "evaluated_ocr_full_detail_retry_count": 1,
            "evaluated_ocr_full_detail_retry_rows": ["phoenix-waymo"],
        },
        "scores": [
            {
                "slug": "phoenix-waymo",
                "status": "active",
                "passed": True,
                "iou": 0.95,
                "area_ratio": 1.0,
                "duration_s": 0.95,
                "vertices": 42,
                "style": "bright-blue",
                "georeference_source": "ocr-georeference:nominatim-label-fit",
                "error": None,
                "note": None,
            },
            {
                "slug": "bay-area-waymo",
                "status": "reference_mismatch",
                "passed": True,
                "iou": None,
                "area_ratio": None,
                "duration_s": 1.2,
                "vertices": 42,
                "style": "bright-blue",
                "georeference_source": "ocr-georeference:nominatim-label-fit",
                "error": None,
                "note": "data debt",
            },
        ],
        "inventory": {"references_without_images": []},
    }

    benchmark_module.print_table(report, tmp_path / "report.json")

    output = capsys.readouterr().out
    assert "active stage max: ocr=0.69s@phoenix-waymo, extract=0.28s@dallas-waymo" in output
    assert "smoked skipped stage max: ocr=1.20s@bay-area-waymo" in output
    assert "evaluated stage max: ocr=1.20s@bay-area-waymo, extract=0.28s@dallas-waymo" in output
    assert "active OCR events: Full-detail map labels read=1, Map labels read=3" in output
    assert "active OCR full-detail retries: 1 (phoenix-waymo)" in output
    assert "smoked skipped OCR events: Map labels read=1" in output
    assert "smoked skipped OCR full-detail retries: 0" in output
    assert "evaluated OCR events: Full-detail map labels read=1, Map labels read=4" in output
    assert "evaluated OCR full-detail retries: 1 (phoenix-waymo)" in output


def test_print_table_warns_when_ocr_engine_profiling_affects_durations(
    capsys,
    tmp_path: Path,
) -> None:
    report = {
        "mode": "full",
        "thresholds": {"profile_ocr_engine": True},
        "summary": {
            "passed": True,
            "passed_fixtures": 1,
            "scored_fixtures": 1,
            "skipped_fixtures": 0,
            "average_iou": 0.96,
            "min_iou": 0.96,
            "total_duration_s": 1.1,
        },
        "scores": [
            {
                "slug": "dallas-waymo",
                "status": "active",
                "passed": True,
                "iou": 0.96,
                "area_ratio": 1.0,
                "duration_s": 1.1,
                "vertices": 42,
                "style": "bright-blue",
                "georeference_source": "ocr-georeference:nominatim-label-fit",
                "error": None,
                "note": None,
            }
        ],
        "inventory": {"references_without_images": []},
    }

    benchmark_module.print_table(report, tmp_path / "report.json")

    output = capsys.readouterr().out
    assert "OCR engine profiling is enabled; fixture durations include profiling overhead" in output


def test_report_latency_budget_check_passes_repeat_profile_budgets() -> None:
    report = {
        "summary": {"total_duration_s": 2.5},
        "scores": [{"slug": "nashville-waymo", "status": "active", "duration_s": 0.7}],
        "repeat_profile": {
            "summary": {
                "analyzed_samples": 4,
                "passed_samples": 4,
                "subsecond_samples": 3,
                "unstable_signature_fixtures": [],
                "max_duration_s": 0.98,
                "median_duration_s": 0.7,
                "p95_duration_s": 0.94,
                "stage_duration_s": {
                    "ocr": {"max_duration_s": 0.72},
                },
                "ocr_engine_stage_duration_s": {
                    "det_elapsed_s": {"p95_duration_s": 0.25},
                    "total_s": {"p95_duration_s": 0.48},
                },
                "ocr_engine_count_metric": {
                    "selected_box_count": {"p95_count": 28.0},
                },
            }
        },
    }

    check = check_report_latency_budgets(
        report,
        max_repeat_profile_duration_s=1.0,
        max_repeat_profile_median_duration_s=0.8,
        max_repeat_profile_p95_duration_s=1.0,
        max_repeat_profile_stage_duration_s={"ocr": 0.8},
        max_repeat_profile_ocr_engine_p95_duration_s={"total_s": 0.5},
        max_repeat_profile_ocr_engine_p95_count={"selected_box_count": 30},
        min_repeat_profile_pass_ratio=1.0,
        min_repeat_profile_subsecond_ratio=0.75,
        fail_on_repeat_profile_signature_drift=True,
    )

    assert check["passed"] is True
    assert check["repeat_profile_analyzed_samples"] == 4
    assert check["repeat_profile_max_duration_s"] == 0.98
    assert check["repeat_profile_median_duration_s"] == 0.7
    assert check["repeat_profile_p95_duration_s"] == 0.94
    assert check["repeat_profile_stage_duration_s"] == {"ocr": 0.72}
    assert check["repeat_profile_ocr_engine_p95_duration_s"] == {
        "det_elapsed_s": 0.25,
        "total_s": 0.48,
    }
    assert check["repeat_profile_ocr_engine_p95_count"] == {"selected_box_count": 28.0}
    assert check["repeat_profile_pass_ratio"] == 1.0
    assert check["repeat_profile_subsecond_ratio"] == 0.75
    assert check["repeat_profile_signature_drift_fixtures"] == []
    assert check["issues"] == []


def test_report_latency_budget_check_flags_repeat_profile_budget_failures() -> None:
    report = {
        "summary": {"total_duration_s": 2.5},
        "scores": [{"slug": "nashville-waymo", "status": "active", "duration_s": 0.7}],
        "repeat_profile": {
            "summary": {
                "analyzed_samples": 4,
                "passed_samples": 3,
                "subsecond_samples": 2,
                "max_duration_s": 1.2,
                "median_duration_s": 0.9,
                "p95_duration_s": 1.15,
            }
        },
    }

    check = check_report_latency_budgets(
        report,
        max_repeat_profile_duration_s=1.0,
        max_repeat_profile_median_duration_s=0.8,
        max_repeat_profile_p95_duration_s=1.0,
        min_repeat_profile_pass_ratio=1.0,
        min_repeat_profile_subsecond_ratio=0.75,
    )

    assert check["passed"] is False
    assert check["repeat_profile_pass_ratio"] == 0.75
    assert check["repeat_profile_subsecond_ratio"] == 0.5
    assert check["issues"] == [
        {
            "kind": "repeat_profile_duration_budget_exceeded",
            "repeat_profile_max_duration_s": 1.2,
            "max_repeat_profile_duration_s": 1.0,
            "excess_s": 0.2,
        },
        {
            "kind": "repeat_profile_median_duration_budget_exceeded",
            "repeat_profile_median_duration_s": 0.9,
            "max_repeat_profile_median_duration_s": 0.8,
            "excess_s": 0.1,
        },
        {
            "kind": "repeat_profile_p95_duration_budget_exceeded",
            "repeat_profile_p95_duration_s": 1.15,
            "max_repeat_profile_p95_duration_s": 1.0,
            "excess_s": 0.15,
        },
        {
            "kind": "repeat_profile_pass_ratio_below_min",
            "repeat_profile_pass_ratio": 0.75,
            "min_repeat_profile_pass_ratio": 1.0,
            "shortfall": 0.25,
        },
        {
            "kind": "repeat_profile_subsecond_ratio_below_min",
            "repeat_profile_subsecond_ratio": 0.5,
            "min_repeat_profile_subsecond_ratio": 0.75,
            "shortfall": 0.25,
        },
    ]


def test_report_latency_budget_check_flags_repeat_profile_signature_drift() -> None:
    report = {
        "summary": {"total_duration_s": 2.5},
        "scores": [{"slug": "phoenix-waymo", "status": "active", "duration_s": 0.7}],
        "repeat_profile": {
            "summary": {
                "analyzed_samples": 4,
                "passed_samples": 4,
                "subsecond_samples": 4,
                "unstable_signature_fixtures": ["phoenix-waymo"],
            }
        },
    }

    check = check_report_latency_budgets(
        report,
        fail_on_repeat_profile_signature_drift=True,
    )

    assert check["passed"] is False
    assert check["fail_on_repeat_profile_signature_drift"] is True
    assert check["repeat_profile_signature_drift_fixtures"] == ["phoenix-waymo"]
    assert check["issues"] == [
        {
            "kind": "repeat_profile_signature_drift",
            "unstable_signature_fixtures": ["phoenix-waymo"],
        }
    ]


def test_report_latency_budget_check_flags_repeat_profile_stage_budget_failures() -> None:
    report = {
        "summary": {"total_duration_s": 2.5},
        "scores": [{"slug": "nashville-waymo", "status": "active", "duration_s": 0.7}],
        "repeat_profile": {
            "summary": {
                "analyzed_samples": 4,
                "stage_duration_s": {
                    "ocr": {"max_duration_s": 1.2},
                },
            }
        },
    }

    check = check_report_latency_budgets(
        report,
        max_repeat_profile_stage_duration_s={"extract": 0.2, "ocr": 1.0},
    )

    assert check["passed"] is False
    assert check["max_repeat_profile_stage_duration_s"] == {"extract": 0.2, "ocr": 1.0}
    assert check["repeat_profile_stage_duration_s"] == {"ocr": 1.2}
    assert check["issues"] == [
        {
            "stage": "extract",
            "kind": "repeat_profile_stage_duration_missing",
            "max_repeat_profile_stage_duration_s": 0.2,
        },
        {
            "stage": "ocr",
            "kind": "repeat_profile_stage_duration_budget_exceeded",
            "repeat_profile_stage_duration_s": 1.2,
            "max_repeat_profile_stage_duration_s": 1.0,
            "excess_s": 0.2,
        },
    ]


def test_report_latency_budget_check_flags_repeat_profile_ocr_engine_budget_failures() -> None:
    report = {
        "summary": {"total_duration_s": 2.5},
        "scores": [{"slug": "los-angeles-waymo", "status": "active", "duration_s": 0.7}],
        "repeat_profile": {
            "summary": {
                "analyzed_samples": 4,
                "ocr_engine_stage_duration_s": {
                    "total_s": {"p95_duration_s": 0.71},
                },
                "ocr_engine_count_metric": {
                    "selected_box_count": {"p95_count": 31.0},
                },
            }
        },
    }

    check = check_report_latency_budgets(
        report,
        max_repeat_profile_ocr_engine_p95_duration_s={
            "det_elapsed_s": 0.3,
            "total_s": 0.65,
        },
        max_repeat_profile_ocr_engine_p95_count={
            "raw_box_count": 50,
            "selected_box_count": 30,
        },
    )

    assert check["passed"] is False
    assert check["max_repeat_profile_ocr_engine_p95_duration_s"] == {
        "det_elapsed_s": 0.3,
        "total_s": 0.65,
    }
    assert check["max_repeat_profile_ocr_engine_p95_count"] == {
        "raw_box_count": 50.0,
        "selected_box_count": 30.0,
    }
    assert check["repeat_profile_ocr_engine_p95_duration_s"] == {"total_s": 0.71}
    assert check["repeat_profile_ocr_engine_p95_count"] == {"selected_box_count": 31.0}
    assert check["issues"] == [
        {
            "metric": "det_elapsed_s",
            "kind": "repeat_profile_ocr_engine_p95_duration_missing",
            "max_repeat_profile_ocr_engine_p95_duration_s": 0.3,
        },
        {
            "metric": "total_s",
            "kind": "repeat_profile_ocr_engine_p95_duration_budget_exceeded",
            "repeat_profile_ocr_engine_p95_duration_s": 0.71,
            "max_repeat_profile_ocr_engine_p95_duration_s": 0.65,
            "excess_s": 0.06,
        },
        {
            "metric": "raw_box_count",
            "kind": "repeat_profile_ocr_engine_p95_count_missing",
            "max_repeat_profile_ocr_engine_p95_count": 50.0,
        },
        {
            "metric": "selected_box_count",
            "kind": "repeat_profile_ocr_engine_p95_count_budget_exceeded",
            "repeat_profile_ocr_engine_p95_count": 31.0,
            "max_repeat_profile_ocr_engine_p95_count": 30.0,
            "excess_count": 1.0,
        },
    ]


def test_report_latency_budget_check_flags_missing_repeat_profile() -> None:
    check = check_report_latency_budgets(
        {"summary": {"total_duration_s": 2.5}, "scores": []},
        max_repeat_profile_stage_duration_s={"ocr": 1.0},
    )

    assert check["passed"] is False
    assert check["repeat_profile_analyzed_samples"] == 0
    assert check["repeat_profile_max_duration_s"] is None
    assert check["issues"] == [{"kind": "repeat_profile_missing"}]


def test_report_latency_budget_check_flags_incomplete_repeat_profile_metrics() -> None:
    report = {
        "summary": {"total_duration_s": 2.5},
        "scores": [],
        "repeat_profile": {
            "summary": {
                "analyzed_samples": 2,
            }
        },
    }

    check = check_report_latency_budgets(
        report,
        max_repeat_profile_duration_s=1.0,
        max_repeat_profile_median_duration_s=0.8,
        max_repeat_profile_p95_duration_s=1.0,
        max_repeat_profile_stage_duration_s={"ocr": 0.8},
        min_repeat_profile_pass_ratio=1.0,
        min_repeat_profile_subsecond_ratio=0.75,
    )

    assert check["passed"] is False
    assert check["issues"] == [
        {
            "kind": "repeat_profile_duration_missing",
            "max_repeat_profile_duration_s": 1.0,
        },
        {
            "kind": "repeat_profile_median_duration_missing",
            "max_repeat_profile_median_duration_s": 0.8,
        },
        {
            "kind": "repeat_profile_p95_duration_missing",
            "max_repeat_profile_p95_duration_s": 1.0,
        },
        {
            "stage": "ocr",
            "kind": "repeat_profile_stage_duration_missing",
            "max_repeat_profile_stage_duration_s": 0.8,
        },
        {
            "kind": "repeat_profile_pass_ratio_missing",
            "min_repeat_profile_pass_ratio": 1.0,
        },
        {
            "kind": "repeat_profile_subsecond_ratio_missing",
            "min_repeat_profile_subsecond_ratio": 0.75,
        },
    ]


def test_subprocess_full_fixture_preserves_cli_failure_profile(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Phoenix.png"
    reference_path = tmp_path / "phoenix-waymo.json"
    image_path.write_bytes(b"unused by patched subprocess")
    reference_path.write_text("{}\n")
    fixture = BenchmarkFixture(
        slug="phoenix-waymo",
        provider="waymo",
        area="Phoenix",
        image_path=image_path,
        reference_path=reference_path,
    )
    seen_commands = []

    def fake_run(command, *, text, capture_output, timeout, check):
        seen_commands.append(command)
        return SimpleNamespace(
            returncode=1,
            stdout=json.dumps(
                {
                    "status": "failed",
                    "error": "could not infer a reliable map location",
                    "event_profile": {
                        "stage_elapsed_s": {"inspect": 0.01, "ocr": 0.2},
                        "events": [
                            {
                                "stage": "ocr",
                                "message": "Map labels read",
                                "details": {
                                    "label_count": 8,
                                    "top_labels": ["Phoenix", "Tempe"],
                                },
                            }
                        ],
                    },
                }
            ),
            stderr="map-boundary-builder: error: could not infer a reliable map location\n",
        )

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)

    score = benchmark_module.score_full_fixture(
        fixture,
        out_dir=tmp_path / "out",
        min_iou=0.99,
        timeout_seconds=30,
        city_overrides=False,
        no_catalog=True,
        catalog_probe_missed=True,
        execution="subprocess",
        debug_artifacts=False,
        neutral_filename_hint=True,
    )

    assert "--print-summary" in seen_commands[0]
    assert "--profile-events" in seen_commands[0]
    assert "--catalog-probe-missed" in seen_commands[0]
    assert seen_commands[0][-2:] == ["--filename-hint", "uploaded-map.png"]
    assert score.passed is False
    assert score.error == "could not infer a reliable map location"
    assert score.stage_elapsed_s == {"inspect": 0.01, "ocr": 0.2}
    assert score.ocr_label_count == 8
    assert score.ocr_top_labels == ["Phoenix", "Tempe"]
    assert score.ocr_label_event == "Map labels read"
    assert score.ocr_label_events == [
        {"message": "Map labels read", "label_count": 8, "top_labels": ["Phoenix", "Tempe"]}
    ]
    assert score.ocr_full_detail_retry is False


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
                "catalog_probe_missed": options.catalog_probe_missed,
                "filename_hint": options.filename_hint,
                "write_mask_artifact": options.write_mask_artifact,
            }
        )
        progress(
            {"stage": "inspect", "message": "Reading image metadata", "percent": 5, "status": "running"}
        )
        progress(
            {
                "stage": "ocr",
                "message": "Map labels read",
                "percent": 47,
                "status": "running",
                "details": {
                    "label_count": 10,
                    "top_labels": ["Phoenix", "Scottsdale", "Tempe"],
                },
            }
        )
        progress(
            {
                "stage": "ocr",
                "message": "Full-detail map labels read",
                "percent": 48,
                "status": "running",
                "details": {
                    "label_count": 16,
                    "top_labels": ["Phoenix", "Scottsdale", "Tempe", "Mesa"],
                },
            }
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
                "road_match_score": 0.681518,
                "road_match_elapsed_s": 0.195375,
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
        catalog_probe_missed=True,
        debug_artifacts=False,
    )

    assert score.passed is True
    assert score.iou == 1.0
    assert score.georeference_source == "ocr-georeference:nominatim-label-fit"
    assert score.road_match_score == 0.681518
    assert score.road_match_elapsed_s == 0.195375
    assert score.ocr_label_count == 16
    assert score.ocr_top_labels == ["Phoenix", "Scottsdale", "Tempe", "Mesa"]
    assert score.ocr_label_event == "Full-detail map labels read"
    assert score.ocr_label_events == [
        {"message": "Map labels read", "label_count": 10, "top_labels": ["Phoenix", "Scottsdale", "Tempe"]},
        {
            "message": "Full-detail map labels read",
            "label_count": 16,
            "top_labels": ["Phoenix", "Scottsdale", "Tempe", "Mesa"],
        },
    ]
    assert score.ocr_full_detail_retry is True
    assert calls == [
        {
            "image": image_path,
            "city": "Phoenix",
            "debug_dir": None,
            "allow_catalog": False,
            "catalog_probe_missed": True,
            "filename_hint": image_path.name,
            "write_mask_artifact": False,
        }
    ]


def test_in_process_full_fixture_can_record_ocr_engine_profile(tmp_path: Path, monkeypatch) -> None:
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

    class FakeRapidOcrProfileContext:
        def __enter__(self):
            self.events = [
                {
                    "det_elapsed_s": 0.32,
                    "rec_elapsed_s": 0.21,
                    "raw_box_count": 30,
                    "selected_box_count": 18,
                    "label_count": 16,
                }
            ]
            return self.events

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_collect_rapidocr_profiles():
        return FakeRapidOcrProfileContext()

    def fake_build_boundary(_image, _city, _output_path, *, debug_dir, options, progress):
        progress(
            {
                "stage": "ocr",
                "message": "Map labels read",
                "percent": 47,
                "status": "running",
                "details": {"label_count": 16, "top_labels": ["Phoenix"]},
            }
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

    monkeypatch.setattr("map_boundary_builder.ocr.collect_rapidocr_profiles", fake_collect_rapidocr_profiles)
    monkeypatch.setattr("map_boundary_builder.runner.build_boundary", fake_build_boundary)

    score = score_full_fixture_in_process(
        fixture,
        output_path=tmp_path / "out" / "boundary.geojson",
        debug_dir=None,
        min_iou=0.99,
        city_overrides=False,
        no_catalog=True,
        catalog_probe_missed=False,
        debug_artifacts=False,
        profile_ocr_engine=True,
    )

    assert score.passed is True
    assert score.ocr_engine_profile == {
        "calls": 1,
        "det_elapsed_s": 0.32,
        "rec_elapsed_s": 0.21,
        "raw_box_count": 30,
        "selected_box_count": 18,
        "label_count": 16,
        "calls_detail": [
            {
                "det_elapsed_s": 0.32,
                "rec_elapsed_s": 0.21,
                "raw_box_count": 30,
                "selected_box_count": 18,
                "label_count": 16,
            }
        ],
    }


def test_in_process_full_fixture_can_use_neutral_filename_hint(tmp_path: Path, monkeypatch) -> None:
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
    filename_hints = []

    def fake_build_boundary(_image, _city, _output_path, *, debug_dir, options, progress):
        filename_hints.append(options.filename_hint)
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
        city_overrides=False,
        no_catalog=True,
        catalog_probe_missed=False,
        debug_artifacts=False,
        neutral_filename_hint=True,
    )

    assert score.passed is True
    assert filename_hints == ["uploaded-map.png"]
