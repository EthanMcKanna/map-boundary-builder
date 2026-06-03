import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import map_boundary_builder.stress_benchmark as stress_module
from map_boundary_builder.extract import EXTRACTION_CACHE_ENV
from map_boundary_builder.runner import RUNNER_OCR_CACHE_ENV


def test_run_stress_case_records_success_summary(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    out_dir = tmp_path / "out"

    def fake_run(command, *, text, capture_output, timeout, check):
        assert "--no-catalog" in command
        assert command[command.index("--filename-hint") + 1] == "upload.png"
        assert text is True
        assert capture_output is True
        assert timeout == 7
        assert check is False
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "city": "Dallas",
                    "style": "purple-fill",
                    "georeference_source": "ocr-georeference:nominatim-label-fit",
                    "combined_confidence": 0.91,
                    "georeference_confidence": 0.88,
                    "control_points": 5,
                    "road_match_score": 0.706233,
                    "road_match_base_score": 0.506854,
                    "road_match_sampled_points": 1348,
                    "road_match_elapsed_s": 0.22955,
                    "bbox": [-97, 32, -96, 33],
                    "pipeline_version": "pipeline-test",
                    "event_profile": {
                        "total_elapsed_s": 0.612345,
                        "stage_elapsed_s": {"ocr": 0.4},
                        "events": [
                            {
                                "stage": "extract",
                                "message": "Extracting service-area pixels",
                                "details": {"width": 1200, "height": 900},
                            },
                            {
                                "stage": "extract",
                                "message": "Pixel polygon extracted",
                                "details": {
                                    "style": "purple-fill",
                                    "coverage_ratio": 0.123456,
                                    "contour_count": 2,
                                },
                            },
                            {
                                "stage": "ocr",
                                "message": "Map labels read",
                                "details": {"label_count": 8, "top_labels": ["Dallas", "Deep Ellum"]},
                            },
                        ],
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(stress_module.subprocess, "run", fake_run)

    row = stress_module.run_stress_case(
        {
            "slug": "avride-dallas",
            "image": str(image),
            "expect": {
                "status": "complete",
                "source_prefix": "ocr-georeference:",
                "city_equals": "Dallas",
                "min_control_points": 5,
            },
        },
        out_dir,
        timeout_seconds=7,
        write_debug=False,
        python_executable="python",
    )

    assert row["observed_status"] == "complete"
    assert row["status"] == "complete"
    assert row["expectation_passed"] is True
    assert row["source"] == "ocr-georeference:nominatim-label-fit"
    assert row["total_elapsed_s"] == 0.612345
    assert row["stages"] == {"ocr": 0.4}
    assert row["road_match_score"] == 0.706233
    assert row["road_match_base_score"] == 0.506854
    assert row["road_match_sampled_points"] == 1348
    assert row["road_match_elapsed_s"] == 0.22955
    assert row["pipeline_version"] == "pipeline-test"
    assert row["image_width"] == 1200
    assert row["image_height"] == 900
    assert row["coverage_ratio"] == 0.123456
    assert row["contour_count"] == 2
    assert row["ocr_label_count"] == 8
    assert row["ocr_top_labels"] == ["Dallas", "Deep Ellum"]
    assert row["ocr_label_event"] == "Map labels read"
    assert row["ocr_label_events"] == [
        {"message": "Map labels read", "label_count": 8, "top_labels": ["Dallas", "Deep Ellum"]}
    ]
    assert row["ocr_full_detail_retry"] is False


def test_row_from_process_records_route_ui_reject_details() -> None:
    row = stress_module.row_from_process(
        {"slug": "route-ui", "image": "route.png"},
        command=["in-process"],
        completed=subprocess.CompletedProcess(["in-process"], 1, stdout="", stderr="route error"),
        wall_s=0.4,
        summary={
            "status": "failed",
            "error": "Could not infer a service-area boundary from ride-route UI.",
            "event_profile": {
                "total_elapsed_s": 0.4,
                "stage_elapsed_s": {"ocr": 0.2},
                "events": [
                    {
                        "stage": "ocr",
                        "message": "Map labels read",
                        "details": {"label_count": 22, "top_labels": ["Ride is 16 min away"]},
                    },
                    {
                        "stage": "georeference",
                        "message": "Rejecting ride-route UI",
                        "details": {
                            "route_ui_categories": ["pickup", "plate", "ride"],
                            "route_metric_labels": ["Ride is 16 min away"],
                        },
                    },
                ],
            },
        },
        parse_error=None,
        expected_status="failed",
    )

    assert row["route_ui_categories"] == ["pickup", "plate", "ride"]
    assert row["route_metric_labels"] == ["Ride is 16 min away"]


def test_row_from_process_records_non_map_ui_reject_details() -> None:
    row = stress_module.row_from_process(
        {"slug": "non-map-ui", "image": "settings.png"},
        command=["in-process"],
        completed=subprocess.CompletedProcess(["in-process"], 1, stdout="", stderr="non-map error"),
        wall_s=0.3,
        summary={
            "status": "failed",
            "error": "Could not infer a service-area boundary from non-map app UI.",
            "event_profile": {
                "total_elapsed_s": 0.3,
                "stage_elapsed_s": {"ocr": 0.2},
                "events": [
                    {
                        "stage": "ocr",
                        "message": "Map labels read",
                        "details": {"label_count": 20, "top_labels": ["Tesla Sync"]},
                    },
                    {
                        "stage": "georeference",
                        "message": "Rejecting non-map app UI",
                        "details": {
                            "non_map_ui_categories": ["account", "privacy", "sync"],
                            "non_map_ui_labels": ["Tesla Sync", "Secure&Private"],
                        },
                    },
                ],
            },
        },
        parse_error=None,
        expected_status="failed",
    )

    assert row["non_map_ui_categories"] == ["account", "privacy", "sync"]
    assert row["non_map_ui_labels"] == ["Tesla Sync", "Secure&Private"]


def test_row_from_process_records_thematic_map_reject_details() -> None:
    row = stress_module.row_from_process(
        {"slug": "thematic", "image": "climate.jpg"},
        command=["in-process"],
        completed=subprocess.CompletedProcess(["in-process"], 1, stdout="", stderr="thematic error"),
        wall_s=0.2,
        summary={
            "status": "failed",
            "error": "Could not infer a service-area boundary from a thematic map.",
            "event_profile": {
                "total_elapsed_s": 0.2,
                "stage_elapsed_s": {"ocr": 0.1},
                "events": [
                    {
                        "stage": "ocr",
                        "message": "Map labels read",
                        "details": {"label_count": 2, "top_labels": ["Vulnerability Index"]},
                    },
                    {
                        "stage": "georeference",
                        "message": "Rejecting thematic map",
                        "details": {"thematic_map_labels": ["Climate Change Vulnerability Index"]},
                    },
                ],
            },
        },
        parse_error=None,
        expected_status="failed",
    )

    assert row["thematic_map_labels"] == ["Climate Change Vulnerability Index"]


def test_repeat_profile_ocr_engine_count_metric_text_includes_confidence_counts() -> None:
    text = stress_module.repeat_profile_ocr_engine_count_metric_text(
        {
            "label_count": {"p95_count": 12.0, "max_count": 14.0},
            "label_confidence_lt_70_count": {"p95_count": 0.0, "max_count": 0.0},
            "label_confidence_lt_80_count": {"p95_count": 1.5, "max_count": 2.0},
            "unused_metric": {"p95_count": 99.0, "max_count": 100.0},
        }
    )

    assert text == (
        "label_count=p95 12.0 max 14.0, "
        "conf_lt70=p95 0.0 max 0.0, "
        "conf_lt80=p95 1.5 max 2.0"
    )


def test_print_stress_table_reports_confidence_count_metrics(capsys) -> None:
    report = {
        "manifest_contracts": {
            "total_cases": 2,
            "ocr_call_contract_rows": 2,
            "ocr_positive_call_contract_rows": 1,
            "ocr_zero_call_contract_rows": 1,
            "ocr_count_contract_rows": 1,
            "ocr_positive_call_rows_without_count_contract": [],
        },
        "manifest_contract_budget": {
            "passed": True,
            "violations": [],
            "min_ocr_call_contract_rows": 2,
            "min_ocr_count_contract_rows": 1,
        },
        "summary": {
            "total": 1,
            "expectation_passed": 1,
            "statuses": {"complete": 1},
            "max_total_elapsed_s": 0.5,
        },
        "repeat_profile": {
            "summary": {
                "analyzed_samples": 1,
                "expectation_passed_samples": 1,
                "subsecond_samples": 1,
                "median_total_elapsed_s": 0.5,
                "p95_total_elapsed_s": 0.5,
                "max_total_elapsed_s": 0.5,
                "slowest_samples": [
                    {
                        "slug": "kept",
                        "repeat_index": 2,
                        "total_elapsed_s": 0.5,
                        "ocr_engine": {
                            "rec_elapsed_s": 0.12,
                            "total_s": 0.3,
                            "label_confidence_p50": 88.25,
                            "label_confidence_lt_80_count": 1,
                        },
                    }
                ],
                "ocr_engine_count_metric": {
                    "label_count": {"p95_count": 12.0, "max_count": 14.0},
                    "label_confidence_lt_70_count": {"p95_count": 0.0, "max_count": 0.0},
                    "label_confidence_lt_80_count": {"p95_count": 1.0, "max_count": 2.0},
                },
            }
        },
        "rows": [
            {
                "slug": "kept",
                "observed_status": "complete",
                "expectation_passed": True,
                "source": "ocr-georeference:nominatim-label-fit",
                "control_points": 4,
                "total_elapsed_s": 0.5,
            }
        ],
    }

    stress_module.print_stress_table(report)

    output = capsys.readouterr().out
    assert "manifest OCR contracts: calls=2/2, count-capped=1/1 positive-call rows" in output
    assert "manifest contract budget: passed" in output
    assert "repeat slowest: kept#2=0.500s rec=0.120s ocr_total=0.300s conf_p50=88.2 conf_lt80=1" in output
    assert "repeat ocr engine counts: label_count=p95 12.0 max 14.0" in output
    assert "conf_lt70=p95 0.0 max 0.0" in output
    assert "conf_lt80=p95 1.0 max 2.0" in output


def test_stress_benchmark_can_profile_ocr_engine(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "stress.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "profiled-map",
                        "image": str(image),
                        "expect": {"status": "failed", "error_contains": "sparse OCR labels"},
                    }
                ]
            }
        )
    )

    def fake_run(command, *, text, capture_output, timeout, check):
        assert "--profile-ocr-engine" in command
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(
                {
                    "status": "failed",
                    "error": "Could not infer a reliable map location from sparse OCR labels.",
                    "ocr_engine_profile": {
                        "calls": 1,
                        "det_elapsed_s": 0.2,
                        "rec_elapsed_s": 0.3,
                        "total_s": 0.55,
                        "raw_box_count": 4,
                        "selected_box_count": 3,
                    },
                    "event_profile": {"total_elapsed_s": 0.7, "stage_elapsed_s": {"ocr": 0.6}},
                }
            ),
            stderr="map-boundary-builder: error",
        )

    monkeypatch.setattr(stress_module.subprocess, "run", fake_run)

    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        profile_ocr_engine=True,
        python_executable="python",
    )

    row = report["rows"][0]
    assert row["observed_status"] == "failed"
    assert row["status"] == "failed"
    assert row["expectation_passed"] is True
    assert row["ocr_engine_profile"]["det_elapsed_s"] == 0.2
    assert report["summary"]["ocr_engine_profile"] == {
        "fixtures": 1,
        "calls": 1,
        "det_elapsed_s": 0.2,
        "rec_elapsed_s": 0.3,
        "total_s": 0.55,
        "raw_box_count": 4,
        "selected_box_count": 3,
    }
    assert report["summary"]["ocr_engine_stage_max_rows"] == {
        "det_elapsed_s": {
            "slug": "profiled-map",
            "elapsed_s": 0.2,
            "raw_box_count": 4,
            "selected_box_count": 3,
        },
        "rec_elapsed_s": {
            "slug": "profiled-map",
            "elapsed_s": 0.3,
            "raw_box_count": 4,
            "selected_box_count": 3,
        },
        "total_s": {
            "slug": "profiled-map",
            "elapsed_s": 0.55,
            "raw_box_count": 4,
            "selected_box_count": 3,
        },
    }


def test_run_stress_case_can_disable_runner_ocr_cache_for_subprocess(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")

    def fake_run(command, *, text, capture_output, timeout, check, env):
        assert env[RUNNER_OCR_CACHE_ENV] == "0"
        assert RUNNER_OCR_CACHE_ENV not in os.environ
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "city": "Houston",
                    "georeference_source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "event_profile": {"total_elapsed_s": 0.6},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(stress_module.subprocess, "run", fake_run)

    row = stress_module.run_stress_case(
        {
            "slug": "houston",
            "image": str(image),
            "expect": {
                "status": "complete",
                "source_prefix": "ocr-georeference:",
                "min_control_points": 5,
            },
        },
        tmp_path / "out",
        timeout_seconds=5,
        write_debug=False,
        runner_ocr_cache=False,
        python_executable="python",
    )

    assert row["expectation_passed"] is True
    assert row["runner_ocr_cache"] is False


def test_run_stress_case_can_disable_extraction_cache_for_subprocess(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")

    def fake_run(command, *, text, capture_output, timeout, check, env):
        assert env[EXTRACTION_CACHE_ENV] == "0"
        assert EXTRACTION_CACHE_ENV not in os.environ
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "city": "Houston",
                    "georeference_source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "event_profile": {"total_elapsed_s": 0.6},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(stress_module.subprocess, "run", fake_run)

    row = stress_module.run_stress_case(
        {
            "slug": "houston",
            "image": str(image),
            "expect": {
                "status": "complete",
                "source_prefix": "ocr-georeference:",
                "min_control_points": 5,
            },
        },
        tmp_path / "out",
        timeout_seconds=5,
        write_debug=False,
        extraction_cache=False,
        python_executable="python",
    )

    assert row["expectation_passed"] is True
    assert row["extraction_cache"] is False


def test_run_stress_case_reports_city_drift(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")

    def fake_run(command, *, text, capture_output, timeout, check):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "city": "Yost Ice Arena",
                    "georeference_source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 4,
                    "event_profile": {"total_elapsed_s": 0.7},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(stress_module.subprocess, "run", fake_run)

    row = stress_module.run_stress_case(
        {
            "slug": "ann-arbor",
            "image": str(image),
            "expect": {
                "status": "complete",
                "source_prefix": "ocr-georeference:",
                "city_equals": "Ann Arbor",
                "min_control_points": 3,
            },
        },
        tmp_path / "out",
        timeout_seconds=5,
        write_debug=False,
        python_executable="python",
    )

    assert row["expectation_passed"] is False
    assert row["expectation_issues"] == ["city 'Yost Ice Arena' did not equal 'Ann Arbor'"]


def test_check_expectations_accepts_bbox_within_meter_tolerance() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "city": "Los Angeles",
            "control_points": 18,
            "bbox": [-118.52404, 33.93024, -118.21938, 34.11846],
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "city_equals": "Los Angeles",
            "min_control_points": 18,
            "bbox_approx": [-118.5240391, 33.9302354, -118.2193825, 34.1184628],
            "max_bbox_error_m": 20,
        },
    )

    assert issues == []


def test_check_expectations_rejects_source_equals_drift() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "city": "Nashville",
            "control_points": 3,
        },
        {
            "status": "complete",
            "source_equals": "ocr-georeference:nominatim-label-fit+osm-road-refine",
            "source_prefix": "ocr-georeference:",
            "city_equals": "Nashville",
            "min_control_points": 3,
        },
    )

    assert issues == [
        "source 'ocr-georeference:nominatim-label-fit' did not equal "
        "'ocr-georeference:nominatim-label-fit+osm-road-refine'"
    ]


def test_check_expectations_rejects_low_ocr_label_count() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "city": "Miami",
            "control_points": 5,
            "ocr_label_count": 17,
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "city_equals": "Miami",
            "min_control_points": 5,
            "min_ocr_labels": 18,
        },
    )

    assert issues == ["ocr_label_count 17 below 18"]


def test_check_expectations_accepts_catalog_slug_and_ocr_call_budget() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "catalog-shape-match",
            "city": "Dallas",
            "catalog_slug": "dallas-waymo",
            "ocr_engine_profile": {"calls": 0},
        },
        {
            "status": "complete",
            "source_equals": "catalog-shape-match",
            "city_equals": "Dallas",
            "catalog_slug_equals": "dallas-waymo",
            "max_ocr_engine_calls": 0,
        },
    )

    assert issues == []


def test_check_expectations_rejects_catalog_slug_drift() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "catalog-shape-match",
            "city": "Dallas",
            "catalog_slug": "houston-waymo",
        },
        {
            "status": "complete",
            "source_equals": "catalog-shape-match",
            "city_equals": "Dallas",
            "catalog_slug_equals": "dallas-waymo",
        },
    )

    assert issues == ["catalog_slug 'houston-waymo' did not equal 'dallas-waymo'"]


def test_check_expectations_rejects_excess_ocr_engine_calls() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "catalog-shape-match",
            "city": "Orlando",
            "catalog_slug": "orlando-waymo",
            "ocr_engine_profile": {"calls": 1},
        },
        {
            "status": "complete",
            "source_equals": "catalog-shape-match",
            "city_equals": "Orlando",
            "catalog_slug_equals": "orlando-waymo",
            "max_ocr_engine_calls": 0,
        },
    )

    assert issues == ["ocr_engine_profile.calls 1 above 0"]


def test_check_expectations_accepts_ocr_engine_count_budgets() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "ocr_engine_profile": {
                "calls": 1,
                "selected_box_count": 30,
                "result_count": 29,
            },
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "max_ocr_engine_calls": 1,
            "max_ocr_engine_counts": {
                "selected_box_count": 30,
                "result_count": 29,
            },
        },
    )

    assert issues == []


def test_check_expectations_rejects_excess_ocr_engine_count_budget() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "failed",
            "error": "Could not infer a service-area boundary from ride-route UI.",
            "ocr_engine_profile": {
                "calls": 1,
                "selected_box_count": 31,
            },
        },
        {
            "status": "failed",
            "error_contains": "ride-route UI",
            "max_ocr_engine_counts": {
                "selected_box_count": 30,
            },
        },
    )

    assert issues == ["ocr_engine_profile.selected_box_count 31 above 30"]


def test_check_expectations_rejects_invalid_ocr_engine_count_budget_metric() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "ocr_engine_profile": {
                "calls": 1,
                "selected_box_count": 12,
            },
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "max_ocr_engine_counts": {
                "unknown_count": 30,
            },
        },
    )

    assert issues == ["ocr_engine_profile metric 'unknown_count' is invalid"]


def test_summarize_manifest_contracts_reports_ocr_count_coverage() -> None:
    contracts = stress_module.summarize_manifest_contracts(
        [
            {
                "slug": "count-capped-ocr",
                "expect": {
                    "max_ocr_engine_calls": 1,
                    "max_ocr_engine_counts": {
                        "raw_box_count": 12,
                        "selected_box_count": 8,
                    },
                },
            },
            {
                "slug": "call-only-ocr",
                "expect": {
                    "max_ocr_engine_calls": 1,
                },
            },
            {
                "slug": "pure-catalog",
                "expect": {
                    "max_ocr_engine_calls": 0,
                },
            },
            {
                "slug": "missing-call-contract",
                "expect": {},
            },
            {
                "slug": "invalid-count-contract",
                "expect": {
                    "max_ocr_engine_calls": 1,
                    "max_ocr_engine_counts": {
                        "unknown_count": 4,
                    },
                },
            },
        ]
    )

    assert contracts == {
        "total_cases": 5,
        "ocr_call_contract_rows": 4,
        "ocr_call_contract_missing_rows": ["missing-call-contract"],
        "ocr_positive_call_contract_rows": 3,
        "ocr_zero_call_contract_rows": 1,
        "ocr_count_contract_rows": 1,
        "ocr_count_contract_slugs": ["count-capped-ocr"],
        "ocr_positive_call_rows_without_count_contract": [
            "call-only-ocr",
            "invalid-count-contract",
        ],
        "ocr_count_contract_metric_counts": {
            "raw_box_count": 1,
            "selected_box_count": 1,
        },
        "invalid_ocr_count_contract_rows": [
            {"slug": "invalid-count-contract", "metric": "unknown_count"},
        ],
    }


def test_manifest_contract_budget_flags_coverage_gaps() -> None:
    budget = stress_module.build_manifest_contract_budget_summary(
        {
            "ocr_call_contract_rows": 3,
            "ocr_call_contract_missing_rows": ["missing-call"],
            "ocr_count_contract_rows": 1,
            "ocr_positive_call_rows_without_count_contract": ["call-only-a", "call-only-b"],
            "invalid_ocr_count_contract_rows": [
                {"slug": "bad-count", "metric": "unknown_count"},
            ],
        },
        min_ocr_call_contract_rows=4,
        min_ocr_count_contract_rows=2,
        max_positive_ocr_call_only_rows=1,
        fail_on_invalid_ocr_count_contracts=True,
    )

    assert budget == {
        "passed": False,
        "violations": [
            {
                "kind": "ocr_call_contract_rows_below_min",
                "ocr_call_contract_rows": 3,
                "min_ocr_call_contract_rows": 4,
                "missing_rows": ["missing-call"],
            },
            {
                "kind": "ocr_count_contract_rows_below_min",
                "ocr_count_contract_rows": 1,
                "min_ocr_count_contract_rows": 2,
            },
            {
                "kind": "positive_ocr_call_only_rows_above_max",
                "positive_ocr_call_only_rows": 2,
                "max_positive_ocr_call_only_rows": 1,
                "rows": ["call-only-a", "call-only-b"],
            },
            {
                "kind": "invalid_ocr_count_contracts",
                "invalid_ocr_count_contract_rows": [
                    {"slug": "bad-count", "metric": "unknown_count"},
                ],
            },
        ],
        "min_ocr_call_contract_rows": 4,
        "min_ocr_count_contract_rows": 2,
        "max_positive_ocr_call_only_rows": 1,
        "fail_on_invalid_ocr_count_contracts": True,
    }


def test_manifest_contract_budget_passes_when_coverage_meets_thresholds() -> None:
    budget = stress_module.build_manifest_contract_budget_summary(
        {
            "ocr_call_contract_rows": 49,
            "ocr_call_contract_missing_rows": [],
            "ocr_count_contract_rows": 12,
            "ocr_positive_call_rows_without_count_contract": ["call-only"] * 26,
            "invalid_ocr_count_contract_rows": [],
        },
        min_ocr_call_contract_rows=49,
        min_ocr_count_contract_rows=12,
        max_positive_ocr_call_only_rows=26,
        fail_on_invalid_ocr_count_contracts=True,
    )

    assert budget == {
        "passed": True,
        "violations": [],
        "min_ocr_call_contract_rows": 49,
        "min_ocr_count_contract_rows": 12,
        "max_positive_ocr_call_only_rows": 26,
        "fail_on_invalid_ocr_count_contracts": True,
    }


def test_real_screenshot_manifest_preserves_ocr_contract_coverage() -> None:
    manifest = stress_module.load_manifest(Path("benchmarks/real-screenshot-stress.json"))
    contracts = stress_module.summarize_manifest_contracts(manifest["cases"])
    budget = stress_module.build_manifest_contract_budget_summary(
        contracts,
        min_ocr_call_contract_rows=49,
        min_ocr_count_contract_rows=12,
        max_positive_ocr_call_only_rows=26,
        fail_on_invalid_ocr_count_contracts=True,
    )
    expected_count_contract_slugs = {
        "tesla-austin-route-receipt-long",
        "tesla-austin-route-receipt-gray-long",
        "profile-app-non-map-ui",
        "dallas-waymo",
        "san-antonio-waymo",
        "bay-area-waymo",
        "bay-area-waymo-jpeg-small",
        "los-angeles-waymo",
        "houston-waymo",
        "orlando-waymo",
        "nashville-waymo",
        "miami-waymo",
    }

    assert contracts["total_cases"] == 49
    assert contracts["ocr_call_contract_rows"] == contracts["total_cases"]
    assert contracts["ocr_call_contract_missing_rows"] == []
    assert contracts["ocr_positive_call_contract_rows"] == 38
    assert contracts["ocr_zero_call_contract_rows"] == 11
    assert set(contracts["ocr_count_contract_slugs"]) >= expected_count_contract_slugs
    assert contracts["ocr_count_contract_rows"] >= len(expected_count_contract_slugs)
    assert len(contracts["ocr_positive_call_rows_without_count_contract"]) <= 26
    assert contracts["invalid_ocr_count_contract_rows"] == []
    assert budget["passed"] is True


def test_check_expectations_rejects_route_ui_evidence_drift() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "failed",
            "error": "Could not infer a service-area boundary from ride-route UI.",
            "route_ui_categories": ["pickup", "ride"],
            "route_metric_labels": [],
        },
        {
            "status": "failed",
            "error_contains": "ride-route UI",
            "route_ui_categories_include": ["pickup", "plate", "ride"],
            "min_route_metric_labels": 1,
        },
    )

    assert issues == [
        "route_ui_categories missing ['plate']",
        "route_metric_labels count 0 below 1",
    ]


def test_check_expectations_rejects_non_map_ui_evidence_drift() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "failed",
            "error": "Could not infer a service-area boundary from non-map app UI.",
            "non_map_ui_categories": ["privacy", "sync"],
            "non_map_ui_labels": [],
        },
        {
            "status": "failed",
            "error_contains": "non-map app UI",
            "non_map_ui_categories_include": ["privacy", "stats", "sync"],
            "min_non_map_ui_labels": 1,
        },
    )

    assert issues == [
        "non_map_ui_categories missing ['stats']",
        "non_map_ui_labels count 0 below 1",
    ]


def test_check_expectations_rejects_thematic_map_evidence_drift() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "failed",
            "error": "Could not infer a service-area boundary from a thematic map.",
            "thematic_map_labels": ["Khartoum"],
        },
        {
            "status": "failed",
            "error_contains": "thematic map",
            "min_thematic_map_labels": 2,
            "thematic_map_labels_contain": ["vulnerability index"],
        },
    )

    assert issues == [
        "thematic_map_labels count 1 below 2",
        "thematic_map_labels missing snippets ['vulnerability index']",
    ]


def test_check_expectations_rejects_ocr_top_label_evidence_drift() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "failed",
            "error": "Could not infer a reliable map location from sparse OCR labels.",
            "ocr_label_count": 3,
            "ocr_top_labels": ["Downtown", "Arts District"],
        },
        {
            "status": "failed",
            "error_contains": "sparse OCR labels",
            "min_ocr_labels": 3,
            "ocr_top_labels_contain": ["las vegas", "paradise"],
        },
    )

    assert issues == ["ocr_top_labels missing snippets ['las vegas', 'paradise']"]


def test_check_expectations_rejects_missing_ocr_top_label_evidence() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "ocr_label_count": 4,
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "ocr_top_labels_contain": ["dallas"],
        },
    )

    assert issues == ["ocr_top_labels missing"]


def test_check_expectations_rejects_low_confidence() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "city": "Los Angeles",
            "control_points": 18,
            "combined_confidence": 0.72,
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "city_equals": "Los Angeles",
            "min_control_points": 18,
            "min_combined_confidence": 0.83,
            "min_georeference_confidence": 0.83,
        },
    )

    assert issues == [
        "combined_confidence 0.72 below 0.83",
        "georeference_confidence None below 0.83",
    ]


def test_check_expectations_rejects_bbox_outside_meter_tolerance() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "city": "Los Angeles",
            "control_points": 18,
            "bbox": [-118.50, 33.93, -118.20, 34.12],
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "city_equals": "Los Angeles",
            "min_control_points": 18,
            "bbox_approx": [-118.5240391, 33.9302354, -118.2193825, 34.1184628],
            "max_bbox_error_m": 500,
        },
    )

    assert len(issues) == 1
    assert issues[0].startswith("bbox max corner error ")
    assert issues[0].endswith("m above 500m")


def test_check_expectations_rejects_invalid_bbox() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "city": "Los Angeles",
            "control_points": 18,
            "bbox": None,
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "city_equals": "Los Angeles",
            "min_control_points": 18,
            "bbox_approx": [-118.5240391, 33.9302354, -118.2193825, 34.1184628],
            "max_bbox_error_m": 500,
        },
    )

    assert issues == ["bbox was missing or invalid"]


def test_check_expectations_rejects_slow_expected_failure() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "failed",
            "error": "Could not infer a reliable map location from sparse OCR labels.",
            "total_elapsed_s": 1.2,
        },
        {
            "status": "failed",
            "error_contains": "sparse OCR labels",
            "max_total_elapsed_s": 1.0,
        },
    )

    assert issues == ["total_elapsed_s 1.2 above 1.0"]


def test_check_expectations_can_guard_ocr_labels_on_expected_failure() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "failed",
            "error": "Could not infer a reliable map location from sparse OCR labels.",
            "ocr_label_count": 2,
        },
        {
            "status": "failed",
            "error_contains": "sparse OCR labels",
            "min_ocr_labels": 3,
        },
    )

    assert issues == ["ocr_label_count 2 below 3"]


def test_check_expectations_rejects_slow_complete() -> None:
    issues = stress_module.check_expectations(
        {
            "observed_status": "complete",
            "source": "ocr-georeference:nominatim-label-fit",
            "city": "Bay Area",
            "control_points": 16,
            "total_elapsed_s": 1.01,
        },
        {
            "status": "complete",
            "source_prefix": "ocr-georeference:",
            "min_control_points": 16,
            "max_total_elapsed_s": 1.0,
        },
    )

    assert issues == ["total_elapsed_s 1.01 above 1.0"]


def test_run_stress_case_accepts_expected_fail_closed(tmp_path, monkeypatch) -> None:
    image = tmp_path / "zoox.png"
    image.write_bytes(b"not a real image")

    def fake_run(command, *, text, capture_output, timeout, check):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(
                {
                    "status": "failed",
                    "error": "Could not infer a reliable map location and georeference from sparse OCR labels.",
                    "event_profile": {
                        "total_elapsed_s": 1.2,
                        "stage_elapsed_s": {"ocr": 0.9},
                        "events": [
                            {
                                "stage": "extract",
                                "message": "Extracting service-area pixels",
                                "details": {"width": 734, "height": 1596},
                            },
                            {
                                "stage": "extract",
                                "message": "Pixel polygon extracted",
                                "details": {
                                    "style": "dark-teal",
                                    "coverage_ratio": 0.152668,
                                    "contour_count": 1,
                                },
                            },
                            {
                                "stage": "ocr",
                                "message": "Focused map labels read",
                                "details": {
                                    "label_count": 2,
                                    "top_labels": ["Las Vegas", "Enterprise"],
                                },
                            },
                            {
                                "stage": "ocr",
                                "message": "Map labels read",
                                "details": {
                                    "label_count": 2,
                                    "top_labels": ["Las Vegas", "Enterprise"],
                                },
                            },
                            {
                                "stage": "ocr",
                                "message": "Full-detail map labels read",
                                "details": {
                                    "label_count": 5,
                                    "top_labels": ["Help", "Zoox", "Las Vegas"],
                                },
                            },
                        ],
                    },
                }
            ),
            stderr="map-boundary-builder: error",
        )

    monkeypatch.setattr(stress_module.subprocess, "run", fake_run)

    row = stress_module.run_stress_case(
        {
            "slug": "zoox-mobile",
            "image": str(image),
            "expect": {"status": "failed", "error_contains": "sparse OCR labels"},
        },
        tmp_path / "out",
        timeout_seconds=5,
        write_debug=False,
        python_executable="python",
    )

    assert row["observed_status"] == "failed"
    assert row["expectation_passed"] is True
    assert row["error"].endswith("sparse OCR labels.")
    assert row["style"] == "dark-teal"
    assert row["image_width"] == 734
    assert row["image_height"] == 1596
    assert row["coverage_ratio"] == 0.152668
    assert row["ocr_label_count"] == 5
    assert row["ocr_top_labels"] == ["Help", "Zoox", "Las Vegas"]
    assert row["ocr_label_event"] == "Full-detail map labels read"
    assert row["ocr_label_events"] == [
        {"message": "Focused map labels read", "label_count": 2, "top_labels": ["Las Vegas", "Enterprise"]},
        {"message": "Map labels read", "label_count": 2, "top_labels": ["Las Vegas", "Enterprise"]},
        {"message": "Full-detail map labels read", "label_count": 5, "top_labels": ["Help", "Zoox", "Las Vegas"]},
    ]
    assert row["ocr_full_detail_retry"] is True


def test_run_stress_case_reports_missing_without_subprocess(tmp_path, monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess should not run for missing images")

    monkeypatch.setattr(stress_module.subprocess, "run", fail_run)

    row = stress_module.run_stress_case(
        {
            "slug": "missing-map",
            "image": str(tmp_path / "missing.png"),
            "expect": {"status": "complete"},
        },
        tmp_path / "out",
        timeout_seconds=5,
        write_debug=False,
        python_executable="python",
    )

    assert row["observed_status"] == "missing"
    assert row["status"] == "missing"
    assert row["expectation_passed"] is False
    assert row["expectation_issues"] == ["expected complete, got missing"]


def test_run_stress_benchmark_writes_report_and_summarizes(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "kept",
                        "image": str(image),
                        "expect": {"status": "complete", "source_prefix": "ocr-georeference:"},
                    },
                    {
                        "slug": "skipped",
                        "image": str(image),
                        "expect": {"status": "complete"},
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(stress_module, "get_pipeline_version", lambda: "pipeline-test")
    monkeypatch.setattr(
        stress_module,
        "pipeline_version_dependency_versions",
        lambda: [("rapidocr-onnxruntime", "1.4.4"), ("onnxruntime", "1.26.0")],
    )
    monkeypatch.setattr(
        stress_module,
        "ocr_runtime_config",
        lambda: {"rapidocr_max_dimension": 1600},
    )

    def fake_run_case(
        case,
        out_dir,
        *,
        timeout_seconds,
        write_debug,
        profile_ocr_engine,
        runner_ocr_cache,
        extraction_cache,
        execution,
        python_executable,
    ):
        assert profile_ocr_engine is False
        assert runner_ocr_cache is True
        assert extraction_cache is True
        assert execution == "subprocess"
        return {
            "slug": case["slug"],
            "image": case["image"],
            "expected_status": "complete",
            "observed_status": "complete",
            "expectation_passed": True,
            "source": "ocr-georeference:nominatim-label-fit",
            "total_elapsed_s": 0.5,
        }

    monkeypatch.setattr(stress_module, "run_stress_case", fake_run_case)
    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        only_slugs=["kept"],
        timeout_seconds=3,
        write_debug=True,
        python_executable="python",
    )

    saved = json.loads((tmp_path / "out" / "stress-summary.json").read_text())
    assert [row["slug"] for row in report["rows"]] == ["kept"]
    assert saved["summary"]["expectation_passed"] == 1
    assert saved["summary"]["sources"] == {"ocr-georeference:nominatim-label-fit": 1}
    assert saved["summary"]["ocr_label_event_counts"] == {}
    assert saved["summary"]["ocr_full_detail_retry_count"] == 0
    assert saved["summary"]["ocr_full_detail_retry_rows"] == []
    assert saved["summary"]["ocr_engine_stage_max_rows"] == {}
    assert saved["summary"]["stage_duration_s"] == {}
    assert saved["summary"]["stage_max_rows"] == {}
    assert saved["runtime_config"]["pipeline_version"] == "pipeline-test"
    assert saved["runtime_config"]["runtime_dependencies"] == {
        "rapidocr-onnxruntime": "1.4.4",
        "onnxruntime": "1.26.0",
    }
    assert saved["runtime_config"]["ocr"] == {"rapidocr_max_dimension": 1600}
    assert saved["runtime_config"]["generation_env"][RUNNER_OCR_CACHE_ENV] == "1"
    assert saved["runtime_config"]["generation_env"][EXTRACTION_CACHE_ENV] == "1"


def test_run_stress_benchmark_runtime_config_records_cache_policy(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "kept",
                        "image": str(image),
                        "expect": {"status": "complete", "source_prefix": "ocr-georeference:"},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(stress_module, "get_pipeline_version", lambda: "pipeline-test")
    monkeypatch.setattr(
        stress_module,
        "pipeline_version_dependency_versions",
        lambda: [("rapidocr-onnxruntime", "1.4.4"), ("onnxruntime", "1.26.0")],
    )
    monkeypatch.setattr(
        stress_module,
        "ocr_runtime_config",
        lambda: {"rapidocr_max_dimension": 1400},
    )
    monkeypatch.delenv(RUNNER_OCR_CACHE_ENV, raising=False)
    monkeypatch.delenv(EXTRACTION_CACHE_ENV, raising=False)

    def fake_run_case(
        case,
        out_dir,
        *,
        timeout_seconds,
        write_debug,
        profile_ocr_engine,
        runner_ocr_cache,
        extraction_cache,
        execution,
        python_executable,
    ):
        assert runner_ocr_cache is False
        assert extraction_cache is False
        assert RUNNER_OCR_CACHE_ENV not in os.environ
        assert EXTRACTION_CACHE_ENV not in os.environ
        return {
            "slug": case["slug"],
            "image": case["image"],
            "expected_status": "complete",
            "observed_status": "complete",
            "expectation_passed": True,
            "source": "ocr-georeference:nominatim-label-fit",
            "total_elapsed_s": 0.5,
        }

    monkeypatch.setattr(stress_module, "run_stress_case", fake_run_case)
    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        runner_ocr_cache=False,
        extraction_cache=False,
        python_executable="python",
    )

    saved = json.loads((tmp_path / "out" / "stress-summary.json").read_text())
    runtime_config = saved["runtime_config"]
    assert report["runtime_config"] == runtime_config
    assert runtime_config["pipeline_version"] == "pipeline-test"
    assert runtime_config["runtime_dependencies"] == {
        "rapidocr-onnxruntime": "1.4.4",
        "onnxruntime": "1.26.0",
    }
    assert runtime_config["ocr"] == {"rapidocr_max_dimension": 1400}
    assert runtime_config["generation_env"][RUNNER_OCR_CACHE_ENV] == "0"
    assert runtime_config["generation_env"][EXTRACTION_CACHE_ENV] == "0"
    assert runtime_config["generation_env"]["MAP_BOUNDARY_ROAD_REFINE_CACHE_MAX_PIXELS"] == "3000000"
    assert os.environ.get(RUNNER_OCR_CACHE_ENV) is None
    assert os.environ.get(EXTRACTION_CACHE_ENV) is None


def test_run_stress_benchmark_can_gate_manifest_contract_coverage(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "count-capped",
                        "image": str(image),
                        "expect": {
                            "status": "complete",
                            "source_prefix": "ocr-georeference:",
                            "max_ocr_engine_calls": 1,
                            "max_ocr_engine_counts": {"selected_box_count": 3},
                        },
                    },
                    {
                        "slug": "call-only",
                        "image": str(image),
                        "expect": {
                            "status": "complete",
                            "source_prefix": "ocr-georeference:",
                            "max_ocr_engine_calls": 1,
                        },
                    },
                ]
            }
        )
    )

    def fake_run_case(
        case,
        out_dir,
        *,
        timeout_seconds,
        write_debug,
        profile_ocr_engine,
        runner_ocr_cache,
        extraction_cache,
        execution,
        python_executable,
    ):
        return {
            "slug": case["slug"],
            "image": case["image"],
            "expected_status": "complete",
            "observed_status": "complete",
            "expectation_passed": True,
            "source": "ocr-georeference:nominatim-label-fit",
            "total_elapsed_s": 0.5,
        }

    monkeypatch.setattr(stress_module, "run_stress_case", fake_run_case)

    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        min_ocr_call_contract_rows=2,
        min_ocr_count_contract_rows=2,
        max_positive_ocr_call_only_rows=0,
        fail_on_invalid_ocr_count_contracts=True,
        python_executable="python",
    )

    saved = json.loads((tmp_path / "out" / "stress-summary.json").read_text())
    assert saved["manifest_contracts"] == report["manifest_contracts"]
    assert report["manifest_contracts"]["ocr_call_contract_rows"] == 2
    assert report["manifest_contracts"]["ocr_count_contract_rows"] == 1
    assert report["manifest_contracts"]["ocr_positive_call_rows_without_count_contract"] == ["call-only"]
    assert saved["manifest_contract_budget"] == report["manifest_contract_budget"]
    assert report["manifest_contract_budget"] == {
        "passed": False,
        "violations": [
            {
                "kind": "ocr_count_contract_rows_below_min",
                "ocr_count_contract_rows": 1,
                "min_ocr_count_contract_rows": 2,
            },
            {
                "kind": "positive_ocr_call_only_rows_above_max",
                "positive_ocr_call_only_rows": 1,
                "max_positive_ocr_call_only_rows": 0,
                "rows": ["call-only"],
            },
        ],
        "min_ocr_call_contract_rows": 2,
        "min_ocr_count_contract_rows": 2,
        "max_positive_ocr_call_only_rows": 0,
        "fail_on_invalid_ocr_count_contracts": True,
    }


def test_run_stress_benchmark_can_record_generation_prewarm(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "kept",
                        "image": str(image),
                        "expect": {"status": "complete", "source_prefix": "ocr-georeference:"},
                    }
                ]
            }
        )
    )
    prewarm_calls = []

    def fake_prewarm(*, runner_ocr_cache, extraction_cache):
        prewarm_calls.append((runner_ocr_cache, extraction_cache))
        return {
            "status": "ok",
            "rapidocr_inference_warmed": True,
            "rapidocr_s": 0.09,
            "total_s": 0.12,
        }

    def fake_run_case(
        case,
        out_dir,
        *,
        timeout_seconds,
        write_debug,
        profile_ocr_engine,
        runner_ocr_cache,
        extraction_cache,
        execution,
        python_executable,
    ):
        return {
            "slug": case["slug"],
            "image": case["image"],
            "expected_status": "complete",
            "observed_status": "complete",
            "expectation_passed": True,
            "source": "ocr-georeference:nominatim-label-fit",
            "total_elapsed_s": 0.5,
        }

    monkeypatch.setattr(stress_module, "run_generation_prewarm", fake_prewarm)
    monkeypatch.setattr(stress_module, "run_stress_case", fake_run_case)
    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        prewarm_runtime=True,
        max_prewarm_runtime_s=0.2,
        max_prewarm_stage_s={"rapidocr_s": 0.1},
        runner_ocr_cache=False,
        extraction_cache=False,
        python_executable="python",
    )

    saved = json.loads((tmp_path / "out" / "stress-summary.json").read_text())
    assert prewarm_calls == [(False, False)]
    assert report["prewarm_runtime"] is True
    assert saved["prewarm"] == {
        "status": "ok",
        "rapidocr_inference_warmed": True,
        "rapidocr_s": 0.09,
        "total_s": 0.12,
    }
    assert saved["latency_budget"] == {
        "passed": True,
        "primary_violations": [],
        "repeat_violations": [],
        "max_prewarm_runtime_s": 0.2,
        "prewarm_violations": [],
        "max_prewarm_stage_s": {"rapidocr_s": 0.1},
        "prewarm_stage_violations": [],
    }


def test_run_stress_benchmark_repeat_profile_records_samples(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "kept",
                        "image": str(image),
                        "expect": {"status": "complete", "source_prefix": "ocr-georeference:"},
                    }
                ]
            }
        )
    )
    durations = iter([1.4, 1.2, 0.8])
    calls = []

    def fake_run_case(
        case,
        out_dir,
        *,
        timeout_seconds,
        write_debug,
        profile_ocr_engine,
        runner_ocr_cache,
        extraction_cache,
        execution,
        python_executable,
    ):
        assert runner_ocr_cache is True
        assert extraction_cache is True
        assert execution == "subprocess"
        calls.append((case["slug"], out_dir.name, timeout_seconds, write_debug, profile_ocr_engine))
        duration = next(durations)
        return {
            "slug": case["slug"],
            "image": case["image"],
            "expected_status": "complete",
            "observed_status": "complete",
            "expectation_passed": True,
            "source": "ocr-georeference:nominatim-label-fit",
            "total_elapsed_s": duration,
            "stages": {"ocr": round(duration / 2, 6)},
            "ocr_engine_profile": {
                "calls": 1,
                "det_elapsed_s": round(duration / 10, 6),
                "rec_elapsed_s": round(duration / 5, 6),
                "total_s": round(duration / 4, 6),
                "raw_box_count": 3,
                "label_confidence_lt_50_count": 0,
                "label_confidence_lt_70_count": 0,
                "label_confidence_lt_80_count": 1,
                "label_confidence_lt_90_count": 2,
                "calls_detail": [
                    {
                        "det_elapsed_s": round(duration / 10, 6),
                        "rec_elapsed_s": round(duration / 5, 6),
                        "total_s": round(duration / 4, 6),
                        "input_shape": [900, 1200],
                        "detector_limit": 608,
                        "selected_box_count": 3,
                        "label_confidence_min": 72.0,
                        "label_confidence_p50": 84.0,
                        "label_confidence_p90": 93.2,
                        "label_confidence_max": 96.0,
                        "label_confidence_lt_50_count": 0,
                        "label_confidence_lt_70_count": 0,
                        "label_confidence_lt_80_count": 1,
                        "label_confidence_lt_90_count": 2,
                    }
                ],
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_case", fake_run_case)
    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        timeout_seconds=3,
        write_debug=True,
        profile_ocr_engine=True,
        repeat_profile_runs=2,
        repeat_profile_warmups=1,
        max_total_elapsed_s=1.0,
        python_executable="python",
    )

    saved = json.loads((tmp_path / "out" / "stress-summary.json").read_text())
    assert calls == [
        ("kept", "out", 3, True, True),
        ("kept", "run-1", 3, True, True),
        ("kept", "run-2", 3, True, True),
    ]
    assert report["repeat_profile_runs"] == 2
    assert report["repeat_profile_warmups"] == 1
    assert saved["repeat_profile"]["runs_per_case"] == 2
    repeat_profile = report["repeat_profile"]
    assert repeat_profile["summary"]["samples"] == 2
    assert repeat_profile["summary"]["analyzed_samples"] == 1
    assert repeat_profile["summary"]["expectation_passed_samples"] == 1
    assert repeat_profile["summary"]["unexpected_samples"] == 0
    assert repeat_profile["summary"]["subsecond_samples"] == 1
    assert repeat_profile["summary"]["subsecond_case_min_total_count"] == 1
    assert repeat_profile["summary"]["stable_signature_cases"] == 1
    assert repeat_profile["summary"]["unstable_signature_cases"] == []
    assert repeat_profile["summary"]["min_total_elapsed_s"] == 0.8
    assert repeat_profile["summary"]["median_total_elapsed_s"] == 0.8
    assert repeat_profile["summary"]["p90_total_elapsed_s"] == 0.8
    assert repeat_profile["summary"]["p95_total_elapsed_s"] == 0.8
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
    assert repeat_profile["summary"]["slowest_samples"] == [
        {
            "slug": "kept",
            "repeat_index": 2,
            "total_elapsed_s": 0.8,
            "observed_status": "complete",
            "expectation_passed": True,
            "top_stage": {"stage": "ocr", "elapsed_s": 0.4},
            "ocr_engine": {
                "det_elapsed_s": 0.08,
                "rec_elapsed_s": 0.16,
                "total_s": 0.2,
                "calls": 1,
                "raw_box_count": 3,
                "label_confidence_min": 72.0,
                "label_confidence_p50": 84.0,
                "label_confidence_p90": 93.2,
                "label_confidence_max": 96.0,
                "label_confidence_lt_50_count": 0,
                "label_confidence_lt_70_count": 0,
                "label_confidence_lt_80_count": 1,
                "label_confidence_lt_90_count": 2,
            },
        }
    ]
    assert repeat_profile["summary"]["ocr_engine_profile"] == {
        "fixtures": 1,
        "calls": 1,
        "det_elapsed_s": 0.08,
        "rec_elapsed_s": 0.16,
        "total_s": 0.2,
        "raw_box_count": 3,
        "label_confidence_lt_50_count": 0,
        "label_confidence_lt_70_count": 0,
        "label_confidence_lt_80_count": 1,
        "label_confidence_lt_90_count": 2,
    }
    assert repeat_profile["summary"]["ocr_engine_stage_duration_s"] == {
        "det_elapsed_s": {
            "samples": 1,
            "min_duration_s": 0.08,
            "median_duration_s": 0.08,
            "average_duration_s": 0.08,
            "p90_duration_s": 0.08,
            "p95_duration_s": 0.08,
            "max_duration_s": 0.08,
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
            "min_duration_s": 0.2,
            "median_duration_s": 0.2,
            "average_duration_s": 0.2,
            "p90_duration_s": 0.2,
            "p95_duration_s": 0.2,
            "max_duration_s": 0.2,
        },
    }
    assert repeat_profile["summary"]["ocr_engine_count_metric"] == {
        "raw_box_count": {
            "samples": 1,
            "min_count": 3,
            "median_count": 3,
            "average_count": 3,
            "p90_count": 3,
            "p95_count": 3,
            "max_count": 3,
        },
        "selected_box_count": {
            "samples": 1,
            "min_count": 3,
            "median_count": 3,
            "average_count": 3,
            "p90_count": 3,
            "p95_count": 3,
            "max_count": 3,
        },
        "label_confidence_lt_50_count": {
            "samples": 1,
            "min_count": 0,
            "median_count": 0,
            "average_count": 0,
            "p90_count": 0,
            "p95_count": 0,
            "max_count": 0,
        },
        "label_confidence_lt_70_count": {
            "samples": 1,
            "min_count": 0,
            "median_count": 0,
            "average_count": 0,
            "p90_count": 0,
            "p95_count": 0,
            "max_count": 0,
        },
        "label_confidence_lt_80_count": {
            "samples": 1,
            "min_count": 1,
            "median_count": 1,
            "average_count": 1,
            "p90_count": 1,
            "p95_count": 1,
            "max_count": 1,
        },
        "label_confidence_lt_90_count": {
            "samples": 1,
            "min_count": 2,
            "median_count": 2,
            "average_count": 2,
            "p90_count": 2,
            "p95_count": 2,
            "max_count": 2,
        },
    }
    assert repeat_profile["summary"]["ocr_engine_stage_max_rows"] == {
        "det_elapsed_s": {
            "slug": "kept",
            "elapsed_s": 0.08,
            "input_shape": [900, 1200],
            "detector_limit": 608,
            "selected_box_count": 3,
            "label_confidence_min": 72.0,
            "label_confidence_p50": 84.0,
            "label_confidence_p90": 93.2,
            "label_confidence_max": 96.0,
            "label_confidence_lt_50_count": 0,
            "label_confidence_lt_70_count": 0,
            "label_confidence_lt_80_count": 1,
            "label_confidence_lt_90_count": 2,
        },
        "rec_elapsed_s": {
            "slug": "kept",
            "elapsed_s": 0.16,
            "input_shape": [900, 1200],
            "detector_limit": 608,
            "selected_box_count": 3,
            "label_confidence_min": 72.0,
            "label_confidence_p50": 84.0,
            "label_confidence_p90": 93.2,
            "label_confidence_max": 96.0,
            "label_confidence_lt_50_count": 0,
            "label_confidence_lt_70_count": 0,
            "label_confidence_lt_80_count": 1,
            "label_confidence_lt_90_count": 2,
        },
        "total_s": {
            "slug": "kept",
            "elapsed_s": 0.2,
            "input_shape": [900, 1200],
            "detector_limit": 608,
            "selected_box_count": 3,
            "label_confidence_min": 72.0,
            "label_confidence_p50": 84.0,
            "label_confidence_p90": 93.2,
            "label_confidence_max": 96.0,
            "label_confidence_lt_50_count": 0,
            "label_confidence_lt_70_count": 0,
            "label_confidence_lt_80_count": 1,
            "label_confidence_lt_90_count": 2,
        },
    }
    assert repeat_profile["cases"]["kept"]["signature_stability"] == {
        "samples": 1,
        "stable": True,
        "unique_signatures": 1,
        "signatures": [
            {
                "count": 1,
                "observed_status": "complete",
                "city": None,
                "source": "ocr-georeference:nominatim-label-fit",
                "control_points": None,
                "bbox": None,
                "geojson_geometry_hash": None,
                "geojson_coordinate_count": None,
                "combined_confidence": None,
                "georeference_confidence": None,
                "road_match_score": None,
                "road_match_base_score": None,
                "road_match_sampled_points": None,
                "ocr_label_count": None,
                "ocr_label_event": None,
                "ocr_full_detail_retry": None,
                "ocr_top_labels": None,
                "error": None,
            }
        ],
    }
    assert repeat_profile["cases"]["kept"]["max_total_elapsed_s"] == 0.8
    assert repeat_profile["samples"][0]["warmup"] is True
    assert repeat_profile["samples"][1]["repeat_index"] == 2
    assert saved["latency_budget"] == report["latency_budget"]
    assert report["latency_budget"] == {
        "max_total_elapsed_s": 1.0,
        "passed": False,
        "primary_violations": [
            {
                "slug": "kept",
                "total_elapsed_s": 1.4,
                "over_by_s": 0.4,
                "observed_status": "complete",
            }
        ],
        "repeat_violations": [],
    }


def test_parse_metric_count_budgets_accepts_known_ocr_engine_counts() -> None:
    assert stress_module.parse_metric_count_budgets(
        ["selected_box_count=30,raw_box_count=50", "result_count=29"]
    ) == {
        "raw_box_count": 50.0,
        "result_count": 29.0,
        "selected_box_count": 30.0,
    }


def test_parse_metric_count_budgets_rejects_unknown_counts() -> None:
    with pytest.raises(ValueError, match="Unknown OCR engine count metric"):
        stress_module.parse_metric_count_budgets(["total_elapsed_s=1"])


def test_latency_budget_flags_repeat_ocr_engine_count_p95_excess() -> None:
    report = stress_module.build_latency_budget_summary(
        rows=[],
        repeat_profile={
            "summary": {
                "ocr_engine_count_metric": {
                    "selected_box_count": {"p95_count": 30.0},
                    "raw_box_count": {"p95_count": 50.0},
                }
            }
        },
        max_repeat_ocr_engine_p95_count={
            "selected_box_count": 28.0,
            "raw_box_count": 55.0,
            "result_count": 29.0,
        },
    )

    assert report["passed"] is False
    assert report["max_repeat_ocr_engine_p95_count"] == {
        "raw_box_count": 55.0,
        "result_count": 29.0,
        "selected_box_count": 28.0,
    }
    assert report["repeat_ocr_engine_count_p95_violations"] == [
        {
            "kind": "repeat_ocr_engine_count_p95_missing",
            "metric": "result_count",
            "max_repeat_ocr_engine_p95_count": 29.0,
        },
        {
            "kind": "repeat_ocr_engine_count_p95_budget_exceeded",
            "metric": "selected_box_count",
            "p95_count": 30.0,
            "max_repeat_ocr_engine_p95_count": 28.0,
            "excess_count": 2.0,
        },
    ]


def test_latency_budget_flags_repeat_ocr_engine_count_max_excess() -> None:
    report = stress_module.build_latency_budget_summary(
        rows=[],
        repeat_profile={
            "summary": {
                "ocr_engine_count_metric": {
                    "selected_box_count": {"max_count": 31.0},
                    "raw_box_count": {"max_count": 50.0},
                }
            }
        },
        max_repeat_ocr_engine_max_count={
            "selected_box_count": 30.0,
            "raw_box_count": 55.0,
            "result_count": 29.0,
        },
    )

    assert report["passed"] is False
    assert report["max_repeat_ocr_engine_max_count"] == {
        "raw_box_count": 55.0,
        "result_count": 29.0,
        "selected_box_count": 30.0,
    }
    assert report["repeat_ocr_engine_count_max_violations"] == [
        {
            "kind": "repeat_ocr_engine_count_max_missing",
            "metric": "result_count",
            "max_repeat_ocr_engine_max_count": 29.0,
        },
        {
            "kind": "repeat_ocr_engine_count_max_budget_exceeded",
            "metric": "selected_box_count",
            "max_count": 31.0,
            "max_repeat_ocr_engine_max_count": 30.0,
            "excess_count": 1.0,
        },
    ]


def test_latency_budget_skips_primary_ocr_engine_budgets_for_zero_call_profiles() -> None:
    report = stress_module.build_latency_budget_summary(
        rows=[
            {
                "slug": "pure-catalog",
                "observed_status": "complete",
                "source": "catalog-shape-match",
                "ocr_engine_profile": {"calls": 0, "calls_detail": []},
            },
            {
                "slug": "missing-profile",
                "observed_status": "complete",
                "source": "ocr-georeference:nominatim-label-fit",
            },
            {
                "slug": "slow-profile",
                "observed_status": "complete",
                "source": "ocr-georeference:nominatim-label-fit",
                "ocr_engine_profile": {
                    "calls": 1,
                    "total_s": 0.95,
                    "selected_box_count": 31,
                },
            },
        ],
        repeat_profile=None,
        max_ocr_engine_duration_s={"total_s": 0.9},
        max_ocr_engine_count={"selected_box_count": 30},
    )

    assert report["passed"] is False
    assert report["ocr_engine_violations"] == [
        {
            "kind": "ocr_engine_duration_missing",
            "slug": "missing-profile",
            "metric": "total_s",
            "observed_status": "complete",
            "max_ocr_engine_duration_s": 0.9,
        },
        {
            "kind": "ocr_engine_duration_budget_exceeded",
            "slug": "slow-profile",
            "metric": "total_s",
            "observed_status": "complete",
            "duration_s": 0.95,
            "max_ocr_engine_duration_s": 0.9,
            "excess_s": 0.05,
        },
    ]
    assert report["ocr_engine_count_violations"] == [
        {
            "kind": "ocr_engine_count_missing",
            "slug": "missing-profile",
            "metric": "selected_box_count",
            "observed_status": "complete",
            "max_ocr_engine_count": 30.0,
        },
        {
            "kind": "ocr_engine_count_budget_exceeded",
            "slug": "slow-profile",
            "metric": "selected_box_count",
            "observed_status": "complete",
            "count": 31.0,
            "max_ocr_engine_count": 30.0,
            "excess_count": 1.0,
        },
    ]


def test_latency_budget_skips_repeat_ocr_engine_p95_for_zero_call_profiles() -> None:
    report = stress_module.build_latency_budget_summary(
        rows=[],
        repeat_profile={
            "summary": {
                "ocr_engine_profile": {"calls": 0},
                "ocr_engine_stage_duration_s": {},
                "ocr_engine_count_metric": {},
            },
            "samples": [
                {
                    "slug": "pure-catalog",
                    "warmup": False,
                    "ocr_engine_profile": {"calls": 0, "calls_detail": []},
                },
                {
                    "slug": "pure-catalog",
                    "warmup": False,
                    "ocr_engine_profile": {"calls": 0, "calls_detail": []},
                },
            ],
        },
        max_repeat_ocr_engine_p95_duration_s={"total_s": 0.9},
        max_repeat_ocr_engine_p95_count={"selected_box_count": 30},
        max_repeat_ocr_engine_max_count={"selected_box_count": 30},
    )

    assert report["passed"] is True
    assert report["repeat_ocr_engine_p95_violations"] == []
    assert report["repeat_ocr_engine_count_p95_violations"] == []
    assert report["repeat_ocr_engine_count_max_violations"] == []


def test_repeat_profile_flags_output_signature_drift() -> None:
    samples = [
        {
            "slug": "drifty",
            "repeat_index": 1,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.52,
            "city": "Grand Rapids",
            "source": "ocr-georeference:nominatim-label-fit",
            "control_points": 5,
            "ocr_label_count": 56,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["Highway", "GENTEX"],
        },
        {
            "slug": "drifty",
            "repeat_index": 2,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.58,
            "city": "Inferred map area",
            "source": "ocr-georeference:nominatim-label-fit",
            "control_points": 4,
            "ocr_label_count": 49,
            "ocr_label_event": "Full-detail map labels read",
            "ocr_full_detail_retry": True,
            "ocr_top_labels": ["Highway"],
        },
    ]

    repeat_profile = stress_module.summarize_repeat_profile_samples(
        samples,
        runs_per_case=2,
        warmup_runs_per_case=0,
    )

    assert repeat_profile["summary"]["stable_signature_cases"] == 0
    assert repeat_profile["summary"]["unstable_signature_cases"] == ["drifty"]
    stability = repeat_profile["cases"]["drifty"]["signature_stability"]
    assert stability["stable"] is False
    assert stability["unique_signatures"] == 2
    assert [signature["count"] for signature in stability["signatures"]] == [1, 1]


def test_repeat_profile_flags_bbox_signature_drift() -> None:
    samples = [
        {
            "slug": "drifty-bbox",
            "repeat_index": 1,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.42,
            "city": "Miami",
            "source": "ocr-georeference:nominatim-label-fit",
            "control_points": 5,
            "bbox": [-80.33880121, 25.68860351, -80.11168631, 25.98301471],
            "ocr_label_count": 20,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["Miami"],
        },
        {
            "slug": "drifty-bbox",
            "repeat_index": 2,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.43,
            "city": "Miami",
            "source": "ocr-georeference:nominatim-label-fit",
            "control_points": 5,
            "bbox": [-80.3188012, 25.6886035, -80.0916863, 25.9830147],
            "ocr_label_count": 20,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["Miami"],
        },
    ]

    repeat_profile = stress_module.summarize_repeat_profile_samples(
        samples,
        runs_per_case=2,
        warmup_runs_per_case=0,
    )

    stability = repeat_profile["cases"]["drifty-bbox"]["signature_stability"]
    assert stability["stable"] is False
    assert stability["unique_signatures"] == 2
    assert sorted(signature["bbox"] for signature in stability["signatures"]) == [
        [-80.338801, 25.688604, -80.111686, 25.983015],
        [-80.318801, 25.688603, -80.091686, 25.983015],
    ]


def test_repeat_profile_flags_confidence_signature_drift() -> None:
    samples = [
        {
            "slug": "drifty-confidence",
            "repeat_index": 1,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.42,
            "city": "Los Angeles",
            "source": "ocr-georeference:nominatim-label-fit",
            "control_points": 18,
            "bbox": [-118.5240391, 33.9302354, -118.2193825, 34.1184628],
            "combined_confidence": 0.8794444,
            "georeference_confidence": 0.8794444,
            "ocr_label_count": 40,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["Brentwood"],
        },
        {
            "slug": "drifty-confidence",
            "repeat_index": 2,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.43,
            "city": "Los Angeles",
            "source": "ocr-georeference:nominatim-label-fit",
            "control_points": 18,
            "bbox": [-118.5240391, 33.9302354, -118.2193825, 34.1184628],
            "combined_confidence": 0.721,
            "georeference_confidence": 0.721,
            "ocr_label_count": 40,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["Brentwood"],
        },
    ]

    repeat_profile = stress_module.summarize_repeat_profile_samples(
        samples,
        runs_per_case=2,
        warmup_runs_per_case=0,
    )

    stability = repeat_profile["cases"]["drifty-confidence"]["signature_stability"]
    assert stability["stable"] is False
    assert stability["unique_signatures"] == 2
    assert sorted(signature["combined_confidence"] for signature in stability["signatures"]) == [
        0.721,
        0.879444,
    ]


def test_repeat_profile_flags_geojson_geometry_signature_drift() -> None:
    samples = [
        {
            "slug": "drifty-geometry",
            "repeat_index": 1,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.42,
            "city": "San Francisco",
            "source": "ocr-georeference:nominatim-label-fit",
            "control_points": 16,
            "bbox": [-122.4411264, 37.7478098, -122.3905477, 37.8058554],
            "geojson_geometry_hash": "stable-shape-a",
            "geojson_coordinate_count": 42,
            "combined_confidence": 0.953,
            "georeference_confidence": 0.953,
            "ocr_label_count": 21,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["San Francisco"],
        },
        {
            "slug": "drifty-geometry",
            "repeat_index": 2,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.43,
            "city": "San Francisco",
            "source": "ocr-georeference:nominatim-label-fit",
            "control_points": 16,
            "bbox": [-122.4411264, 37.7478098, -122.3905477, 37.8058554],
            "geojson_geometry_hash": "stable-shape-b",
            "geojson_coordinate_count": 42,
            "combined_confidence": 0.953,
            "georeference_confidence": 0.953,
            "ocr_label_count": 21,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["San Francisco"],
        },
    ]

    repeat_profile = stress_module.summarize_repeat_profile_samples(
        samples,
        runs_per_case=2,
        warmup_runs_per_case=0,
    )

    stability = repeat_profile["cases"]["drifty-geometry"]["signature_stability"]
    assert stability["stable"] is False
    assert stability["unique_signatures"] == 2
    assert sorted(signature["geojson_geometry_hash"] for signature in stability["signatures"]) == [
        "stable-shape-a",
        "stable-shape-b",
    ]


def test_repeat_profile_flags_road_match_signature_drift_but_ignores_elapsed_time() -> None:
    samples = [
        {
            "slug": "road-score",
            "repeat_index": 1,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.42,
            "city": "Nashville",
            "source": "ocr-georeference:nominatim-label-fit+osm-road-refine",
            "control_points": 3,
            "bbox": [-86.8475058, 36.1153856, -86.6964806, 36.2435026],
            "geojson_geometry_hash": "stable-shape",
            "geojson_coordinate_count": 90,
            "combined_confidence": 0.746501,
            "georeference_confidence": 0.746501,
            "road_match_score": 0.334126,
            "road_match_base_score": 0.260624,
            "road_match_sampled_points": 1348,
            "road_match_elapsed_s": 0.23,
            "ocr_label_count": 16,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["Nashville"],
        },
        {
            "slug": "road-score",
            "repeat_index": 2,
            "warmup": False,
            "expectation_passed": True,
            "observed_status": "complete",
            "total_elapsed_s": 0.43,
            "city": "Nashville",
            "source": "ocr-georeference:nominatim-label-fit+osm-road-refine",
            "control_points": 3,
            "bbox": [-86.8475058, 36.1153856, -86.6964806, 36.2435026],
            "geojson_geometry_hash": "stable-shape",
            "geojson_coordinate_count": 90,
            "combined_confidence": 0.746501,
            "georeference_confidence": 0.746501,
            "road_match_score": 0.340001,
            "road_match_base_score": 0.260624,
            "road_match_sampled_points": 1348,
            "road_match_elapsed_s": 0.99,
            "ocr_label_count": 16,
            "ocr_label_event": "Map labels read",
            "ocr_full_detail_retry": False,
            "ocr_top_labels": ["Nashville"],
        },
    ]

    repeat_profile = stress_module.summarize_repeat_profile_samples(
        samples,
        runs_per_case=2,
        warmup_runs_per_case=0,
    )

    stability = repeat_profile["cases"]["road-score"]["signature_stability"]
    assert stability["stable"] is False
    assert stability["unique_signatures"] == 2
    assert sorted(signature["road_match_score"] for signature in stability["signatures"]) == [
        0.334126,
        0.340001,
    ]
    assert "road_match_elapsed_s" not in stability["signatures"][0]


def test_geojson_geometry_summary_hashes_rounded_geometry(tmp_path) -> None:
    geojson = tmp_path / "shape.geojson"
    geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "ignored": "metadata",
                            "road_match_score": 0.334126,
                            "road_match_base_score": 0.260624,
                            "road_match_sampled_points": 1348,
                            "road_match_elapsed_s": 0.22955,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-122.44112641, 37.74780981],
                                    [-122.39054771, 37.80585541],
                                    [-122.44112641, 37.74780981],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )

    summary = stress_module.geojson_geometry_summary(geojson)
    geojson.write_text(geojson.read_text().replace("metadata", "changed metadata"))
    summary_after_metadata_change = stress_module.geojson_geometry_summary(geojson)

    assert summary["geojson_coordinate_count"] == 3
    assert isinstance(summary["geojson_geometry_hash"], str)
    assert len(summary["geojson_geometry_hash"]) == 16
    assert summary["road_match_score"] == 0.334126
    assert summary["road_match_base_score"] == 0.260624
    assert summary["road_match_sampled_points"] == 1348
    assert summary["road_match_elapsed_s"] == 0.22955
    assert summary_after_metadata_change == summary


def test_repeat_profile_signature_drift_cases_reads_summary() -> None:
    assert stress_module.repeat_profile_signature_drift_cases(
        {"repeat_profile": {"summary": {"unstable_signature_cases": ["drifty", 42]}}}
    ) == ["drifty", "42"]
    assert stress_module.repeat_profile_signature_drift_cases({}) == []
    assert stress_module.repeat_profile_signature_drift_cases(
        {"repeat_profile": {"summary": {"unstable_signature_cases": "drifty"}}}
    ) == []


def test_repeat_profile_unexpected_helpers_read_summary_and_cases() -> None:
    report = {
        "repeat_profile": {
            "summary": {"unexpected_samples": 3},
            "cases": {
                "stable": {"unexpected_samples": 0},
                "flaky": {"unexpected_samples": 2},
                "missing": {},
            },
        }
    }

    assert stress_module.repeat_profile_unexpected_sample_count(report) == 3
    assert stress_module.repeat_profile_unexpected_cases(report) == ["flaky"]

    assert (
        stress_module.repeat_profile_unexpected_sample_count(
            {"repeat_profile": {"cases": {"fallback": {"unexpected_samples": 2}}}}
        )
        == 2
    )
    assert stress_module.repeat_profile_unexpected_sample_count({}) == 0


def test_main_requires_repeat_runs_for_repeat_signature_drift_gate(monkeypatch) -> None:
    def fake_run_stress_benchmark(*args, **kwargs):
        raise AssertionError("stress benchmark should not run without repeat samples")

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    with pytest.raises(SystemExit) as exc:
        stress_module.main(["--fail-on-repeat-signature-drift"])

    assert exc.value.code == 2


def test_main_fails_when_repeat_signature_drift_is_detected(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert manifest_path == Path("manifest.json")
        assert out_dir == tmp_path / "out"
        assert kwargs["repeat_profile_runs"] == 2
        assert kwargs["repeat_profile_warmups"] == 1
        return {
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 0.7,
            },
            "rows": [
                {
                    "slug": "drifty-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 0.7,
                }
            ],
            "repeat_profile": {
                "summary": {
                    "analyzed_samples": 1,
                    "expectation_passed_samples": 1,
                    "subsecond_samples": 1,
                    "median_total_elapsed_s": 0.72,
                    "p95_total_elapsed_s": 0.72,
                    "max_total_elapsed_s": 0.72,
                    "unstable_signature_cases": ["drifty-map"],
                }
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--repeat-profile-runs",
            "2",
            "--repeat-profile-warmups",
            "1",
            "--fail-on-repeat-signature-drift",
        ]
    )

    assert exit_code == 1


def test_main_passes_repeat_signature_drift_gate_when_stable(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert kwargs["repeat_profile_runs"] == 1
        return {
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [
                {
                    "slug": "stable-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 0.6,
                }
            ],
            "repeat_profile": {
                "summary": {
                    "analyzed_samples": 1,
                    "expectation_passed_samples": 1,
                    "subsecond_samples": 1,
                    "median_total_elapsed_s": 0.62,
                    "p95_total_elapsed_s": 0.62,
                    "max_total_elapsed_s": 0.62,
                    "unstable_signature_cases": [],
                }
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--repeat-profile-runs",
            "1",
            "--fail-on-repeat-signature-drift",
        ]
    )

    assert exit_code == 0


def test_main_fails_when_requested_prewarm_fails(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert kwargs["prewarm_runtime"] is True
        return {
            "prewarm": {"status": "error", "error": "warmup failed"},
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [
                {
                    "slug": "stable-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 0.6,
                }
            ],
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--prewarm-runtime",
        ]
    )

    assert exit_code == 1


def test_main_accepts_successful_requested_prewarm(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert kwargs["prewarm_runtime"] is True
        assert kwargs["max_prewarm_runtime_s"] == 0.5
        assert kwargs["max_prewarm_stage_s"] == {
            "rapidocr_s": 1.2,
            "total_s": 1.5,
        }
        return {
            "prewarm": {"status": "ok", "total_s": 0.1},
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [
                {
                    "slug": "stable-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 0.6,
                }
            ],
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--prewarm-runtime",
            "--max-prewarm-runtime-s",
            "0.5",
            "--max-prewarm-stage-s",
            "rapidocr_s=1.2,total_elapsed_s=1.5",
        ]
    )

    assert exit_code == 0


def test_main_fails_on_repeat_profile_unexpected_when_requested(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert manifest_path == Path("manifest.json")
        assert out_dir == tmp_path / "out"
        assert kwargs["repeat_profile_runs"] == 2
        return {
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 0.7,
            },
            "rows": [
                {
                    "slug": "flaky-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 0.7,
                }
            ],
            "repeat_profile": {
                "summary": {
                    "analyzed_samples": 2,
                    "expectation_passed_samples": 1,
                    "unexpected_samples": 1,
                    "subsecond_samples": 2,
                    "median_total_elapsed_s": 0.64,
                    "p95_total_elapsed_s": 0.68,
                    "max_total_elapsed_s": 0.68,
                    "unstable_signature_cases": [],
                },
                "cases": {"flaky-map": {"unexpected_samples": 1}},
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--repeat-profile-runs",
            "2",
            "--fail-on-unexpected",
        ]
    )

    assert exit_code == 1


def test_main_fails_when_latency_budget_is_exceeded(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert manifest_path == Path("manifest.json")
        assert out_dir == tmp_path / "out"
        assert kwargs["max_total_elapsed_s"] == 1.0
        return {
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 1.2,
            },
            "rows": [
                {
                    "slug": "slow-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 1.2,
                }
            ],
            "latency_budget": {
                "max_total_elapsed_s": 1.0,
                "passed": False,
                "primary_violations": [{"slug": "slow-map", "total_elapsed_s": 1.2}],
                "repeat_violations": [],
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--max-total-elapsed-s",
            "1.0",
        ]
    )

    assert exit_code == 1


def test_main_fails_when_manifest_contract_budget_is_exceeded(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert manifest_path == Path("manifest.json")
        assert out_dir == tmp_path / "out"
        assert kwargs["min_ocr_call_contract_rows"] == 49
        assert kwargs["min_ocr_count_contract_rows"] == 12
        assert kwargs["max_positive_ocr_call_only_rows"] == 25
        assert kwargs["fail_on_invalid_ocr_count_contracts"] is True
        return {
            "manifest_contracts": {
                "total_cases": 49,
                "ocr_call_contract_rows": 49,
                "ocr_positive_call_contract_rows": 38,
                "ocr_zero_call_contract_rows": 11,
                "ocr_count_contract_rows": 12,
                "ocr_positive_call_rows_without_count_contract": ["call-only"] * 26,
            },
            "manifest_contract_budget": {
                "passed": False,
                "violations": [
                    {
                        "kind": "positive_ocr_call_only_rows_above_max",
                        "positive_ocr_call_only_rows": 26,
                        "max_positive_ocr_call_only_rows": 25,
                        "rows": ["call-only"] * 26,
                    }
                ],
            },
            "summary": {
                "total": 49,
                "expectation_passed": 49,
                "unexpected": [],
                "statuses": {"complete": 38, "failed": 11},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [],
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--min-ocr-call-contract-rows",
            "49",
            "--min-ocr-count-contract-rows",
            "12",
            "--max-positive-ocr-call-only-rows",
            "25",
            "--fail-on-invalid-ocr-count-contracts",
        ]
    )

    assert exit_code == 1


def test_main_applies_real_screenshot_hard_gate_preset(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert manifest_path == Path("benchmarks/real-screenshot-stress.json")
        assert out_dir == tmp_path / "out"
        assert kwargs["execution"] == "in-process"
        assert kwargs["profile_ocr_engine"] is True
        assert kwargs["prewarm_runtime"] is True
        assert kwargs["runner_ocr_cache"] is False
        assert kwargs["extraction_cache"] is False
        assert kwargs["repeat_profile_runs"] == 3
        assert kwargs["repeat_profile_warmups"] == 1
        assert kwargs["max_total_elapsed_s"] == 1.0
        assert kwargs["max_repeat_profile_p95_duration_s"] == 0.8
        assert kwargs["max_prewarm_runtime_s"] == 2.0
        assert kwargs["max_prewarm_stage_s"] == {
            "rapidocr_s": 1.8,
            "total_s": 2.0,
        }
        assert kwargs["max_repeat_ocr_engine_p95_duration_s"] == {
            "total_s": 0.7,
        }
        expected_count_budget = {
            "label_confidence_lt_90_count": 3.0,
            "label_count": 29.0,
            "raw_box_count": 50.0,
            "result_count": 29.0,
            "selected_box_count": 30.0,
        }
        assert kwargs["max_repeat_ocr_engine_p95_count"] == expected_count_budget
        assert kwargs["max_repeat_ocr_engine_max_count"] == expected_count_budget
        assert kwargs["min_ocr_call_contract_rows"] == 49
        assert kwargs["min_ocr_count_contract_rows"] == 12
        assert kwargs["max_positive_ocr_call_only_rows"] == 26
        assert kwargs["fail_on_invalid_ocr_count_contracts"] is True
        return {
            "prewarm": {"status": "ok", "total_s": 1.0},
            "manifest_contracts": {
                "total_cases": 49,
                "ocr_call_contract_rows": 49,
                "ocr_positive_call_contract_rows": 38,
                "ocr_zero_call_contract_rows": 11,
                "ocr_count_contract_rows": 12,
                "ocr_positive_call_rows_without_count_contract": ["call-only"] * 26,
            },
            "manifest_contract_budget": {
                "passed": True,
                "violations": [],
            },
            "summary": {
                "total": 49,
                "expectation_passed": 49,
                "unexpected": [],
                "statuses": {"complete": 38, "failed": 11},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [],
            "repeat_profile": {
                "summary": {
                    "analyzed_samples": 98,
                    "expectation_passed_samples": 98,
                    "unexpected_samples": 0,
                    "subsecond_samples": 98,
                    "median_total_elapsed_s": 0.42,
                    "p95_total_elapsed_s": 0.47,
                    "max_total_elapsed_s": 0.51,
                    "unstable_signature_cases": [],
                }
            },
            "latency_budget": {
                "passed": True,
                "primary_violations": [],
                "repeat_violations": [],
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--real-screenshot-hard-gate",
        ]
    )

    assert exit_code == 0


def test_real_screenshot_hard_gate_merges_metric_budget_overrides(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert kwargs["max_prewarm_stage_s"] == {
            "rapidocr_s": 1.7,
            "total_s": 2.0,
        }
        assert kwargs["max_repeat_ocr_engine_p95_duration_s"] == {
            "total_s": 0.6,
        }
        assert kwargs["max_repeat_ocr_engine_p95_count"] == {
            "label_confidence_lt_90_count": 3.0,
            "label_count": 29.0,
            "raw_box_count": 50.0,
            "result_count": 29.0,
            "selected_box_count": 28.0,
        }
        assert kwargs["max_repeat_ocr_engine_max_count"] == {
            "label_confidence_lt_90_count": 3.0,
            "label_count": 28.0,
            "raw_box_count": 50.0,
            "result_count": 29.0,
            "selected_box_count": 30.0,
        }
        return {
            "prewarm": {"status": "ok", "total_s": 1.0},
            "manifest_contract_budget": {
                "passed": True,
                "violations": [],
            },
            "summary": {
                "total": 49,
                "expectation_passed": 49,
                "unexpected": [],
                "statuses": {"complete": 38, "failed": 11},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [],
            "repeat_profile": {
                "summary": {
                    "unexpected_samples": 0,
                    "unstable_signature_cases": [],
                }
            },
            "latency_budget": {
                "passed": True,
                "primary_violations": [],
                "repeat_violations": [],
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--real-screenshot-hard-gate",
            "--max-prewarm-stage-s",
            "rapidocr_s=1.7",
            "--max-repeat-ocr-engine-p95-duration-s",
            "total_s=0.6",
            "--max-repeat-ocr-engine-p95-count",
            "selected_box_count=28",
            "--max-repeat-ocr-engine-max-count",
            "label_count=28",
        ]
    )

    assert exit_code == 0


def test_main_rejects_real_screenshot_hard_gate_with_only(monkeypatch) -> None:
    def fake_run_stress_benchmark(*args, **kwargs):
        raise AssertionError("stress benchmark should not run when hard gate is combined with --only")

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    with pytest.raises(SystemExit) as exc:
        stress_module.main(["--real-screenshot-hard-gate", "--only", "dallas-waymo"])

    assert exc.value.code == 2


def test_main_rejects_nonpositive_repeat_profile_p95_budget(monkeypatch) -> None:
    def fake_run_stress_benchmark(*args, **kwargs):
        raise AssertionError("stress benchmark should not run with an invalid p95 budget")

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    with pytest.raises(SystemExit) as exc:
        stress_module.main(["--max-repeat-profile-p95-duration-s", "0"])

    assert exc.value.code == 2


def test_main_rejects_nonpositive_prewarm_budget(monkeypatch) -> None:
    def fake_run_stress_benchmark(*args, **kwargs):
        raise AssertionError("stress benchmark should not run with an invalid prewarm budget")

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    with pytest.raises(SystemExit) as exc:
        stress_module.main(["--max-prewarm-runtime-s", "0"])

    assert exc.value.code == 2


def test_main_rejects_invalid_prewarm_stage_budget(monkeypatch) -> None:
    def fake_run_stress_benchmark(*args, **kwargs):
        raise AssertionError("stress benchmark should not run with an invalid prewarm stage budget")

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    with pytest.raises(SystemExit) as exc:
        stress_module.main(["--max-prewarm-stage-s", "rapidocr_s=0"])

    assert exc.value.code == 2


def test_main_rejects_negative_manifest_contract_budgets(monkeypatch) -> None:
    def fake_run_stress_benchmark(*args, **kwargs):
        raise AssertionError("stress benchmark should not run with invalid manifest contract budgets")

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    for flag in (
        "--min-ocr-call-contract-rows",
        "--min-ocr-count-contract-rows",
        "--max-positive-ocr-call-only-rows",
    ):
        with pytest.raises(SystemExit) as exc:
            stress_module.main([flag, "-1"])
        assert exc.value.code == 2


def test_latency_budget_flags_repeat_profile_p95_excess_and_missing() -> None:
    repeat_profile = {
        "summary": {
            "p95_total_elapsed_s": 0.92,
        },
        "samples": [
            {"slug": "zoox-tall", "warmup": False, "total_elapsed_s": 0.8},
        ],
    }

    budget = stress_module.build_latency_budget_summary(
        [],
        repeat_profile,
        max_repeat_profile_p95_duration_s=0.8,
    )

    assert budget["passed"] is False
    assert budget["max_repeat_profile_p95_duration_s"] == 0.8
    assert budget["repeat_p95_violations"] == [
        {
            "kind": "repeat_profile_p95_budget_exceeded",
            "p95_total_elapsed_s": 0.92,
            "max_repeat_profile_p95_duration_s": 0.8,
            "excess_s": 0.12,
        }
    ]

    missing_budget = stress_module.build_latency_budget_summary(
        [],
        None,
        max_repeat_profile_p95_duration_s=0.8,
    )

    assert missing_budget["passed"] is False
    assert missing_budget["repeat_p95_violations"] == [
        {
            "kind": "repeat_profile_p95_missing",
            "max_repeat_profile_p95_duration_s": 0.8,
        }
    ]


def test_latency_budget_flags_prewarm_excess_and_missing() -> None:
    budget = stress_module.build_latency_budget_summary(
        [],
        None,
        prewarm={"status": "ok", "total_s": 1.23},
        max_prewarm_runtime_s=1.0,
    )

    assert budget["passed"] is False
    assert budget["max_prewarm_runtime_s"] == 1.0
    assert budget["prewarm_violations"] == [
        {
            "kind": "prewarm_runtime_budget_exceeded",
            "prewarm_total_s": 1.23,
            "max_prewarm_runtime_s": 1.0,
            "excess_s": 0.23,
        }
    ]

    missing_budget = stress_module.build_latency_budget_summary(
        [],
        None,
        max_prewarm_runtime_s=1.0,
    )

    assert missing_budget["passed"] is False
    assert missing_budget["prewarm_violations"] == [
        {
            "kind": "prewarm_runtime_missing",
            "max_prewarm_runtime_s": 1.0,
        }
    ]


def test_parse_prewarm_stage_duration_budgets_accepts_repeated_and_comma_values() -> None:
    assert stress_module.parse_prewarm_stage_duration_budgets(
        ["rapidocr_s=1.2, extraction_s=0.05", "total_elapsed_s=1.5"]
    ) == {
        "extraction_s": 0.05,
        "rapidocr_s": 1.2,
        "total_s": 1.5,
    }


def test_parse_prewarm_stage_duration_budgets_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="METRIC=SECONDS"):
        stress_module.parse_prewarm_stage_duration_budgets(["rapidocr_s:1.2"])
    with pytest.raises(ValueError, match="positive"):
        stress_module.parse_prewarm_stage_duration_budgets(["rapidocr_s=0"])
    with pytest.raises(ValueError, match="Unknown prewarm stage metric"):
        stress_module.parse_prewarm_stage_duration_budgets(["det_elapsed_s=0.6"])


def test_latency_budget_flags_prewarm_stage_excess_and_missing() -> None:
    budget = stress_module.build_latency_budget_summary(
        [],
        None,
        prewarm={"status": "ok", "rapidocr_s": 1.32},
        max_prewarm_stage_s={
            "extraction_s": 0.05,
            "rapidocr_s": 1.2,
        },
    )

    assert budget["passed"] is False
    assert budget["max_prewarm_stage_s"] == {
        "extraction_s": 0.05,
        "rapidocr_s": 1.2,
    }
    assert budget["prewarm_stage_violations"] == [
        {
            "kind": "prewarm_stage_missing",
            "metric": "extraction_s",
            "max_prewarm_stage_s": 0.05,
        },
        {
            "kind": "prewarm_stage_budget_exceeded",
            "metric": "rapidocr_s",
            "duration_s": 1.32,
            "max_prewarm_stage_s": 1.2,
            "excess_s": 0.12,
        },
    ]

    missing_budget = stress_module.build_latency_budget_summary(
        [],
        None,
        max_prewarm_stage_s={"rapidocr_s": 1.2},
    )

    assert missing_budget["passed"] is False
    assert missing_budget["prewarm_stage_violations"] == [
        {
            "kind": "prewarm_stage_missing",
            "metric": "rapidocr_s",
            "max_prewarm_stage_s": 1.2,
        }
    ]


def test_stress_benchmark_can_gate_repeat_profile_p95(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "kept",
                        "image": str(image),
                        "expect": {"status": "complete", "source_prefix": "ocr-georeference:"},
                    }
                ]
            }
        )
    )
    durations = iter([0.7, 0.9, 0.8])

    def fake_run_case(
        case,
        out_dir,
        *,
        timeout_seconds,
        write_debug,
        profile_ocr_engine,
        runner_ocr_cache,
        extraction_cache,
        execution,
        python_executable,
    ):
        duration = next(durations)
        return {
            "slug": case["slug"],
            "image": case["image"],
            "expected_status": "complete",
            "observed_status": "complete",
            "expectation_passed": True,
            "source": "ocr-georeference:nominatim-label-fit",
            "total_elapsed_s": duration,
        }

    monkeypatch.setattr(stress_module, "run_stress_case", fake_run_case)

    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        repeat_profile_runs=2,
        max_repeat_profile_p95_duration_s=0.82,
        python_executable="python",
    )

    assert report["latency_budget"]["passed"] is False
    assert report["latency_budget"]["repeat_p95_violations"] == [
        {
            "kind": "repeat_profile_p95_budget_exceeded",
            "p95_total_elapsed_s": 0.895,
            "max_repeat_profile_p95_duration_s": 0.82,
            "excess_s": 0.075,
        }
    ]


def test_main_passes_repeat_profile_p95_budget_to_runner(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert manifest_path == Path("manifest.json")
        assert out_dir == tmp_path / "out"
        assert kwargs["max_repeat_profile_p95_duration_s"] == 0.8
        return {
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [
                {
                    "slug": "stable-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 0.6,
                }
            ],
            "latency_budget": {
                "passed": True,
                "primary_violations": [],
                "repeat_violations": [],
                "max_repeat_profile_p95_duration_s": 0.8,
                "repeat_p95_violations": [],
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--repeat-profile-runs",
            "2",
            "--max-repeat-profile-p95-duration-s",
            "0.8",
        ]
    )

    assert exit_code == 0


def test_parse_metric_duration_budgets_accepts_repeated_and_comma_values() -> None:
    assert stress_module.parse_metric_duration_budgets(
        ["det_elapsed_s=0.3, rec_elapsed_s=0.6", "total_elapsed_s=0.9"]
    ) == {
        "det_elapsed_s": 0.3,
        "rec_elapsed_s": 0.6,
        "total_s": 0.9,
    }


def test_parse_metric_duration_budgets_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="METRIC=SECONDS"):
        stress_module.parse_metric_duration_budgets(["rec_elapsed_s:0.6"])
    with pytest.raises(ValueError, match="positive"):
        stress_module.parse_metric_duration_budgets(["rec_elapsed_s=0"])
    with pytest.raises(ValueError, match="Unknown OCR engine duration metric"):
        stress_module.parse_metric_duration_budgets(["unknown_elapsed_s=0.6"])


def test_latency_budget_flags_repeat_ocr_engine_p95_excess_and_missing() -> None:
    repeat_profile = {
        "summary": {
            "ocr_engine_stage_duration_s": {
                "rec_elapsed_s": {
                    "samples": 3,
                    "p95_duration_s": 0.72,
                }
            }
        },
        "samples": [
            {"slug": "zoox-tall", "warmup": False, "total_elapsed_s": 0.8},
        ],
    }

    budget = stress_module.build_latency_budget_summary(
        [],
        repeat_profile,
        max_repeat_ocr_engine_p95_duration_s={
            "det_elapsed_s": 0.3,
            "rec_elapsed_s": 0.6,
        },
    )

    assert budget["passed"] is False
    assert budget["primary_violations"] == []
    assert budget["repeat_violations"] == []
    assert budget["max_repeat_ocr_engine_p95_duration_s"] == {
        "det_elapsed_s": 0.3,
        "rec_elapsed_s": 0.6,
    }
    assert budget["repeat_ocr_engine_p95_violations"] == [
        {
            "kind": "repeat_ocr_engine_p95_missing",
            "metric": "det_elapsed_s",
            "max_repeat_ocr_engine_p95_duration_s": 0.3,
        },
        {
            "kind": "repeat_ocr_engine_p95_budget_exceeded",
            "metric": "rec_elapsed_s",
            "p95_duration_s": 0.72,
            "max_repeat_ocr_engine_p95_duration_s": 0.6,
            "excess_s": 0.12,
        },
    ]


def test_latency_budget_flags_primary_ocr_engine_excess_and_missing() -> None:
    budget = stress_module.build_latency_budget_summary(
        [
            {
                "slug": "slow-map",
                "observed_status": "complete",
                "ocr_engine_profile": {
                    "total_s": 0.72,
                    "selected_box_count": 31,
                },
            },
            {
                "slug": "unprofiled-map",
                "observed_status": "failed",
            },
        ],
        repeat_profile=None,
        max_ocr_engine_duration_s={"total_s": 0.6},
        max_ocr_engine_count={"selected_box_count": 30},
    )

    assert budget["passed"] is False
    assert budget["max_ocr_engine_duration_s"] == {"total_s": 0.6}
    assert budget["ocr_engine_violations"] == [
        {
            "kind": "ocr_engine_duration_budget_exceeded",
            "slug": "slow-map",
            "metric": "total_s",
            "observed_status": "complete",
            "duration_s": 0.72,
            "max_ocr_engine_duration_s": 0.6,
            "excess_s": 0.12,
        },
        {
            "kind": "ocr_engine_duration_missing",
            "slug": "unprofiled-map",
            "metric": "total_s",
            "observed_status": "failed",
            "max_ocr_engine_duration_s": 0.6,
        },
    ]
    assert budget["max_ocr_engine_count"] == {"selected_box_count": 30.0}
    assert budget["ocr_engine_count_violations"] == [
        {
            "kind": "ocr_engine_count_budget_exceeded",
            "slug": "slow-map",
            "metric": "selected_box_count",
            "observed_status": "complete",
            "count": 31.0,
            "max_ocr_engine_count": 30.0,
            "excess_count": 1.0,
        },
        {
            "kind": "ocr_engine_count_missing",
            "slug": "unprofiled-map",
            "metric": "selected_box_count",
            "observed_status": "failed",
            "max_ocr_engine_count": 30.0,
        },
    ]


def test_stress_benchmark_can_gate_primary_ocr_engine_duration(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "kept",
                        "image": str(image),
                        "expect": {"status": "complete", "source_prefix": "ocr-georeference:"},
                    }
                ]
            }
        )
    )

    def fake_run_case(
        case,
        out_dir,
        *,
        timeout_seconds,
        write_debug,
        profile_ocr_engine,
        runner_ocr_cache,
        extraction_cache,
        execution,
        python_executable,
    ):
        return {
            "slug": case["slug"],
            "image": case["image"],
            "expected_status": "complete",
            "observed_status": "complete",
            "expectation_passed": True,
            "source": "ocr-georeference:nominatim-label-fit",
            "total_elapsed_s": 0.8,
            "ocr_engine_profile": {
                "calls": 1,
                "total_s": 0.7,
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_case", fake_run_case)

    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        profile_ocr_engine=True,
        max_ocr_engine_duration_s={"total_s": 0.6},
        python_executable="python",
    )

    assert report["latency_budget"]["passed"] is False
    assert report["latency_budget"]["ocr_engine_violations"] == [
        {
            "kind": "ocr_engine_duration_budget_exceeded",
            "slug": "kept",
            "metric": "total_s",
            "observed_status": "complete",
            "duration_s": 0.7,
            "max_ocr_engine_duration_s": 0.6,
            "excess_s": 0.1,
        }
    ]


def test_stress_benchmark_can_gate_repeat_ocr_engine_p95(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "kept",
                        "image": str(image),
                        "expect": {"status": "complete", "source_prefix": "ocr-georeference:"},
                    }
                ]
            }
        )
    )
    durations = iter([0.7, 0.9, 0.8])

    def fake_run_case(
        case,
        out_dir,
        *,
        timeout_seconds,
        write_debug,
        profile_ocr_engine,
        runner_ocr_cache,
        extraction_cache,
        execution,
        python_executable,
    ):
        duration = next(durations)
        return {
            "slug": case["slug"],
            "image": case["image"],
            "expected_status": "complete",
            "observed_status": "complete",
            "expectation_passed": True,
            "source": "ocr-georeference:nominatim-label-fit",
            "total_elapsed_s": duration,
            "ocr_engine_profile": {
                "calls": 1,
                "det_elapsed_s": 0.2,
                "rec_elapsed_s": duration,
                "total_s": duration,
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_case", fake_run_case)

    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        repeat_profile_runs=2,
        profile_ocr_engine=True,
        max_repeat_ocr_engine_p95_duration_s={"rec_elapsed_s": 0.82},
        python_executable="python",
    )

    assert report["latency_budget"]["passed"] is False
    assert report["latency_budget"]["repeat_ocr_engine_p95_violations"] == [
        {
            "kind": "repeat_ocr_engine_p95_budget_exceeded",
            "metric": "rec_elapsed_s",
            "p95_duration_s": 0.895,
            "max_repeat_ocr_engine_p95_duration_s": 0.82,
            "excess_s": 0.075,
        }
    ]


def test_main_passes_repeat_ocr_engine_p95_budget_to_runner(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert manifest_path == Path("manifest.json")
        assert out_dir == tmp_path / "out"
        assert kwargs["max_repeat_ocr_engine_p95_duration_s"] == {
            "rec_elapsed_s": 0.6,
            "total_s": 0.8,
        }
        return {
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [
                {
                    "slug": "stable-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 0.6,
                }
            ],
            "latency_budget": {
                "passed": True,
                "primary_violations": [],
                "repeat_violations": [],
                "max_repeat_ocr_engine_p95_duration_s": {
                    "rec_elapsed_s": 0.6,
                    "total_s": 0.8,
                },
                "repeat_ocr_engine_p95_violations": [],
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--repeat-profile-runs",
            "2",
            "--max-repeat-ocr-engine-p95-duration-s",
            "rec_elapsed_s=0.6,total_elapsed_s=0.8",
        ]
    )

    assert exit_code == 0


def test_main_passes_primary_ocr_engine_budgets_to_runner(tmp_path, monkeypatch) -> None:
    def fake_run_stress_benchmark(manifest_path, out_dir, **kwargs):
        assert manifest_path == Path("manifest.json")
        assert out_dir == tmp_path / "out"
        assert kwargs["max_ocr_engine_duration_s"] == {
            "det_elapsed_s": 0.3,
            "total_s": 0.6,
        }
        assert kwargs["max_ocr_engine_count"] == {
            "raw_box_count": 50.0,
            "selected_box_count": 30.0,
        }
        return {
            "summary": {
                "total": 1,
                "expectation_passed": 1,
                "unexpected": [],
                "statuses": {"complete": 1},
                "max_total_elapsed_s": 0.6,
            },
            "rows": [
                {
                    "slug": "stable-map",
                    "observed_status": "complete",
                    "expectation_passed": True,
                    "source": "ocr-georeference:nominatim-label-fit",
                    "control_points": 5,
                    "total_elapsed_s": 0.6,
                }
            ],
            "latency_budget": {
                "passed": True,
                "primary_violations": [],
                "repeat_violations": [],
                "max_ocr_engine_duration_s": {
                    "det_elapsed_s": 0.3,
                    "total_s": 0.6,
                },
                "ocr_engine_violations": [],
                "max_ocr_engine_count": {
                    "raw_box_count": 50.0,
                    "selected_box_count": 30.0,
                },
                "ocr_engine_count_violations": [],
            },
        }

    monkeypatch.setattr(stress_module, "run_stress_benchmark", fake_run_stress_benchmark)

    exit_code = stress_module.main(
        [
            "--manifest",
            "manifest.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--max-ocr-engine-duration-s",
            "det_elapsed_s=0.3,total_elapsed_s=0.6",
            "--max-ocr-engine-count",
            "selected_box_count=30,raw_box_count=50",
        ]
    )

    assert exit_code == 0


def test_repeat_profile_duration_stats_record_tail_percentiles() -> None:
    stats = stress_module.repeat_profile_total_elapsed_stats(
        [
            {"total_elapsed_s": 1.0},
            {"total_elapsed_s": 2.0},
            {"total_elapsed_s": 4.0},
        ]
    )

    assert stats["p90_total_elapsed_s"] == 3.6
    assert stats["p95_total_elapsed_s"] == 3.8
    assert stress_module.repeat_profile_stage_duration_distribution([0.1, 0.2, 0.4]) == {
        "min_duration_s": 0.1,
        "median_duration_s": 0.2,
        "average_duration_s": 0.233333,
        "p90_duration_s": 0.36,
        "p95_duration_s": 0.38,
        "max_duration_s": 0.4,
    }


def test_repeat_profile_slowest_samples_summarizes_actionable_context() -> None:
    samples = [
        {
            "slug": "fast",
            "repeat_index": 1,
            "total_elapsed_s": 0.31,
            "observed_status": "complete",
            "expectation_passed": True,
            "stages": {"ocr": 0.12, "extract": 0.08},
        },
        {
            "slug": "slow",
            "repeat_index": 2,
            "total_elapsed_s": 0.91,
            "observed_status": "complete",
            "expectation_passed": True,
            "stages": {"ocr": 0.74, "extract": 0.05},
            "ocr_label_count": 37,
            "ocr_label_event": "Map labels read",
            "ocr_top_labels": ["Zoox", "Las Vegas", "Paradise", "Airport", "Strip", "Extra"],
            "ocr_engine_profile": {
                "calls": 1,
                "det_elapsed_s": 0.2,
                "rec_elapsed_s": 0.45,
                "total_s": 0.72,
                "selected_box_count": 37,
                "useful_label_count": 18,
                "selected_box_area_p50": 1180.0,
                "selected_box_area_lt_1300_count": 21,
            },
        },
    ]

    slowest = stress_module.repeat_profile_slowest_samples(samples, limit=1)

    assert slowest == [
        {
            "slug": "slow",
            "repeat_index": 2,
            "total_elapsed_s": 0.91,
            "observed_status": "complete",
            "expectation_passed": True,
            "top_stage": {"stage": "ocr", "elapsed_s": 0.74},
            "ocr_label_count": 37,
            "ocr_label_event": "Map labels read",
            "ocr_top_labels": ["Zoox", "Las Vegas", "Paradise", "Airport", "Strip"],
            "ocr_engine": {
                "det_elapsed_s": 0.2,
                "rec_elapsed_s": 0.45,
                "total_s": 0.72,
                "calls": 1,
                "selected_box_count": 37,
                "useful_label_count": 18,
                "selected_box_area_p50": 1180.0,
                "selected_box_area_lt_1300_count": 21,
            },
        }
    ]
    assert (
        stress_module.repeat_profile_slow_sample_text(slowest[0])
        == "slow#2=0.910s ocr=0.740s rec=0.450s ocr_total=0.720s sel_area_p50=1180 sel_lt1300=21"
    )


def test_run_stress_benchmark_supports_in_process_execution(tmp_path, monkeypatch) -> None:
    image = tmp_path / "map.png"
    image.write_bytes(b"not a real image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "in-process-map",
                        "image": str(image),
                        "filename_hint": "custom-upload.png",
                        "source_was_svg": True,
                        "expect": {
                            "status": "complete",
                            "source_prefix": "ocr-georeference:",
                            "city_equals": "Houston",
                            "min_control_points": 4,
                        },
                    }
                ]
            }
        )
    )
    calls = []

    def fake_build_boundary(image_path, city, output_path, *, debug_dir, options, progress):
        assert os.environ[RUNNER_OCR_CACHE_ENV] == "0"
        assert os.environ[EXTRACTION_CACHE_ENV] == "0"
        calls.append(
            {
                "image_path": image_path,
                "city": city,
                "output_path": output_path,
                "debug_dir": debug_dir,
                "allow_catalog": options.allow_catalog,
                "filename_hint": options.filename_hint,
                "source_was_svg": options.source_was_svg,
            }
        )
        progress({"stage": "inspect", "message": "Reading image metadata", "percent": 5})
        progress(
            {
                "stage": "extract",
                "message": "Extracting service-area pixels",
                "percent": 35,
                "details": {"width": 1200, "height": 900},
            }
        )
        progress(
            {
                "stage": "ocr",
                "message": "Map labels read",
                "percent": 47,
                "details": {"label_count": 6, "top_labels": ["Houston", "Montrose"]},
            }
        )
        progress({"stage": "complete", "message": "Boundary ready", "percent": 100})
        return SimpleNamespace(
            summary={
                "city": "Houston",
                "style": "bright-blue",
                "georeference_source": "ocr-georeference:nominatim-label-fit",
                "combined_confidence": 0.92,
                "georeference_confidence": 0.89,
                "control_points": 5,
                "catalog_slug": "houston-waymo",
                "catalog_shape_iou": 0.972838,
                "catalog_shape_margin": 0.288825,
                "catalog_area_ratio": 0.990459,
                "bbox": [-96, 29, -95, 30],
            }
        )

    monkeypatch.setattr(stress_module, "build_boundary", fake_build_boundary)

    report = stress_module.run_stress_benchmark(
        manifest,
        tmp_path / "out",
        execution="in-process",
        runner_ocr_cache=False,
        extraction_cache=False,
        python_executable="python",
    )

    row = report["rows"][0]
    assert report["execution"] == "in-process"
    assert report["runner_ocr_cache"] is False
    assert report["extraction_cache"] is False
    assert row["execution"] == "in-process"
    assert row["runner_ocr_cache"] is False
    assert row["extraction_cache"] is False
    assert row["no_catalog"] is True
    assert row["filename_hint"] == "custom-upload.png"
    assert row["source_was_svg"] is True
    assert row["expectation_passed"] is True
    assert row["observed_status"] == "complete"
    assert row["source"] == "ocr-georeference:nominatim-label-fit"
    assert row["city"] == "Houston"
    assert row["control_points"] == 5
    assert row["catalog_slug"] == "houston-waymo"
    assert row["catalog_shape_iou"] == 0.972838
    assert row["catalog_shape_margin"] == 0.288825
    assert row["catalog_area_ratio"] == 0.990459
    assert row["image_width"] == 1200
    assert row["image_height"] == 900
    assert row["ocr_label_count"] == 6
    assert row["ocr_top_labels"] == ["Houston", "Montrose"]
    assert row["command"][0] == "in-process"
    assert "--source-was-svg" in row["command"]
    assert row["timeout_seconds"] == 30.0
    assert calls == [
        {
            "image_path": image,
            "city": None,
            "output_path": tmp_path / "out" / "in-process-map.geojson",
            "debug_dir": None,
            "allow_catalog": False,
            "filename_hint": "custom-upload.png",
            "source_was_svg": True,
        }
    ]
    assert set(row["stages"]) >= {"extract", "inspect", "ocr"}
    assert RUNNER_OCR_CACHE_ENV not in os.environ


def test_stress_cli_command_can_pass_source_was_svg(tmp_path) -> None:
    image = tmp_path / "browser-rasterized.png"
    output = tmp_path / "boundary.geojson"

    command = stress_module.build_cli_command(
        {
            "slug": "browser-svg",
            "image": str(image),
            "filename_hint": "a.png",
            "source_was_svg": True,
        },
        image,
        output,
        tmp_path,
        write_debug=False,
        profile_ocr_engine=False,
        python_executable="python",
    )

    assert "--source-was-svg" in command


def test_summarize_rows_records_stage_totals_ocr_events_and_max_cases() -> None:
    summary = stress_module.summarize_rows(
        [
            {
                "slug": "ann-arbor",
                "observed_status": "complete",
                "expectation_passed": True,
                "source": "ocr-georeference:nominatim-label-fit",
                "total_elapsed_s": 1.3,
                "stages": {"ocr": 0.9, "extract": 0.3},
                "ocr_label_events": [
                    {"message": "Focused map labels read", "label_count": 4},
                    {"message": "Map labels read", "label_count": 4},
                ],
                "ocr_full_detail_retry": False,
            },
            {
                "slug": "bay-area",
                "observed_status": "complete",
                "expectation_passed": True,
                "source": "ocr-georeference:nominatim-label-fit",
                "total_elapsed_s": 1.5,
                "stages": {"ocr": 1.1, "extract": 0.2},
                "ocr_label_events": [
                    {"message": "Focused map labels read", "label_count": 2},
                    {"message": "Full-detail map labels read", "label_count": 12},
                ],
                "ocr_full_detail_retry": True,
            },
        ]
    )

    assert summary["ocr_label_event_counts"] == {
        "Focused map labels read": 2,
        "Full-detail map labels read": 1,
        "Map labels read": 1,
    }
    assert summary["ocr_full_detail_retry_count"] == 1
    assert summary["ocr_full_detail_retry_rows"] == ["bay-area"]
    assert summary["stage_duration_s"] == {"extract": 0.5, "ocr": 2.0}
    assert summary["stage_max_rows"] == {
        "extract": {"slug": "ann-arbor", "elapsed_s": 0.3},
        "ocr": {"slug": "bay-area", "elapsed_s": 1.1},
    }


def test_select_cases_rejects_unknown_slug() -> None:
    try:
        stress_module.select_cases([{"slug": "known"}], ["missing"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected unknown slug to raise ValueError")
