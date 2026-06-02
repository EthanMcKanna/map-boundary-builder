import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
    assert row["expectation_passed"] is True
    assert row["source"] == "ocr-georeference:nominatim-label-fit"
    assert row["total_elapsed_s"] == 0.612345
    assert row["stages"] == {"ocr": 0.4}
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
                "calls_detail": [
                    {
                        "det_elapsed_s": round(duration / 10, 6),
                        "rec_elapsed_s": round(duration / 5, 6),
                        "total_s": round(duration / 4, 6),
                        "input_shape": [900, 1200],
                        "detector_limit": 608,
                        "selected_box_count": 3,
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
    assert repeat_profile["summary"]["min_total_elapsed_s"] == 0.8
    assert repeat_profile["summary"]["median_total_elapsed_s"] == 0.8
    assert repeat_profile["summary"]["stage_duration_s"] == {
        "ocr": {
            "samples": 1,
            "min_duration_s": 0.4,
            "median_duration_s": 0.4,
            "average_duration_s": 0.4,
            "max_duration_s": 0.4,
        }
    }
    assert repeat_profile["summary"]["ocr_engine_profile"] == {
        "fixtures": 1,
        "calls": 1,
        "det_elapsed_s": 0.08,
        "rec_elapsed_s": 0.16,
        "total_s": 0.2,
        "raw_box_count": 3,
    }
    assert repeat_profile["summary"]["ocr_engine_stage_max_rows"] == {
        "det_elapsed_s": {
            "slug": "kept",
            "elapsed_s": 0.08,
            "input_shape": [900, 1200],
            "detector_limit": 608,
            "selected_box_count": 3,
        },
        "rec_elapsed_s": {
            "slug": "kept",
            "elapsed_s": 0.16,
            "input_shape": [900, 1200],
            "detector_limit": 608,
            "selected_box_count": 3,
        },
        "total_s": {
            "slug": "kept",
            "elapsed_s": 0.2,
            "input_shape": [900, 1200],
            "detector_limit": 608,
            "selected_box_count": 3,
        },
    }
    assert repeat_profile["cases"]["kept"]["max_total_elapsed_s"] == 0.8
    assert repeat_profile["samples"][0]["warmup"] is True
    assert repeat_profile["samples"][1]["repeat_index"] == 2


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
    assert row["expectation_passed"] is True
    assert row["observed_status"] == "complete"
    assert row["source"] == "ocr-georeference:nominatim-label-fit"
    assert row["city"] == "Houston"
    assert row["control_points"] == 5
    assert row["image_width"] == 1200
    assert row["image_height"] == 900
    assert row["ocr_label_count"] == 6
    assert row["ocr_top_labels"] == ["Houston", "Montrose"]
    assert row["command"][0] == "in-process"
    assert row["timeout_seconds"] == 30.0
    assert calls == [
        {
            "image_path": image,
            "city": None,
            "output_path": tmp_path / "out" / "in-process-map.geojson",
            "debug_dir": None,
            "allow_catalog": False,
            "filename_hint": "custom-upload.png",
        }
    ]
    assert set(row["stages"]) >= {"extract", "inspect", "ocr"}
    assert RUNNER_OCR_CACHE_ENV not in os.environ


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
