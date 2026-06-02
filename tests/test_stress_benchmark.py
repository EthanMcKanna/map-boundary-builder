import json
import subprocess
from pathlib import Path

import map_boundary_builder.stress_benchmark as stress_module


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
                                "message": "Map labels read",
                                "details": {
                                    "label_count": 2,
                                    "top_labels": ["Zoox", "Las Vegas"],
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
    assert row["ocr_label_count"] == 2
    assert row["ocr_top_labels"] == ["Zoox", "Las Vegas"]


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

    def fake_run_case(case, out_dir, *, timeout_seconds, write_debug, python_executable):
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
    assert saved["summary"]["stage_duration_s"] == {}
    assert saved["summary"]["stage_max_rows"] == {}


def test_summarize_rows_records_stage_totals_and_max_cases() -> None:
    summary = stress_module.summarize_rows(
        [
            {
                "slug": "ann-arbor",
                "observed_status": "complete",
                "expectation_passed": True,
                "source": "ocr-georeference:nominatim-label-fit",
                "total_elapsed_s": 1.3,
                "stages": {"ocr": 0.9, "extract": 0.3},
            },
            {
                "slug": "bay-area",
                "observed_status": "complete",
                "expectation_passed": True,
                "source": "ocr-georeference:nominatim-label-fit",
                "total_elapsed_s": 1.5,
                "stages": {"ocr": 1.1, "extract": 0.2},
            },
        ]
    )

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
