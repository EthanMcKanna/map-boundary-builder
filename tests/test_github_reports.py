from map_boundary_builder.github_reports import GenerationReport, issue_body, safe_report_extension


def test_safe_report_extension_preserves_avif() -> None:
    assert safe_report_extension("uploaded-map.avif") == ".avif"


def test_safe_report_extension_preserves_bmp() -> None:
    assert safe_report_extension("uploaded-map.bmp") == ".bmp"


def test_safe_report_extension_preserves_tiff() -> None:
    assert safe_report_extension("uploaded-map.tif") == ".tif"
    assert safe_report_extension("uploaded-map.tiff") == ".tiff"


def test_issue_body_includes_runtime_profile() -> None:
    report = GenerationReport(
        filename="slow-map.png",
        image_bytes=b"image",
        error="Boundary took too long.",
        run_id="run-1",
        profile={
            "upload_bytes": 750000,
            "build_boundary_s": 0.52,
            "build_stage_elapsed_s": {"ocr": 0.24},
        },
    )

    body = issue_body(
        "debug-reports",
        "debug-reports/2026-05-29/run/input.png",
        "https://example.test/input.png",
        report,
    )

    assert "## Runtime Profile" in body
    assert '"upload_bytes": 750000' in body
    assert '"build_boundary_s": 0.52' in body
    assert '"ocr": 0.24' in body
