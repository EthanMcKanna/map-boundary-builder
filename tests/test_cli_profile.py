import json
from contextlib import contextmanager

import map_boundary_builder.cli as cli_module
from map_boundary_builder.cli import stage_elapsed_seconds


def test_stage_elapsed_seconds_sums_adjacent_event_deltas_by_stage() -> None:
    events = [
        {"elapsed_s": 0.0, "stage": "inspect"},
        {"elapsed_s": 0.1, "stage": "extract"},
        {"elapsed_s": 0.4, "stage": "extract"},
        {"elapsed_s": 0.7, "stage": "ocr"},
        {"elapsed_s": 1.2, "stage": "georeference"},
        {"elapsed_s": 1.5, "stage": "complete"},
    ]

    assert stage_elapsed_seconds(events) == {
        "inspect": 0.1,
        "extract": 0.6,
        "ocr": 0.5,
        "georeference": 0.3,
    }


def test_stage_elapsed_seconds_uses_valid_adjacent_pairs() -> None:
    events = [
        {"elapsed_s": 0.0, "stage": "extract"},
        {"stage": "extract"},
        {"elapsed_s": 0.3, "stage": "ocr"},
        {"elapsed_s": 0.5},
    ]

    assert stage_elapsed_seconds(events) == {"ocr": 0.2}


def test_print_summary_failure_includes_profile_events(tmp_path, monkeypatch, capsys) -> None:
    image_path = tmp_path / "bad-map.png"
    output_path = tmp_path / "boundary.geojson"
    image_path.write_bytes(b"not a real image")

    def fake_build_boundary(image, city, output, *, debug_dir, options, progress):
        assert image == image_path
        assert city is None
        assert output == str(output_path)
        progress({"stage": "inspect", "message": "Reading image metadata", "percent": 5})
        progress({"stage": "ocr", "message": "Reading labels", "percent": 60})
        progress({"stage": "georeference", "message": "Fitting labels", "percent": 80})
        raise RuntimeError("could not infer a reliable map location")

    monkeypatch.setattr(cli_module, "build_boundary", fake_build_boundary)

    exit_code = cli_module.main(
        [
            "--image",
            str(image_path),
            "--output",
            str(output_path),
            "--print-summary",
            "--profile-events",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 1
    assert summary["status"] == "failed"
    assert summary["error"] == "could not infer a reliable map location"
    assert summary["event_profile"]["stage_elapsed_s"].keys() >= {"inspect", "ocr"}
    assert [event["stage"] for event in summary["event_profile"]["events"]] == [
        "inspect",
        "ocr",
        "georeference",
    ]
    assert "map-boundary-builder: error: could not infer a reliable map location" in captured.err


def test_print_summary_failure_can_include_ocr_engine_profile(tmp_path, monkeypatch, capsys) -> None:
    image_path = tmp_path / "bad-map.png"
    output_path = tmp_path / "boundary.geojson"
    image_path.write_bytes(b"not a real image")

    @contextmanager
    def fake_collect_rapidocr_profiles():
        yield [
            {
                "det_elapsed_s": 0.2,
                "rec_elapsed_s": 0.3,
                "raw_box_count": 4,
                "selected_box_count": 2,
            }
        ]

    def fake_build_boundary(image, city, output, *, debug_dir, options, progress):
        raise RuntimeError("could not infer a reliable map location")

    monkeypatch.setattr(cli_module, "collect_rapidocr_profiles", fake_collect_rapidocr_profiles)
    monkeypatch.setattr(cli_module, "build_boundary", fake_build_boundary)

    exit_code = cli_module.main(
        [
            "--image",
            str(image_path),
            "--output",
            str(output_path),
            "--print-summary",
            "--profile-ocr-engine",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 1
    assert summary["ocr_engine_profile"]["calls"] == 1
    assert summary["ocr_engine_profile"]["det_elapsed_s"] == 0.2
    assert summary["ocr_engine_profile"]["rec_elapsed_s"] == 0.3
    assert summary["ocr_engine_profile"]["raw_box_count"] == 4
    assert summary["ocr_engine_profile"]["selected_box_count"] == 2


def test_filename_hint_override_reaches_runner(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "Waymo Phoenix.png"
    output_path = tmp_path / "boundary.geojson"
    image_path.write_bytes(b"not a real image")
    seen_hints = []

    def fake_build_boundary(image, city, output, *, debug_dir, options, progress):
        seen_hints.append(options.filename_hint)
        raise RuntimeError("stop after options")

    monkeypatch.setattr(cli_module, "build_boundary", fake_build_boundary)

    exit_code = cli_module.main(
        [
            "--image",
            str(image_path),
            "--output",
            str(output_path),
            "--filename-hint",
            "uploaded-map.png",
        ]
    )

    assert exit_code == 1
    assert seen_hints == ["uploaded-map.png"]
