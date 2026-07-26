from __future__ import annotations

import json

import pytest

import map_boundary_builder.cli as cli_module
from map_boundary_builder.cli import EXIT_FAILED, EXIT_NEEDS_CITY, EXIT_OK, main
from map_boundary_builder.pipeline import NeedsCityDetail, PipelineResult


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"stub")
    return path


def result_with(status: str, *, reason: str | None = None, summary: dict | None = None) -> PipelineResult:
    needs_city = None
    if status == "needs_city":
        needs_city = NeedsCityDetail(
            reason="no_city_context",
            ocr_label_count=2,
            sample_labels=("Main St", "Oak Ave"),
        )
    return PipelineResult(
        status=status,  # type: ignore[arg-type]
        reason=reason,
        summary=summary or {"status": status},
        needs_city=needs_city,
    )


def test_complete_run_exits_zero(image_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda *args, **kwargs: result_with("complete", summary={"status": "complete", "city": "Austin, TX"}),
    )
    code = main(["--image", str(image_path), "--output", str(tmp_path / "out.geojson"), "--print-summary"])
    assert code == EXIT_OK
    summary = json.loads(capsys.readouterr().out)
    assert summary["city"] == "Austin, TX"
    assert "pipeline_version" in summary


def test_needs_city_exits_three_without_tty(image_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "run_pipeline", lambda *args, **kwargs: result_with("needs_city"))
    code = main(["--image", str(image_path), "--output", str(tmp_path / "out.geojson"), "--no-input"])
    assert code == EXIT_NEEDS_CITY
    err = capsys.readouterr().err
    assert "--city" in err
    assert "Main St" in err


def test_needs_city_prompts_on_tty(image_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "run_pipeline", lambda *args, **kwargs: result_with("needs_city"))
    completions: list[str] = []

    def fake_complete(image, prior, city, **kwargs):
        completions.append(city)
        return result_with("complete", summary={"status": "complete", "city": city})

    monkeypatch.setattr(cli_module, "complete_with_city", fake_complete)
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "Austin, TX")
    code = main(["--image", str(image_path), "--output", str(tmp_path / "out.geojson")])
    assert code == EXIT_OK
    assert completions == ["Austin, TX"]


def test_needs_city_prompt_abort_keeps_exit_three(image_path, tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module, "run_pipeline", lambda *args, **kwargs: result_with("needs_city"))
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    code = main(["--image", str(image_path), "--output", str(tmp_path / "out.geojson")])
    assert code == EXIT_NEEDS_CITY


def test_failure_exits_one(image_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda *args, **kwargs: result_with(
            "failed",
            reason="no_boundary_found",
            summary={"status": "failed", "message": "No service-area polygon could be extracted"},
        ),
    )
    code = main(["--image", str(image_path), "--output", str(tmp_path / "out.geojson")])
    assert code == EXIT_FAILED
    assert "No service-area polygon" in capsys.readouterr().err


def test_seed_requires_both_coordinates(image_path, tmp_path):
    with pytest.raises(SystemExit):
        main(["--image", str(image_path), "--output", str(tmp_path / "out.geojson"), "--seed-x", "10"])
