import json
import os
from pathlib import Path
from types import SimpleNamespace

from shapely.geometry import Polygon

import map_boundary_builder.benchmark as benchmark_module
from map_boundary_builder.benchmark import (
    BenchmarkFixture,
    BenchmarkScore,
    check_report_latency_budgets,
    compare_report_regressions,
    discover_fixtures,
    load_fixture_config,
    parse_image_name,
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
        duration_s=1.00049,
    ).as_dict()

    assert row["duration_s"] == 1.00049


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
            "catalog_shape_iou": None,
            "catalog_area_ratio": None,
            "stage_elapsed_s": None,
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
    assert report["scores"][0]["status"] == "reference_mismatch"
    assert report["scores"][0]["iou"] is None
    assert report["scores"][0]["georeference_source"] == "ocr-georeference:nominatim-label-fit"


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
    assert report["scores"][0]["catalog_slug"] == "miami-waymo"
    assert "expected OCR/georeference catalog miss" in report["scores"][0]["error"]


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


def test_report_latency_budget_check_flags_absolute_duration_excess() -> None:
    report = {
        "summary": {"total_duration_s": 4.2},
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
    )

    assert check["passed"] is False
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
    ]


def test_report_latency_budget_check_passes_when_within_budget() -> None:
    report = {
        "summary": {"total_duration_s": 2.9},
        "scores": [{"slug": "dallas-tesla", "status": "active", "duration_s": 0.19}],
    }

    check = check_report_latency_budgets(report, max_duration_s=1.0, max_total_duration_s=3.0)

    assert check["passed"] is True
    assert check["issues"] == []


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
                        "events": [],
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
        catalog_probe_missed=True,
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
            "catalog_probe_missed": True,
            "filename_hint": image_path.name,
            "write_mask_artifact": False,
        }
    ]


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
