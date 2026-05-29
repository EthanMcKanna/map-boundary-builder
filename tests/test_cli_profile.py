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
