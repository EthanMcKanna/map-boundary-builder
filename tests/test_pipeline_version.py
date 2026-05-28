from map_boundary_builder.pipeline_version import get_pipeline_version


def test_pipeline_version_is_stable_hash() -> None:
    first = get_pipeline_version()
    second = get_pipeline_version()

    assert first == second
    assert first.startswith("pipeline-")
    assert len(first) == len("pipeline-") + 16
