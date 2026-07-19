from genaudit.records.schema import (
    AttemptRecord,
    ObjectDisplacement,
    read_jsonl,
    write_jsonl,
)


def _record(index: int, success: bool) -> AttemptRecord:
    return AttemptRecord(
        task="threading",
        variant="D2E",
        attempt_id=f"demo_{index}@demo.hdf5",
        source_demo_id=index % 10,
        success=success,
        episode_length=311,
        displacements=(
            ObjectDisplacement("needle", 0.12, 0.4),
            ObjectDisplacement("tripod", 0.05, 0.0),
        ),
        d_raw=0.17,
        d_pos=0.21,
        d_rot=0.13,
        extras={"proprio_grasp_delta": 0.03},
    )


def test_jsonl_round_trip(tmp_path):
    records = [_record(i, success=(i % 3 == 0)) for i in range(25)]
    path = tmp_path / "attempts.jsonl"
    assert write_jsonl(records, path) == 25
    loaded = list(read_jsonl(path))
    assert loaded == records
    assert loaded[3].extras["proprio_grasp_delta"] == 0.03


def test_records_are_immutable():
    record = _record(0, True)
    try:
        record.success = False  # type: ignore[misc]
        raised = False
    except AttributeError:
        raised = True
    assert raised


def test_extras_keys_must_be_strings():
    import pytest

    with pytest.raises(TypeError, match="extras keys must be strings"):
        AttemptRecord(
            task="t",
            variant="v",
            attempt_id="demo_0@x",
            source_demo_id=0,
            success=True,
            episode_length=1,
            displacements=(),
            d_raw=0.0,
            d_pos=0.0,
            d_rot=0.0,
            extras={3: 1.0},
        )


def test_distance_value_resolves_schema_and_extras_loudly():
    import pytest

    from genaudit.records.schema import distance_value

    record = _record(0, True)
    assert distance_value(record, "d_pos") == record.d_pos
    assert distance_value(record, "proprio_grasp_delta") == 0.03  # extras column
    with pytest.raises(KeyError, match="unknown distance key"):
        distance_value(record, "no_such_axis")
