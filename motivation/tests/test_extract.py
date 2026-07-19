"""End-to-end extraction test against a synthetic mimicgen-shaped hdf5."""
import math

import h5py
import numpy as np
import pytest

from genaudit.envs.bounds import BOUNDS
from genaudit.factors.initial_condition import build_task_geometry
from genaudit.records.extract import extract_attempt_records, load_source_initial_states

OBJECTS = ("needle", "tripod")


def _pose(x: float, y: float, yaw: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 0] = math.cos(yaw)
    pose[0, 1] = -math.sin(yaw)
    pose[1, 0] = math.sin(yaw)
    pose[1, 1] = math.cos(yaw)
    pose[0, 3] = x
    pose[1, 3] = y
    return pose


def _write_demo(handle, name, poses, src_inds, num_actions=7):
    group = handle.create_group(f"data/{name}")
    group.create_dataset("actions", data=np.zeros((num_actions, 7)))
    group.create_dataset("src_demo_inds", data=np.asarray(src_inds))
    for object_name, (x, y, yaw) in poses.items():
        stacked = np.stack([_pose(x, y, yaw)] * 3)  # frames 0..2
        group.create_dataset(f"datagen_info/object_poses/{object_name}", data=stacked)


@pytest.fixture()
def synthetic_run(tmp_path):
    source_path = tmp_path / "source.hdf5"
    with h5py.File(source_path, "w") as handle:
        # demo_10 before demo_2 on purpose: sorting must be numeric, not lexical
        _write_demo(handle, "demo_10", {"needle": (-0.1, 0.2, 0.5), "tripod": (0.0, -0.15, 1.0)}, [0])
        _write_demo(handle, "demo_2", {"needle": (-0.15, 0.18, 0.0), "tripod": (0.0, -0.15, 1.57)}, [0])
        for index in range(3, 12):
            _write_demo(
                handle,
                f"demo_{index + 10}",
                {"needle": (-0.12, 0.2, 0.1), "tripod": (0.01, -0.14, 1.5)},
                [0],
            )

    demo_path = tmp_path / "demo.hdf5"
    with h5py.File(demo_path, "w") as handle:
        _write_demo(handle, "demo_0", {"needle": (-0.1, 0.2, 0.5), "tripod": (0.0, -0.15, 1.0)}, [0, 0])

    failed_path = tmp_path / "demo_failed.hdf5"
    with h5py.File(failed_path, "w") as handle:
        _write_demo(handle, "demo_0", {"needle": (0.05, 0.28, -0.5), "tripod": (0.18, -0.2, 0.4)}, [1, 1])
        _write_demo(handle, "demo_1", {"needle": (0.0, 0.25, 0.0), "tripod": (0.1, -0.1, 1.0)}, [2, 3])

    return source_path, demo_path, failed_path


def _geometry():
    return build_task_geometry(
        "threading", BOUNDS["threading"]["D2E"], {"needle": 1, "tripod": 1}
    )


def test_source_loading_orders_numerically(synthetic_run):
    source_path, _, _ = synthetic_run
    xy, yaw = load_source_initial_states(source_path, OBJECTS)
    assert len(xy) == 11
    assert xy[0]["needle"] == pytest.approx((-0.15, 0.18))  # demo_2 first
    assert xy[1]["needle"] == pytest.approx((-0.1, 0.2))  # then demo_10
    assert yaw[1]["tripod"] == pytest.approx(1.0)


def test_extraction_produces_consistent_records(synthetic_run):
    source_path, demo_path, failed_path = synthetic_run
    geometry = _geometry()
    source_xy, source_yaw = load_source_initial_states(source_path, OBJECTS)
    records = extract_attempt_records(
        task="threading",
        variant="D2E",
        geometry=geometry,
        source_xy=source_xy,
        source_yaw=source_yaw,
        demo_hdf5=demo_path,
        failed_hdf5=failed_path,
    )
    assert len(records) == 3
    by_id = {record.attempt_id: record for record in records}

    success = by_id["demo_0@demo.hdf5"]
    assert success.success and success.source_demo_id == 0
    # identical to source demo_2? no — source index 0 is demo_2 (numeric order),
    # whose needle sits at (-0.15, 0.18); displacement is small but nonzero.
    assert success.d_pos > 0.0
    assert success.episode_length == 7

    failure = by_id["demo_0@demo_failed.hdf5"]
    assert not failure.success and failure.source_demo_id == 1
    # success attempt used src 0 twice -> no mixed flag; demo_1 used [2, 3]
    assert "mixed_source_subtasks" not in success.extras
    assert by_id["demo_1@demo_failed.hdf5"].extras.get("mixed_source_subtasks") is True
    # d_raw is the plain meter sum, d_pos the diagonal-normalized mean
    assert 0.0 < failure.d_pos <= 1.0
    assert failure.d_raw == pytest.approx(sum(d.dxy_m for d in failure.displacements))
    expected_d_pos = sum(
        d.dxy_m / geometry.normalizers_m[d.object_name] for d in failure.displacements
    ) / len(failure.displacements)
    assert failure.d_pos == pytest.approx(expected_d_pos)


def test_out_of_range_source_index_fails(synthetic_run, tmp_path):
    source_path, _, _ = synthetic_run
    geometry = _geometry()
    source_xy, source_yaw = load_source_initial_states(source_path, OBJECTS)
    bad_path = tmp_path / "bad.hdf5"
    with h5py.File(bad_path, "w") as handle:
        _write_demo(handle, "demo_0", {"needle": (0, 0.2, 0), "tripod": (0, -0.1, 0)}, [99])
    with pytest.raises(IndexError, match="out of range"):
        extract_attempt_records(
            task="threading",
            variant="D2E",
            geometry=geometry,
            source_xy=source_xy,
            source_yaw=source_yaw,
            demo_hdf5=bad_path,
        )


def test_unprepared_source_fails_with_guidance(tmp_path):
    raw_path = tmp_path / "raw_source.hdf5"
    with h5py.File(raw_path, "w") as handle:
        group = handle.create_group("data/demo_0")
        group.create_dataset("actions", data=np.zeros((5, 7)))
    with pytest.raises(KeyError, match="prepare_src_dataset"):
        load_source_initial_states(raw_path, OBJECTS)
