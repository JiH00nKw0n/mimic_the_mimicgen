import math

import numpy as np
import pytest

from genaudit.envs.bounds import BOUNDS, PlacementBounds
from genaudit.factors.initial_condition import (
    build_task_geometry,
    max_normalized_displacement,
    nearest_source_distance,
    transform_distances,
    wrapped_symmetric_angle,
)


def _threading_geometry():
    return build_task_geometry(
        "threading",
        BOUNDS["threading"]["D2E"],
        symmetry_orders={"needle": 1, "tripod": 1},
    )


def test_normalizers_are_placement_diagonals_and_source_independent():
    geometry = _threading_geometry()
    assert geometry.normalizers_m["needle"] == pytest.approx(math.hypot(0.35, 0.20))
    assert geometry.normalizers_m["tripod"] == pytest.approx(math.hypot(0.35, 0.20))


def test_fixed_objects_are_excluded_from_movable_set():
    geometry = build_task_geometry(
        "square",
        BOUNDS["square"]["D0"],  # peg fixed in D0
        symmetry_orders={"square_nut": 1, "square_peg": 1},
    )
    assert geometry.movable_objects == ("square_nut",)


def test_d_pos_is_within_unit_interval_for_in_region_samples():
    geometry = _threading_geometry()
    rng = np.random.default_rng(0)
    bounds = BOUNDS["threading"]["D2E"]

    def sample(b: PlacementBounds):
        return (rng.uniform(*b.x), rng.uniform(*b.y))

    for _ in range(500):
        new_xy = {name: sample(b) for name, b in bounds.items()}
        src_xy = {name: sample(b) for name, b in bounds.items()}
        result = transform_distances(
            geometry, new_xy, src_xy, {"needle": 0, "tripod": 0}, {"needle": 0, "tripod": 0}
        )
        assert 0.0 <= result.d_pos <= 1.0
        assert result.d_raw >= 0.0


def test_d_pos_reaches_one_only_at_full_diagonal_displacement():
    geometry = _threading_geometry()
    bounds = BOUNDS["threading"]["D2E"]
    new_xy = {n: (b.x[1], b.y[1]) for n, b in bounds.items()}
    src_xy = {n: (b.x[0], b.y[0]) for n, b in bounds.items()}
    result = transform_distances(
        geometry, new_xy, src_xy, {"needle": 0, "tripod": 0}, {"needle": 0, "tripod": 0}
    )
    assert result.d_pos == pytest.approx(1.0)
    assert result.d_raw == pytest.approx(2 * math.hypot(0.35, 0.20))


def test_wrapped_symmetric_angle_cube():
    # A cube (4-fold): 92 degrees is equivalent to 2 degrees.
    assert wrapped_symmetric_angle(math.radians(92), 4) == pytest.approx(math.radians(2))
    # No symmetry: max distance is pi.
    assert wrapped_symmetric_angle(math.pi, 1) == pytest.approx(math.pi)
    assert wrapped_symmetric_angle(-math.pi / 2, 1) == pytest.approx(math.pi / 2)
    assert wrapped_symmetric_angle(2 * math.pi, 1) == pytest.approx(0.0)


def test_d_rot_normalization_hits_one_at_max_symmetric_angle():
    bounds = {
        "cube": PlacementBounds((-0.1, 0.1), (-0.1, 0.1), (0.0, 2 * math.pi)),
    }
    geometry = build_task_geometry("toy", bounds, symmetry_orders={"cube": 4})
    result = transform_distances(
        geometry,
        {"cube": (0.0, 0.0)},
        {"cube": (0.0, 0.0)},
        {"cube": math.pi / 4},  # max distance modulo 4-fold symmetry
        {"cube": 0.0},
    )
    assert result.d_rot == pytest.approx(1.0)
    assert result.d_pos == 0.0


def test_rotation_fixed_objects_do_not_enter_d_rot():
    bounds = {
        "pod": PlacementBounds((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.0)),  # rotation fixed
    }
    geometry = build_task_geometry("toy", bounds, symmetry_orders={"pod": 1})
    result = transform_distances(
        geometry, {"pod": (0.1, 0.0)}, {"pod": (0.0, 0.0)}, {}, {}
    )
    assert result.d_rot == 0.0


def test_max_aggregation_robustness_variant():
    geometry = _threading_geometry()
    length = geometry.normalizers_m["needle"]
    new_xy = {"needle": (length / 2, 0.0), "tripod": (0.0, 0.0)}
    src_xy = {"needle": (0.0, 0.0), "tripod": (0.0, 0.0)}
    assert max_normalized_displacement(geometry, new_xy, src_xy) == pytest.approx(0.5)


def test_nearest_source_distance_picks_argmin():
    geometry = _threading_geometry()
    episode = {"needle": (0.0, 0.2), "tripod": (0.0, -0.15)}
    yaw = {"needle": 0.0, "tripod": 0.0}
    far = {"needle": (0.3, 0.9), "tripod": (0.3, 0.9)}
    near = {"needle": (0.01, 0.2), "tripod": (0.0, -0.15)}
    distance, index = nearest_source_distance(
        geometry, episode, yaw, [far, near], [yaw, yaw]
    )
    assert index == 1
    assert distance == pytest.approx(0.01 / geometry.normalizers_m["needle"] / 2)


def test_missing_object_fails_loudly():
    geometry = _threading_geometry()
    with pytest.raises(KeyError, match="missing from src_xy"):
        transform_distances(geometry, {"needle": (0, 0), "tripod": (0, 0)}, {}, {}, {})
