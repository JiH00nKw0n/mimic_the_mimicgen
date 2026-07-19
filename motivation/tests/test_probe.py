import pytest

from genaudit.envs.bounds import get_variant
from genaudit.envs.probe import (
    INTERIOR,
    plan_probe_runs,
    probe_bounds,
    probe_box,
    probe_class_name,
)


def test_probe_box_corners_and_edges_are_clipped_inside():
    full = get_variant("threading", "D2E")["needle"]  # x(-0.25,0.10) y(0.10,0.30)
    corner = probe_box(full, "corner_00")
    assert corner.x == pytest.approx((-0.25, -0.23)) and corner.y == pytest.approx((0.10, 0.12))
    far = probe_box(full, "corner_11")
    assert far.x == pytest.approx((0.08, 0.10)) and far.y == pytest.approx((0.28, 0.30))
    edge = probe_box(full, "edge_y1")
    assert edge.y == pytest.approx((0.28, 0.30))
    assert edge.x == pytest.approx((-0.095, -0.055))  # centered on x midpoint
    assert corner.z_rot == full.z_rot  # rotation range untouched


def test_probe_bounds_pins_only_the_target_object():
    pinned = probe_bounds("threading", "D2E", "needle", "corner_00")
    full = get_variant("threading", "D2E")
    assert pinned["tripod"] == full["tripod"]
    assert pinned["needle"] != full["needle"]
    assert pinned["needle"].diagonal < full["needle"].diagonal


def test_plan_counts_primary_gets_edges_secondary_corners():
    runs = plan_probe_runs("threading", "D2E")
    assert len(runs) == 1 + 8 + 4  # interior + needle(8) + tripod(4)
    runs3 = plan_probe_runs("stack_three", "D2E")
    assert len(runs3) == 1 + 8 + 4 + 4  # cubeA primary, cubeB/cubeC corners
    assert runs[0]["position"] == INTERIOR


def test_fixed_objects_are_not_probed():
    # coffee D0: machine fixed -> only the pod would be probed
    runs = plan_probe_runs("coffee", "D0")
    probed_objects = {run["object"] for run in runs if run["object"]}
    assert probed_objects == {"coffee_pod"}


def test_probe_class_names():
    assert probe_class_name("threading", "D2E", None, INTERIOR) == "Threading_D2E"
    assert (
        probe_class_name("threading", "D2E", "needle", "corner_01")
        == "Threading_D2E_PROBE_needle_corner_01"
    )
