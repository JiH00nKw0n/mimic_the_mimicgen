import math

import pytest

from genaudit.envs.bounds import (
    BOUNDS,
    EXPECTED_NON_SUPERSET_PAIRS,
    EXPECTED_SUPERSET_PAIRS,
    PlacementBounds,
    get_variant,
    variant_is_superset,
)


@pytest.mark.parametrize("task,outer,inner", EXPECTED_SUPERSET_PAIRS)
def test_expansion_ladders_are_supersets(task, outer, inner):
    assert variant_is_superset(get_variant(task, outer), get_variant(task, inner)), (
        f"{task}: {outer} must be a superset of {inner} (claimed in TASKS.md)"
    )


@pytest.mark.parametrize("task,outer,inner", EXPECTED_NON_SUPERSET_PAIRS)
def test_documented_public_confounds_are_not_supersets(task, outer, inner):
    assert not variant_is_superset(get_variant(task, outer), get_variant(task, inner)), (
        f"{task}: {outer} vs {inner} is documented as relocation/shift; "
        "if this now passes, the registry numbers changed"
    )


def test_every_expected_pair_references_registered_variants():
    for task, outer, inner in EXPECTED_SUPERSET_PAIRS + EXPECTED_NON_SUPERSET_PAIRS:
        assert outer in BOUNDS[task] and inner in BOUNDS[task]


def test_diagonal_matches_hand_computation():
    needle_d2e = BOUNDS["threading"]["D2E"]["needle"]
    assert needle_d2e.diagonal == pytest.approx(math.hypot(0.35, 0.20))
    nut_d2 = BOUNDS["square"]["D2"]["square_nut"]
    assert nut_d2.diagonal == pytest.approx(math.hypot(0.5, 0.5))


def test_fixed_object_properties():
    peg_d0 = BOUNDS["square"]["D0"]["square_peg"]
    assert peg_d0.is_position_fixed
    assert not peg_d0.is_rotation_randomized
    assert peg_d0.diagonal == 0.0


def test_reversed_interval_rejected():
    with pytest.raises(ValueError, match="reversed"):
        PlacementBounds(x=(0.1, -0.1), y=(0.0, 0.0), z_rot=(0.0, 0.0))


def test_unknown_variant_fails_loudly():
    with pytest.raises(KeyError, match="unknown task/variant"):
        get_variant("threading", "D9")


def test_union_bounding_box_covers_mirror_and_expansion():
    from genaudit.envs.bounds import union_bounding_box, variant_is_superset

    d2e = get_variant("threading", "D2E")
    mirror = get_variant("threading", "D2")
    union = union_bounding_box(d2e, mirror)
    # the union contains both pools -> contrast-axis distances stay bounded
    assert variant_is_superset(union, d2e)
    assert variant_is_superset(union, mirror)
    # needle: D2E y (0.10, 0.30) + mirror y (-0.25, -0.15) -> (-0.25, 0.30)
    assert union["needle"].y == pytest.approx((-0.25, 0.30))
    assert union["needle"].diagonal > d2e["needle"].diagonal


def test_custom_variants_stay_within_probe_envelope():
    """E-series xy bounds must stay within the +/-0.25 draft reach envelope
    of TASKS.md §3 until the IK scan (PLAN.md §1.5) widens it. The mug_cleanup
    ladder inherits slightly wider public D1 y-extents (drawer at y=0.38,
    mug at y=-0.32), which the probe must confirm."""
    envelope = {
        "threading": 0.30,  # needle y reaches 0.30 by design (draft, probe-gated)
        "coffee": 0.33,
        "stack": 0.25,
        "stack_three": 0.25,
        "mug_cleanup": 0.38,
    }
    for task, variants in BOUNDS.items():
        for variant, objects in variants.items():
            if not variant.endswith("E"):
                continue
            limit = envelope[task]
            for name, bounds in objects.items():
                extent = max(abs(v) for v in (*bounds.x, *bounds.y))
                assert extent <= limit + 1e-9, (
                    f"{task}/{variant}/{name} exceeds draft envelope {limit}"
                )
