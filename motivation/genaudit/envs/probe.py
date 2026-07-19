"""Reachability-probe geometry (PLAN.md §1.5).

Before an E-variant's bounds are frozen, we pin one object into a small box at
each extreme of its placement region (4 corners; for the manipulated object
also 4 edge midpoints) while the other objects sample their full region, and
generate ~50 attempts per position. The acceptance signal is the FIRST-subtask
completion rate (did the gripper reach/grasp at all), compared against an
interior reference run — task success would conflate reachability with
transform difficulty, which is exactly what we must not gate on.

Pure geometry here (testable without a simulator); class registration is
guarded like the rest of genaudit.envs.
"""
from __future__ import annotations

from genaudit.envs.bounds import PlacementBounds, get_variant

# The object the gripper must grasp first — probed at 8 positions; other
# objects (place targets) get the 4 corners only.
PRIMARY_OBJECT = {
    "threading": "needle",
    "coffee": "coffee_pod",
    "stack": "cubeA",
    "stack_three": "cubeA",
}

CORNER_POSITIONS = ("corner_00", "corner_01", "corner_10", "corner_11")
EDGE_POSITIONS = ("edge_x0", "edge_x1", "edge_y0", "edge_y1")
INTERIOR = "interior"


def probe_box(
    full: PlacementBounds, position: str, half_width: float = 0.02
) -> PlacementBounds:
    """A small sampling box pinned at an extreme of the full region, clipped
    inside it. Rotation keeps the full range (rotation reach is part of what
    the probe must exercise)."""
    x0, x1 = full.x
    y0, y1 = full.y
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    anchors = {
        "corner_00": (x0, y0),
        "corner_01": (x0, y1),
        "corner_10": (x1, y0),
        "corner_11": (x1, y1),
        "edge_x0": (x0, ym),
        "edge_x1": (x1, ym),
        "edge_y0": (xm, y0),
        "edge_y1": (xm, y1),
    }
    if position not in anchors:
        raise ValueError(f"unknown probe position {position!r}; known: {sorted(anchors)}")
    ax, ay = anchors[position]
    return PlacementBounds(
        x=(max(x0, ax - half_width), min(x1, ax + half_width)),
        y=(max(y0, ay - half_width), min(y1, ay + half_width)),
        z_rot=full.z_rot,
    )


def probe_bounds(
    task: str, variant: str, object_name: str, position: str
) -> dict[str, PlacementBounds]:
    """Variant bounds with one object pinned at a probe position."""
    full = get_variant(task, variant)
    if object_name not in full:
        raise KeyError(f"{task}/{variant}: no object {object_name!r} ({sorted(full)})")
    pinned = dict(full)
    pinned[object_name] = probe_box(full[object_name], position)
    return pinned


def plan_probe_runs(task: str, variant: str) -> list[dict]:
    """Enumerate the probe runs for one (task, variant): interior reference +
    per-object extreme positions."""
    if task not in PRIMARY_OBJECT:
        raise KeyError(f"no primary object registered for task {task!r}")
    full = get_variant(task, variant)
    primary = PRIMARY_OBJECT[task]
    if primary not in full:
        raise KeyError(f"{task}/{variant}: primary object {primary!r} missing")
    runs: list[dict] = [{"object": None, "position": INTERIOR}]
    for name, bounds in sorted(full.items()):
        if bounds.is_position_fixed:
            continue
        positions = CORNER_POSITIONS + (EDGE_POSITIONS if name == primary else ())
        runs.extend({"object": name, "position": position} for position in positions)
    return runs


def probe_class_name(task: str, variant: str, object_name: str | None, position: str) -> str:
    from genaudit.envs.robosuite_variants import variant_class_name

    base = variant_class_name(task, variant)
    if position == INTERIOR:
        return base  # interior reference = the plain variant
    return f"{base}_PROBE_{object_name}_{position}"


def register_probe_variant(
    task: str, variant: str, object_name: str, position: str
) -> type:
    """Create and register the pinned env class (server-side, mimicgen needed).

    Subclasses the (already registered) E-variant so all task machinery is
    inherited; only the placement bounds change.
    """
    import importlib

    from genaudit.envs.bounds import REFERENCE_OFFSETS
    from genaudit.envs.robosuite_variants import (
        CUSTOM_VARIANT_PARENTS,
        register_custom_variants,
        to_mimicgen_bounds,
        variant_class_name,
    )

    created = register_custom_variants()
    base_name = variant_class_name(task, variant)
    if base_name in created:
        parent = created[base_name]
        module_name = CUSTOM_VARIANT_PARENTS[(task, variant)][0]
    else:  # public variant probe
        module_name = CUSTOM_VARIANT_PARENTS.get(
            (task, "D2E"), (f"mimicgen.envs.robosuite.{task}", "")
        )[0]
        parent = getattr(importlib.import_module(module_name), base_name)

    class_name = probe_class_name(task, variant, object_name, position)
    module = importlib.import_module(module_name)
    if hasattr(module, class_name):
        return getattr(module, class_name)
    bounds = to_mimicgen_bounds(
        probe_bounds(task, variant, object_name, position), REFERENCE_OFFSETS[task]
    )

    def _get_initial_placement_bounds(self):  # noqa: ARG001
        return bounds

    probe_class = type(
        class_name,
        (parent,),
        {
            "_get_initial_placement_bounds": _get_initial_placement_bounds,
            "__doc__": f"Reachability probe: {object_name} pinned at {position}.",
            "__module__": module_name,
        },
    )
    setattr(module, class_name, probe_class)
    return probe_class
