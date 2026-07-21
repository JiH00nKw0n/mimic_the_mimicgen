"""motivation_new — isotropic, rotation-free concentric-square variant ladder.

Redesign of the D0/D1/D2 ladder so the transform axis is clean and comparable
across tasks:
  * every movable object expands as a CENTERED SQUARE about its original D0
    center: half-widths 0.05 / 0.15 / 0.25 m  (boxes 0.1 / 0.3 / 0.5 m),
  * expansion is isotropic (same in +x/-x/+y/-y) — no directional bias,
  * rotation is FIXED per object (no randomization) — transform = pure xy,
  * sizes are the TARGET; the reachability+collision probe fills CAPS per
    (task, object) where 0.5 is not physically reachable (multi-object
    collision / off-table / arm reach). Fixtures that cannot roam stay fixed.

Internal variant keys are N0/N1/N2 (to coexist with the untouched public
D0/D1/D2 registry); they are LABELLED D0/D1/D2 in the report.
"""
from __future__ import annotations

from genaudit.envs.bounds import BOUNDS, PlacementBounds

# Reachable table window (top-down grasp): the arm reaches a central region of
# the 0.8x0.8 m table, NOT the corners. REACH_WINDOW keeps every box inside a
# central square (0.05 m from the table edge); CENTRAL_CAP is the largest
# verified half-width (stack reached +-0.23). Per object the max half-width is
#   h = min(CENTRAL_CAP, REACH_WINDOW - |cx|, REACH_WINDOW - |cy|)
# and the ladder is the SAME relative fractions of h for every task.
REACH_WINDOW = 0.35
CENTRAL_CAP = 0.23
LADDER_FRACTIONS = {"N0": 0.20, "N1": 0.60, "N2": 1.00}

# objects that physically cannot roam (fixtures / in-drawer) — kept fixed.
KEEP_FIXED: set[tuple[str, str]] = {
    ("coffee_preparation", "drawer"),
    ("coffee_preparation", "coffee_pod"),
}

# optional manual override of the per-(task, object) max half-width, e.g. if a
# quick generation smoke shows a box is unreachable. Absent = computed h.
CAPS: dict[tuple[str, str], float] = {}


def _max_half(cx: float, cy: float) -> float:
    return min(CENTRAL_CAP, REACH_WINDOW - abs(cx), REACH_WINDOW - abs(cy))

LADDER_TASKS = (
    "square", "threading", "coffee", "three_piece_assembly", "stack",
    "stack_three", "mug_cleanup", "hammer_cleanup", "coffee_preparation",
)


def _center(pb: PlacementBounds) -> tuple[float, float]:
    return ((pb.x[0] + pb.x[1]) / 2.0, (pb.y[0] + pb.y[1]) / 2.0)


def _rot_fixed(pb: PlacementBounds) -> float:
    return (pb.z_rot[0] + pb.z_rot[1]) / 2.0


def build_new_bounds() -> dict[str, dict[str, dict[str, PlacementBounds]]]:
    """Generate the N0/N1/N2 ladder for every task: an isotropic square centered
    on each object's D0 position, sized to the reachable table window, with the
    same relative D0/D1/D2 fractions for every task. Rotation fixed."""
    out: dict[str, dict[str, dict[str, PlacementBounds]]] = {}
    for task in LADDER_TASKS:
        d0 = BOUNDS[task]["D0"]
        out[task] = {}
        for vname, frac in LADDER_FRACTIONS.items():
            box: dict[str, PlacementBounds] = {}
            for obj, pb in d0.items():
                cx, cy = _center(pb)
                rot = _rot_fixed(pb)
                if (task, obj) in KEEP_FIXED:
                    box[obj] = PlacementBounds((cx, cx), (cy, cy), (rot, rot))
                    continue
                h_max = min(_max_half(cx, cy), CAPS.get((task, obj), _max_half(cx, cy)))
                h = frac * h_max
                box[obj] = PlacementBounds((cx - h, cx + h), (cy - h, cy + h), (rot, rot))
            out[task][vname] = box
    return out


NEW_BOUNDS = build_new_bounds()


def register_into(bounds: dict) -> None:
    """Merge the N0/N1/N2 ladder into a BOUNDS-shaped dict (for env auto-gen)."""
    for task, variants in NEW_BOUNDS.items():
        bounds.setdefault(task, {}).update(variants)


if __name__ == "__main__":
    import math
    for task in LADDER_TASKS:
        print(f"\n== {task} ==")
        for v in ("N0", "N1", "N2"):
            parts = []
            for obj, pb in NEW_BOUNDS[task][v].items():
                w = pb.width_x
                if pb.is_position_fixed:
                    parts.append(f"{obj}=FIXED")
                else:
                    parts.append(f"{obj} {w:.2f}m² rot{math.degrees(pb.z_rot[0]):+.0f}°")
            print(f"   {v}: " + " | ".join(parts))
