"""Placement-bounds registry — the single source of truth for reset regions.

Pure data + geometry utilities, importable without any simulator. The public
variant numbers were extracted verbatim from mimicgen v1.0.1
(`mimicgen/envs/robosuite/*.py`, `_get_initial_placement_bounds()`); the
E-series variants are ours (PLAN.md §1.1, TASKS.md §3). The robosuite env
classes in `genaudit.envs.robosuite_variants` are derived from this registry,
never the other way around.

Conventions: coordinates are relative to each env's table reference offset
(`REFERENCE_OFFSETS`); angles are radians; a fixed object is a degenerate
interval (lo == hi).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TWO_PI = 2.0 * math.pi
PI = math.pi


@dataclass(frozen=True)
class PlacementBounds:
    x: tuple[float, float]
    y: tuple[float, float]
    z_rot: tuple[float, float]

    def __post_init__(self) -> None:
        for name, (lo, hi) in (("x", self.x), ("y", self.y), ("z_rot", self.z_rot)):
            if hi < lo:
                raise ValueError(f"{name} interval reversed: ({lo}, {hi})")

    @property
    def width_x(self) -> float:
        return self.x[1] - self.x[0]

    @property
    def width_y(self) -> float:
        return self.y[1] - self.y[0]

    @property
    def diagonal(self) -> float:
        """Diagonal of the xy placement box — the normalizer L_m of PLAN.md §1.3."""
        return math.hypot(self.width_x, self.width_y)

    @property
    def is_position_fixed(self) -> bool:
        return self.diagonal == 0.0

    @property
    def is_rotation_randomized(self) -> bool:
        return self.z_rot[1] > self.z_rot[0]

    def corners(self) -> tuple[tuple[float, float], ...]:
        return (
            (self.x[0], self.y[0]),
            (self.x[0], self.y[1]),
            (self.x[1], self.y[0]),
            (self.x[1], self.y[1]),
        )

    def contains_position(self, x: float, y: float, tol: float = 0.0) -> bool:
        return (
            self.x[0] - tol <= x <= self.x[1] + tol
            and self.y[0] - tol <= y <= self.y[1] + tol
        )


def _interval_contains(outer: tuple[float, float], inner: tuple[float, float], tol: float) -> bool:
    return outer[0] - tol <= inner[0] and inner[1] <= outer[1] + tol


def is_superset(outer: PlacementBounds, inner: PlacementBounds, tol: float = 1e-9) -> bool:
    """True iff `inner` fits inside `outer` on x, y, and z_rot (boundary contact allowed)."""
    return (
        _interval_contains(outer.x, inner.x, tol)
        and _interval_contains(outer.y, inner.y, tol)
        and _interval_contains(outer.z_rot, inner.z_rot, tol)
    )


def variant_is_superset(
    outer: dict[str, PlacementBounds], inner: dict[str, PlacementBounds], tol: float = 1e-9
) -> bool:
    """A variant expands another iff every object's bounds are a superset."""
    if set(outer) != set(inner):
        raise ValueError(f"variant object sets differ: {sorted(outer)} vs {sorted(inner)}")
    return all(is_superset(outer[name], inner[name], tol) for name in outer)


# World-frame reference each env adds to the sampled offsets (informational; the
# distance computation uses world-frame poses so cross-task offsets cancel out).
REFERENCE_OFFSETS: dict[str, tuple[float, float, float]] = {
    "square": (0.0, 0.0, 0.82),
    "threading": (0.0, 0.0, 0.8),
    "coffee": (0.0, 0.0, 0.8),
    "stack": (0.0, 0.0, 0.8),
    "stack_three": (0.0, 0.0, 0.8),
    "three_piece_assembly": (0.0, 0.0, 0.8),
    "mug_cleanup": (0.0, 0.0, 0.8),
    "hammer_cleanup": (-0.2, 0.0, 0.90),
    "coffee_preparation": (0.0, 0.0, 0.8),  # coffee_pod is drawer-local frame
}

_FIXED = (0.0, 0.0)


def _fixed(x: float, y: float, z_rot: float = 0.0) -> PlacementBounds:
    return PlacementBounds(x=(x, x), y=(y, y), z_rot=(z_rot, z_rot))


# BOUNDS[task][variant][object] -> PlacementBounds
BOUNDS: dict[str, dict[str, dict[str, PlacementBounds]]] = {
    "square": {
        "D0": {
            "square_nut": PlacementBounds((-0.115, -0.11), (0.11, 0.225), (0.0, TWO_PI)),
            "square_peg": _fixed(0.23, 0.10),
        },
        "D1": {
            "square_nut": PlacementBounds((-0.115, 0.115), (-0.255, 0.255), (0.0, TWO_PI)),
            "square_peg": PlacementBounds((-0.10, 0.30), (-0.20, 0.20), _FIXED),
        },
        "D2": {
            "square_nut": PlacementBounds((-0.25, 0.25), (-0.25, 0.25), (0.0, TWO_PI)),
            "square_peg": PlacementBounds((-0.25, 0.25), (-0.25, 0.25), (0.0, PI / 2)),
        },
    },
    "threading": {
        "D0": {
            "needle": PlacementBounds((-0.20, -0.05), (0.15, 0.25), (-2 * PI / 3, -PI / 3)),
            "tripod": _fixed(0.0, -0.15, PI / 2),
        },
        "D1": {
            "needle": PlacementBounds((-0.20, 0.05), (0.15, 0.25), (-7 * PI / 6, PI / 6)),
            "tripod": PlacementBounds((-0.10, 0.15), (-0.20, -0.10), (PI / 6, 5 * PI / 6)),
        },
        # Public D2 mirrors both objects across y=0 — the relocation confound.
        "D2": {
            "needle": PlacementBounds((-0.20, 0.05), (-0.25, -0.15), (-7 * PI / 6, PI / 6)),
            "tripod": PlacementBounds((-0.10, 0.15), (0.10, 0.20), (-5 * PI / 6, -PI / 6)),
        },
        "D2E": {
            "needle": PlacementBounds((-0.25, 0.10), (0.10, 0.30), (-7 * PI / 6, PI / 6)),
            "tripod": PlacementBounds((-0.15, 0.20), (-0.25, -0.05), (PI / 12, 11 * PI / 12)),
        },
    },
    "coffee": {
        "D0": {
            "coffee_machine": _fixed(0.0, -0.10, -PI / 6),
            "coffee_pod": PlacementBounds((-0.13, -0.07), (0.17, 0.23), _FIXED),
        },
        # Public D1 shifts the machine off the D0 point (x=0 not in (0.05, 0.15)).
        "D1": {
            "coffee_machine": PlacementBounds((0.05, 0.15), (-0.20, -0.10), (-PI / 6, PI / 3)),
            "coffee_pod": PlacementBounds((-0.20, 0.05), (0.17, 0.30), _FIXED),
        },
        # Public D2 mirrors both objects across y=0.
        "D2": {
            "coffee_machine": PlacementBounds((-0.05, 0.05), (0.10, 0.20), (2 * PI / 3, 7 * PI / 6)),
            "coffee_pod": PlacementBounds((-0.20, 0.05), (-0.30, -0.17), _FIXED),
        },
        "D1E": {
            "coffee_machine": PlacementBounds((0.0, 0.15), (-0.20, -0.10), (-PI / 6, PI / 3)),
            "coffee_pod": PlacementBounds((-0.20, 0.05), (0.17, 0.30), _FIXED),
        },
        # B1 probe round 1 (2026-07-19): the two inner-facing corners failed the
        # reach gate (machine at robot-side x + centerline y blocks the arm's
        # path to the pod region; pod at right x + centerline y) — both axes of
        # each failing corner shrunk by the pre-registered 0.02 m step.
        "D2E": {
            "coffee_machine": PlacementBounds((-0.08, 0.20), (-0.25, -0.07), (-PI / 3, PI / 2)),
            "coffee_pod": PlacementBounds((-0.25, 0.08), (0.14, 0.33), _FIXED),
        },
    },
    "stack": {
        "D0": {
            "cubeA": PlacementBounds((-0.08, 0.08), (-0.08, 0.08), (0.0, TWO_PI)),
            "cubeB": PlacementBounds((-0.08, 0.08), (-0.08, 0.08), (0.0, TWO_PI)),
        },
        "D1": {
            "cubeA": PlacementBounds((-0.20, 0.20), (-0.20, 0.20), (0.0, TWO_PI)),
            "cubeB": PlacementBounds((-0.20, 0.20), (-0.20, 0.20), (0.0, TWO_PI)),
        },
        # B1 probe: the far corner (0.25, 0.25) sits ~0.85 m from the robot
        # base (reach limit) and failed the gate (0.66 vs interior 0.92) —
        # upper x/y shrunk by the pre-registered 0.02 m step.
        "D2E": {
            "cubeA": PlacementBounds((-0.25, 0.23), (-0.25, 0.23), (0.0, TWO_PI)),
            "cubeB": PlacementBounds((-0.25, 0.23), (-0.25, 0.23), (0.0, TWO_PI)),
        },
    },
    "stack_three": {
        "D0": {
            "cubeA": PlacementBounds((-0.10, 0.10), (-0.10, 0.10), (0.0, TWO_PI)),
            "cubeB": PlacementBounds((-0.10, 0.10), (-0.10, 0.10), (0.0, TWO_PI)),
            "cubeC": PlacementBounds((-0.10, 0.10), (-0.10, 0.10), (0.0, TWO_PI)),
        },
        "D1": {
            "cubeA": PlacementBounds((-0.20, 0.20), (-0.20, 0.20), (0.0, TWO_PI)),
            "cubeB": PlacementBounds((-0.20, 0.20), (-0.20, 0.20), (0.0, TWO_PI)),
            "cubeC": PlacementBounds((-0.20, 0.20), (-0.20, 0.20), (0.0, TWO_PI)),
        },
        "D2E": {
            "cubeA": PlacementBounds((-0.25, 0.25), (-0.25, 0.25), (0.0, TWO_PI)),
            "cubeB": PlacementBounds((-0.25, 0.25), (-0.25, 0.25), (0.0, TWO_PI)),
            "cubeC": PlacementBounds((-0.25, 0.25), (-0.25, 0.25), (0.0, TWO_PI)),
        },
    },
    "three_piece_assembly": {
        # Upstream stores the piece rotation as the literal 1.5708 (its own
        # approximation of pi/2); kept verbatim so registry-vs-code parity
        # checks pass to the last decimal.
        "D0": {
            "base": _fixed(0.0, 0.0),
            "piece_1": PlacementBounds((-0.22, 0.22), (-0.22, 0.22), (1.5708, 1.5708)),
            "piece_2": PlacementBounds((-0.22, 0.22), (-0.22, 0.22), (1.5708, 1.5708)),
        },
        "D1": {
            "base": PlacementBounds((-0.22, 0.22), (-0.22, 0.22), _FIXED),
            "piece_1": PlacementBounds((-0.22, 0.22), (-0.22, 0.22), (1.5708, 1.5708)),
            "piece_2": PlacementBounds((-0.22, 0.22), (-0.22, 0.22), (1.5708, 1.5708)),
        },
        "D2": {
            "base": PlacementBounds((-0.22, 0.22), (-0.22, 0.22), (-PI / 4, PI / 4)),
            "piece_1": PlacementBounds((-0.22, 0.22), (-0.22, 0.22), (1.5708 - PI / 2, 1.5708 + PI / 2)),
            "piece_2": PlacementBounds((-0.22, 0.22), (-0.22, 0.22), (1.5708 - PI / 2, 1.5708 + PI / 2)),
        },
    },
    "mug_cleanup": {
        "D0": {
            "drawer": _fixed(0.0, 0.30),
            "object": PlacementBounds((-0.15, 0.15), (-0.25, -0.10), (0.0, TWO_PI)),
        },
        # Public D1 loses the mug's (-0.15, -0.10) y-sliver from D0.
        "D1": {
            "drawer": PlacementBounds((-0.15, 0.05), (0.25, 0.35), (-PI / 6, PI / 6)),
            "object": PlacementBounds((-0.25, 0.15), (-0.30, -0.15), (0.0, TWO_PI)),
        },
        "D1E": {
            "drawer": PlacementBounds((-0.15, 0.05), (0.25, 0.35), (-PI / 6, PI / 6)),
            "object": PlacementBounds((-0.25, 0.15), (-0.30, -0.10), (0.0, TWO_PI)),
        },
        "D2E": {
            "drawer": PlacementBounds((-0.20, 0.10), (0.22, 0.38), (-PI / 4, PI / 4)),
            "object": PlacementBounds((-0.25, 0.20), (-0.32, -0.08), (0.0, TWO_PI)),
        },
    },
}

BOUNDS["hammer_cleanup"] = {
    # D0 distribution is hardcoded in mimicgen's _load_model (no bounds
    # method); numbers verified from mimicgen hammer_cleanup.py L179-191.
    # CAUTION: hammer rotation is about z in D0 but about y in D1 (init_quat
    # change) — rotation containment/d_rot is NOT comparable across the ladder;
    # only the positional axes enter the distance (documented in TASKS.md).
    "D0": {
        "hammer": PlacementBounds((0.10, 0.18), (-0.20, -0.13), (-0.1, 0.1)),
        "drawer": _fixed(0.2, 0.30),
    },
    "D1": {
        "hammer": PlacementBounds((-0.2, 0.2), (-0.25, -0.13), (0.0, TWO_PI)),
        "drawer": PlacementBounds((0.0, 0.2), (0.2, 0.3), (-PI / 6, PI / 6)),
    },
}

BOUNDS["coffee_preparation"] = {
    # coffee_pod samples in the drawer-local frame (reference (0,0,0)) — its
    # diagonal is frame-free so the distance machinery is unaffected.
    "D0": {
        "drawer": _fixed(0.15, -0.35, PI),
        "coffee_machine": _fixed(-0.15, -0.25, -PI / 6),
        "mug": PlacementBounds((0.05, 0.20), (0.05, 0.25), _FIXED),
        "coffee_pod": PlacementBounds((-0.03, 0.03), (-0.05, 0.03), _FIXED),
    },
    "D1": {
        "drawer": _fixed(0.15, -0.35, PI),
        "coffee_machine": PlacementBounds((-0.25, -0.15), (-0.30, -0.25), (-PI / 6, PI / 3)),
        "mug": PlacementBounds((-0.15, 0.20), (0.05, 0.25), (0.0, TWO_PI)),
        "coffee_pod": PlacementBounds((-0.03, 0.03), (-0.05, 0.03), _FIXED),
    },
}

# The expansion ladders we claim in PLAN.md/TASKS.md: each pair (outer, inner)
# must satisfy variant_is_superset. Tested in tests/test_bounds.py.
EXPECTED_SUPERSET_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("square", "D1", "D0"),
    ("square", "D2", "D0"),
    ("threading", "D1", "D0"),
    ("threading", "D2E", "D1"),
    ("coffee", "D1E", "D0"),
    ("coffee", "D1E", "D1"),
    ("coffee", "D2E", "D1E"),
    ("stack", "D1", "D0"),
    ("stack", "D2E", "D1"),
    ("stack_three", "D1", "D0"),
    ("stack_three", "D2E", "D1"),
    ("three_piece_assembly", "D1", "D0"),
    ("three_piece_assembly", "D2", "D1"),
    ("mug_cleanup", "D1E", "D0"),
    ("mug_cleanup", "D1E", "D1"),
    ("mug_cleanup", "D2E", "D1E"),
    ("coffee_preparation", "D1", "D0"),
    # hammer_cleanup D1 is positionally a superset of D0 but the hammer's
    # rotation axis changes (z -> y), so it is deliberately absent from the
    # strict-superset claims.
)

# Documented violations in the public ladders — the confounds E1 removes.
EXPECTED_NON_SUPERSET_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("threading", "D2", "D1"),  # mirror relocation
    ("coffee", "D2", "D1"),  # mirror relocation
    ("coffee", "D1", "D0"),  # machine shifted off the D0 point
    ("square", "D2", "D1"),  # nut-y 5mm + peg-x upper 50mm truncations
    ("mug_cleanup", "D1", "D0"),  # mug y-sliver lost
)


def union_bounding_box(
    *variants: dict[str, PlacementBounds],
) -> dict[str, PlacementBounds]:
    """Per-object bounding box of several variants' regions.

    The contrast axis of PLAN.md §1.4: when overlaying a relocation pool
    (public mirror D2) with an expansion pool (D2E) the normalizer must cover
    both regions, otherwise the mirror pool's distances exceed 1.
    """
    if not variants:
        raise ValueError("need at least one variant")
    names = set(variants[0])
    for other in variants[1:]:
        if set(other) != names:
            raise ValueError(
                f"variant object sets differ: {sorted(names)} vs {sorted(other)}"
            )
    union: dict[str, PlacementBounds] = {}
    for name in names:
        boxes = [variant[name] for variant in variants]
        union[name] = PlacementBounds(
            x=(min(b.x[0] for b in boxes), max(b.x[1] for b in boxes)),
            y=(min(b.y[0] for b in boxes), max(b.y[1] for b in boxes)),
            z_rot=(min(b.z_rot[0] for b in boxes), max(b.z_rot[1] for b in boxes)),
        )
    return union


def get_variant(task: str, variant: str) -> dict[str, PlacementBounds]:
    try:
        return BOUNDS[task][variant]
    except KeyError as error:
        known = sorted(BOUNDS.get(task, {})) or sorted(BOUNDS)
        raise KeyError(
            f"unknown task/variant {task!r}/{variant!r}; known here: {known}"
        ) from error
