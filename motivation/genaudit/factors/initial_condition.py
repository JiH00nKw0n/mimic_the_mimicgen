"""Initial-condition factor axis — transform distances of PLAN.md §1.3.

All functions are pure computations over plain floats/arrays so the definitions
are unit-testable without a simulator. Positions are world-frame xy (meters),
angles are radians.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from genaudit.envs.bounds import PlacementBounds
from genaudit.records.schema import ObjectDisplacement

Position = tuple[float, float]


@dataclass(frozen=True)
class TaskGeometry:
    """Frozen per-task geometry the distance definitions depend on.

    Built once from the widest variant (D2/D2E) of the task ladder so that
    every variant in the ladder shares the same axis. Source-independent by
    construction (PLAN.md §1.3): normalizers are placement-box diagonals.
    """

    task: str
    normalizers_m: Mapping[str, float]  # L_m = diag(V_m); movable objects only
    symmetry_orders: Mapping[str, int]  # n_m; keys = all task objects
    rotation_randomized: frozenset[str]  # objects with z_rot width > 0 in V

    @property
    def movable_objects(self) -> tuple[str, ...]:
        return tuple(sorted(self.normalizers_m))


def build_task_geometry(
    task: str,
    widest_variant_bounds: Mapping[str, PlacementBounds],
    symmetry_orders: Mapping[str, int],
) -> TaskGeometry:
    """Derive the frozen geometry from the widest variant's bounds."""
    normalizers = {
        name: bounds.diagonal
        for name, bounds in widest_variant_bounds.items()
        if not bounds.is_position_fixed
    }
    if not normalizers:
        raise ValueError(f"task {task!r}: no movable object in the widest variant")
    unknown = set(widest_variant_bounds) - set(symmetry_orders)
    if unknown:
        raise ValueError(
            f"task {task!r}: symmetry_order missing for objects {sorted(unknown)}"
        )
    bad_orders = {k: v for k, v in symmetry_orders.items() if v < 1}
    if bad_orders:
        raise ValueError(f"task {task!r}: symmetry_order must be >= 1, got {bad_orders}")
    rotation_randomized = frozenset(
        name
        for name, bounds in widest_variant_bounds.items()
        if bounds.is_rotation_randomized
    )
    return TaskGeometry(
        task=task,
        normalizers_m=dict(normalizers),
        symmetry_orders=dict(symmetry_orders),
        rotation_randomized=rotation_randomized,
    )


def wrapped_symmetric_angle(delta_rad: float, symmetry_order: int) -> float:
    """Fold an angle difference by the object's n-fold rotational symmetry.

    Returns the smallest equivalent |angle| in [0, pi/n]: for a cube (n=4) a
    92-degree turn is a 2-degree turn.
    """
    if symmetry_order < 1:
        raise ValueError(f"symmetry_order must be >= 1, got {symmetry_order}")
    period = TWO_PI / symmetry_order
    folded = math.fmod(delta_rad, period)
    if folded < 0:
        folded += period
    return min(folded, period - folded)


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TransformDistances:
    d_raw: float
    d_pos: float
    d_rot: float
    displacements: tuple[ObjectDisplacement, ...]


def transform_distances(
    geometry: TaskGeometry,
    new_xy: Mapping[str, Position],
    src_xy: Mapping[str, Position],
    new_yaw: Mapping[str, float],
    src_yaw: Mapping[str, float],
) -> TransformDistances:
    """The three distance definitions of PLAN.md §1.3, computed together.

    d_raw = sum_m ||dxy||                    (meters, Phase-0 continuity)
    d_pos = (1/M) sum_m ||dxy|| / L_m        (M = movable objects, in [0, 1])
    d_rot = (1/M_rot) sum_m fold(dyaw)/(pi/n_m)   (in [0, 1]; 0 if M_rot empty)
    """
    displacements = []
    d_raw = 0.0
    pos_terms = []
    rot_terms = []
    for name in geometry.movable_objects:
        _require(name, new_xy, "new_xy")
        _require(name, src_xy, "src_xy")
        dxy = math.dist(new_xy[name], src_xy[name])
        d_raw += dxy
        pos_terms.append(dxy / geometry.normalizers_m[name])
        dyaw = 0.0
        if name in geometry.rotation_randomized:
            _require(name, new_yaw, "new_yaw")
            _require(name, src_yaw, "src_yaw")
            order = geometry.symmetry_orders[name]
            dyaw = wrapped_symmetric_angle(new_yaw[name] - src_yaw[name], order)
            rot_terms.append(dyaw / (math.pi / order))
        displacements.append(
            ObjectDisplacement(object_name=name, dxy_m=dxy, dyaw_rad=dyaw)
        )
    d_pos = sum(pos_terms) / len(pos_terms)
    d_rot = sum(rot_terms) / len(rot_terms) if rot_terms else 0.0
    return TransformDistances(
        d_raw=d_raw, d_pos=d_pos, d_rot=d_rot, displacements=tuple(displacements)
    )


def max_normalized_displacement(
    geometry: TaskGeometry,
    new_xy: Mapping[str, Position],
    src_xy: Mapping[str, Position],
) -> float:
    """Max-aggregation robustness variant of d_pos (PLAN.md §1.3)."""
    return max(
        math.dist(new_xy[name], src_xy[name]) / geometry.normalizers_m[name]
        for name in geometry.movable_objects
    )


def nearest_source_distance(
    geometry: TaskGeometry,
    episode_xy: Mapping[str, Position],
    episode_yaw: Mapping[str, float],
    sources_xy: Sequence[Mapping[str, Position]],
    sources_yaw: Sequence[Mapping[str, float]],
) -> tuple[float, int]:
    """d_eval = min over source demos of d_pos(episode, source) (PLAN.md §2.6).

    Returns (distance, index of the nearest source demo).
    """
    if not sources_xy:
        raise ValueError("sources_xy is empty")
    if len(sources_xy) != len(sources_yaw):
        raise ValueError("sources_xy and sources_yaw length mismatch")
    best = (math.inf, -1)
    for index, (src_xy, src_yaw) in enumerate(zip(sources_xy, sources_yaw)):
        d = transform_distances(geometry, episode_xy, src_xy, episode_yaw, src_yaw).d_pos
        if d < best[0]:
            best = (d, index)
    return best


def _require(name: str, mapping: Mapping[str, object], label: str) -> None:
    if name not in mapping:
        raise KeyError(f"object {name!r} missing from {label} (have {sorted(mapping)})")
