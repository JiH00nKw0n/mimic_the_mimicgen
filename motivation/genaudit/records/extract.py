"""Extract AttemptRecords from MimicGen keep_failed output hdf5s.

Input layout (mimicgen `generate_dataset.py`, keep_failed=True):
  demo.hdf5        data/demo_i/... successful attempts
  demo_failed.hdf5 data/demo_i/... failed attempts
Each demo group carries per-timestep `datagen_info` (object_poses 4x4 world
frame) and the selected `src_demo_inds`. Frame 0 of object_poses is the
attempt's initial condition; the annotated SOURCE dataset provides each source
demo's initial condition the same way.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping, Sequence

from genaudit.factors.initial_condition import TaskGeometry, transform_distances
from genaudit.records.schema import AttemptRecord


def _require_h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - env-dependent
        raise ImportError(
            "h5py is required for hdf5 extraction: pip install 'genaudit[data]'"
        ) from error
    return h5py


def _pose_to_xy_yaw(pose_4x4) -> tuple[float, float, float]:
    x = float(pose_4x4[0, 3])
    y = float(pose_4x4[1, 3])
    yaw = math.atan2(float(pose_4x4[1, 0]), float(pose_4x4[0, 0]))
    return x, y, yaw


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else -1, name)


def _initial_state(group, objects: Sequence[str]) -> tuple[dict, dict]:
    """Frame-0 (x, y) and yaw per object from a demo group's datagen_info."""
    poses = group["datagen_info/object_poses"]
    xy: dict[str, tuple[float, float]] = {}
    yaw: dict[str, float] = {}
    for name in objects:
        if name not in poses:
            raise KeyError(
                f"object {name!r} missing from datagen_info/object_poses "
                f"(have {sorted(poses.keys())})"
            )
        x, y, theta = _pose_to_xy_yaw(poses[name][0])
        xy[name] = (x, y)
        yaw[name] = theta
    return xy, yaw


def _source_demo_index(group, attempt_id: str) -> tuple[int, bool]:
    """Selected source demo index; flags per-subtask disagreement.

    With select_src_per_subtask=False (our protocol) all subtasks share one
    index. If indices disagree the attempt came from a per-subtask run — we
    record the first and mark it so analysis can exclude or handle it.
    """
    if "src_demo_inds" not in group:
        raise KeyError(f"{attempt_id}: src_demo_inds missing — ancestry unavailable")
    indices = [int(v) for v in group["src_demo_inds"][()]]
    return indices[0], len(set(indices)) > 1


def load_source_initial_states(
    source_hdf5: str | Path, objects: Sequence[str]
) -> tuple[list[dict], list[dict]]:
    """Per-source-demo initial (xy, yaw), ordered by demo index.

    Requires the ANNOTATED source dataset (after prepare_src_dataset.py).
    """
    h5py = _require_h5py()
    xy_list: list[dict] = []
    yaw_list: list[dict] = []
    with h5py.File(source_hdf5, "r") as handle:
        names = sorted(handle["data"].keys(), key=_demo_sort_key)
        for name in names:
            group = handle[f"data/{name}"]
            if "datagen_info" not in group:
                raise KeyError(
                    f"source demo {name} has no datagen_info — run "
                    "prepare_src_dataset.py first (PLAN.md §4 B0)"
                )
            xy, yaw = _initial_state(group, objects)
            xy_list.append(xy)
            yaw_list.append(yaw)
    if not xy_list:
        raise ValueError(f"no demos found in {source_hdf5}")
    return xy_list, yaw_list


def extract_attempt_records(
    task: str,
    variant: str,
    geometry: TaskGeometry,
    source_xy: Sequence[Mapping],
    source_yaw: Sequence[Mapping],
    demo_hdf5: str | Path | None = None,
    failed_hdf5: str | Path | None = None,
) -> list[AttemptRecord]:
    """Turn a keep_failed generation run into AttemptRecords.

    Note: the global attempt ORDER across the two files is not recoverable
    from mimicgen output; records carry stable per-file ids instead, which is
    sufficient for everything downstream (order never enters the analysis).
    """
    h5py = _require_h5py()
    if demo_hdf5 is None and failed_hdf5 is None:
        raise ValueError("provide at least one of demo_hdf5 / failed_hdf5")
    records: list[AttemptRecord] = []
    for path, success in ((demo_hdf5, True), (failed_hdf5, False)):
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            names = sorted(handle["data"].keys(), key=_demo_sort_key)
            for name in names:
                group = handle[f"data/{name}"]
                attempt_id = f"{name}@{path.name}"
                new_xy, new_yaw = _initial_state(group, list(geometry.symmetry_orders))
                source_id, mixed_sources = _source_demo_index(group, attempt_id)
                if not 0 <= source_id < len(source_xy):
                    raise IndexError(
                        f"{attempt_id}: src_demo_ind {source_id} out of range "
                        f"for {len(source_xy)} source demos"
                    )
                distances = transform_distances(
                    geometry, new_xy, source_xy[source_id], new_yaw, source_yaw[source_id]
                )
                extras: dict = {}
                if mixed_sources:
                    extras["mixed_source_subtasks"] = True
                records.append(
                    AttemptRecord(
                        task=task,
                        variant=variant,
                        attempt_id=attempt_id,
                        source_demo_id=source_id,
                        success=success,
                        episode_length=int(group["actions"].shape[0])
                        if "actions" in group
                        else -1,
                        displacements=distances.displacements,
                        d_raw=distances.d_raw,
                        d_pos=distances.d_pos,
                        d_rot=distances.d_rot,
                        extras=extras,
                    )
                )
    return records
