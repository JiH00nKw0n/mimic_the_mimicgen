"""Synthetic-data writer to our HDF5 schema (robomimic / Isaac Lab Mimic compatible).

Layout (robomimic-style):
    data/
      demo_0/
        actions           (T, A)
        obs/<key>         (T, ...)
        rewards           (T,)
      demo_1/ ...
    data.attrs["total"], env args, etc.

Isaac Lab Mimic consumes the same demo_i/{actions,obs} convention, so this writer is
the bridge point: keep our action/obs keys aligned with the target policy trainer.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class HDF5DataWriter:
    """Buffers episodes and flushes successful ones to a robomimic-style HDF5."""

    def __init__(self, only_successful: bool = True):
        self.only_successful = only_successful
        self._episodes: List[dict] = []
        self._cur: Optional[dict] = None

    def begin_episode(self, meta: dict) -> None:
        self._cur = {"meta": dict(meta), "actions": [], "obs": [], "rewards": []}

    def record(self, step: dict) -> None:
        assert self._cur is not None, "begin_episode() first"
        self._cur["actions"].append(np.asarray(step["action"], dtype=np.float32))
        self._cur["obs"].append(step.get("obs", {}))
        self._cur["rewards"].append(float(step.get("reward", 0.0)))

    def end_episode(self, success: bool) -> None:
        assert self._cur is not None
        self._cur["success"] = success
        self._episodes.append(self._cur)
        self._cur = None

    def save(self, path: str) -> None:
        try:
            import h5py
        except ImportError as e:  # pragma: no cover
            raise ImportError("h5py required to save HDF5 (pip install h5py)") from e

        eps = [e for e in self._episodes if (e["success"] or not self.only_successful)]
        with h5py.File(path, "w") as f:
            data_grp = f.create_group("data")
            total = 0
            for i, ep in enumerate(eps):
                g = data_grp.create_group(f"demo_{i}")
                actions = np.asarray(ep["actions"], dtype=np.float32)
                g.create_dataset("actions", data=actions)
                g.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32))
                # obs: transpose list-of-dicts -> dict-of-arrays
                obs_grp = g.create_group("obs")
                if ep["obs"] and isinstance(ep["obs"][0], dict):
                    for key in ep["obs"][0].keys():
                        arr = np.asarray([np.asarray(o[key]) for o in ep["obs"]])
                        obs_grp.create_dataset(key, data=arr)
                g.attrs["num_samples"] = int(actions.shape[0])
                for k, v in ep["meta"].items():
                    try:
                        g.attrs[k] = v
                    except (TypeError, ValueError):
                        g.attrs[k] = str(v)
                total += int(actions.shape[0])
            data_grp.attrs["total"] = total
            data_grp.attrs["num_demos"] = len(eps)

    @property
    def num_success(self) -> int:
        return sum(1 for e in self._episodes if e.get("success"))
