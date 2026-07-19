"""Write robomimic filter keys (`mask/<key>`) into a generated hdf5.

This is the whole E2 A/B mechanism on the training side: one physical dataset,
one filter key per arm, identical training configs that differ only in
`train.hdf5_filter_key`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _require_h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - env-dependent
        raise ImportError(
            "h5py is required for filter-key writing: pip install 'genaudit[data]'"
        ) from error
    return h5py


def write_filter_key(hdf5_path: str | Path, key: str, demo_names: Sequence[str]) -> None:
    """Create or replace `mask/<key>` with the given demo names.

    Every name must exist under data/ — a mask of nonexistent demos would fail
    only later inside robomimic on the server, after generation compute is
    spent, so we fail here instead.
    """
    h5py = _require_h5py()
    if not demo_names:
        raise ValueError(f"refusing to write empty filter key {key!r}")
    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "a") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise KeyError(f"{path} has no data/ group — not a robomimic dataset")
        missing = [name for name in demo_names if name not in data_group]
        if missing:
            raise KeyError(
                f"filter key {key!r}: {len(missing)} demo names not in data/ "
                f"(first few: {missing[:5]}) — pass demo names, not attempt_ids"
            )
        mask = handle.require_group("mask")
        if key in mask:
            del mask[key]
        mask.create_dataset(
            key, data=np.array([name.encode("utf-8") for name in demo_names])
        )


def read_filter_key(hdf5_path: str | Path, key: str) -> list[str]:
    h5py = _require_h5py()
    with h5py.File(hdf5_path, "r") as handle:
        if f"mask/{key}" not in handle:
            available = sorted(handle["mask"].keys()) if "mask" in handle else []
            raise KeyError(f"filter key {key!r} not found; available: {available}")
        return [name.decode("utf-8") for name in handle[f"mask/{key}"][()]]
