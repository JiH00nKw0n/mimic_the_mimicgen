"""Build a filtered COPY of the annotated source dataset (SOURCE_DEMO_FILTER).

WHY THIS EXISTS
---------------
Some seed demos are dead weight: on gen_flip_25 (201 attempts, post base-frame fix)
source index 0 produced 0 successes out of 37 attempts while the other three measured
sources produced 25 between them (see `assets/source_yield.json`). Every attempt that
nearest-neighbour selection routes to a dead source is ~3.2 s of GPU time that cannot
become an episode, so dropping it lifts the measured yield from 12.4% to 15.2% for free.

WHY A COPY AND NOT A FILTER KEY
-------------------------------
robomimic filters a dataset with a `mask/<key>` dataset and `train.hdf5_filter_key`.
isaaclab_mimic has no equivalent: the source pool is built by
`DataGenInfoPool.load_from_dataset_file(path)`, which walks
`HDF5DatasetFileHandler.get_episode_names()` — literally `h5file["data"].keys()` — and
loads every episode it finds. `datagen_config.source_dataset_path` selects the FILE and
nothing finer; there is no filter-key parameter anywhere in the datagen config
(METHODOLOGY.md §12 lists the loader; the cfg exposes only the path). So the only
mechanism that actually changes which demos MimicGen sees is the set of demos physically
present under `data/`.

We therefore do both, which costs nothing and keeps everyone happy:
  * physically copy only the selected demos into a new file (this is what takes effect),
  * and still write `mask/train` listing the kept demo names, as INTERFACE.md §7 asks,
    so robomimic-style tooling reading the copy sees a consistent filter key.

The original `assets/fwd_annotated.hdf5` is opened read-only and never touched.

INDEX SEMANTICS (important for provenance)
------------------------------------------
A "source index" is the position of a demo in `list(h5file["data"].keys())` — the same
order the pool enumerates, so provenance's `src_ind` indexes it. After filtering, a run's
`src_ind` indexes the KEPT list, not the original file. The mapping back is written into
the copy's `hf80k_source_filter` root attribute and into a `<copy>.filter.json` sidecar.

BEWARE: h5py hands back group keys in ALPHABETICAL order, so in a 13-demo file the order
is demo_0, demo_1, demo_10, demo_11, demo_12, demo_2, … — index 2 is `demo_10`, not
`demo_2`. That is genuinely what MimicGen calls index 2, so index-addressed filtering
stays consistent with provenance, but a human typing `SOURCE_DEMO_FILTER=0,1,2` while
thinking of demo names will get something else. Every run prints the resolved
index -> name mapping so the mistake is visible in the log.

Usage (module):
    import source_filter
    path = source_filter.build_filtered_source("assets/fwd_annotated.hdf5")

Usage (CLI — logs go to stderr, the resulting path is the only thing on stdout):
    SRC=$(python source_filter.py --source assets/fwd_annotated.hdf5 \
          --filter exclude_zero_yield --out-dir "$WORK_DIR/source_filtered")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# INTERFACE.md §1. `all` keeps everything, `exclude_zero_yield` drops the measured
# zero-yield sources, anything else is read as an explicit index list ("0,1,2").
FILTER_ENV = "SOURCE_DEMO_FILTER"
DEFAULT_FILTER = "exclude_zero_yield"

# Root attribute stamped on every copy. Doubles as the "already filtered" marker so a
# second, accidental filtering pass cannot shift the indices a second time.
MARKER_ATTR = "hf80k_source_filter"
FILTER_SCHEMA = "fr3_cube.hf80k.source_filter.v1"

# INTERFACE.md does not name a variable for these two paths, so we pick the obvious
# defaults (repo-relative) and allow an override for container layouts that differ.
_HF80K_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YIELD_JSON = _HF80K_ROOT / "assets" / "source_yield.json"


def _log(message: str) -> None:
    """Log to stderr so the CLI can put the resulting path alone on stdout."""
    print(f"[source_filter] {message}", file=sys.stderr, flush=True)


def load_yield_table(path: str | os.PathLike | None = None) -> dict:
    """Read `assets/source_yield.json` (or LAB_SOURCE_YIELD_JSON / an explicit path)."""
    if path is None:
        path = os.environ.get("LAB_SOURCE_YIELD_JSON", str(DEFAULT_YIELD_JSON))
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"source yield table not found: {path}. SOURCE_DEMO_FILTER=exclude_zero_yield "
            f"needs it; use SOURCE_DEMO_FILTER=all or point LAB_SOURCE_YIELD_JSON at it."
        )
    with path.open(encoding="utf-8") as stream:
        table = json.load(stream)
    table["_path"] = str(path.resolve())
    return table


def zero_yield_indices(table: dict) -> list[int]:
    """Source indices that were MEASURED and never produced a success.

    Sources absent from the table were never selected during the measurement run, which
    is not evidence that they are bad — they stay in.
    """
    measured = [row for row in table.get("sources", []) if int(row.get("attempts", 0)) > 0]
    dropped = sorted(int(row["src_ind"]) for row in measured if int(row.get("successes", 0)) == 0)
    declared = table.get("excluded_indices")
    if declared is not None and sorted(int(v) for v in declared) != dropped:
        _log(f"WARNING: yield table lists excluded_indices={declared} but the per-source "
             f"counts imply {dropped}; using the counts")
    return dropped


def resolve_kept_indices(
    num_demos: int, setting: str, table: dict | None = None
) -> tuple[list[int], str]:
    """Turn a SOURCE_DEMO_FILTER setting into the list of source indices to keep."""
    setting = (setting or "").strip()
    if setting in ("", "all"):
        return list(range(num_demos)), "all demos kept"

    if setting == "exclude_zero_yield":
        if table is None:
            table = load_yield_table()
        zero = zero_yield_indices(table)
        drop = [i for i in zero if i < num_demos]
        stale = [i for i in zero if i >= num_demos]
        if stale:
            _log(f"WARNING: yield table names source indices {stale} but the file only has "
                 f"{num_demos} demos; ignoring those entries")
        measured = {int(row["src_ind"]) for row in table.get("sources", [])}
        unmeasured = sorted(set(range(num_demos)) - measured)
        if unmeasured:
            _log(f"indices {unmeasured} have no measurement in {table.get('_path')} and are "
                 f"KEPT (never selected != zero yield)")
        kept = [i for i in range(num_demos) if i not in drop]
        return kept, f"dropped measured zero-yield sources {drop}"

    tokens = [tok for tok in re.split(r"[,\s]+", setting) if tok]
    try:
        wanted = sorted({int(tok) for tok in tokens})
    except ValueError:
        raise ValueError(
            f"{FILTER_ENV}={setting!r} is not 'all', 'exclude_zero_yield', or a comma "
            f"separated index list like '0,1,2'"
        ) from None
    bad = [i for i in wanted if not 0 <= i < num_demos]
    if bad:
        raise ValueError(f"{FILTER_ENV}={setting!r} names indices {bad} but the source file "
                         f"has {num_demos} demos (valid 0..{num_demos - 1})")
    return wanted, f"explicit index list {wanted}"


def _default_out_path(source: Path, setting: str) -> Path:
    """`$WORK_DIR/source_filtered/<stem>.<tag>.hdf5` — never next to the original."""
    tag = re.sub(r"[^0-9a-zA-Z]+", "-", setting.strip()) or "all"
    work_dir = Path(os.environ.get("WORK_DIR", "/work"))
    return work_dir / "source_filtered" / f"{source.stem}.{tag}.hdf5"


def _copy_subset(source: Path, out: Path, kept_names: list[str], meta: dict,
                 mask_key: str) -> None:
    """Write `out` containing only `kept_names`, plus a robomimic-style mask group."""
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    try:
        with h5py.File(source, "r") as src, h5py.File(tmp, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            data_group = dst.create_group("data")
            for key, value in src["data"].attrs.items():
                data_group.attrs[key] = value
            total = 0
            have_totals = True
            for name in kept_names:
                src.copy(src["data"][name], data_group, name=name)
                episode = src["data"][name]
                if "num_samples" in episode.attrs:
                    total += int(episode.attrs["num_samples"])
                else:
                    have_totals = False
            # `data.attrs["total"]` is a sample count over the demos in the file; leave it
            # alone unless every kept demo told us its length.
            if have_totals and "total" in src["data"].attrs:
                data_group.attrs["total"] = total
            mask_group = dst.create_group("mask")
            mask_group.create_dataset(mask_key, data=np.array(kept_names, dtype="S"))
            dst.attrs[MARKER_ATTR] = json.dumps(meta, sort_keys=True)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_filtered_source(
    source_path: str | os.PathLike,
    setting: str | None = None,
    out_path: str | os.PathLike | None = None,
    yield_path: str | os.PathLike | None = None,
    mask_key: str = "train",
    force: bool = False,
) -> str:
    """Return the path MimicGen should read for this SOURCE_DEMO_FILTER setting.

    Writes a filtered copy when filtering actually removes something, and returns the
    ORIGINAL path when it does not (a byte-identical copy would only waste disk and
    confuse provenance). The original file is opened read-only in every branch.
    """
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"annotated source dataset not found: {source}")
    if setting is None:
        setting = os.environ.get(FILTER_ENV, DEFAULT_FILTER)
    setting = setting.strip()

    with h5py.File(source, "r") as handle:
        if MARKER_ATTR in handle.attrs:
            _log(f"{source} is already a filtered copy; leaving it alone")
            return str(source)
        if "data" not in handle:
            raise ValueError(f"{source} has no 'data' group — not an annotated dataset")
        names = list(handle["data"].keys())

    table = None
    if setting == "exclude_zero_yield":
        table = load_yield_table(yield_path)
    kept_indices, reason = resolve_kept_indices(len(names), setting, table)
    kept_names = [names[i] for i in kept_indices]
    if not kept_names:
        raise ValueError(f"{FILTER_ENV}={setting!r} would keep zero demos out of {len(names)}")
    if len(kept_names) == len(names):
        _log(f"filter={setting!r} keeps all {len(names)} demos; using the original file")
        return str(source)

    out = (Path(out_path).expanduser() if out_path is not None
           else _default_out_path(source, setting))
    out.parent.mkdir(parents=True, exist_ok=True)

    kept_set = set(kept_indices)
    meta = {
        "schema_version": FILTER_SCHEMA,
        "filter": setting,
        "reason": reason,
        "source_path": str(source),
        "source_demo_names": names,
        "kept_source_indices": kept_indices,
        "kept_demo_names": kept_names,
        "dropped_source_indices": [i for i in range(len(names)) if i not in kept_set],
        "dropped_demo_names": [n for i, n in enumerate(names) if i not in kept_set],
        "mask_key": mask_key,
        "yield_table": table.get("_path") if table else None,
        "provenance_note": "a run reading this file reports src_ind against "
                           "kept_source_indices, not against the original file",
    }

    if out.is_file() and not force:
        try:
            with h5py.File(out, "r") as handle:
                cached = json.loads(handle.attrs.get(MARKER_ATTR, "{}"))
        except OSError:
            cached = {}
        if (cached.get("kept_demo_names") == kept_names
                and cached.get("source_path") == str(source)):
            _log(f"reusing {out} ({len(kept_names)}/{len(names)} demos)")
            return str(out)
        _log(f"{out} exists but was built from a different selection; rebuilding")

    _copy_subset(source, out, kept_names, meta, mask_key)
    sidecar = out.parent / (out.name + ".filter.json")
    with sidecar.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
    dropped = ", ".join(f"{i}={names[i]}" for i in meta["dropped_source_indices"])
    _log(f"filter={setting!r} -> {reason}; kept {len(kept_names)}/{len(names)} demos")
    _log(f"dropped source index=name: {dropped}")
    if names != sorted(names, key=lambda n: (len(n), n)):
        _log("NOTE: h5py orders demo keys alphabetically, so index N is not necessarily "
             "'demo_N' — the mapping above is authoritative")
    _log(f"wrote {out}")
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a filtered copy of an annotated MimicGen source dataset.")
    parser.add_argument("--source", required=True, help="annotated HDF5 (never modified)")
    parser.add_argument("--filter", default=None,
                        help=f"filter setting; defaults to ${FILTER_ENV} or {DEFAULT_FILTER}")
    parser.add_argument("--out", default=None, help="explicit output path for the copy")
    parser.add_argument("--out-dir", default=None,
                        help="directory for the copy (name derived from source + filter)")
    parser.add_argument("--yield-json", default=None, help="override assets/source_yield.json")
    parser.add_argument("--mask-key", default="train", help="name of the mask/<key> dataset")
    parser.add_argument("--force", action="store_true", help="rebuild even if the copy exists")
    args = parser.parse_args()

    out = args.out
    if out is None and args.out_dir is not None:
        setting = args.filter if args.filter is not None else os.environ.get(FILTER_ENV,
                                                                            DEFAULT_FILTER)
        out = Path(args.out_dir) / _default_out_path(Path(args.source), setting).name
    path = build_filtered_source(
        args.source, setting=args.filter, out_path=out, yield_path=args.yield_json,
        mask_key=args.mask_key, force=args.force,
    )
    print(path)


if __name__ == "__main__":
    main()
