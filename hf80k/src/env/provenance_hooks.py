"""Provenance + source-usage metadata capture for generation (no shared-source edits).

The official `DataGenerator.generate()` computes which source demo feeds each subtask
(`selected_src_demo_inds[eef]`, updated inside `generate_eef_subtask_trajectory`) but
then DISCARDS it — its result dict is only {initial_state, states, observations, actions,
success}. We monkey-patch from our own runner (imported like `lab_register`, after Isaac
launches) to accumulate, per attempt:

  - which source demo index was selected for each (eef, subtask), and
  - whether that attempt succeeded,

so we can report, per source seed demo and per subtask, how many / what % of the kept
(successful) generated demos it contributed to. Written to a JSON that
`provenance_report.py` turns into the contribution table.

Output JSON (LAB_PROVENANCE_OUT, default /tmp/provenance.json):
  {
    "n_success": int, "n_attempts": int,
    "input_file": str,                         # the annotated source dataset
    "counts_success": [{"eef","subtask","src_ind","count"}],   # over kept demos
    "counts_all":     [{"eef","subtask","src_ind","count"}],   # over all attempts
    "per_demo": [ [ {"eef","subtask","src_ind"} , ... ] , ... ]  # one list per SUCCESSFUL demo
  }
"""

from __future__ import annotations

import atexit
import collections
import json
import os

from isaaclab_mimic.datagen.data_generator import DataGenerator

_OUT = os.environ.get("LAB_PROVENANCE_OUT", "/tmp/provenance.json")
_INPUT = os.environ.get("LAB_PROVENANCE_INPUT", "")
_DUMP_EVERY = int(os.environ.get("LAB_PROVENANCE_DUMP_EVERY", "20"))

_pending = collections.defaultdict(list)          # env_id -> [(eef, subtask_ind, src_ind), ...]
_counts_success = collections.Counter()           # (eef, subtask_ind, src_ind) -> count (kept demos)
_counts_all = collections.Counter()               # (eef, subtask_ind, src_ind) -> count (all attempts)
_per_demo_success = []                             # one entry per successful demo
_n_success = 0
_n_attempts = 0


def _dump():
    def rows(counter):
        return [
            {"eef": eef, "subtask": st, "src_ind": si, "count": c}
            for (eef, st, si), c in sorted(counter.items())
        ]
    with open(_OUT, "w") as fh:
        json.dump(
            {
                "n_success": _n_success,
                "n_attempts": _n_attempts,
                "input_file": _INPUT,
                "counts_success": rows(_counts_success),
                "counts_all": rows(_counts_all),
                "per_demo": _per_demo_success,
            },
            fh,
            indent=2,
        )


_orig_traj = DataGenerator.generate_eef_subtask_trajectory


def _traj(self, env_id, eef_name, subtask_ind, boundaries, constraints, selected_src_demo_inds):
    res = _orig_traj(self, env_id, eef_name, subtask_ind, boundaries, constraints, selected_src_demo_inds)
    src_ind = selected_src_demo_inds.get(eef_name)
    _pending[env_id].append((str(eef_name), int(subtask_ind), int(src_ind) if src_ind is not None else -1))
    return res


_orig_gen = DataGenerator.generate


async def _gen(self, env_id, *args, **kwargs):
    res = await _orig_gen(self, env_id, *args, **kwargs)
    global _n_success, _n_attempts
    items = _pending.pop(env_id, [])
    _n_attempts += 1
    for eef, st, si in items:
        _counts_all[(eef, st, si)] += 1
    succeeded = isinstance(res, dict) and res.get("success")
    if succeeded:
        _n_success += 1
        _per_demo_success.append([{"eef": e, "subtask": s, "src_ind": i} for (e, s, i) in items])
        for eef, st, si in items:
            _counts_success[(eef, st, si)] += 1
    # Dump on every success (so the kept-demo table is never lost) and periodically.
    # atexit is unreliable under Isaac (the app may hard-exit), so we cannot rely on it.
    if succeeded or _n_attempts % _DUMP_EVERY == 0:
        _dump()
    return res


DataGenerator.generate_eef_subtask_trajectory = _traj
DataGenerator.generate = _gen
atexit.register(_dump)
print(f"[provenance_hooks] capturing source-usage provenance -> {_OUT}")
