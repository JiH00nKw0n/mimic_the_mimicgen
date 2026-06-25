"""Count a generated demo as success ONLY if its FINAL state is a clean canonical stack.

MimicGen marks success stickily inside DataGenerator.generate:
    generated_success = generated_success or exec_results["success"]
so a demo where the tower momentarily forms (success term fires) and then collapses
during post-placement settling is still exported as a success. On our lab FR3 setup
~20% of "successes" end with a toppled / partial tower (cubes back on the desk),
which is bad supervision.

This hook re-evaluates the stack at the FINAL state (directly from cube poses) and gates
success on it, at two consistent points:
  1. RecorderManager.set_success_to_episodes  -> the EXPORTED episode is success only if
     the final state is a clean stack (so collapsed demos are not written to the kept file).
  2. DataGenerator.generate (return value)     -> the env_loop success counter / the
     generation_guarantee target count only clean demos, so requesting N trials yields N
     clean demos (not N sticky-successes of which ~20% are junk).

Both read the SAME final-state check. Imported (after Isaac launches) by run_generate.sh,
BEFORE provenance_hooks, so provenance counts the corrected (clean) successes.
Shared Isaac Lab source is left untouched.

Canonical clean stack = cube_1 < cube_2 < cube_3 in z, each gap ~ one cube height
(|gap - 0.0468| < 0.01), and the upper cubes xy-aligned over the lower (< 0.04 m) — i.e.
the same geometry the env's `cubes_stacked` success term checks, evaluated at the last step.
"""

from __future__ import annotations

import torch

from isaaclab.managers.recorder_manager import RecorderManager
from isaaclab_mimic.datagen.data_generator import DataGenerator

import os

GAP_LO, GAP_HI = 0.0368, 0.0568   # one cube height 0.0468 +/- 0.01
# 0.04 was LOOSER than the physical topple limit (half-cube edge ~0.0235 m): a stack whose COM sits
# past the supporting cube's edge (~2.9 cm off) still read "clean", got marked success stickily, then
# toppled. Tighten to 0.02 so only physically stable stacks pass. (Toppled demos already fail the
# z-order/gap check — cubes settle flat to z~0.745 — so this mainly rejects metastable near-edge ones.)
XY_TOL = float(os.environ.get("LAB_SKILLGEN_XY_TOL", "0.02"))


def _final_stack_ok(env, env_id: int) -> bool:
    """True iff env_id's cubes are a clean canonical 3-stack (cube_1<cube_2<cube_3) right now."""
    try:
        org = env.scene.env_origins[env_id]
        z, xy = {}, {}
        for i in (1, 2, 3):
            p = env.scene[f"cube_{i}"].data.root_pos_w[env_id] - org
            z[i] = float(p[2])
            xy[i] = p[:2]
        if not (z[1] < z[2] < z[3]):
            return False
        g1, g2 = z[2] - z[1], z[3] - z[2]
        if not (GAP_LO < g1 < GAP_HI and GAP_LO < g2 < GAP_HI):
            return False
        if float(torch.linalg.vector_norm(xy[2] - xy[1])) > XY_TOL:
            return False
        if float(torch.linalg.vector_norm(xy[3] - xy[2])) > XY_TOL:
            return False
        return True
    except Exception:
        return True  # never harden a transient read error into a dropped demo


def _ids(env, env_ids):
    if env_ids is None:
        return list(range(env.num_envs))
    if isinstance(env_ids, torch.Tensor):
        return env_ids.tolist()
    return list(env_ids)


_orig_set = RecorderManager.set_success_to_episodes


def _set_success(self, env_ids, success_values):
    for vi, eid in enumerate(_ids(self._env, env_ids)):
        if vi < len(success_values) and bool(success_values[vi].item()) and not _final_stack_ok(self._env, eid):
            success_values[vi] = False
    return _orig_set(self, env_ids, success_values)


_orig_gen = DataGenerator.generate


async def _gen(self, env_id, *args, **kwargs):
    res = await _orig_gen(self, env_id, *args, **kwargs)
    if isinstance(res, dict) and res.get("success") and not _final_stack_ok(self.env, env_id):
        res["success"] = False
    return res


RecorderManager.set_success_to_episodes = _set_success
DataGenerator.generate = _gen
print("[clean_success_hook] success gated on clean final-state canonical stack")
