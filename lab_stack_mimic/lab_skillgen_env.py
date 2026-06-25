"""Lab FR3 SkillGen env — base-frame IK fix + 4-subtask term/start signals.

SkillGen (vs vanilla MimicGen) needs TWO extra things from the env:
  1. the FINAL subtask's termination signal (stack_2) — MimicGen only needs N-1 boundaries,
     SkillGen needs all N so it can isolate the last skill.
  2. per-subtask START signals (get_subtask_start_signals) — these split each subtask into a
     free-space TRANSITION (planned by cuRobo) followed by the contact-rich SKILL that is
     replayed/transformed. No stock Isaac env implements this, so the official SkillGen path is
     manual keyboard annotation; we implement it to enable AUTO annotation.

Start-signal definition (stateless, gives clean ordered 0->1 edges + real transitions):
    start_i = (subtask i-1 has terminated  OR  i == 0)  AND  (EE within APPROACH_DIST of object_ref_i)
The grasped/stacked "done" conditions persist (a grasped cube stays grasped until release; a
stacked cube stays stacked), so gating on the *current* previous-term value is sufficient — no
history/state needed. The four object_refs for the canonical forward stack are
cube_2 (grasp) -> cube_1 (place under) -> cube_3 (grasp) -> cube_2 (place under), so the edges
fire in order as the hand visits home->cube_2->cube_1->cube_3->cube_2, and each preceding
free-space hop becomes a cuRobo-planned transition.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from lab_mimic_env import LabFR3CubeStackIKRelMimicEnv

# Distance (m) from the ee_frame TCP to a subtask's reference object at which the contact SKILL
# is considered to begin (everything before it is the cuRobo-planned free-space transition).
# NOTE: the SkillGen scene sets the ee_frame offset to 0, so the TCP sits at the wrist (fr3_hand),
# ~0.10 m above the grasp point. At a place moment the wrist is ~0.15 m from the lower cube
# (cube height ~0.047 + wrist 0.10), so a 0.15 threshold only grazes the touchdown and most
# stack starts are missed. Use 0.25 to robustly capture the approach with the offset-0 frame.
APPROACH_DIST = float(os.environ.get("LAB_SKILL_APPROACH_DIST", "0.25"))


class LabFR3CubeStackIKRelSkillgenEnv(LabFR3CubeStackIKRelMimicEnv):
    """FR3 stack env for SkillGen: base-frame IK + 4 term signals + auto start signals."""

    _SIGNAL_NAMES = ["grasp_1", "stack_1", "grasp_2", "stack_2"]
    # reference object whose approach marks the start of each subtask's contact skill
    _SUBTASK_OBJECTS = ["cube_2", "cube_1", "cube_3", "cube_2"]

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        st = self.obs_buf["subtask_terms"]
        # SkillGen needs all four, including the final stack_2 (MimicGen omits it).
        return {n: st[n][env_ids] for n in self._SIGNAL_NAMES}

    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        st = self.obs_buf["subtask_terms"]
        ee = self.obs_buf["policy"]["eef_pos"][env_ids]  # world/env-origin frame
        origin = self.scene.env_origins[env_ids]

        signals: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self._SIGNAL_NAMES):
            obj = self.scene[self._SUBTASK_OBJECTS[i]].data.root_pos_w[env_ids] - origin
            near = torch.linalg.vector_norm(ee - obj, dim=-1) < APPROACH_DIST  # (N,)
            if i == 0:
                started = near
            else:
                prev_term = st[self._SIGNAL_NAMES[i - 1]][env_ids].reshape(near.shape).bool()
                started = torch.logical_and(prev_term, near)
            signals[name] = started.to(st[name].dtype).reshape(st[name][env_ids].shape)
        return signals
