"""Lab FR3 SkillGen env cfg — official Franka SkillGen cfg retargeted to the lab FR3.

Bases on `FrankaCubeStackIKRelSkillgenEnvCfg` (NOT the MimicGen cfg) because the SkillGen cfg
brings the two things SkillGen needs and the MimicGen cfg lacks:
  * a 4th subtask config whose `subtask_term_signal="stack_2"` (final skill end), and
  * a `subtask_terms` observation group that includes the `stack_2` term.
We then apply the SAME lab overrides as the MimicGen path (FR3 robot, lab desk, FR3 finger
joints for the grasp/stack checks, teleport reset, cube-spawn range matched to the source) so
the scene/robot are identical to our vanilla run — only the data-gen method (motion-planned
transitions) differs.

The official SkillGen subtask order is already our canonical forward stack
(grasp cube_2 -> stack on cube_1 -> grasp cube_3 -> stack on cube_2), so no reverse variant is
needed: we canonicalize the seeds to forward order before annotating (same as the vanilla run).
"""

from __future__ import annotations

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_mimic.envs.franka_stack_ik_rel_skillgen_env_cfg import FrankaCubeStackIKRelSkillgenEnvCfg

from lab_mimic_cfg import _apply_lab_overrides, _apply_threshold_fixes, GRASP_DIFF_THRESHOLD

# The SkillGen scene cfg deliberately sets the ee_frame "end_effector" target offset to
# [0,0,0] (so the MimicGen eef reference frame coincides with cuRobo's tool frame, fr3_hand).
# That makes the ee_frame TCP sit at the wrist (~0.10 m above the grasp point), so the stock
# grasp check (object_grasped: ||cube - ee_frame|| < diff_threshold ~0.06-0.08) NEVER fires for
# a real grasp and auto-annotation drops every demo. We keep SkillGen's offset-0 frame (needed
# for cuRobo) and instead widen the grasp proximity threshold to cover the wrist->cube distance.
# Grasp is still gated on the gripper being CLOSED, so a looser distance can't false-positive.
SKILLGEN_GRASP_DIFF_THRESHOLD = float(os.environ.get("LAB_SKILLGEN_GRASP_DIFF", "0.13"))


@configclass
class LabFR3CubeStackIKRelSkillgenEnvCfg(FrankaCubeStackIKRelSkillgenEnvCfg):
    """Forward-order FR3 SkillGen cfg (canonical: cube_1 bottom < cube_2 < cube_3 top)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_lab_overrides(self)
        _apply_threshold_fixes(self)
        # Override the grasp proximity for the SkillGen offset-0 ee_frame (see note above).
        self.observations.subtask_terms.grasp_1.params["diff_threshold"] = SKILLGEN_GRASP_DIFF_THRESHOLD
        self.observations.subtask_terms.grasp_2.params["diff_threshold"] = SKILLGEN_GRASP_DIFF_THRESHOLD

        # --- restore the vanilla post-contact SETTLE/DWELL that NVIDIA's SkillGen base cfg zeroed ---
        # franka_stack_ik_rel_skillgen_env_cfg.py ships every subtask with
        #   subtask_term_offset_range=(0,0)  # (10,20)     and     num_interpolation_steps=0  # 5
        # i.e. the vanilla MimicGen values are present but COMMENTED OUT. With (0,0) the place
        # segment is cut the instant the stack term-signal fires -> the cube is released while still
        # descending / before per-step action-noise has averaged -> the stack is left ~2.9-3.9 cm
        # off-centre (vs ~1.0 cm for vanilla) and topples (COM past the 2.35 cm half-cube edge).
        # subtask_term_offset_range extends each subtask's END by N source frames = the human's
        # post-contact dwell (cube held on the stack, gripper still closed, contact settling) BEFORE
        # the next subtask's release. Vanilla MimicGen — identical transform / IK-rel / ARM_SCALE /
        # NN-selection — gives (10,20) to subtasks 0-2 and keeps the FINAL subtask at (0,0), and its
        # stacks are stable. We restore exactly that. LAB_SKILLGEN_DWELL=0 reverts for A/B.
        if os.environ.get("LAB_SKILLGEN_DWELL", "1") != "0":
            eef = list(self.subtask_configs.keys())[0]
            scs = self.subtask_configs[eef]
            # Vanilla uses (10,20), but datagen asserts subtask i's extended end must not cross
            # subtask i+1's start signal: term_i + offset_hi < start_(i+1). Our canonicalized seeds
            # are tighter than vanilla's — the min term->next-start gap is 19 frames (gaps [19,24,29]
            # for the three transitions) — so offset_hi must stay < 19. Use (8,15): a real settle
            # (8-15 source dwell frames on the stack before release) with margin under 19.
            lo = int(os.environ.get("LAB_SKILLGEN_DWELL_LO", "8"))
            hi = int(os.environ.get("LAB_SKILLGEN_DWELL_HI", "15"))
            for i, sc in enumerate(scs):
                sc.num_interpolation_steps = 5            # vanilla entry-smoothing
                if i < len(scs) - 1:                      # final subtask must stay (0,0) (datagen assert)
                    sc.subtask_term_offset_range = (lo, hi)
            # RANK-2 (optional) tighten source selection so the place skill comes from the
            # nearest-object demo (less carried grasp-in-gripper offset). Vanilla uses nn_k=3; only
            # enable nn_k=1 if the final placement still topples after the dwell restore.
            if os.environ.get("LAB_SKILLGEN_NNK1", "0") == "1":
                for sc in scs:
                    if "stack" in str(sc.subtask_term_signal).lower():
                        sc.selection_strategy_kwargs = {"nn_k": 1}

        # Debug knob: the SkillGen base cfg sets generation_guarantee=True, which makes the
        # generator retry FOREVER until it collects the requested number of successes. While
        # debugging cuRobo, a single broken plan then spins for 30+ min. LAB_SKILLGEN_GUARANTEE=0
        # turns guarantee off and caps failures so a bad run exits in a couple of minutes.
        if os.environ.get("LAB_SKILLGEN_GUARANTEE", "1") == "0":
            self.datagen_config.generation_guarantee = False
            self.datagen_config.max_num_failures = 3
