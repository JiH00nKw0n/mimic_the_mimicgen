# Lab 3-cube stack — MimicGen methodology, metrics, and how it differs from the official example

This explains, end to end, how we run MimicGen (Isaac Lab Mimic) on **our lab's**
3-cube stacking demos: what a "subtask" is, how auto-annotation works, exactly how
our setup differs from NVIDIA's official Franka cube-stack example, which metrics we
report (and what they mean), and what provenance metadata we log into the generated
data. Read top to bottom; every claim is tied to a paper quote or a code path.

- Paper: MimicGen, Mandlekar et al., 2023 — arXiv 2310.17596.
- Code (official, on the server): `/home/ubuntu/jake/UWLab/_isaaclab/IsaacLab/` (NVIDIA's
  Isaac Lab Mimic is a direct reimplementation of the paper).
- Our seed data: 13 of 29 teleop demos that reproduce a stack on replay (see
  [replay results](#9-replay-success-result)).

---

## 1. What MimicGen does, in one paragraph

MimicGen takes a few human demonstrations of a multi-step manipulation task and
multiplies them into a large synthetic dataset. It assumes the task is a fixed
sequence of **object-centric subtasks** — each subtask is one phase of motion
expressed entirely in **one object's coordinate frame**. It slices every human demo
into those subtask segments, then for each new randomized scene it (a) picks a source
segment per subtask, (b) rigidly transforms that segment so it keeps the same pose
relative to the object's *new* location, (c) stitches the segments together with short
interpolation bridges, (d) executes the result in simulation, and (e) keeps the
attempt only if the task actually succeeds. Our task is the paper's **Stack Three**
(3 blocks, 4 subtasks); MimicGen reports a **71.3%** data-generation rate for it.

The transform that makes this work (paper Sec. 4.2), where `O_0` is the object pose at
the start of the source segment, `O'_0` is the object pose in the new scene, and `C_t`
are the controller target poses:

```
T_W^{C'_t} = T_W^{O_0} · (T_W^{O'_0})^{-1} · T_W^{C_t}
```

This requires a **delta end-effector action space** (paper Assumption 1). Our teleop
used exactly that — IK-relative (`DifferentialInverseKinematicsActionCfg`, relative
mode) — so the method applies directly.

---

## 2. The 4 subtasks (official Franka cube-stack)

Defined in `source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.py`
(`FrankaCubeStackIKRelMimicEnvCfg.__post_init__`). Cube identities (fixed by USD asset):
**`cube_1` = blue, `cube_2` = red, `cube_3` = green**; intended tower is blue (bottom)
→ red (middle) → green (top).

| # | `object_ref` (motion frame) | `subtask_term_signal` | offset | meaning |
|---|---|---|---|---|
| 0 | `cube_2` (red) | `grasp_1` | (10, 20) | grasp the red cube |
| 1 | `cube_1` (blue) | `stack_1` | (10, 20) | place red onto blue |
| 2 | `cube_3` (green) | `grasp_2` | (10, 20) | grasp the green cube |
| 3 | `cube_2` (red) | `None` | (0, 0) | place green onto red (final) |

Notes that matter:
- For a **place** subtask, `object_ref` is the **lower** (target) cube — the frame you
  place *onto*. That is why subtask 1 references `cube_1` and subtask 3 references `cube_2`.
- The **final** subtask has **no** term signal: `N` subtasks need only `N−1` detected
  boundaries; the last one runs to the end of the demo.
- Generation config: `selection_strategy = "nearest_neighbor_object"` with `nn_k = 3`,
  `generation_select_src_per_subtask = True` (a possibly-different source demo per
  subtask), `action_noise = 0.03`, `num_interpolation_steps = 5`. This matches the
  paper's Stack Three settings (nearest-neighbor, per-subtask selection).

### 2.1 The per-step term signals (exact conditions)

The signals are computed every step by observation functions in
`source/isaaclab_tasks/.../stack/mdp/observations.py` and surfaced via the
`subtask_terms` observation group → `get_subtask_term_signals()`.

- **`grasp_1` / `grasp_2`** = `mdp.object_grasped` (line 292). True when the end
  effector is within **0.06 m** of the cube center **and** both gripper fingers have
  closed away from the open value (`gripper_open_val = 0.04`) by more than
  `gripper_threshold = 0.005`.
- **`stack_1`** = `mdp.object_stacked` (line 340). True when the upper cube is over the
  lower cube (xy offset **< 0.05 m**), the vertical spacing is ≈ one cube
  (`|Δz| − 0.0468 < 0.005`), **and** the gripper is fully open (cube released). The
  open-gripper requirement makes the signal fire at the *moment of release*, not while
  the cube is still held above the target.

There is intentionally **no `stack_2` signal** (the 4th subtask ends at the demo end).

### 2.2 Success — order-specific (`cubes_stacked`)

`source/isaaclab_tasks/.../stack/mdp/terminations.py::cubes_stacked` (wired as the
`success` termination). It checks **two ordered pairs** plus the gripper:

- blue↔red: `z(cube_1) < z(cube_2)`, xy `< 0.04`, `|Δz| ≈ 0.0468`
- red↔green: `z(cube_2) < z(cube_3)`, xy `< 0.04`, `|Δz| ≈ 0.0468`
- both fingers back at the open value (released)

The `z(cube_1) < z(cube_2) < z(cube_3)` constraint makes success **identity- and
order-specific**: a tower in any other vertical order fails even though the cubes are
physically piled. This is the single most important difference for us (see §5–6).

---

## 3. Auto-annotation (`annotate_demos.py --auto`)

What the step actually does, per demo:

1. **Replay** the recorded actions from the recorded initial state (all terminations
   disabled so it runs to completion), recording each step's `datagen_info`
   (object poses, eef pose, target eef pose) and the dense binary `subtask_term_signals`
   (`grasp_1`, `stack_1`, `grasp_2`).
2. **Validate.** Keep the demo only if (a) `cubes_stacked` is true at the end **and**
   (b) every term signal fired at least once. Otherwise it is dropped
   ("The final task was not completed" / "Did not detect completion for subtask …").

So a rough or order-mismatched demo that does not replay into a clean canonical stack
is rejected here — exactly what our replay test measures up front.

The dense binary signals are turned into segment boundaries later, at generation load
time (`datagen/datagen_info_pool.py`): the **end of subtask _i_ is the first 0→1 edge**
of its signal (`diffs.nonzero()[0][0]+1`), subtask _i_ starts where _i−1_ ended, and the
final subtask runs to the last action. During generation each non-final boundary is
**jittered by +10…20 steps** (`subtask_term_offset_range`) for variety.

---

## 4. Data generation & the generation success rate

`scripts/imitation_learning/isaaclab_mimic/generate_dataset.py` →
`datagen/generation.py::run_data_generator` (one async task per env). The trial loop:

```
num_success, num_failures, num_attempts   # module-level counters
while True:
    results = await data_generator.generate(...)     # one full demo attempt
    num_success  += bool(results["success"])
    num_attempts += 1
generation_success_rate = num_success / num_attempts
```

- An **attempt** = one `generate()`; **success** = the task success term true after
  executing all stitched segments. With `generation_guarantee = True`, generation keeps
  attempting until `num_success ≥ generation_num_trials` (it retries failures).
- `generation_keep_failed` only decides whether failed attempts are also written (to a
  separate file); it does not change the counters.

**This `num_success / num_attempts` is our Data Generation Rate (DGR).**

---

## 5. Official example vs OURS — the differences

| Aspect | Official Isaac Lab example | Ours (lab 3-cube stack) |
|---|---|---|
| Robot | Franka Panda (`panda_*`) | **FR3** (`fr3_*`), at the lab desk position |
| Scene | tutorial table | **lab `table_scene.usdc`** + invisible work-surface collision pad |
| Cubes | blue/red/green at fixed tutorial poses | 3× 50 mm cubes on the lab desk |
| Action space | IK-rel (`DifferentialInverseKinematics`, relative) | **same** (IK-rel) — method transfers directly |
| Seed demos | NVIDIA's provided demos | **our teleop** (SpaceMouse/MetaQuest over WebRTC) |
| Seed stacking order | canonical (blue→red→green) | **any order** (the human stacked in any color order) |
| Runtime | `isaac-lab-base` Docker container | **UWLab native env** (`env_uwlab`), where the lab assets live |
| Success criterion | `cubes_stacked`, order-specific | order-agnostic when judging seeds; canonicalized to order-specific for the pipeline (see §6) |

The robot/scene differences are handled by re-using the exact env overrides from
`aidas/3cube_stack/teleop/lab_teleop.py` (FR3 retarget + lab table + FR3 finger joints
for the gripper checks). The **stacking-order** difference is the one that needs real
work, below.

---

## 6. Order-agnostic seeds → canonicalization (our annotation difference)

Our seeds were success-filtered offline (`aidas/3cube_stack/teleop/filter_success.py`)
as **"any 3-cube tower"** (gripper released; sorting cubes by height, both consecutive
z-gaps ≈ one cube; xy aligned), regardless of color order. The official subtask signals
and `cubes_stacked` success are **order-specific** (§2.2), so a demo the human stacked
in, say, reverse color order would be dropped by auto-annotation and could not feed a
consistent `object_ref` schema during generation.

Our fix — **per-demo canonicalization** (keeps the official subtask schema unchanged):

1. For each seed demo, read the final recorded cube poses and determine the actual
   bottom → middle → top order (same height-sort as the success filter).
2. **Permute the cube identities** in the recorded data so that `cube_1` = bottom,
   `cube_2` = middle, `cube_3` = top — swapping the per-cube channels everywhere
   identity matters: `obs/cube_positions`, `obs/cube_orientations`, `obs/object`, and
   `states/rigid_object/cube_{1,2,3}` (and the `initial_state`). The **actions are left
   untouched** (they are end-effector deltas, identity-agnostic).

After this, every kept demo looks like a canonical blue→red→green stack: when replayed,
`reset_to` places the physical cubes at the permuted positions and the unchanged gripper
trajectory builds a canonical-identity tower, so `grasp_1 → stack_1 → grasp_2` fire in
order and `cubes_stacked` succeeds. Generation then reads a consistent `object_ref`
schema across all seeds. This recovers the reversed-order demos — concretely **+4 demos
(13 vs 9)** in our replay test — without forking the official env logic.

We still need a **lab Mimic env config** = the official `FrankaCubeStackIKRelMimicEnvCfg`
(its 4 `subtask_configs` + `get_subtask_term_signals`) **plus** the lab overrides (FR3,
lab table, FR3 finger joints for the grasp/stack/success checks). The subtask schema and
success logic are reused verbatim; only the robot/scene are swapped.

---

## 7. Metrics we report

### Bucket A — from generation alone (no policy training) — report these

| Metric | Definition | Reference (Stack-family) |
|---|---|---|
| **N** source demos | size of the seed set | ours: **13** (10 in the paper) |
| **M** generated demos | successful demos kept | ours target: **2000** (1000 in the paper) |
| **Attempts** | total `generate()` calls = `M / DGR` | report it |
| **DGR** (data generation rate) | `num_success / num_attempts` | Stack Three D0 = **71.3%**, Stack = 94.3% |
| **M_sub** subtasks | task decomposition | **4** |
| **Per-subtask DGR** | where generation fails along the 4 subtasks | localizes failures |
| **Source-demo usage** | how the M demos distribute over the 13 seeds | non-uniform with nearest-neighbor selection (expected) |

### Bucket B — needs a trained policy (separate phase, NOT from generation)

Task success rate of a BC/BC-RNN policy trained on the generated data, source-only vs
generated improvement, dataset-size scaling, human-parity. Out of scope for the
generation run; flagged here so the numbers are not confused with DGR.

### The caveat to keep in mind

**DGR and policy success rate are decoupled.** The paper's Gear Assembly has DGR
46.9 / 8.2 / 7.1% (D0/D1/D2) yet trained-policy success 92.7 / 76.0 / 64.0%. A modest
DGR on our cube task is **not** a quality failure — it only means generation needs more
attempts (more wall-clock), not that the data is worse. Generation-seed variance is
tiny (< 0.6% across seeds), so a single generation seed is fine.

---

## 8. Provenance metadata we log into the generated data

The paper notes you can log "which source demo each generated episode came from"; the
distribution is informative. In the official code this provenance is **computed and then
discarded** — `DataGenerator.select_source_demo` returns the chosen source index, it
lands in the dict `current_eef_selected_src_demo_indices`, and the `generate()` result
dict drops it. We capture it.

What we log, per generated demo (into the HDF5 `data/demo_i` attrs as JSON):
- `src_demo_provenance`: ordered list of `{eef, subtask_index, src_demo_ind, src_demo_name}`
  — i.e. **which source demo each of the 4 subtask segments was taken from**.

And once per dataset (into the `data` group attrs):
- `src_demo_usage_counts`: per source demo, how many times it was used.
- `subtask_usage_counts`: per `(eef, subtask)`, how many times it was used.

**How we attach it without editing the colleague's shared Isaac Lab source:** runtime
hooks (monkey-patch) applied from our own runner before generation starts —
`DataGenInfoPool` to map source index → demo name, `DataGenerator.generate` to
accumulate the per-subtask provenance and stash it on the exported `EpisodeData`, and
`HDF5DatasetFileHandler.write_episode` to JSON-dump it into the demo attrs (the same
path that already writes `success`/`seed`/`num_samples`). The shared install is left
untouched. (The exact capture point is `DataGenerator.select_source_demo` →
`current_eef_selected_src_demo_indices` → `generate()`.)

---

## 9. Replay-success result

Replaying the 29 success-filtered seed demos open-loop in the rebuilt lab env
(`replay_count.py`, CPU):

- **any-order 3-tower: 13 / 29**
- canonical (cube_1<cube_2<cube_3): 9 / 29

So only **13** seeds actually reproduce a stack on replay (the offline pose filter
over-counts; open-loop replay drifts). Auto-annotation applies the same replay+success
test, so **~13 is the usable seed count** going into generation. Order-agnostic
handling (§6) is what turns 9 → 13. Caveat: seeds were recorded on GPU and replayed on
CPU; since our annotate/generate pipeline runs on CPU, the CPU count is the relevant one.

---

## 10. Task difficulty distributions (D0/D1/D2) — and what our setup corresponds to

MimicGen evaluates each task at increasingly broad **reset distributions** — the region
and orientation over which objects are initialized. This is the single biggest lever on
the data-generation rate, and it explains our DGR directly.

**The variants** (paper Sec. 5). Only **D0 and D1** exist for the Stack family — there is
no Stack / Stack-Three D2 (the paper tables and the code registry both stop at D1):

- **D0** — the *default* distribution. **All source demos are collected on D0.** Note D0
  is already a *region*, not a single fixed pose.
- **D1** — a broader region (objects initialized over a much larger area).
- **D2** — broader still; only some other tasks (e.g. Threading, Coffee) have one.

**Exact bounds** (`NVlabs/mimicgen` → `mimicgen/envs/robosuite/stack.py`,
`_get_initial_placement_bounds`; the ± values are half-ranges, so ±0.10 = a 0.20 m span;
cross-checked against the paper's Appendix L prose):

| Variant | cube x,y (half-range) | full region | cube z-rotation |
|---|---|---|---|
| Stack D0 | ±0.08 m | 0.16 × 0.16 m | **0–360°** |
| Stack D1 | ±0.20 m | 0.40 × 0.40 m | 0–360° |
| Stack Three D0 | ±0.10 m | 0.20 × 0.20 m | **0–360°** |
| Stack Three D1 | ±0.20 m | 0.40 × 0.40 m | 0–360° |

Every variant randomizes a **full 360° top-down cube rotation**, and the 10 source demos
are recorded *across* this distribution — so the source already spans D0, rotation included.

**Results** (paper; max over 3 seeds):

| Task | DGR D0 | DGR D1 | policy: source-only → D0 → D1 |
|---|---|---|---|
| Stack | 94.3% | 90.0% | 26.0% → 100.0% → 99.3% |
| Stack Three | **71.3%** | 68.9% | 0.7% → 92.7% → 86.7% |

(DGR = Appendix P / Table P.1; policy = image BC-RNN trained on 1000 generated demos,
Sec. 6.1. Source-only = a policy trained on just the 10 human demos.)

Two things to take from this:

1. **DGR barely drops D0 → D1** (71.3 → 68.9) even though the region *quadruples* (0.20² →
   0.40² m). MimicGen's object-frame transform extrapolates well across region **size** —
   *when the source spans the rotation distribution.*
2. **Source-only policies are near-useless** (Stack Three: 0.7%); the generated data is what
   makes them work (92.7%). DGR and policy quality are decoupled (§7) — a low DGR costs
   wall-clock, not data quality.

### What OUR setup corresponds to

This is the real reason our fwd DGR is ~13% (scale 1.0) / ~20% (scale 0.5) rather than ~70%:

| | source distribution | generation distribution |
|---|---|---|
| paper Stack Three | **spans D0**: 0.20×0.20 m region **+ 0–360° rotation**, the 10 demos cover it | D0 (same) or D1 (2× region) |
| ours | **a single fixed pose**: cubes at x=0.25, y=0.038/0.138/0.238, **yaw=0** (all 29 demos identical) | x∈[0.22,0.40] (0.18 m), y∈[0.00,0.28] (0.28 m), **yaw ±0.5 rad ≈ ±29°** |

Our *generation region* (≈0.18×0.28 m, ±29°) is actually **between D0 and D1 in size and far
less rotation than the paper's 360°** — by the paper's own knobs it is *easier* than even D1.
Our DGR is nonetheless much lower because of the **source side**: the paper's source spans its
generation distribution (especially rotation), whereas **our 29 seeds are a single fixed
layout with zero rotation** (jake recorded teleop with cube randomization off; see §9). So
every randomized, rotated scene we generate is pure *extrapolation* from one configuration,
and the cube-rotation extrapolation in particular collides with the FR3 wrist near-singularity
(see `GENERATION_DEBUG.md`) — which is exactly where the dominant failure sits (subtask 1,
stacking the middle cube, ~48% pass; the remaining failures split grasp-green/stack-green and
grasp-red).

### Implication

To raise DGR toward the paper's regime, the highest-leverage fix is to make the **source span
the generation distribution** — re-collect a handful of teleop seeds with cube randomization
**on, including rotation**, mirroring how the paper collects its D0 source. Failing that,
**narrow the generation range** toward the seeds' single pose and shrink the yaw range
(approach D0), trading diversity for DGR. Keeping the broad range is also legitimate — it is
MimicGen's intended *single-config → many* amplification, just at a lower DGR (more attempts),
and DGR does not bound the trained policy's quality.

---

## 11. How to run

Runtime is the **UWLab native env** (`env_uwlab`), not the docker container — the lab
assets and `isaaclab_mimic` live there. `run_*.sh` set the environment (mirrors
`aidas/3cube_stack/run_live.sh`).

```bash
# 0) replay-success count (done)
bash run_replay.sh <success.hdf5> --device cpu --report /tmp/replay_success.json

# 1) canonicalize seeds (any-order -> canonical cube identities)
python canonicalize.py --src <success.hdf5> --dst <canonical.hdf5>

# 2) annotate (lab Mimic env; --auto), then 3) generate (smoke -> full 2000)
#    with provenance hooks. (scripts added in the implementation step.)
```

---

## 12. Key file references (official, on `arpa-l40s`)

Root: `/home/ubuntu/jake/UWLab/_isaaclab/IsaacLab/`

- Subtask configs: `source/isaaclab_mimic/.../envs/franka_stack_ik_rel_mimic_env_cfg.py`
- Mimic env API (`get_subtask_term_signals`, `target_eef_pose_to_action`):
  `source/isaaclab_mimic/.../envs/franka_stack_ik_rel_mimic_env.py`
- Subtask-term observations + `success`:
  `source/isaaclab_tasks/.../manipulation/stack/stack_env_cfg.py`
- `object_grasped` / `object_stacked`: `.../stack/mdp/observations.py` (lines 292 / 340)
- `cubes_stacked` (success): `.../stack/mdp/terminations.py` (line 25)
- Cube colors / gripper constants: `.../stack/config/franka/stack_joint_pos_env_cfg.py`
- Auto-annotation: `scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`
- 0→1 edge → boundary: `source/isaaclab_mimic/.../datagen/datagen_info_pool.py`
- Generation loop + counters: `source/isaaclab_mimic/.../datagen/generation.py`
- Selection + per-subtask source pick: `source/isaaclab_mimic/.../datagen/data_generator.py`
- HDF5 writer (where attrs land): `source/isaaclab/.../utils/datasets/hdf5_dataset_file_handler.py`
