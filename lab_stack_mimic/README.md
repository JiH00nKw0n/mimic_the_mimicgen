# lab_stack_mimic — MimicGen on the lab FR3 3-cube stack

Work-in-progress: run MimicGen on **our lab's** 3-cube stacking demos (not the
NVIDIA tutorial scene). The seed demos were teleoperated by a labmate in the lab
sim setup (FR3 single arm at the lab desk, provided by 주상) and stack 3 cubes in
**any order** (colour/identity-agnostic).

## Source data (on the GPU server `arpa-l40s` / spdp-l2)

- `/home/ubuntu/jake/aidas/3cube_stack/datasets/teleop_dataset_success.hdf5`
  — 29 demos, success-filtered offline to "any 3-cube tower" by
  `aidas/3cube_stack/teleop/filter_success.py`.
- `..._canonical.hdf5` — the stricter subset that stacked in identity order
  cube_1 < cube_2 < cube_3.

The dataset has no registered `env_name`, so we rebuild the recording env from
`aidas/3cube_stack/teleop/lab_teleop.py` (the IK-rel Franka stack env retargeted
to the FR3 + lab desk).

## Runtime — IMPORTANT

This runs in the **UWLab native Isaac Lab env** (`/home/ubuntu/jake/env_uwlab`),
NOT the `isaac-lab-base` docker container the rest of this repo uses. That is
where the lab assets, FR3, UWLab, and `isaaclab_mimic` live. `run_replay.sh` sets
the same environment as `aidas/3cube_stack/run_live.sh`.

## Steps

### 1. Replay-success count (done first)

How many of the 29 demos actually reproduce a stack when their recorded actions
are replayed open-loop in sim (replay can drift, even though they passed the
offline pose filter). Reports both the order-agnostic and canonical counts.

```bash
bash run_replay.sh \
  /home/ubuntu/jake/aidas/3cube_stack/datasets/teleop_dataset_success.hdf5 \
  --report /tmp/replay_success.json
```

- `replay_count.py` — rebuilds the lab env, replays each demo, judges success.
- `success_criteria.py` — the any-order / canonical tower criterion, ported
  verbatim (same thresholds) from `filter_success.py`.

### 2. Order-agnostic auto-annotation + MimicGen generation (next)

The stock Franka stack Mimic env defines **order-specific** subtask term signals
(grasp cube_2 → stack on cube_1 → grasp cube_3 → stack on cube_2) and a canonical
success term, so `annotate_demos.py --auto` would drop any demo stacked in a
different order. Plan: per-demo canonicalization (relabel cube identities to the
demo's actual bottom→mid→top roles) so every demo fits the canonical subtask
graph, then run the stock annotate → generate. Tracked in the parent task.

## Notes

- Nothing is hard-coded: pass `--dataset_file` (and `--table_usd` if it moves).
- Replays headless; default device is the AppLauncher default (cuda). Add
  `--device cpu` to compare (CPU replay can be more faithful for some physics).
