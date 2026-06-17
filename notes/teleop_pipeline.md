# Isaac Lab Mimic - underlying commands (cheat sheet)

Distilled from the official tutorial:
https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html

Our step scripts (`scripts/0*.py`) wrap these so you don't have to type them,
but this is what they actually run inside the container.

## Task (environment) names

| Purpose | Task name |
|---------|-----------|
| Record / replay / play (plain env) | `Isaac-Stack-Cube-Franka-IK-Rel-v0` |
| Annotate / generate (Mimic variant) | `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0` |
| Visuomotor (camera) variants | `...-Visuomotor-v0` / `...-Visuomotor-Mimic-v0` |

## Source dataset (10 human demos)

Public S3, no login:
```
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Mimic/franka_stack_datasets/dataset.hdf5
```

## Raw commands (run from /workspace/isaaclab inside the container)

Annotate (auto subtask detection):
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 --auto \
  --input_file ./datasets/source_dataset.hdf5 \
  --output_file ./datasets/annotated_dataset.hdf5
```

Generate - small sanity run (10 trials):
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu --num_envs 10 --generation_num_trials 10 \
  --input_file ./datasets/annotated_dataset.hdf5 \
  --output_file ./datasets/generated_dataset_small.hdf5
```

Generate - full run (1000 trials, headless):
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu --headless --num_envs 10 --generation_num_trials 1000 \
  --input_file ./datasets/annotated_dataset.hdf5 \
  --output_file ./datasets/generated_dataset.hdf5
```

Replay collected data (needs a display or livestream; we record MP4 instead):
```bash
./isaaclab.sh -p scripts/tools/replay_demos.py \
  --device cpu --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --dataset_file ./datasets/generated_dataset.hdf5
```

## Video recording

The Mimic scripts have no `--video` flag. Isaac Lab can still render offscreen
MP4s on a headless server when launched with `--enable_cameras`. Our
`scripts/_record_video_inproc.py` uses this: it replays recorded actions and
calls `env.render()` each step to grab RGB frames, then writes them with
imageio. See the gymnasium RecordVideo how-to for the general pattern:
https://isaac-sim.github.io/IsaacLab/main/source/how-to/record_video.html

## Training (out of scope for now)

If we later want a policy: install robomimic (`./isaaclab.sh -i robomimic`),
then `scripts/imitation_learning/robomimic/train.py` (BC) and `.../play.py`
(`play.py` supports `--video` directly).
