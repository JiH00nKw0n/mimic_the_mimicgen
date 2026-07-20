"""Frozen paired evaluation (PLAN.md §2.6).

A fixed set of initial states is drawn once from a seeded env and stored;
every arm and seed is then evaluated on the SAME episodes via `reset_to`,
enabling paired (McNemar) analysis. Requires robomimic + the task's sim stack
— runs on the server; this module is written to be smoke-tested there.
"""
from __future__ import annotations

import json
from pathlib import Path


def _require_robomimic():
    try:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
    except ImportError as error:  # pragma: no cover - env-dependent
        raise ImportError(
            "robomimic is required for frozen-reset evaluation — run on the server"
        ) from error
    return EnvUtils, FileUtils


def _require_h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - env-dependent
        raise ImportError("h5py is required: pip install 'genaudit[data]'") from error
    return h5py


def _register_variants() -> None:
    """E-series env names (Threading_D2E, ...) live in dataset env_meta and
    checkpoints; every process that builds an env from them must register the
    variants first or robosuite raises 'unknown environment'."""
    from genaudit.envs.robosuite_variants import register_custom_variants

    register_custom_variants()


def create_frozen_resets(
    dataset_path: str | Path, num_resets: int, seed: int, out_path: str | Path
) -> Path:
    """Sample `num_resets` initial states from the dataset's env (its native
    reset distribution) and store them for reuse by every arm."""
    import numpy as np
    import robomimic.utils.obs_utils as ObsUtils

    EnvUtils, FileUtils = _require_robomimic()
    h5py = _require_h5py()
    _register_variants()

    # robomimic envs require obs-modality registration before reset(); a
    # minimal low-dim spec suffices since we only harvest simulator states.
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}}
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
    np.random.seed(seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as handle:
        handle.attrs["dataset_path"] = str(dataset_path)
        handle.attrs["seed"] = seed
        handle.attrs["env_meta"] = json.dumps(env_meta)
        for index in range(num_resets):
            env.reset()
            state = env.get_state()
            group = handle.create_group(f"reset_{index}")
            group.create_dataset("states", data=state["states"])
            group.attrs["model"] = state["model"]
    return out_path


def evaluate_policy_on_frozen_resets(
    checkpoint_path: str | Path,
    resets_path: str | Path,
    horizon: int,
    out_jsonl: str | Path,
) -> list[dict]:
    """Roll the checkpointed policy from every frozen reset; one JSON line per
    episode: {reset_index, success, steps}."""
    _, FileUtils = _require_robomimic()
    h5py = _require_h5py()
    _register_variants()

    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint_path))
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_path=str(checkpoint_path), render=False, render_offscreen=False
    )

    results: list[dict] = []
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(resets_path, "r") as handle, out_jsonl.open("w") as sink:
        reset_names = sorted(handle.keys(), key=lambda n: int(n.split("_")[1]))
        for name in reset_names:
            group = handle[name]
            policy.start_episode()
            env.reset()
            observation = env.reset_to(
                {"model": group.attrs["model"], "states": group["states"][()]}
            )
            success = False
            steps = 0
            for _ in range(horizon):
                action = policy(ob=observation)
                observation, _, done, _ = env.step(action)
                steps += 1
                if env.is_success()["task"]:
                    success = True
                    break
                if done:
                    break
            record = {
                "reset_index": int(name.split("_")[1]),
                "success": bool(success),
                "steps": steps,
            }
            results.append(record)
            sink.write(json.dumps(record) + "\n")
    return results
