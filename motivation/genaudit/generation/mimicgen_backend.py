"""Build MimicGen generation configs for the E1/E2 protocol (PLAN.md §1.2).

Pure JSON manipulation — no simulator import. The actual run happens through
`genaudit.generation.run_mimicgen` on the server.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


def build_generation_config(
    template: dict,
    *,
    task_name: str,
    source_dataset: str,
    output_folder: str,
    num_attempts: int,
    seed: int,
    experiment_name: str | None = None,
    collect_obs: bool = True,
    camera_names: list[str] | None = None,
) -> dict:
    """Patch a mimicgen task template into our fixed-attempt audit protocol.

    Protocol constants (identical for every task, PLAN.md §1.2):
    guarantee=False (fixed attempt count), keep_failed=True with UNCAPPED
    failures, selection_strategy='random', select_src_per_subtask=False.

    collect_obs stays True because mimicgen v1.0.1's write_demo_to_hdf5
    crashes on observations=None; state-only pools are achieved with
    camera_names=[] (low-dim obs only), matching mimicgen's own core protocol.
    """
    if num_attempts < 1:
        raise ValueError(f"num_attempts must be >= 1, got {num_attempts}")
    config = copy.deepcopy(template)

    # NOTE: config["name"] (top-level) is mimicgen's config-registry key —
    # never touch it; only experiment.name is run bookkeeping.
    experiment = config["experiment"]
    experiment["name"] = experiment_name or f"{task_name}_n{num_attempts}_seed{seed}"
    experiment["seed"] = seed
    experiment["max_num_failures"] = None  # uncapped: keep every failed attempt
    experiment["source"]["dataset_path"] = source_dataset
    experiment["source"]["n"] = None  # use all source demos
    generation = experiment["generation"]
    generation["path"] = output_folder
    generation["num_trials"] = num_attempts
    generation["guarantee"] = False
    generation["keep_failed"] = True
    generation["select_src_per_subtask"] = False

    config["experiment"]["task"]["name"] = task_name

    config["obs"]["collect_obs"] = collect_obs
    config["obs"]["camera_names"] = [] if camera_names is None else camera_names

    task_spec = config["task"]["task_spec"]
    for subtask_key, subtask in task_spec.items():
        if not subtask_key.startswith("subtask_"):
            continue
        subtask["selection_strategy"] = "random"
        subtask["selection_strategy_kwargs"] = None
    return config


def load_template(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_config(config: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=4))
    return path
