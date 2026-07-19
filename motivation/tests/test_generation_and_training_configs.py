import pytest

from genaudit.generation.mimicgen_backend import build_generation_config
from genaudit.training.robomimic_config import patch_training_config


def _mimicgen_template() -> dict:
    """Shape mirrors mimicgen/exps/templates/robosuite/threading.json."""
    return {
        "name": "threading",
        "type": "robosuite",
        "experiment": {
            "name": "demo_src_threading_task_D0",
            "source": {"dataset_path": "/old.hdf5", "filter_key": None, "n": 10, "start": None},
            "generation": {
                "path": "/old/out",
                "guarantee": True,
                "keep_failed": True,
                "num_trials": 1000,
                "select_src_per_subtask": False,
                "transform_first_robot_pose": False,
                "interpolate_from_last_target_pose": True,
            },
            "task": {"name": "Threading_D0", "robot": None, "gripper": None},
            "max_num_failures": 25,
            "render_video": True,
            "num_demo_to_render": 10,
            "num_fail_demo_to_render": 25,
            "seed": 1,
        },
        "obs": {"collect_obs": True, "camera_names": ["agentview"], "camera_height": 84, "camera_width": 84},
        "task": {
            "task_spec": {
                "subtask_1": {
                    "object_ref": "needle",
                    "selection_strategy": "nearest_neighbor_object",
                    "selection_strategy_kwargs": {"nn_k": 3},
                    "action_noise": 0.05,
                },
                "subtask_2": {
                    "object_ref": "tripod",
                    "selection_strategy": "random",
                    "selection_strategy_kwargs": None,
                    "action_noise": 0.05,
                },
            }
        },
    }


def test_generation_config_enforces_audit_protocol():
    config = build_generation_config(
        _mimicgen_template(),
        task_name="Threading_D2E",
        source_dataset="/data/source/threading_prepared.hdf5",
        output_folder="/data/e2/threading_D2E",
        num_attempts=6250,
        seed=11,
    )
    generation = config["experiment"]["generation"]
    assert generation["num_trials"] == 6250
    assert generation["guarantee"] is False  # fixed attempt count
    assert generation["keep_failed"] is True
    assert generation["select_src_per_subtask"] is False
    assert config["experiment"]["max_num_failures"] is None  # uncapped
    assert config["experiment"]["seed"] == 11
    assert config["experiment"]["task"]["name"] == "Threading_D2E"
    assert config["experiment"]["source"]["dataset_path"].endswith("prepared.hdf5")
    # config-registry key must never be touched (mimicgen resolves the config
    # class by it); state-only pools = obs on but zero cameras
    assert config["name"] == "threading"
    assert config["obs"]["collect_obs"] is True
    assert config["obs"]["camera_names"] == []
    # video rendering competes with sim workers for CPU — must be off
    assert config["experiment"]["render_video"] is False
    assert config["experiment"]["num_demo_to_render"] == 0
    assert config["experiment"]["num_fail_demo_to_render"] == 0
    for subtask in config["task"]["task_spec"].values():
        assert subtask["selection_strategy"] == "random"
        assert subtask["selection_strategy_kwargs"] is None
    # noise settings untouched (protocol changes selection only)
    assert config["task"]["task_spec"]["subtask_1"]["action_noise"] == 0.05


def test_generation_config_does_not_mutate_template():
    template = _mimicgen_template()
    build_generation_config(
        template,
        task_name="X",
        source_dataset="s",
        output_folder="o",
        num_attempts=10,
        seed=1,
    )
    assert template["experiment"]["generation"]["guarantee"] is True


def _robomimic_base() -> dict:
    return {
        "experiment": {
            "name": "bc_rnn_low_dim",
            "rollout": {"enabled": True, "n": 50, "horizon": 400, "rate": 50, "terminate_on_success": True},
        },
        "train": {"data": "/old.hdf5", "hdf5_filter_key": None, "seed": 101, "output_dir": "/old"},
        "algo": {"rnn": {"enabled": True}},
    }


def test_training_config_varies_only_intended_fields():
    base = _robomimic_base()
    config = patch_training_config(
        base,
        dataset_path="/data/pool.hdf5",
        filter_key="transform_uniform_seed101",
        seed=7,
        experiment_name="e2_threading_uniform_s101",
        output_dir="/results/e2",
        rollout_overrides={"rate": 50},
    )
    assert config["train"]["hdf5_filter_key"] == "transform_uniform_seed101"
    assert config["train"]["seed"] == 7
    assert config["algo"] == base["algo"]  # recipe untouched
    assert base["train"]["data"] == "/old.hdf5"  # no mutation


def test_training_config_rejects_unknown_rollout_keys():
    with pytest.raises(ValueError, match="unknown rollout override"):
        patch_training_config(
            _robomimic_base(),
            dataset_path="d",
            filter_key="k",
            seed=1,
            experiment_name="n",
            output_dir="o",
            rollout_overrides={"num_rollouts": 100},
        )
