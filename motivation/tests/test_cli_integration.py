"""Config-driven pipeline integration: YAML -> curate -> analyze on synthetic data."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from genaudit.cli import main
from genaudit.config import load_experiment_spec, load_task_spec
from genaudit.records.schema import AttemptRecord, write_jsonl

# noqa notes: numpy used by the RNG-independence tests
import numpy as np  # noqa: E402

REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("path", sorted((REPO_CONFIG_DIR / "tasks").glob("*.yaml")))
def test_repo_task_configs_validate_against_registry(path):
    spec = load_task_spec(path)
    assert spec.widest_variant in spec.ladder
    assert spec.rollout_horizon > 0


def _synthetic_records(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    records = []
    for index in range(n):
        d_pos = float(rng.beta(1.6, 2.4))
        success = bool(rng.uniform() < 0.75 * (1 - 0.8 * d_pos))
        records.append(
            AttemptRecord(
                task="threading",
                variant="D2E",
                attempt_id=f"demo_{index}@demo{'_failed' if not success else ''}.hdf5",
                source_demo_id=int(rng.integers(10)),
                success=success,
                episode_length=200,
                displacements=(),
                d_raw=d_pos * 0.81,
                d_pos=d_pos,
                d_rot=0.0,
            )
        )
    return records


def _experiment_yaml(tmp_path: Path, records_path: Path) -> Path:
    payload = {
        "experiment": "e2",
        "task": "threading",
        "variant": "D2E",
        "distance": {"primary": "d_pos"},
        "pool": {"num_attempts": 3000, "seeds": [1]},
        "binning": {"k": 5, "fallback_k": 4},
        "certification": {"tv_threshold": 0.02, "min_bin_fraction": 0.9},
        "arms": {
            "baseline": {"size": 500},
            "transform_uniform": {"size": 500, "quota_per_stratum": 100},
            "ancestry_balanced": {"size": 500, "quota_per_stratum": 50},
        },
        "dataset_seeds": [101, 102],
        "paths": {
            "records": str(records_path),
            "edges": str(tmp_path / "edges.json"),
            "out_dir": str(tmp_path / "out"),
        },
    }
    path = tmp_path / "e2_test.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_curate_filter_keys_and_analyze_round_trip(tmp_path):
    import h5py

    records = _synthetic_records()
    records_path = tmp_path / "attempts.jsonl"
    write_jsonl(records, records_path)
    experiment_yaml = _experiment_yaml(tmp_path, records_path)
    task_yaml = REPO_CONFIG_DIR / "tasks" / "threading.yaml"

    main(["curate", "--task-config", str(task_yaml), "--experiment-config", str(experiment_yaml)])
    manifest = json.loads((tmp_path / "out" / "arms_manifest.json").read_text())
    assert manifest["k_used"] == 5
    arms = manifest["arms"]
    assert set(arms) == {
        f"{arm}_seed{seed}"
        for arm in ("baseline", "transform_uniform", "ancestry_balanced")
        for seed in (101, 102)
    }
    uniform = arms["transform_uniform_seed101"]
    assert uniform["certification"]["passed"] is True
    assert sorted(uniform["per_stratum_counts"], reverse=True)[0] <= 110
    assert len(uniform["attempt_ids"]) == 500
    assert len(set(uniform["attempt_ids"])) == 500
    # demo_names are the robomimic-usable form of attempt_ids
    assert uniform["demo_names"][0] == uniform["attempt_ids"][0].split("@")[0]
    # ancestry arm balanced over the 10 configured sources
    assert arms["ancestry_balanced_seed101"]["per_stratum_counts"] == [50] * 10
    # arms must differ across dataset seeds (the dataset-seed axis is real)
    assert arms["baseline_seed101"]["attempt_ids"] != arms["baseline_seed102"]["attempt_ids"]

    # filter-keys: build a demo.hdf5 holding every retained demo group, then
    # verify each arm's mask lands with the right names
    demo_hdf5 = tmp_path / "demo.hdf5"
    retained_names = {r.attempt_id.split("@")[0] for r in records if r.success}
    with h5py.File(demo_hdf5, "w") as handle:
        for name in retained_names:
            handle.create_group(f"data/{name}")
    payload = yaml.safe_load(experiment_yaml.read_text())
    payload["paths"]["demo_hdf5"] = str(demo_hdf5)
    experiment_yaml.write_text(yaml.safe_dump(payload, sort_keys=False))
    main(["filter-keys", "--experiment-config", str(experiment_yaml)])
    from genaudit.curation.filter_keys import read_filter_key

    stored = read_filter_key(demo_hdf5, "transform_uniform_seed101")
    assert stored == uniform["demo_names"]

    main(["analyze", "--experiment-config", str(experiment_yaml)])
    analysis = json.loads((tmp_path / "out" / "analysis.json").read_text())
    assert analysis["n_attempts"] == 3000
    curve = analysis["dgr_curve"]["per_bin_dgr"]
    assert curve[0] > curve[-1]  # synthetic monotone decline recovered
    assert analysis["trend_by_definition"]["d_pos"]["spearman_rho"] < 0
    edges = json.loads((tmp_path / "edges.json").read_text())
    assert edges["k"] == 5 and len(edges["interior_edges"]) == 4


def test_arm_rng_streams_are_independent(tmp_path):
    """Regression for the shared-RNG defect: changing K (the pre-registered
    fallback) or drawing other arms first must not change the ancestry arm."""
    from genaudit.cli import _sample_arm
    from genaudit.config import ArmSpec, load_experiment_spec
    from genaudit.curation.binning import assign_bins, compute_quantile_edges

    records = _synthetic_records()
    records_path = tmp_path / "attempts.jsonl"
    write_jsonl(records, records_path)
    experiment = load_experiment_spec(_experiment_yaml(tmp_path, records_path))
    distances = np.array([r.d_pos for r in records])
    retained = [i for i, r in enumerate(records) if r.success]
    retained_sources = np.array([records[i].source_demo_id for i in retained])
    ancestry = ArmSpec(name="ancestry_balanced", size=500, quota_per_stratum=50)
    uniform = ArmSpec(name="transform_uniform", size=500, quota_per_stratum=100)

    selections = []
    for k in (5, 4):
        bins = assign_bins(distances[retained], compute_quantile_edges(distances, k))
        # draw the transform arm first — must not perturb the ancestry draw
        _sample_arm(uniform, 101, bins, retained_sources, k, experiment)
        result = _sample_arm(ancestry, 101, bins, retained_sources, k, experiment)
        selections.append(result.selected)
    assert selections[0] == selections[1]


def test_ancestry_arm_rejects_out_of_range_source_ids(tmp_path):
    from genaudit.cli import _sample_arm
    from genaudit.config import ArmSpec, load_experiment_spec

    records_path = tmp_path / "attempts.jsonl"
    write_jsonl(_synthetic_records(), records_path)
    experiment = load_experiment_spec(_experiment_yaml(tmp_path, records_path))
    ancestry = ArmSpec(name="ancestry_balanced", size=500, quota_per_stratum=50)
    bad_sources = np.array([12] * 600)  # id beyond the 10 configured sources
    with pytest.raises(ValueError, match="source id 12"):
        _sample_arm(ancestry, 101, np.zeros(600, dtype=int), bad_sources, 5, experiment)


def test_analyze_survives_degenerate_high_dgr_pool(tmp_path):
    """D0 / positive-control pools can be ~100% success; analyze must still
    write DGR + ancestry with the trend marked unavailable."""
    records = [
        AttemptRecord(
            task="threading",
            variant="D0",
            attempt_id=f"demo_{i}@demo.hdf5",
            source_demo_id=i % 10,
            success=True,
            episode_length=100,
            displacements=(),
            d_raw=0.1,
            d_pos=0.1 + i * 1e-4,
            d_rot=0.0,
        )
        for i in range(600)
    ]
    records_path = tmp_path / "attempts.jsonl"
    write_jsonl(records, records_path)
    experiment_yaml = _experiment_yaml(tmp_path, records_path)
    main(["analyze", "--experiment-config", str(experiment_yaml)])
    analysis = json.loads((tmp_path / "out" / "analysis.json").read_text())
    assert analysis["dgr"] == 1.0
    assert analysis["trend_by_definition"] is None
    assert "zero variance" in analysis["trend_unavailable_reason"]
    assert analysis["ancestry"]["n_eff_retained"] == pytest.approx(10.0)


def test_gen_config_e1_sweep_and_e2_pool_split(tmp_path):
    template = {
        "name": "threading",
        "experiment": {
            "name": "x",
            "source": {"dataset_path": "old", "n": 10},
            "generation": {"path": "old", "guarantee": True, "keep_failed": True,
                           "num_trials": 1000, "select_src_per_subtask": False},
            "task": {"name": "Threading_D0"},
            "max_num_failures": 25,
            "seed": 1,
        },
        "obs": {"collect_obs": True, "camera_names": ["agentview"]},
        "task": {"task_spec": {"subtask_1": {"selection_strategy": "random",
                                             "selection_strategy_kwargs": None}}},
    }
    template_path = tmp_path / "threading_template.json"
    template_path.write_text(json.dumps(template))
    task_yaml = tmp_path / "threading.yaml"
    task_yaml.write_text(yaml.safe_dump({
        "task": "threading",
        "objects": {"needle": {"symmetry_order": 1}, "tripod": {"symmetry_order": 1}},
        "source_dataset": "datasets/source/threading.hdf5",
        "env_interface": "MG_Threading",
        "generation_template": str(template_path),
        "rollout_horizon": 400,
        "ladder": ["D0", "D1", "D2E"],
        "widest_variant": "D2E",
    }, sort_keys=False))

    e1_yaml = tmp_path / "e1.yaml"
    e1_yaml.write_text(yaml.safe_dump({
        "experiment": "e1",
        "protocol": {"num_attempts": 500, "seed": 1},
        "tasks": {"threading": ["D0", "D2E"]},
        "binning": {"k": 5},
        "paths": {"out_root": str(tmp_path / "e1_out")},
    }, sort_keys=False))
    main(["gen-config", "--task-config", str(task_yaml),
          "--experiment-config", str(e1_yaml), "--variant", "D2E"])
    written = json.loads((tmp_path / "e1_out" / "threading_D2E" / "mg_D2E_seed1.json").read_text())
    assert written["experiment"]["generation"]["num_trials"] == 500  # E1 protocol, not E2 pool
    assert written["experiment"]["task"]["name"] == "Threading_D2E"

    # E2: total 3000 across 2 pool seeds -> 1500 per seed, one config each
    records_path = tmp_path / "attempts.jsonl"
    write_jsonl(_synthetic_records(), records_path)
    e2_yaml = _experiment_yaml(tmp_path, records_path)
    payload = yaml.safe_load(e2_yaml.read_text())
    payload["pool"]["seeds"] = [1, 2]
    e2_yaml.write_text(yaml.safe_dump(payload, sort_keys=False))
    main(["gen-config", "--task-config", str(task_yaml),
          "--experiment-config", str(e2_yaml), "--variant", "D2E"])
    for seed in (1, 2):
        written = json.loads((tmp_path / "out" / f"mg_D2E_seed{seed}.json").read_text())
        assert written["experiment"]["generation"]["num_trials"] == 1500
        assert written["experiment"]["seed"] == seed


def test_experiment_spec_validation(tmp_path):
    records_path = tmp_path / "r.jsonl"
    experiment_yaml = _experiment_yaml(tmp_path, records_path)
    spec = load_experiment_spec(experiment_yaml)
    assert spec.primary_distance == "d_pos"
    assert [arm.name for arm in spec.arms] == [
        "baseline",
        "transform_uniform",
        "ancestry_balanced",
    ]
