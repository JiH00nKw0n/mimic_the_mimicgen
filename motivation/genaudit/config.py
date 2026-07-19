"""Typed loading of task/experiment YAML configs (assembly layer, Ch11).

Only this module and the CLI know about YAML; everything else takes typed
values. Validation is loud: a config problem must fail before any compute is
spent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from genaudit.envs.bounds import BOUNDS


def _require_yaml():
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - env-dependent
        raise ImportError("pyyaml is required for config loading") from error
    return yaml


@dataclass(frozen=True)
class TaskSpec:
    task: str
    symmetry_orders: dict[str, int]
    source_dataset: str
    env_interface: str
    generation_template: str
    rollout_horizon: int
    ladder: tuple[str, ...]
    widest_variant: str
    contrast_variants: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.task not in BOUNDS:
            raise ValueError(f"unknown task {self.task!r}; registry has {sorted(BOUNDS)}")
        known = BOUNDS[self.task]
        for variant in (*self.ladder, self.widest_variant, *self.contrast_variants):
            if variant not in known:
                raise ValueError(
                    f"task {self.task!r}: variant {variant!r} not in bounds "
                    f"registry ({sorted(known)})"
                )
        objects = set(known[self.widest_variant])
        missing = objects - set(self.symmetry_orders)
        if missing:
            raise ValueError(
                f"task {self.task!r}: symmetry_orders missing for {sorted(missing)}"
            )


@dataclass(frozen=True)
class ArmSpec:
    name: str
    size: int
    quota_per_stratum: int | None = None  # None for the baseline arm

    def __post_init__(self) -> None:
        if self.quota_per_stratum is not None:
            if self.quota_per_stratum < 1:
                raise ValueError(f"arm {self.name}: quota_per_stratum must be >= 1")
            if self.size % self.quota_per_stratum != 0:
                raise ValueError(
                    f"arm {self.name}: size {self.size} not divisible by "
                    f"quota_per_stratum {self.quota_per_stratum}"
                )

    @property
    def num_strata(self) -> int:
        if self.quota_per_stratum is None:
            raise ValueError(f"arm {self.name} has no quota_per_stratum")
        return self.size // self.quota_per_stratum


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    task: str
    variant: str
    primary_distance: str  # fixed after the B0 trend-preservation check
    num_attempts: int  # TOTAL attempts across pool_seeds (split evenly)
    pool_seeds: tuple[int, ...]
    binning_k: int
    fallback_k: int | None
    tv_threshold: float
    min_bin_fraction: float
    arms: tuple[ArmSpec, ...]
    dataset_seeds: tuple[int, ...]
    paths: dict[str, str]  # demo_hdf5 / failed_hdf5 / records / edges / out_dir

    def __post_init__(self) -> None:
        if self.binning_k < 2:
            raise ValueError("binning_k must be >= 2")
        if self.fallback_k is not None and not 2 <= self.fallback_k < self.binning_k:
            raise ValueError(
                f"fallback_k must be in [2, {self.binning_k}), got {self.fallback_k}"
            )
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate arm names: {names}")
        if not self.pool_seeds:
            raise ValueError("pool.seeds must not be empty")
        if self.num_attempts % len(self.pool_seeds) != 0:
            raise ValueError(
                f"pool.num_attempts {self.num_attempts} (total) not divisible by "
                f"{len(self.pool_seeds)} pool seeds — fix the config (no silent split)"
            )

    @property
    def attempts_per_pool_seed(self) -> int:
        return self.num_attempts // len(self.pool_seeds)


@dataclass(frozen=True)
class E1SweepSpec:
    """E1 fixed-attempt DGR sweep: many (task, variant) pools, one protocol."""

    experiment: str
    num_attempts: int  # per (task, variant) pool
    seed: int
    tasks: dict[str, tuple[str, ...]]  # task -> variants to generate
    binning_k: int
    out_root: str

    def __post_init__(self) -> None:
        if self.experiment != "e1":
            raise ValueError(f"E1SweepSpec requires experiment: e1, got {self.experiment!r}")
        if not self.tasks:
            raise ValueError("tasks must not be empty")

    def out_dir(self, task: str, variant: str) -> str:
        return str(Path(self.out_root) / f"{task}_{variant}")


def load_task_spec(path: str | Path) -> TaskSpec:
    yaml = _require_yaml()
    payload = yaml.safe_load(Path(path).read_text())
    try:
        return TaskSpec(
            task=payload["task"],
            symmetry_orders={
                name: spec["symmetry_order"] for name, spec in payload["objects"].items()
            },
            source_dataset=payload["source_dataset"],
            env_interface=payload["env_interface"],
            generation_template=payload["generation_template"],
            rollout_horizon=payload["rollout_horizon"],
            ladder=tuple(payload["ladder"]),
            widest_variant=payload["widest_variant"],
            contrast_variants=tuple(payload.get("contrast_variants", ())),
        )
    except KeyError as error:
        raise KeyError(f"{path}: missing required task-config key {error}") from error


def load_experiment_spec(path: str | Path) -> ExperimentSpec | E1SweepSpec:
    """Load an experiment config, dispatching on its `experiment` field."""
    yaml = _require_yaml()
    payload = yaml.safe_load(Path(path).read_text())
    kind = payload.get("experiment")
    try:
        if kind == "e1":
            return E1SweepSpec(
                experiment=kind,
                num_attempts=payload["protocol"]["num_attempts"],
                seed=payload["protocol"]["seed"],
                tasks={
                    task: tuple(variants) for task, variants in payload["tasks"].items()
                },
                binning_k=payload["binning"]["k"],
                out_root=payload["paths"]["out_root"],
            )
        if kind == "e2":
            arms = tuple(
                ArmSpec(
                    name=name,
                    size=spec["size"],
                    quota_per_stratum=spec.get("quota_per_stratum"),
                )
                for name, spec in payload["arms"].items()
            )
            return ExperimentSpec(
                experiment=kind,
                task=payload["task"],
                variant=payload["variant"],
                primary_distance=payload["distance"]["primary"],
                num_attempts=payload["pool"]["num_attempts"],
                pool_seeds=tuple(payload["pool"]["seeds"]),
                binning_k=payload["binning"]["k"],
                fallback_k=payload["binning"].get("fallback_k"),
                tv_threshold=payload["certification"]["tv_threshold"],
                min_bin_fraction=payload["certification"]["min_bin_fraction"],
                arms=arms,
                dataset_seeds=tuple(payload["dataset_seeds"]),
                paths=dict(payload["paths"]),
            )
    except KeyError as error:
        raise KeyError(f"{path}: missing required experiment-config key {error}") from error
    raise ValueError(f"{path}: experiment must be 'e1' or 'e2', got {kind!r}")
