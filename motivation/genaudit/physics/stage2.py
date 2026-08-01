"""Stage2 FR3 contact-calibration contract -> robosuite/MuJoCo mapping.

Loads `stage2_contact_calibration_v2` (contact_parameter_contract.yaml +
posterior_samples.csv) and provides per-attempt samplers for three profiles:

  nominal    every contract nominal (deterministic_nominal profile)
  posterior  row-wise draws from posterior_samples.csv for the fitted
             parameters (cube-cube friction, cube damping); the rest nominal
  robust     robust_stochastic: 80% posterior-near / 20% full training-range,
             with the contract joint rules enforced by rejection

Mapping semantics (contract values are effective contact-PAIR coefficients;
MuJoCo combines pair friction as the element-wise MAX of the two geoms):

  table_mu  -> table_collision geom sliding mu   (table >= objects, so the
               table geom alone pins the table-object pair value)
  cube_mu   -> cube / nut / peg geom sliding mu  (object-object pair value)
  finger_mu -> gripper finger + pad geom sliding mu (finger-object pair value,
               valid while finger_mu >= cube_mu — enforced by rejection)
  force_scale -> gripper actuator forcerange RATIO force_scale/1.6 (the
               contract 1.6 is FR3-absolute; only the ratio transplants;
               partially inert above the kp*error servo demand ceiling)

NOT mapped (sampled + recorded for the integration report, not applied):
  restitution (MuJoCo has no direct restitution parameter; solref-based),
  table dynamic friction (MuJoCo has a single sliding coefficient; static
  values are used since stiction dominates grasp/stack events),
  gripper speed_scale (would require control-loop changes),
  cube linear/angular damping (MuJoCo free joints take ONE damping scalar for
  all 6 dof; any value large enough to matter linearly over-damps rotation
  ~1000x, and the contract ranges are dynamically negligible in these tasks).

Square (peg-in-hole) uses this as a PROXY mapping (finger_cube->finger-nut,
table_cube->table-nut, cube_cube->nut-peg) — a "new condition" per the
contract's own integration guide.

Pure stdlib (yaml aside) — no numpy, so the sampler runs anywhere.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

FORCE_SCALE_NOMINAL = 1.6
CUBE_MASS_KG = 0.065  # contract cubes 63.3-66.0 g
JOINT_MIN_MU_FORCE = 1.241946  # finger_mu * force_scale >= this (contract rule)

# OmniReset Table 2 actuation rows (P5 arm), multiplicative scales.
OMNI_ACTUATION = {
    "osc_kp_scale": (0.8, 1.2),        # U
    "osc_damping_scale": (0.8, 1.2),   # U
    "joint_fric_scale": (0.8, 1.2),    # U — robot joint frictionloss
    "joint_armature_scale": (0.8, 1.2),
    "joint_damping_scale": (0.8, 1.2),
    "grip_kp_scale": (0.5, 2.0),       # logU — gripper stiffness
    "grip_damping_scale": (0.5, 2.0),  # logU
}


@dataclass(frozen=True)
class Stage2Contract:
    table_mu_nominal: float
    table_mu_range: tuple[float, float]
    cube_mu_nominal: float
    cube_mu_range: tuple[float, float]
    finger_mu_nominal: float
    finger_mu_range: tuple[float, float]
    force_scale_nominal: float
    force_scale_range: tuple[float, float]
    lin_damp_nominal: float
    lin_damp_range: tuple[float, float]
    ang_damp_nominal: float
    ang_damp_range: tuple[float, float]
    posterior_rows: tuple[dict, ...]  # cube_mu / lin_damp / ang_damp columns


def load_contract(contract_dir: str | Path) -> Stage2Contract:
    import yaml

    contract_dir = Path(contract_dir)
    doc = yaml.safe_load(
        (contract_dir / "contact_parameter_contract.yaml").read_text()
    )
    params = doc["parameters"]

    def nominal_and_range(section: dict, key: str) -> tuple[float, tuple[float, float]]:
        entry = section[key]
        lo, hi = entry["training_range"]
        return float(entry["nominal"]), (float(lo), float(hi))

    table_nom, table_rng = nominal_and_range(params["table_cube"], "static_friction")
    cube_nom, cube_rng = nominal_and_range(params["cube_cube"], "static_friction")
    finger_nom, finger_rng = nominal_and_range(params["finger_cube"], "static_friction")
    force_nom, force_rng = nominal_and_range(params["finger_cube"], "gripper_force_scale")
    lin_nom, lin_rng = nominal_and_range(params["cube_rigid_body"], "linear_damping")
    ang_nom, ang_rng = nominal_and_range(params["cube_rigid_body"], "angular_damping")

    rows: list[dict] = []
    with (contract_dir / "posterior_samples.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "cube_mu": float(row["cube_cube_static_friction"]),
                    "lin_damp": float(row["cube_linear_damping"]),
                    "ang_damp": float(row["cube_angular_damping"]),
                }
            )
    if not rows:
        raise ValueError(f"no posterior rows in {contract_dir}")
    return Stage2Contract(
        table_mu_nominal=table_nom, table_mu_range=table_rng,
        cube_mu_nominal=cube_nom, cube_mu_range=cube_rng,
        finger_mu_nominal=finger_nom, finger_mu_range=finger_rng,
        force_scale_nominal=force_nom, force_scale_range=force_rng,
        lin_damp_nominal=lin_nom, lin_damp_range=lin_rng,
        ang_damp_nominal=ang_nom, ang_damp_range=ang_rng,
        posterior_rows=tuple(rows),
    )


def _nominal_sample(contract: Stage2Contract) -> dict:
    return {
        "table_mu": contract.table_mu_nominal,
        "cube_mu": contract.cube_mu_nominal,
        "finger_mu": contract.finger_mu_nominal,
        "force_scale": contract.force_scale_nominal,
        "lin_damp": contract.lin_damp_nominal,
        "ang_damp": contract.ang_damp_nominal,
        "mass_scale": 1.0,
        "size_scale": 1.0,
    }


def _joint_rules_ok(sample: dict) -> bool:
    if sample["finger_mu"] * sample["force_scale"] < JOINT_MIN_MU_FORCE:
        return False
    if sample["finger_mu"] < sample["cube_mu"]:  # mujoco max-combine validity
        return False
    return True


def sample_profile(
    contract: Stage2Contract, profile: str, rng: random.Random
) -> dict:
    """One contract-space sample for one generation attempt."""
    if profile == "nominal":
        return _nominal_sample(contract)

    if profile == "posterior":
        sample = _nominal_sample(contract)
        row = contract.posterior_rows[rng.randrange(len(contract.posterior_rows))]
        sample.update(row)  # cube_mu, lin_damp, ang_damp — full row, correlated
        # posterior rows can exceed finger nominal only if cube_mu > 0.8; the
        # fitted 90% interval tops at 0.742 so this never rejects in practice,
        # but guard anyway by clamping to the max-combine validity limit.
        sample["cube_mu"] = min(sample["cube_mu"], sample["finger_mu"])
        return sample

    if profile in ("robust", "robust_omni"):
        for _ in range(1000):
            sample = _nominal_sample(contract)
            if rng.random() < 0.8:
                # posterior-near: fitted params from a posterior row (keeps
                # their correlations), unfitted params triangular around
                # nominal inside the training range.
                row = contract.posterior_rows[rng.randrange(len(contract.posterior_rows))]
                sample.update(row)
                sample["table_mu"] = rng.triangular(*contract.table_mu_range, contract.table_mu_nominal)
                sample["finger_mu"] = rng.triangular(*contract.finger_mu_range, contract.finger_mu_nominal)
                sample["force_scale"] = rng.triangular(*contract.force_scale_range, contract.force_scale_nominal)
            else:
                # full-range: independent uniforms over the training ranges.
                sample["table_mu"] = rng.uniform(*contract.table_mu_range)
                sample["cube_mu"] = rng.uniform(*contract.cube_mu_range)
                sample["finger_mu"] = rng.uniform(*contract.finger_mu_range)
                sample["force_scale"] = rng.uniform(*contract.force_scale_range)
                sample["lin_damp"] = rng.uniform(*contract.lin_damp_range)
                sample["ang_damp"] = rng.uniform(*contract.ang_damp_range)
            sample["mass_scale"] = rng.uniform(0.98, 1.02)   # contract mass spread
            sample["size_scale"] = rng.uniform(0.994, 1.006)  # 50.4-51.0mm on 50.7
            if _joint_rules_ok(sample):
                return sample
        raise RuntimeError("robust sampler: joint-rule rejection did not converge")

    raise ValueError(f"unknown profile {profile!r}")


def sample_omni_actuation(rng: random.Random) -> dict:
    """OmniReset Table 2 actuation scales (P5 arm)."""
    import math

    out = {}
    for key, (lo, hi) in OMNI_ACTUATION.items():
        if key.startswith("grip_"):  # logU
            out[key] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            out[key] = rng.uniform(lo, hi)
    return out


def to_mujoco(sample: dict) -> dict:
    """Contract-space sample -> concrete MuJoCo attribute values."""
    return {
        "table_sliding": sample["table_mu"],
        "object_sliding": sample["cube_mu"],
        "finger_sliding": sample["finger_mu"],
        "gripper_force_scale": sample["force_scale"] / FORCE_SCALE_NOMINAL,
        "mass_scale": sample["mass_scale"],
        "size_scale": sample["size_scale"],
    }


if __name__ == "__main__":  # smoke: python -m genaudit.physics.stage2 <contract_dir>
    import json
    import sys

    contract = load_contract(sys.argv[1])
    rng = random.Random(0)
    for profile in ("nominal", "posterior", "robust"):
        sample = sample_profile(contract, profile, rng)
        assert _joint_rules_ok(sample), profile
        print(profile, json.dumps({k: round(v, 4) for k, v in sample.items()}))
        print("   mj:", json.dumps({k: round(v, 5) for k, v in to_mujoco(sample).items()}))
    print("omni:", json.dumps({k: round(v, 3) for k, v in sample_omni_actuation(rng).items()}))
    counts = {"reject_rate_check": sum(
        not _joint_rules_ok(sample_profile(contract, "robust", rng)) for _ in range(2000)
    )}
    print("post-hoc invalid robust samples (must be 0):", counts)
