"""Non-destructive FR3 cube SysID bundle integration for evaluation.

This module deliberately lives beside the FR3 cube task instead of changing the
generic OmniReset randomization.  Values come from
``fr3_cube_system_calibration_bundle_v1`` and are sampled once per environment.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import carb
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
CUBE_MASSES_KG = {
    # Receiving-scene identity mapping: Cube_1 is red, Cube_2 is blue, and
    # Cube_3 is the remaining (black in the measured set; green preview only).
    "cube_1": 0.0633333333,
    "cube_2": 0.0650333333,
    "cube_3": 0.0660333333,
}


def _read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


class apply_fr3_cube_calibration_bundle(ManagerTermBase):
    """Apply the calibrated dynamics/contact ensemble once at startup.

    ``profile=nominal`` is deterministic. ``profile=posterior_stochastic``
    samples the measured dynamics rows and posterior contact samples only.
    ``profile=robust_stochastic`` adds the contact module's recommended 80/20
    posterior-near/full-range mixture.  The existing OSC is never modified.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.bundle_root = Path(cfg.params["bundle_root"]).expanduser().resolve()
        self.profile = str(cfg.params.get("profile", "robust_stochastic"))
        if self.profile not in {"nominal", "posterior_stochastic", "robust_stochastic"}:
            raise ValueError(f"Unsupported SysID profile: {self.profile}")
        self.robot: Articulation = env.scene[cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]
        self.cubes: dict[str, RigidObject] = {
            name: env.scene[name] for name in cfg.params.get("cube_names", tuple(CUBE_MASSES_KG))
        }
        self.work_surface: RigidObject = env.scene[cfg.params.get("work_surface_name", "work_surface")]
        self.arm_joint_ids = self.robot.find_joints(list(JOINT_NAMES))[0]
        self.finger_joint_ids = self.robot.find_joints(["fr3_finger_joint.*"])[0]
        self.arm_actuator_name = str(cfg.params.get("arm_actuator_name", "arm"))
        self._dynamics_rows = _read_csv(
            self.bundle_root / "modules/dynamics_controller/domain_randomization_samples.csv"
        )
        self._contact_rows = _read_csv(self.bundle_root / "modules/contact/posterior_samples.csv")
        self.applied_summary: dict[str, object] = {}

    @staticmethod
    def _set_materials(asset: RigidObject | Articulation, env_ids_cpu: torch.Tensor, values: torch.Tensor) -> None:
        materials = asset.root_physx_view.get_material_properties()
        # hf80k 수정: max_shapes를 int()로 감싼다. 이 판본의 PhysX 뷰가 이 값을 파이썬
        # 정수가 아니라 0차원 텐서로 돌려주는데, expand()에 텐서를 넣으면
        # "only integer tensors of a single element can be converted to an index"로
        # 죽는다. 환경을 만드는 startup 이벤트에서 터지므로 생성이 통째로 시작도 못 한다.
        max_shapes = int(asset.root_physx_view.max_shapes)
        materials[env_ids_cpu] = values[:, None, :].expand(-1, max_shapes, -1)
        asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _set_finger_materials(self, env_ids_cpu: torch.Tensor, values: torch.Tensor) -> None:
        materials = self.robot.root_physx_view.get_material_properties()
        shape_counts: list[int] = []
        for link_path in self.robot.root_physx_view.link_paths[0]:
            view = self.robot._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
            shape_counts.append(int(view.max_shapes))   # 위와 같은 이유로 int()
        body_names = list(self.robot.body_names)
        for finger_name in ("fr3_leftfinger", "fr3_rightfinger"):
            body_id = body_names.index(finger_name)
            start = sum(shape_counts[:body_id])
            stop = start + shape_counts[body_id]
            materials[env_ids_cpu, start:stop] = values[:, None, :]
        self.robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _sample(self, count: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        if self.profile == "nominal":
            armature = np.full((count, 7), 0.1)
            static = np.tile([0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5], (count, 1))
            dynamic = np.tile([0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4], (count, 1))
            viscous = np.zeros((count, 7))
            delay = np.zeros(count, dtype=np.int64)
            table_static = np.full(count, 1.9)
            table_dynamic = np.full(count, 1.6753858178)
            table_restitution = np.full(count, 0.1301685272)
            cube_static = np.full(count, 0.651759882)
            cube_dynamic = np.full(count, 0.5539958997)
            linear_damping = np.full(count, 0.0682659557)
            angular_damping = np.full(count, 0.0766774104)
            finger_static = np.full(count, 0.8)
            finger_dynamic = np.full(count, 0.68)
            force_scale = np.full(count, 1.6)
            posterior_near = np.ones(count, dtype=bool)
        else:
            dyn_ids = rng.integers(0, len(self._dynamics_rows), count)
            dyn = [self._dynamics_rows[index] for index in dyn_ids]
            armature = np.asarray([[row[f"armature_j{i}"] for i in range(1, 8)] for row in dyn])
            static = np.asarray([[row[f"static_friction_j{i}"] for i in range(1, 8)] for row in dyn])
            dynamic = np.asarray([[row[f"dynamic_friction_j{i}"] for i in range(1, 8)] for row in dyn])
            viscous = np.asarray([[row[f"viscous_friction_j{i}"] for i in range(1, 8)] for row in dyn])
            delay = np.asarray([int(row["motor_delay_steps_120hz"]) for row in dyn], dtype=np.int64)

            posterior_near = (
                np.ones(count, dtype=bool)
                if self.profile == "posterior_stochastic"
                else rng.random(count) < 0.8
            )
            post_ids = rng.integers(0, len(self._contact_rows), count)
            post = [self._contact_rows[index] for index in post_ids]
            table_restitution = np.asarray([row["table_cube_restitution"] for row in post])
            cube_static = np.asarray([row["cube_cube_static_friction"] for row in post])
            cube_ratio = np.asarray([row["cube_cube_dynamic_ratio"] for row in post])
            linear_damping = np.asarray([row["cube_linear_damping"] for row in post])
            angular_damping = np.asarray([row["cube_angular_damping"] for row in post])
            full = ~posterior_near
            table_restitution[full] = rng.uniform(0.0, 0.225, full.sum())
            cube_static[full] = rng.uniform(0.4443944345, 0.8159016711, full.sum())
            cube_ratio[full] = rng.uniform(0.6, 0.98, full.sum())
            linear_damping[full] = rng.uniform(0.0, 0.15, full.sum())
            angular_damping[full] = rng.uniform(0.0, 0.15, full.sum())
            cube_dynamic = cube_static * cube_ratio

            table_static = rng.uniform(1.25, 2.6, count)
            table_dynamic = rng.triangular(1.1, 1.6753858178, 2.15, count)
            table_dynamic = np.minimum(table_dynamic, table_static)
            force_scale = rng.uniform(1.0, 2.0, count)
            finger_static = rng.uniform(0.65, 1.3, count)
            invalid = finger_static * force_scale < 1.241946
            while invalid.any():
                finger_static[invalid] = rng.uniform(0.65, 1.3, invalid.sum())
                force_scale[invalid] = rng.uniform(1.0, 2.0, invalid.sum())
                invalid = finger_static * force_scale < 1.241946
            finger_dynamic = finger_static * rng.uniform(0.6, 0.98, count)

            # PhysX supports at most 64K unique materials.  Quantize contact
            # tuples to a shared 256-bucket ensemble while retaining per-env
            # joint dynamics/delay samples.
            num_buckets = min(256, count)
            bucket_ids = rng.integers(0, num_buckets, count)
            for array in (
                table_static,
                table_dynamic,
                table_restitution,
                cube_static,
                cube_dynamic,
                linear_damping,
                angular_damping,
                finger_static,
                finger_dynamic,
                force_scale,
                posterior_near,
            ):
                array[:] = array[:num_buckets][bucket_ids]

        return {
            "armature": armature,
            "joint_static": static,
            "joint_dynamic": np.minimum(dynamic, static),
            "joint_viscous": viscous,
            "delay": delay,
            "table_static": table_static,
            "table_dynamic": table_dynamic,
            "table_restitution": table_restitution,
            "cube_static": cube_static,
            "cube_dynamic": cube_dynamic,
            "linear_damping": linear_damping,
            "angular_damping": angular_damping,
            "finger_static": finger_static,
            "finger_dynamic": finger_dynamic,
            "force_scale": force_scale,
            "posterior_near": posterior_near,
        }

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        bundle_root: str,
        profile: str = "robust_stochastic",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        cube_names: tuple[str, str, str] = ("cube_1", "cube_2", "cube_3"),
        work_surface_name: str = "work_surface",
        arm_actuator_name: str = "arm",
        sample_seed_offset: int = 73000,
        log_samples: bool = False,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.robot.device)
        # 색인으로 쓸 것이므로 정수 텐서임을 보장한다. startup 이벤트에서는 슬라이스나
        # 리스트가 올 수도 있다.
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.robot.device)
        env_ids_cpu = env_ids.detach().cpu().long()
        seed = int(getattr(env.cfg, "seed", 0) or 0) + int(sample_seed_offset)
        values = self._sample(len(env_ids), seed)

        def dev(name: str, dtype=torch.float32) -> torch.Tensor:
            return torch.as_tensor(values[name], dtype=dtype, device=self.robot.device)

        self.robot.write_joint_armature_to_sim(dev("armature"), joint_ids=self.arm_joint_ids, env_ids=env_ids)
        self.robot.write_joint_friction_coefficient_to_sim(
            dev("joint_static"),
            joint_dynamic_friction_coeff=dev("joint_dynamic"),
            joint_viscous_friction_coeff=dev("joint_viscous"),
            joint_ids=self.arm_joint_ids,
            env_ids=env_ids,
        )
        actuator = self.robot.actuators[self.arm_actuator_name]
        if hasattr(actuator, "positions_delay_buffer"):
            delays = dev("delay", torch.int)
            actuator.positions_delay_buffer.set_time_lag(delays, env_ids)
            actuator.velocities_delay_buffer.set_time_lag(delays, env_ids)
            actuator.efforts_delay_buffer.set_time_lag(delays, env_ids)

        # Effective pair coefficients are converted to material components for
        # PhysX multiply combine: cube*cube, table*cube, finger*cube.
        cube_component = np.sqrt(values["cube_static"])
        cube_dyn_component = np.sqrt(values["cube_dynamic"])
        cube_material = torch.as_tensor(
            np.column_stack((cube_component, cube_dyn_component, np.zeros(len(env_ids)))), dtype=torch.float32
        )
        table_material = torch.as_tensor(
            np.column_stack(
                (
                    values["table_static"] / cube_component,
                    values["table_dynamic"] / cube_dyn_component,
                    np.minimum(1.0, 2.0 * values["table_restitution"]),
                )
            ),
            dtype=torch.float32,
        )
        finger_material = torch.as_tensor(
            np.column_stack(
                (
                    values["finger_static"] / cube_component,
                    values["finger_dynamic"] / cube_dyn_component,
                    np.zeros(len(env_ids)),
                )
            ),
            dtype=torch.float32,
        )
        for cube in self.cubes.values():
            self._set_materials(cube, env_ids_cpu, cube_material)
        self._set_materials(self.work_surface, env_ids_cpu, table_material)
        self._set_finger_materials(env_ids_cpu, finger_material)

        # Exact measured identity masses and uniform-cube inertia at 50.7 mm.
        size = 0.0507
        for name, cube in self.cubes.items():
            masses = cube.root_physx_view.get_masses()
            if masses.ndim == 1:
                masses[env_ids_cpu] = CUBE_MASSES_KG[name]
            else:
                masses[env_ids_cpu, 0] = CUBE_MASSES_KG[name]
            cube.root_physx_view.set_masses(masses, env_ids_cpu)
            inertias = cube.root_physx_view.get_inertias()
            moment = CUBE_MASSES_KG[name] * size * size / 6.0
            if inertias.ndim == 2:
                inertias[env_ids_cpu] = 0.0
                inertias[env_ids_cpu, 0] = moment
                inertias[env_ids_cpu, 4] = moment
                inertias[env_ids_cpu, 8] = moment
            else:
                inertias[env_ids_cpu, 0] = 0.0
                inertias[env_ids_cpu, 0, 0] = moment
                inertias[env_ids_cpu, 0, 4] = moment
                inertias[env_ids_cpu, 0, 8] = moment
            cube.root_physx_view.set_inertias(inertias, env_ids_cpu)

        # Map the rigid D405 payload to the receiving asset's hand-TCP body.
        # The bundle provides mass and CoM but no inertia, so preserve the
        # receiving inertia shape and scale it by mass.
        payload_body = "fr3_hand_tcp" if "fr3_hand_tcp" in self.robot.body_names else "fr3_hand"
        payload_id = list(self.robot.body_names).index(payload_body)
        masses = self.robot.root_physx_view.get_masses()
        old_payload_mass = masses[env_ids_cpu, payload_id].clone()
        masses[env_ids_cpu, payload_id] = 0.0946
        self.robot.root_physx_view.set_masses(masses, env_ids_cpu)
        inertias = self.robot.root_physx_view.get_inertias()
        ratio = (0.0946 / torch.clamp(old_payload_mass, min=1e-6)).unsqueeze(-1)
        inertias[env_ids_cpu, payload_id] *= ratio
        self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
        coms = self.robot.root_physx_view.get_coms()
        coms[env_ids_cpu, payload_id, :3] = torch.tensor(
            [0.0695086838, -0.0630348763, -0.0714033328], dtype=coms.dtype
        )
        self.robot.root_physx_view.set_coms(coms, env_ids_cpu)

        # Gripper force scale is represented by its implicit position actuator
        # stiffness. Speed scale is not explicitly modeled by this task.
        gripper = self.robot.actuators["gripper"]
        force_scale = dev("force_scale").unsqueeze(-1)
        base_stiffness = self.robot.data.default_joint_stiffness[env_ids[:, None], self.finger_joint_ids]
        stiffness = base_stiffness * force_scale
        gripper.stiffness[env_ids] = stiffness
        self.robot.write_joint_stiffness_to_sim(stiffness, joint_ids=self.finger_joint_ids, env_ids=env_ids)

        # Damping is authored at nominal in the receiving RigidBodyCfg because
        # the Isaac 5.1 tensor API has no per-environment damping setter.
        self.applied_summary = {
            "profile": self.profile,
            "seed": seed,
            "num_envs": len(env_ids),
            "posterior_near_fraction": float(np.mean(values["posterior_near"])),
            "cube_mass_mapping_kg": CUBE_MASSES_KG,
            "cube_size_m": size,
            "payload_body": payload_body,
            "payload_previous_mass_mean_kg": float(old_payload_mass.float().mean()),
            "payload_mass_kg": 0.0946,
            "joint_delay_mean_steps": float(np.mean(values["delay"])),
            "cube_cube_static_effective_mean": float(np.mean(values["cube_static"])),
            "table_cube_static_effective_mean": float(np.mean(values["table_static"])),
            "finger_cube_static_effective_mean": float(np.mean(values["finger_static"])),
            "cube_linear_damping_authored_nominal": 0.0682659557,
            "cube_angular_damping_authored_nominal": 0.0766774104,
            "unmodeled": ["per_env_cube_damping", "gripper_speed_scale"],
        }
        env.fr3_cube_calibration_summary = self.applied_summary
        if log_samples:
            log_dir = Path(getattr(env.cfg, "log_dir", "."))
            log_dir.mkdir(parents=True, exist_ok=True)
            sample_path = log_dir / f"calibrated_sysid_samples_seed{seed}.npz"
            np.savez_compressed(sample_path, **values)
            self.applied_summary["sample_log"] = str(sample_path)
        carb.log_warn("[fr3_cube_calibrated_sysid] " + json.dumps(self.applied_summary, sort_keys=True))
