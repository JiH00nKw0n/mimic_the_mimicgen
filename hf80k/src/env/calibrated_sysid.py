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
import warp as wp
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
    """표준 접촉 CSV. 빈 칸은 "이 번들이 재지 않았다"는 뜻이라 건너뛴다."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            out = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    out[key] = float(value)
                except ValueError:
                    continue          # sample_id 같은 글자 열
            rows.append(out)
        return rows


def _read_params(path: Path) -> dict:
    """번들의 표준 파라미터 파일.

    예전에는 대표값과 범위가 이 파일에 숫자로 적혀 있었다. 그 숫자들은 전부 큐브 번들의
    계약 파일에서 베껴 온 것이라, 다른 번들이 다른 값을 줘도 코드가 무시했다. 이제
    tools/normalize_physics_bundle.py가 번들에서 뽑아 만든 parameters.json만 읽는다.
    """
    if not path.is_file():
        raise RuntimeError(
            f"물리 파라미터 파일이 없다: {path}\n"
            f"받은 번들을 tools/normalize_physics_bundle.py로 표준 배치로 옮겨야 한다.")
    with path.open(encoding="utf-8") as stream:
        doc = json.load(stream)
    for section in ("contact", "sampling"):
        if section not in doc:
            raise RuntimeError(f"{path}에 {section} 항목이 없다")
    return doc


def _draw(rng, spec: dict, count: int) -> np.ndarray:
    """파라미터 파일이 적어 둔 분포에서 뽑는다."""
    kind = spec.get("dist", "uniform")
    if kind == "triangular":
        return rng.triangular(spec["low"], spec["mode"], spec["high"], count)
    return rng.uniform(spec["low"], spec["high"], count)


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
        # 장면 물체를 이름이 아니라 역할로 받는다. 큐브 태스크에서는 주된 물체가 큐브 셋과
        # 책상이고, peg 태스크에서는 핀 하나와 데스크다. 예전에는 cube_1부터 cube_3을
        # 코드가 직접 찾아서, 그 이름이 없는 장면에서는 시작하자마자 죽었다.
        self.cubes: dict[str, RigidObject] = {
            name: env.scene[name]
            for name in cfg.params.get("object_names", cfg.params.get("cube_names",
                                                                      tuple(CUBE_MASSES_KG)))
        }
        surface_name = cfg.params.get("surface_name",
                                      cfg.params.get("work_surface_name", "work_surface"))
        self.work_surface: RigidObject = env.scene[surface_name]
        self.arm_joint_ids = self.robot.find_joints(list(JOINT_NAMES))[0]
        self.finger_joint_ids = self.robot.find_joints(["fr3_finger_joint.*"])[0]
        self.arm_actuator_name = str(cfg.params.get("arm_actuator_name", "arm"))
        self._dynamics_rows = _read_csv(
            self.bundle_root / "modules/dynamics_controller/domain_randomization_samples.csv"
        )
        self._contact_rows = _read_csv(self.bundle_root / "modules/contact/posterior_samples.csv")
        self._params = _read_params(self.bundle_root / "parameters.json")
        self._nominal = self._params["contact"].get("nominal", {})
        self._range = self._params["contact"].get("range", {})
        self._constraint = self._params["contact"].get("constraint", {})
        self._sampling = self._params.get("sampling", {})
        self._joint_nominal = (self._params.get("joint") or {}).get("nominal", {})
        # 물체 질량. 번들이 물체별 질량을 주면 그것을 쓰고, 아니면 표준 항목 하나를
        # 모든 주된 물체에 같이 쓴다. 큐브 번들은 물체별로 주므로 예전과 같은 값이 된다.
        self._object_masses = dict(cfg.params.get("object_masses", CUBE_MASSES_KG))
        shared_mass = self._nominal.get("object_mass_kg")
        if shared_mass is not None:
            for name in self.cubes:
                self._object_masses.setdefault(name, float(shared_mass))
        self.applied_summary: dict[str, object] = {}

    @staticmethod
    def _warp_indices(env_ids_cpu: torch.Tensor):
        """PhysX 뷰가 받는 유일한 색인 형태로 바꾼다.

        이 판본에서 set_material_properties는 파이토치 텐서를 색인으로 받으면
        "issubclass() arg 1 must be a class"로 죽는다. warp int32 배열만 받는다.
        """
        return wp.from_torch(env_ids_cpu.to(torch.int32).contiguous(), dtype=wp.int32)

    @staticmethod
    def _get_as_torch(getter):
        """PhysX 뷰의 조회 결과를 수정 가능한 파이토치 텐서로 바꾼다.

        warp 배열이 오면 사본을 뜨고, 이미 파이토치 텐서면 그대로 쓴다. 판본에 따라
        둘 중 하나가 오므로 양쪽을 다 받는다.
        """
        out = getter()
        if isinstance(out, torch.Tensor):
            return out
        return wp.to_torch(out).clone()

    @staticmethod
    def _put(setter, tensor: torch.Tensor, env_ids_cpu: torch.Tensor) -> None:
        """수정한 텐서를 PhysX 뷰에 되돌려 넣는다. 색인은 warp int32여야 한다."""
        setter(wp.from_torch(tensor.contiguous(), dtype=wp.float32),
               apply_fr3_cube_calibration_bundle._warp_indices(env_ids_cpu))

    @staticmethod
    def _materials_as_torch(view):
        """재질 배열을 수정 가능한 파이토치 텐서로 가져온다.

        get_material_properties()는 warp 배열을 돌려준다. warp 배열은 여러 색인으로
        값을 넣는 것이 아예 안 되고, wp.to_torch가 만든 텐서는 메모리를 공유하지 않는
        사본이라 거기에 써도 시뮬레이터에는 반영되지 않는다. 실측으로 확인했다.
        그래서 사본에서 고친 뒤 set으로 되돌려 넣는 방식만 동작한다.
        """
        return apply_fr3_cube_calibration_bundle._get_as_torch(view.get_material_properties)

    @staticmethod
    def _set_materials(asset: RigidObject | Articulation, env_ids_cpu: torch.Tensor, values: torch.Tensor) -> None:
        view = asset.root_physx_view
        materials = apply_fr3_cube_calibration_bundle._materials_as_torch(view)
        max_shapes = int(view.max_shapes)
        materials[env_ids_cpu] = values[:, None, :].expand(-1, max_shapes, -1).to(materials.dtype)
        apply_fr3_cube_calibration_bundle._put(
            view.set_material_properties, materials, env_ids_cpu)

    def _set_finger_materials(self, env_ids_cpu: torch.Tensor, values: torch.Tensor) -> None:
        materials = self._materials_as_torch(self.robot.root_physx_view)
        shape_counts: list[int] = []
        for link_path in self.robot.root_physx_view.link_paths[0]:
            view = self.robot._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
            shape_counts.append(int(view.max_shapes))   # 위와 같은 이유로 int()
        body_names = list(self.robot.body_names)
        for finger_name in ("fr3_leftfinger", "fr3_rightfinger"):
            body_id = body_names.index(finger_name)
            start = sum(shape_counts[:body_id])
            stop = start + shape_counts[body_id]
            materials[env_ids_cpu, start:stop] = values[:, None, :].to(materials.dtype)
        self._put(self.robot.root_physx_view.set_material_properties,
                  materials, env_ids_cpu)

    def _sample(self, count: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        nom, joint_nom = self._nominal, self._joint_nominal
        if self.profile == "nominal":
            armature = np.tile(joint_nom["armature"], (count, 1))
            static = np.tile(joint_nom["static_friction"], (count, 1))
            dynamic = np.tile(joint_nom["dynamic_friction"], (count, 1))
            viscous = np.tile(joint_nom["viscous_friction"], (count, 1))
            delay = np.full(count, int(joint_nom["motor_delay_steps"]), dtype=np.int64)
            table_static = np.full(count, nom["surface_static_friction"])
            table_dynamic = np.full(count, nom["surface_dynamic_friction"])
            table_restitution = np.full(count, nom["surface_restitution"])
            cube_static = np.full(count, nom["pair_primary_static_friction"])
            cube_dynamic = np.full(count, nom["pair_primary_dynamic_friction"])
            linear_damping = np.full(count, nom["object_linear_damping"])
            angular_damping = np.full(count, nom["object_angular_damping"])
            finger_static = np.full(count, nom["finger_static_friction"])
            finger_dynamic = np.full(count, nom["finger_dynamic_friction"])
            force_scale = np.full(count, nom["gripper_force_scale"])
            posterior_near = np.ones(count, dtype=bool)
        else:
            dyn_ids = rng.integers(0, len(self._dynamics_rows), count)
            dyn = [self._dynamics_rows[index] for index in dyn_ids]
            armature = np.asarray([[row[f"armature_j{i}"] for i in range(1, 8)] for row in dyn])
            static = np.asarray([[row[f"static_friction_j{i}"] for i in range(1, 8)] for row in dyn])
            dynamic = np.asarray([[row[f"dynamic_friction_j{i}"] for i in range(1, 8)] for row in dyn])
            viscous = np.asarray([[row[f"viscous_friction_j{i}"] for i in range(1, 8)] for row in dyn])
            delay = np.asarray([int(row["motor_delay_steps_120hz"]) for row in dyn], dtype=np.int64)

            near_fraction = float(self._sampling.get("posterior_near_fraction", 0.8))
            posterior_near = (
                np.ones(count, dtype=bool)
                if self.profile == "posterior_stochastic"
                else rng.random(count) < near_fraction
            )
            post_ids = rng.integers(0, len(self._contact_rows), count)
            post = [self._contact_rows[index] for index in post_ids]
            rng_range = self._range

            def column(name: str, standard: str) -> np.ndarray:
                """표본 행에서 한 열을 꺼낸다. 그 번들이 재지 않은 열이면 대표값으로 채운다."""
                if all(name in row for row in post):
                    return np.asarray([row[name] for row in post], dtype=float)
                return np.full(count, float(nom[standard]))

            table_restitution = column("surface_restitution", "surface_restitution")
            cube_static = column("pair_primary_static_friction", "pair_primary_static_friction")
            cube_ratio = column("pair_primary_dynamic_ratio", "pair_primary_dynamic_ratio")
            linear_damping = column("object_linear_damping", "object_linear_damping")
            angular_damping = column("object_angular_damping", "object_angular_damping")
            full = ~posterior_near
            n_full = int(full.sum())
            table_restitution[full] = _draw(rng, rng_range["surface_restitution"], n_full)
            cube_static[full] = _draw(rng, rng_range["pair_primary_static_friction"], n_full)
            cube_ratio[full] = _draw(rng, rng_range["pair_primary_dynamic_ratio"], n_full)
            linear_damping[full] = _draw(rng, rng_range["object_linear_damping"], n_full)
            angular_damping[full] = _draw(rng, rng_range["object_angular_damping"], n_full)
            cube_dynamic = cube_static * cube_ratio

            table_static = _draw(rng, rng_range["surface_static_friction"], count)
            table_dynamic = _draw(rng, rng_range["surface_dynamic_friction"], count)
            table_dynamic = np.minimum(table_dynamic, table_static)
            force_scale = _draw(rng, rng_range["gripper_force_scale"], count)
            finger_static = _draw(rng, rng_range["finger_static_friction"], count)
            # 그리퍼가 물체를 놓치지 않을 조건. 번들이 부등식으로 적어 둔 하한이다.
            product_min = self._constraint.get("finger_force_product_min")
            if product_min is not None:
                invalid = finger_static * force_scale < product_min
                while invalid.any():
                    n_bad = int(invalid.sum())
                    finger_static[invalid] = _draw(rng, rng_range["finger_static_friction"], n_bad)
                    force_scale[invalid] = _draw(rng, rng_range["gripper_force_scale"], n_bad)
                    invalid = finger_static * force_scale < product_min
            finger_dynamic = finger_static * _draw(rng, rng_range["finger_dynamic_ratio"], count)

            # PhysX supports at most 64K unique materials.  Quantize contact
            # tuples to a shared 256-bucket ensemble while retaining per-env
            # joint dynamics/delay samples.
            num_buckets = min(int(self._sampling.get("material_buckets", 256)), count)
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
            masses = self._get_as_torch(cube.root_physx_view.get_masses)
            if masses.ndim == 1:
                masses[env_ids_cpu] = CUBE_MASSES_KG[name]
            else:
                masses[env_ids_cpu, 0] = CUBE_MASSES_KG[name]
            self._put(cube.root_physx_view.set_masses, masses, env_ids_cpu)
            inertias = self._get_as_torch(cube.root_physx_view.get_inertias)
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
            self._put(cube.root_physx_view.set_inertias, inertias, env_ids_cpu)

        # Map the rigid D405 payload to the receiving asset's hand-TCP body.
        # The bundle provides mass and CoM but no inertia, so preserve the
        # receiving inertia shape and scale it by mass.
        payload_body = "fr3_hand_tcp" if "fr3_hand_tcp" in self.robot.body_names else "fr3_hand"
        payload_id = list(self.robot.body_names).index(payload_body)
        masses = self._get_as_torch(self.robot.root_physx_view.get_masses)
        old_payload_mass = masses[env_ids_cpu, payload_id].clone()
        masses[env_ids_cpu, payload_id] = 0.0946
        self._put(self.robot.root_physx_view.set_masses, masses, env_ids_cpu)
        inertias = self._get_as_torch(self.robot.root_physx_view.get_inertias)
        ratio = (0.0946 / torch.clamp(old_payload_mass, min=1e-6)).unsqueeze(-1)
        inertias[env_ids_cpu, payload_id] *= ratio
        self._put(self.robot.root_physx_view.set_inertias, inertias, env_ids_cpu)
        coms = self._get_as_torch(self.robot.root_physx_view.get_coms)
        coms[env_ids_cpu, payload_id, :3] = torch.tensor(
            [0.0695086838, -0.0630348763, -0.0714033328], dtype=coms.dtype
        )
        self._put(self.robot.root_physx_view.set_coms, coms, env_ids_cpu)

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
            # env.cfg.log_dir은 None인 경우가 많다. Path(None)은 TypeError로 죽으므로
            # 작업 디렉터리 아래를 기본값으로 쓴다. 뽑힌 물리값을 남기는 진단용 경로다.
            log_dir = Path(getattr(env.cfg, "log_dir", None)
                           or os.environ.get("WORK_DIR", ".")) / "sysid_samples"
            log_dir.mkdir(parents=True, exist_ok=True)
            sample_path = log_dir / f"calibrated_sysid_samples_seed{seed}.npz"
            np.savez_compressed(sample_path, **values)
            self.applied_summary["sample_log"] = str(sample_path)
        carb.log_warn("[fr3_cube_calibrated_sysid] " + json.dumps(self.applied_summary, sort_keys=True))
