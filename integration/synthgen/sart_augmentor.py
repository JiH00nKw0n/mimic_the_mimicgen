"""SART (Self-Augmented Robot Trajectory) reimplemented on our SimEnv/IK/collision stack.

Original: robo_manip_aug/common/CollectAugmentedDataBase.py (bound to RoboManipBaselines +
MuJoCo + pinocchio). Here the SAME algorithm runs on ANY SimEnv/IKSolver/CollisionChecker
— i.e. Isaac Lab + cuRobo. Core is unchanged:
    sample EEF pose in an acceptable region (sphere) around a precision waypoint
    -> converge to the waypoint -> IK + (cuRobo) collision-check -> replay -> keep.

Upgrade over the original: collision safety comes from an online CollisionChecker
(cuRobo world w/ hand-camera point cloud) instead of only the hand-annotated sphere, and
episodes pass through the pipeline's task-success filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .interfaces import CollisionChecker, DataWriter, IKSolver, SimEnv
from .math_utils import (
    Pose,
    normalize_quat,
    pose_interp,
    quat_mul,
    random_rotation_quat,
    sample_points_in_sphere,
)
from .runtime import execute_trajectory
from .skills import Demo, SkillSegment


@dataclass
class SartConfig:
    # 각 acceptable-region 구 당 샘플 수 (RoboManipAug --num_sphere_sample)
    num_sphere_sample: int = 16
    # 전체 목표 샘플 수 (설정 시 region 수로 나눠 num_sphere_sample 재계산)
    num_total_sample: Optional[int] = None
    # 주석 반지름 대신 고정 반지름 [m] (--overwrite_radius). None 이면 segment/annotation 값
    radius: Optional[float] = None
    default_radius: float = 0.02
    # 구 내부에서 샘플 (--sample_inside_sphere): 다양성 ↑
    sample_inside: bool = True
    # 위치 고정, 회전만 (--position_fix): 기어 축 정합에 유리
    fix_position: bool = False
    # 회전 랜덤 각 [deg]. None 이면 rotation_random_scale * radius 사용 (원본 규약)
    rotation_random_angle_deg: Optional[float] = None
    rotation_random_scale: float = 2.0
    # 수렴 궤적 보간 스텝 수 (--interp_duration 를 스텝으로 이산화)
    interp_steps: int = 20
    seed: int = 0


class SartAugmentor:
    def __init__(
        self,
        env: SimEnv,
        ik: IKSolver,
        collision: Optional[CollisionChecker],
        writer: DataWriter,
        cfg: SartConfig = field(default_factory=SartConfig),
    ):
        self.env = env
        self.ik = ik
        self.collision = collision
        self.writer = writer
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _regions(self, demo: Demo) -> List[SkillSegment]:
        """Insert/align segments define the acceptable regions (precision waypoints)."""
        return [s for s in demo.segments if s.is_insert]

    def _region_radius(self, seg: SkillSegment, demo: Demo) -> float:
        if self.cfg.radius is not None:
            return self.cfg.radius
        # per-segment annotated radius, else demo meta, else default
        r = demo.meta.get(f"radius::{seg.name}")
        return float(r) if r is not None else self.cfg.default_radius

    def augment(self, demo: Demo, world_idx: Optional[int] = None) -> int:
        """Generate local precision-augmented episodes. Returns #successful written."""
        regions = self._regions(demo)
        if not regions:
            return 0

        n_per = self.cfg.num_sphere_sample
        if self.cfg.num_total_sample is not None:
            n_per = int(np.ceil(self.cfg.num_total_sample / len(regions)))

        widx = world_idx if world_idx is not None else demo.world_idx
        n_success = 0
        for r_idx, seg in enumerate(regions):
            conv = demo.waypoints[seg.t_end]        # convergence (precision) waypoint
            center_pos = conv.eef_pose[:3]
            center_quat = conv.eef_pose[3:7]
            radius = self._region_radius(seg, demo)

            sample_pos = sample_points_in_sphere(
                center_pos, radius, n_per, surface=not self.cfg.sample_inside, rng=self.rng
            )
            max_angle = (
                np.deg2rad(self.cfg.rotation_random_angle_deg)
                if self.cfg.rotation_random_angle_deg is not None
                else self.cfg.rotation_random_scale * radius
            )

            for s_idx in range(n_per):
                pos = center_pos if self.cfg.fix_position else sample_pos[s_idx]
                rot = quat_mul(random_rotation_quat(max_angle, self.rng), center_quat)
                start_pose: Pose = np.concatenate([pos, normalize_quat(rot)])
                # converging trajectory: from the sampled offset back INTO the demo waypoint
                poses = [
                    pose_interp(start_pose, conv.eef_pose, t)
                    for t in np.linspace(0.0, 1.0, self.cfg.interp_steps)
                ]
                grippers = [conv.gripper] * len(poses)
                meta = {
                    "source": "sart",
                    "region": seg.name,
                    "region_idx": r_idx,
                    "sample_idx": s_idx,
                    "world_idx": widx,
                    "seed": self.cfg.seed,
                }
                if execute_trajectory(
                    self.env, self.ik, self.collision, self.writer,
                    poses, grippers, meta, world_idx=widx,
                ):
                    n_success += 1
        return n_success
