"""SE(3) / quaternion helpers (numpy only).

Pose convention: 7-vector [px, py, pz, qx, qy, qz, qw] (position + xyzw quaternion).
Kept dependency-light on purpose so the pure-python core runs without pinocchio/scipy.
"""
from __future__ import annotations

import numpy as np

Pose = np.ndarray  # shape (7,): [x,y,z, qx,qy,qz,qw]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def random_rotation_quat(max_angle: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random rotation about a random axis with |angle| <= max_angle (SART-style)."""
    rng = rng or np.random.default_rng()
    axis = rng.standard_normal(3)
    angle = rng.uniform(-max_angle, max_angle)
    return axis_angle_to_quat(axis, angle)


def rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Robust rotation-matrix -> xyzw quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quat(np.array([x, y, z, w]))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # shortest path
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # near-parallel -> lerp
        return normalize_quat(q0 + t * (q1 - q0))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    q2 = normalize_quat(q1 - q0 * dot)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def mat_from_pose(pose: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotmat_from_quat(pose[3:7])
    T[:3, 3] = pose[:3]
    return T


def pose_from_mat(T: np.ndarray) -> Pose:
    return np.concatenate([T[:3, 3], quat_from_rotmat(T[:3, :3])])


def pose_compose(a: Pose, b: Pose) -> Pose:
    """Return pose of (T_a * T_b)."""
    return pose_from_mat(mat_from_pose(a) @ mat_from_pose(b))


def pose_inv(pose: Pose) -> Pose:
    return pose_from_mat(np.linalg.inv(mat_from_pose(pose)))


def pose_interp(p0: Pose, p1: Pose, t: float) -> Pose:
    t = float(np.clip(t, 0.0, 1.0))
    pos = (1 - t) * p0[:3] + t * p1[:3]
    quat = quat_slerp(p0[3:7], p1[3:7], t)
    return np.concatenate([pos, quat])


def sample_points_in_sphere(
    center: np.ndarray,
    radius: float,
    num: int,
    surface: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample `num` points on (surface=True) or inside (surface=False) a sphere.

    Faithful to RoboManipAug's sample_points_on_sphere; `surface=False` reproduces
    the `--sample_inside_sphere` radius*U^(1/3) volumetric fill.
    """
    rng = rng or np.random.default_rng()
    phi = rng.uniform(0, 2 * np.pi, num)
    cos_theta = rng.uniform(-1, 1, num)
    theta = np.arccos(cos_theta)
    dirs = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
    )
    if surface:
        r = radius
    else:
        r = radius * rng.random(num) ** (1.0 / 3.0)
        r = r[:, None]
    return dirs * r + np.asarray(center)[None, :]


def pos_error(p0: Pose, p1: Pose) -> float:
    return float(np.linalg.norm(p0[:3] - p1[:3]))
