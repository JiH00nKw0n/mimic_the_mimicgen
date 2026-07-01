"""Smoke tests for the pure-python core (mocks only; no Isaac Lab/cuRobo).

Run:  python -m pytest -q   (or)   python tests/test_smoke.py
"""
import numpy as np

from synthgen import (
    CpGenConfig,
    Demo,
    GeometrySample,
    KeypointTrajectoryTransform,
    PipelineConfig,
    SartAugmentor,
    SartConfig,
    SkillGenPipeline,
    rigid_object_transform,
)
from synthgen.math_utils import (
    mat_from_pose,
    pose_from_mat,
    quat_slerp,
    sample_points_in_sphere,
)
from synthgen.mocks import (
    InMemoryDataWriter,
    MockCollision,
    MockEnv,
    MockIK,
    MockPlanner,
    make_toy_demo,
    toy_scene_sampler,
)


def test_math_roundtrip():
    p = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(pose_from_mat(mat_from_pose(p)), p, atol=1e-6)


def test_slerp_endpoints():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(0.5), np.cos(0.5)])
    assert np.allclose(quat_slerp(q0, q1, 0.0), q0, atol=1e-6)
    assert np.allclose(np.abs(quat_slerp(q0, q1, 1.0)), np.abs(q1), atol=1e-6)


def test_sphere_sampling_radius():
    c = np.array([1.0, 2.0, 3.0])
    pts = sample_points_in_sphere(c, 0.05, 200, surface=True)
    d = np.linalg.norm(pts - c, axis=1)
    assert np.allclose(d, 0.05, atol=1e-6)
    inside = sample_points_in_sphere(c, 0.05, 200, surface=False)
    assert np.all(np.linalg.norm(inside - c, axis=1) <= 0.05 + 1e-9)


def test_rigid_transform_identity():
    eef = np.array([0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    obj = np.array([0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    out = rigid_object_transform(eef, obj, obj)  # same obj pose -> unchanged
    assert np.allclose(out, eef, atol=1e-6)


def test_keypoint_transform_translates_with_object():
    tf = KeypointTrajectoryTransform(CpGenConfig(apply_geometry=False))
    demo = make_toy_demo()
    seg = [s for s in demo.segments if s.is_insert][0]
    new_poses = dict(demo.object_poses)
    shifted = demo.object_poses["socket"].copy()
    shifted[0] += 0.1  # move socket +10cm in x
    new_poses["socket"] = shifted
    out = tf.transform_segment(demo, seg, new_poses, GeometrySample.identity())
    src = [w.eef_pose for w in demo.segment_waypoints(seg)]
    # EEF should follow the object by ~+0.1 in x
    assert np.allclose(out[-1][0] - src[-1][0], 0.1, atol=1e-6)


def test_sart_augment_runs():
    demo = make_toy_demo()
    conv = demo.waypoints[-1].eef_pose
    env = MockEnv(goal_pose=conv, success_pos_tol=0.05)
    writer = InMemoryDataWriter()
    aug = SartAugmentor(env, MockIK(), MockCollision(), writer,
                        SartConfig(num_sphere_sample=8, radius=0.01, seed=1))
    n = aug.augment(demo)
    assert len(writer.episodes) == 8
    assert n >= 1  # converging trajectories reach the goal


def test_pipeline_generates():
    demo = make_toy_demo()
    env = MockEnv(goal_pose=None)  # success detector stand-in
    writer = InMemoryDataWriter()
    pipe = SkillGenPipeline(
        env, MockIK(), writer, planner=MockPlanner(), collision=MockCollision(),
        transform=KeypointTrajectoryTransform(CpGenConfig(seed=0)),
        sart=SartAugmentor(env, MockIK(), MockCollision(), writer,
                           SartConfig(num_sphere_sample=3, radius=0.01)),
        cfg=PipelineConfig(transform_mode="keypoint", num_success=3, sart_boost=True),
    )
    total = pipe.generate(demo, toy_scene_sampler)
    assert total >= 3
    assert writer.num_success >= 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} smoke tests passed.")
