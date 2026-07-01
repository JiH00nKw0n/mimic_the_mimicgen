"""CP-Gen keypoint-transform generation + optional SART boost — runnable on mocks.

Demonstrates the unified pipeline: object-centric + geometry transform (CP-Gen mode) for
new scenes, replay + success filter, then SART local variants on the insert skill.

    python examples/run_cpgen_generate.py
"""
from synthgen import (
    CpGenConfig,
    KeypointTrajectoryTransform,
    PipelineConfig,
    SartAugmentor,
    SartConfig,
    SkillGenPipeline,
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


def main():
    demo = make_toy_demo()

    # goal_pose=None -> MockEnv reports success every attempt (stands in for the real
    # task-success detector); the point here is to exercise transform+replay+SART wiring.
    env = MockEnv(goal_pose=None)
    writer = InMemoryDataWriter()

    sart = SartAugmentor(
        env=env, ik=MockIK(), collision=MockCollision(), writer=writer,
        cfg=SartConfig(num_sphere_sample=5, radius=0.01, seed=0),
    )
    pipe = SkillGenPipeline(
        env=env, ik=MockIK(), writer=writer,
        planner=MockPlanner(), collision=MockCollision(),
        transform=KeypointTrajectoryTransform(CpGenConfig(scale_range=(0.85, 1.15), seed=0)),
        sart=sart,
        cfg=PipelineConfig(transform_mode="keypoint", num_success=5, sart_boost=True, seed=0),
    )

    total = pipe.generate(demo, toy_scene_sampler)
    print(f"[CP-Gen+SART] total successful episodes written: {total} "
          f"(base scenes + SART boost); writer holds {len(writer.episodes)} episodes")
    assert total > 0, "pipeline produced nothing"


if __name__ == "__main__":
    main()
