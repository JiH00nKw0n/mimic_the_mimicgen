"""SART local precision augmentation — runnable on mocks (no Isaac Lab/cuRobo needed).

Real run: swap MockEnv->IsaacLabEnv, MockIK->CuroboIK, MockCollision->CuroboCollision,
InMemoryDataWriter->HDF5DataWriter. The SartAugmentor code does not change.

    python examples/run_sart_augment.py
"""
from synthgen import SartAugmentor, SartConfig
from synthgen.mocks import (
    InMemoryDataWriter,
    MockCollision,
    MockEnv,
    MockIK,
    make_toy_demo,
)


def main():
    demo = make_toy_demo()
    conv = demo.waypoints[-1].eef_pose  # insert convergence pose = success goal

    env = MockEnv(goal_pose=conv, success_pos_tol=0.02)
    writer = InMemoryDataWriter()
    aug = SartAugmentor(
        env=env, ik=MockIK(), collision=MockCollision(), writer=writer,
        cfg=SartConfig(num_sphere_sample=12, radius=0.01, sample_inside=True, seed=0),
    )

    n = aug.augment(demo)
    print(f"[SART] insert regions augmented -> {n} successful episodes "
          f"({writer.num_success}/{len(writer.episodes)} kept)")
    assert writer.episodes, "SART produced no episodes"


if __name__ == "__main__":
    main()
