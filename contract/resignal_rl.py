"""Re-annotate RL-source subtask signals with lift-based grasp predicates.

Why: the original grasp predicate (fingers < 0.03 AND TCP within 10 cm)
fires mid-swoop on dynamic RL grasps — at signal onset the TCP is still
3.8 cm off the cube center in xy, 2.6 cm above, moving at ~19 cm/s, and the
fingers (20 mm mean) are narrower than a held 50.7 mm cube allows, i.e. the
cube is NOT yet in hand (human sources at onset: 1.1 cm xy, stationary,
holding). MimicGen transforms the next subtask segment relative to the NEXT
object, so a boundary placed before the hold is secure rigidly displaces
the actual grasp-completion frames — every generated attempt closes a few
cm off the cube and never lifts it (c2 lift 0.0 cm across all sampled
failures, 0/3117 and 0/226 densified).

Fix: a grasp subtask ends only when the cube demonstrably follows the hand:
cube lifted > 1 cm from its resting height, fingers inside the physically
consistent hold window [18, 33] mm, TCP within 8 cm. stack_1 keeps the
verified stacked-predicate (identity run: 10/137 vs human 10/122).

Second fix (gripper schedule): the RL policy pumps the gripper — the action
sign flips ~10 times per 60 steps and the finger state itself oscillates
40->6->40 mm during approach. The demos succeed because the policy is
closed-loop and catches the cube on a lucky cycle; replayed open-loop by the
generator the pump lands off-schedule and every attempt closes empty (min
finger width 0.0 mm across all sampled failures). We therefore REPLACE the
actions' gripper channel with a clean pick-place square wave derived from
the recomputed signals: close CLOSE_RAMP frames before each grasp onset,
open at the actual release frame observed in the finger state.

Pure h5py/numpy — runs on the host:
  python3 resignal_rl.py <ann.hdf5> <out_srcOK.hdf5> [min_gap=4]
Writes only demos passing the annotation gate (all signals fire, ordered,
onset gaps >= min_gap) and prints per-demo onsets + grasp-moment geometry.
"""
from __future__ import annotations

import sys

import h5py
import numpy as np

LIFT_M = 0.01
HOLD_LO, HOLD_HI = 0.018, 0.033
NEAR_M = 0.08
OPEN_M = 0.036       # fingers wider than this = cube released
CLOSE_RAMP = 6       # frames (20 Hz) to command close before a grasp onset
FINAL_OPEN_TAIL = 12  # fallback: open this many last frames if no release seen


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    min_gap = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    src = h5py.File(src_path, "r")
    dst = h5py.File(dst_path, "w")
    data = dst.create_group("data")
    for key, value in src["data"].attrs.items():
        data.attrs[key] = value

    kept, dropped, total = 0, [], 0
    for name in src["data"].keys():
        g = src[f"data/{name}"]
        ee = g["obs/eef_pos"][()]
        cp = g["obs/cube_positions"][()].reshape(-1, 3, 3)
        fingers = np.abs(g["obs/gripper_pos"][()]).mean(axis=1)
        T = len(ee)

        hold = (fingers > HOLD_LO) & (fingers < HOLD_HI)
        released = fingers > 0.03

        def near(c):
            return np.linalg.norm(ee - cp[:, c], axis=1) < NEAR_M

        def lifted(c):
            return (cp[:, c, 2] - cp[0, c, 2]) > LIFT_M

        grasp_1 = lifted(1) & hold & near(1)
        stack_1 = ((np.linalg.norm(cp[:, 1, :2] - cp[:, 0, :2], axis=1) < 0.04)
                   & (cp[:, 1, 2] - cp[:, 0, 2] > 0.035)
                   & (cp[:, 1, 2] - cp[:, 0, 2] < 0.065)
                   & released)
        grasp_2 = (lifted(2) & hold & near(2)
                   & stack_1.cumsum().astype(bool))

        def onset(mask):
            hits = np.flatnonzero(mask)
            return int(hits[0]) if len(hits) else -1

        o1, o2, o3 = onset(grasp_1), onset(stack_1), onset(grasp_2)
        ok = (0 <= o1 < o2 < o3
              and o2 - o1 >= min_gap and o3 - o2 >= min_gap
              and T - 1 - o3 >= min_gap)
        if o1 >= 0:
            d = ee[o1] - cp[o1, 1]
            speed = np.linalg.norm(ee[min(o1 + 1, T - 1)] - ee[o1])
            geom = (f"xy={np.linalg.norm(d[:2]) * 100:.1f}cm "
                    f"z={d[2] * 100:.1f}cm fingers={fingers[o1] * 1000:.0f}mm "
                    f"v={speed * 1000:.1f}mm/f")
        else:
            geom = "no grasp_1"
        print(f"{name}: g1={o1} s1={o2} g2={o3} T={T} "
              f"{'OK' if ok else 'DROP'} | {geom}")
        if not ok:
            dropped.append(name)
            continue

        src.copy(g, data, name=name)
        ep = data[name]
        sig_group = ep["obs/datagen_info/subtask_term_signals"]
        for key, arr in (("grasp_1", grasp_1), ("stack_1", stack_1),
                         ("grasp_2", grasp_2)):
            del sig_group[key]
            sig_group.create_dataset(key, data=arr)

        # clean pick-place gripper schedule replacing the pumped channel
        def first_after(t0, mask):
            hits = np.flatnonzero(mask & (np.arange(T) > t0))
            return int(hits[0]) if len(hits) else -1

        release_1 = first_after(o1, fingers > OPEN_M)
        if release_1 < 0 or release_1 >= o3:
            release_1 = max(o1 + 1, o2 - 2)
        release_2 = first_after(o3, fingers > OPEN_M)
        if release_2 < 0:
            release_2 = T - FINAL_OPEN_TAIL
        sched = np.ones(T, dtype=np.float32)
        sched[max(0, o1 - CLOSE_RAMP):release_1] = -1.0
        sched[max(release_1 + 2, o3 - CLOSE_RAMP):release_2] = -1.0
        acts = ep["actions"][()]
        acts[:, 6] = sched[:len(acts)]
        del ep["actions"]
        ep.create_dataset("actions", data=acts)
        print(f"  grip: close1 [{max(0, o1 - CLOSE_RAMP)},{release_1}) "
              f"close2 [{max(release_1 + 2, o3 - CLOSE_RAMP)},{release_2})")
        kept += 1
        total += int(ep.attrs["num_samples"])

    data.attrs["total"] = total
    src.close()
    dst.close()
    print(f"kept {kept} dropped {len(dropped)}: {dropped}")
    print("wrote", dst_path)


if __name__ == "__main__":
    main()
