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

Third fix (cube orientation normalization): the RL scene's cube_2 asset
frame is modeled 90 deg rotated (recorded R[2,2] = 0.0 while the cube sits
flat on the table; cube_1/cube_3 and all human cubes read 1.0). A cube is
visually symmetric so nothing looks wrong, but MimicGen composes the
subtask transform as new_obj_pose o inv(src_obj_pose) — the spurious 90 deg
rotates the whole grasp segment about the cube (generated hand tilts of
70-130 deg vs the plan's 19-33 deg, 5 cm hover, 0/3117 .. 0/306 across five
runs). Since any of a cube's 24 symmetric orientations is physically
equivalent, we project every object rotation to its nearest yaw-only
rotation (yaw from the most-horizontal column, mod 90 deg).

Pure h5py/numpy — runs on the host:
  python3 resignal_rl.py <ann.hdf5> <out_srcOK.hdf5> [min_gap=4]
Writes only demos passing the annotation gate (all signals fire, ordered,
onset gaps >= min_gap) and prints per-demo onsets + grasp-moment geometry.
"""
from __future__ import annotations

import math
import sys

import h5py
import numpy as np


def vertical_hand_track(mats):
    """Project (T,4,4) hand poses to yaw-preserving vertical top-grasp.

    Residual orientation error under IK tracking of the RL wrist dynamics
    (achieved tilt 39-83 deg vs the plan's 19-33) displaces the TCP by
    ~10.34 cm x sin(err) = 4-7 cm — exactly the observed miss floor. Cubes
    are symmetric, so a vertical grasp at the same TCP position is
    physically equivalent and matches what the (working) human demos do.
    """
    out = mats.copy()
    for t in range(len(mats)):
        x = mats[t, :3, 0].astype(np.float64)
        xh = np.array([x[0], x[1], 0.0])
        n = np.linalg.norm(xh)
        xh = xh / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        z = np.array([0.0, 0.0, -1.0])
        y = np.cross(z, xh)
        out[t, :3, :3] = np.column_stack([xh, y, z]).astype(mats.dtype)
    return out


def yaw_only_track(mats):
    """Project (T,4,4) object poses to yaw-only rotations (cube symmetry)."""
    out = mats.copy()
    for t in range(len(mats)):
        R = mats[t, :3, :3]
        for c in (0, 1, 2):
            if abs(R[2, c]) < 0.7:
                theta = math.atan2(R[1, c], R[0, c]) % (math.pi / 2)
                break
        else:
            theta = 0.0
        cs, sn = math.cos(theta), math.sin(theta)
        out[t, :3, :3] = np.array([[cs, -sn, 0.0], [sn, cs, 0.0],
                                   [0.0, 0.0, 1.0]], dtype=mats.dtype)
    return out

LIFT_M = 0.01
HOLD_LO, HOLD_HI = 0.018, 0.033
NEAR_M = 0.08
OPEN_M = 0.036       # fingers wider than this = cube released
DWELL = 20           # frames (20 Hz, pre-densify) frozen at each grasp pose
CLOSE_AT = 12        # close the gripper this many frames INTO the dwell:
                     # the approach lag is still 3-6 cm at dwell entry, and
                     # fingers seal in ~0.3 s — closing at dwell start loses
                     # the race and the shut fist bulldozes the cube (v10)
HEAD = 20            # frames frozen at the start pose: the RL start pose is
                     # far from the env home, so the generator's short
                     # interpolation leaves a 30-50 cm tracking deficit that
                     # persists as schedule lag through the whole approach
                     # (v6: closest pass reached only at episode end)
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

        # ---- dwell insertion: freeze the grasp waypoint for DWELL frames so
        # the generator closes the gripper while STATIONARY at the cube. The
        # RL close-on-the-fly has zero timing margin under open-loop replay
        # (v4: fingers sealed 5-45 cm short of the cube, min approach 4.6 cm).
        d1 = max(1, o1 - 2)
        d2r = max(o2 + 1, o3 - 2)
        # head-trim: RL demos start with the arm fully extended overhead (all
        # joints ~0) — a differential-IK singularity. FK ground truth showed
        # the generated robot thrashing 0.5-1 m from every cube while chasing
        # waypoints through that region (the recorded eef obs tracks the
        # TARGET, which is why numeric probes looked "close"). Cut the prefix
        # and start at the approach entry, low and near the first cube.
        descend = ((ee[:, 2] - cp[0, 1, 2] < 0.30)
                   & (np.linalg.norm(ee - cp[0, 1], axis=1) < 0.45))
        cand = np.flatnonzero(descend[:max(1, d1 - 4)])
        t_trim = int(cand[0]) if len(cand) else 0
        idx = [t_trim] * (HEAD + 1)
        for t in range(t_trim + 1, T):
            idx.append(t)
            if t == d1 or t == d2r:
                idx.extend([t] * DWELL)
        idx = np.asarray(idx)
        newT = len(idx)
        ins1 = HEAD + (d1 - t_trim) + 1     # first inserted row of dwell 1
        ins2 = HEAD + (d2r - t_trim) + DWELL + 1  # first row of dwell 2
        dwelled = np.zeros(newT, dtype=bool)
        dwelled[1:HEAD + 1] = True
        dwelled[ins1:ins1 + DWELL] = True
        dwelled[ins2:ins2 + DWELL] = True

        ep = data.create_group(name)
        for key, value in g.attrs.items():
            ep.attrs[key] = value
        ep.attrs["num_samples"] = newT
        ep.attrs["dwell_frames"] = DWELL
        ep.attrs["head_trim"] = t_trim
        src.copy(g["initial_state"], ep, name="initial_state")
        # reset from the LAB home config (bent elbow, well-conditioned), not
        # the RL candle pose: differential IK started at the singularity
        # never recovers. The generator's interpolation bridges home -> the
        # (now low) first waypoint.
        jp = ep["initial_state/articulation/robot/joint_position"]
        jp[...] = np.array([[0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741,
                             0.04, 0.04]], dtype=np.float32)
        jv = ep["initial_state/articulation/robot/joint_velocity"]
        jv[...] = np.zeros((1, 9), dtype=np.float32)

        obs_in, obs = g["obs"], ep.create_group("obs")
        for key in obs_in:
            if key == "datagen_info":
                continue
            obs.create_dataset(key, data=obs_in[key][()][idx])
        info_in = obs_in["datagen_info"]
        info = obs.create_group("datagen_info")
        eef44 = vertical_hand_track(info_in["eef_pose/franka"][()][idx])
        # lower each grasp dwell to the RESTING cube height: the dwell frame
        # (lift onset - 2) has the source cube already ~1 cm off the table,
        # so relative to a resting cube the waypoint sits 2-2.5 cm high —
        # v9 executed grasps closed at a constant z +3.4 cm with xy < 1 cm.
        eef44[ins1 - 1:ins1 + DWELL, 2, 3] = cp[0, 1, 2] + 0.005
        eef44[ins2 - 1:ins2 + DWELL, 2, 3] = cp[0, 2, 2] + 0.005
        info.create_dataset("eef_pose/franka", data=eef44)
        info.create_dataset("target_eef_pose/franka",
                            data=np.concatenate([eef44[1:], eef44[-1:]]))
        for c in (1, 2, 3):
            info.create_dataset(
                f"object_pose/cube_{c}",
                data=yaw_only_track(info_in[f"object_pose/cube_{c}"][()][idx]))
        for key, arr in (("grasp_1", grasp_1), ("stack_1", stack_1),
                         ("grasp_2", grasp_2)):
            info.create_dataset(f"subtask_term_signals/{key}", data=arr[idx])

        if "states" in g:
            st = ep.create_group("states")

            def walk(gin, gout):
                for key in gin:
                    if isinstance(gin[key], h5py.Group):
                        walk(gin[key], gout.create_group(key))
                    else:
                        arr = gin[key][()]
                        sidx = np.minimum(idx, len(arr) - 1)
                        full = np.concatenate([arr[sidx], arr[-1:]]) \
                            if len(arr) == T + 1 else arr[sidx]
                        gout.create_dataset(key, data=full)
            walk(g["states"], st)

        # gripper schedule on the NEW timeline: approach open, close during
        # the stationary dwell, hold through transport, open at the observed
        # release frame
        fingers_n = np.abs(obs["gripper_pos"][()]).mean(axis=1)

        def first_after(t0, mask):
            hits = np.flatnonzero(mask & (np.arange(newT) > t0))
            return int(hits[0]) if len(hits) else -1

        o1n = o1 - t_trim + HEAD + DWELL
        o3n = o3 - t_trim + HEAD + 2 * DWELL
        release_1 = first_after(o1n, fingers_n > OPEN_M)
        if release_1 < 0 or release_1 >= o3n:
            release_1 = max(o1n + 1, o2 - t_trim + HEAD + DWELL - 2)
        release_2 = first_after(o3n, fingers_n > OPEN_M)
        if release_2 < 0:
            release_2 = newT - FINAL_OPEN_TAIL
        sched = np.ones(newT, dtype=np.float32)
        sched[ins1 + CLOSE_AT:release_1] = -1.0
        sched[ins2 + CLOSE_AT:release_2] = -1.0
        acts = g["actions"][()][np.minimum(idx, len(g["actions"]) - 1)]
        acts[dwelled[:len(acts)], :6] = 0.0
        acts[:, 6] = sched[:len(acts)]
        ep.create_dataset("actions", data=acts)
        print(f"  dwell@{d1}/{d2r} grip: close1 [{ins1 + 1},{release_1}) "
              f"close2 [{ins2 + 1},{release_2}) T {T}->{newT}")
        kept += 1
        total += newT

    data.attrs["total"] = total
    src.close()
    dst.close()
    print(f"kept {kept} dropped {len(dropped)}: {dropped}")
    print("wrote", dst_path)


if __name__ == "__main__":
    main()
