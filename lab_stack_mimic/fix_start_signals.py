#!/usr/bin/env python3
"""Rewrite SkillGen subtask START signals from the TERM signals, with guaranteed valid ordering.

Why: SkillGen's data generator (datagen_info_pool.py) splits every subtask into a planned
free-space TRANSITION + a replayed SKILL using the start signal, and asserts, across the whole
source pool, that boundaries never overlap even at the largest offsets:

    term_transition_i + 1  <  start_transition_{i+1}  <=  term_transition_{i+1}

A geometric start signal (EE within a distance of the next object) cannot guarantee this: when
two cubes spawn close together, the "near next object" condition is already true at the previous
grasp, so start_{i+1} fires at/before term_i and the assert fails for that demo (one bad demo
kills the whole run).

Robust fix: derive the start signal deterministically from the term signals. For subtask i we put
its start at a fraction `frac` of the way from the previous subtask's termination to this one's,
clamped into the strictly-valid open interval. That makes the SKILL the last (1-frac) of each
subtask (the contact-rich part: final approach + grasp / lower + release) and the TRANSITION the
earlier free-space part that cuRobo plans. Ordering is valid by construction for every demo.

    python fix_start_signals.py <in.hdf5> <out.hdf5> [frac=0.6]
"""

import shutil
import sys

import h5py
import numpy as np

SIGNALS = ["grasp_1", "stack_1", "grasp_2", "stack_2"]


def first_rising_edge(arr: np.ndarray) -> int:
    """First index where arr goes 0 -> 1 becomes 1 (matches the generator's transition index)."""
    a = arr.flatten().astype(np.int64)
    diffs = a[1:] - a[:-1]
    nz = np.nonzero(diffs)[0]
    if len(nz) == 0:
        return -1
    return int(nz[0]) + 1


def main():
    src, dst = sys.argv[1], sys.argv[2]
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    shutil.copyfile(src, dst)

    with h5py.File(dst, "a") as f:
        demos = list(f["data"].keys())
        n_fixed = 0
        for d in demos:
            g = f["data"][d]
            term_grp = g["obs"]["datagen_info"]["subtask_term_signals"]
            start_grp = g["obs"]["datagen_info"]["subtask_start_signals"]
            T = term_grp[SIGNALS[0]].shape[0]

            # term transition step for each subtask (where its term signal first becomes 1)
            term_t = [first_rising_edge(term_grp[s][:]) for s in SIGNALS]
            # demos that didn't fully terminate every subtask shouldn't be here (annotate dropped
            # them), but guard anyway.
            if any(t < 0 for t in term_t):
                continue

            prev = -1  # "previous subtask termination" sentinel for subtask 0
            for i, s in enumerate(SIGNALS):
                ti = term_t[i]
                # valid open interval for this start transition: (prev+1, ti]
                lo = max(prev + 2, 1)          # strictly after previous end, and >=1 (need a 0 before)
                hi = ti                         # at the latest, coincides with this term
                base = prev if prev >= 0 else 0
                s_step = base + int(round(frac * (ti - base)))
                s_step = max(lo, min(s_step, hi))
                # build the step signal (match the existing dataset's shape: (T,) or (T,1)):
                # 0 before s_step, 1 from s_step on.
                ds = start_grp[s]
                sig = np.zeros(ds.shape, dtype=ds.dtype)
                sig[s_step:] = 1
                ds[...] = sig
                prev = ti
            n_fixed += 1
        print(f"[fix_start_signals] rewrote start signals for {n_fixed}/{len(demos)} demos (frac={frac})")


if __name__ == "__main__":
    main()
