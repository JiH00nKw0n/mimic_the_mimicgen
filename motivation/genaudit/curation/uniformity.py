"""Uniformity certification over bins (PLAN.md §2.2).

Metric: total-variation distance to the uniform histogram —
TV = 0.5 * sum_k |n_k/N - 1/K| — readable as "the fraction of demonstrations
that would have to change bins to reach exact uniformity". A dataset is
certified transform-uniform iff TV <= tv_threshold AND every bin holds at
least min_bin_fraction of its quota (the side constraint forbids hiding one
starved bin inside an acceptable TV).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def tv_distance_to_uniform(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or len(counts) < 2:
        raise ValueError(f"counts must be 1-D with >= 2 bins, got shape {counts.shape}")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    total = counts.sum()
    if total == 0:
        raise ValueError("empty histogram")
    return float(0.5 * np.abs(counts / total - 1.0 / len(counts)).sum())


@dataclass(frozen=True)
class CertificationReport:
    counts: tuple[int, ...]
    total: int
    quota: int
    tv: float
    tv_threshold: float
    min_count: int
    min_count_required: int
    passed: bool

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] TV={self.tv:.4f} (<= {self.tv_threshold}), "
            f"min bin={self.min_count} (>= {self.min_count_required}), "
            f"counts={list(self.counts)}"
        )


def certify_uniform(
    counts: np.ndarray,
    quota: int,
    tv_threshold: float = 0.02,
    min_bin_fraction: float = 0.9,
) -> CertificationReport:
    counts = np.asarray(counts, dtype=int)
    tv = tv_distance_to_uniform(counts)
    min_count = int(counts.min())
    min_required = int(np.ceil(min_bin_fraction * quota))
    # Epsilon absorbs float noise so the designed boundary case (exactly
    # tv_threshold, e.g. a 10-demo deficit at N=500) certifies.
    passed = tv <= tv_threshold + 1e-9 and min_count >= min_required
    return CertificationReport(
        counts=tuple(int(count) for count in counts),
        total=int(counts.sum()),
        quota=quota,
        tv=tv,
        tv_threshold=tv_threshold,
        min_count=min_count,
        min_count_required=min_required,
        passed=passed,
    )
