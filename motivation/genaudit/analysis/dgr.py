"""DGR analyses over AttemptRecords (PLAN.md §1.4).

Everything operates on the extracted records — no hdf5 access — so E1 figures
and the definition-robustness check run anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from genaudit.curation.binning import FrozenBinEdges, assign_bins, compute_quantile_edges
from genaudit.records.schema import AttemptRecord, distance_value


def dgr(records: Sequence[AttemptRecord]) -> float:
    if not records:
        raise ValueError("no records")
    return sum(record.success for record in records) / len(records)


@dataclass(frozen=True)
class DgrCurve:
    distance_key: str
    bin_centers: tuple[float, ...]  # median attempted distance per bin
    per_bin_dgr: tuple[float, ...]
    per_bin_attempts: tuple[int, ...]
    edges: FrozenBinEdges


def dgr_vs_distance(
    records: Sequence[AttemptRecord],
    distance_key: str = "d_pos",
    k: int = 5,
    edges: FrozenBinEdges | None = None,
) -> DgrCurve:
    values = np.array([distance_value(record, distance_key) for record in records])
    successes = np.array([record.success for record in records], dtype=bool)
    if edges is None:
        edges = compute_quantile_edges(
            values, k, metadata={"distance_key": distance_key, "n": len(records)}
        )
    bins = assign_bins(values, edges)
    centers, rates, counts = [], [], []
    for bin_index in range(edges.k):
        mask = bins == bin_index
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"bin {bin_index} empty under provided edges")
        centers.append(float(np.median(values[mask])))
        rates.append(float(successes[mask].mean()))
        counts.append(count)
    return DgrCurve(
        distance_key=distance_key,
        bin_centers=tuple(centers),
        per_bin_dgr=tuple(rates),
        per_bin_attempts=tuple(counts),
        edges=edges,
    )


@dataclass(frozen=True)
class TrendStats:
    distance_key: str
    point_biserial_r: float  # Pearson r between success (0/1) and distance
    spearman_rho: float
    per_bin_dgr_monotone_decreasing: bool


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # average ties
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    return sums[inverse] / counts[inverse]


def trend_stats(
    records: Sequence[AttemptRecord], distance_key: str = "d_pos", k: int = 5
) -> TrendStats:
    values = np.array([distance_value(record, distance_key) for record in records])
    successes = np.array([record.success for record in records], dtype=float)
    if successes.std() == 0 or values.std() == 0:
        raise ValueError("degenerate pool: success or distance has zero variance")
    point_biserial = float(np.corrcoef(values, successes)[0, 1])
    spearman = float(np.corrcoef(_rank(values), _rank(successes))[0, 1])
    curve = dgr_vs_distance(records, distance_key, k)
    monotone = all(
        later <= earlier + 1e-12
        for earlier, later in zip(curve.per_bin_dgr, curve.per_bin_dgr[1:])
    )
    return TrendStats(
        distance_key=distance_key,
        point_biserial_r=point_biserial,
        spearman_rho=spearman,
        per_bin_dgr_monotone_decreasing=monotone,
    )


def definition_comparison(
    records: Sequence[AttemptRecord], keys: tuple[str, ...] = ("d_raw", "d_pos"), k: int = 5
) -> dict[str, TrendStats]:
    """The §1.3 trend-preservation check: does normalization weaken the trend?"""
    return {key: trend_stats(records, key, k) for key in keys}


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial rate (PLAN.md §2.4)."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} outside [0, {trials}]")
    p = successes / trials
    denom = 1 + z * z / trials
    center = p + z * z / (2 * trials)
    margin = z * ((p * (1 - p) + z * z / (4 * trials)) / trials) ** 0.5
    return max(0.0, (center - margin) / denom)


@dataclass(frozen=True)
class PoolSizePlan:
    distance_key: str
    k: int
    per_bin_retained: tuple[int, ...]
    per_bin_attempts: tuple[int, ...]
    per_bin_dgr: tuple[float, ...]
    scarcest_bin: int
    scarcest_dgr_wilson_lb: float
    planned_total_attempts: int  # ceil(target_retained / wilson_lb), 0 if infeasible


def plan_pool_size(
    records: Sequence[AttemptRecord],
    distance_key: str = "d_pos",
    k: int = 5,
    target_retained_total: int = 500,
    z: float = 1.96,
) -> PoolSizePlan:
    """Size the E2 pool from a pilot pool (Phase-0's 500-attempt runs).

    N = target / WilsonLB(p_min): the scarcest bin's rate lower bound drives
    the whole pool (PLAN.md §2.4 — the K's cancel). A zero-survivor scarcest
    bin makes the plan infeasible at this K (planned_total_attempts = 0):
    switch to K-1 or truncate the top bin, per the pre-registered fallback.
    """
    curve = dgr_vs_distance(records, distance_key, k)
    values = np.array([distance_value(record, distance_key) for record in records])
    successes = np.array([record.success for record in records], dtype=bool)
    bins = assign_bins(values, curve.edges)
    retained = tuple(int(successes[bins == index].sum()) for index in range(k))
    attempts = curve.per_bin_attempts
    lbs = [wilson_lower_bound(retained[i], attempts[i], z) for i in range(k)]
    scarcest = int(np.argmin(lbs))
    lb = lbs[scarcest]
    planned = int(np.ceil(target_retained_total / lb)) if lb > 0 else 0
    return PoolSizePlan(
        distance_key=distance_key,
        k=k,
        per_bin_retained=retained,
        per_bin_attempts=attempts,
        per_bin_dgr=curve.per_bin_dgr,
        scarcest_bin=scarcest,
        scarcest_dgr_wilson_lb=lb,
        planned_total_attempts=planned,
    )
