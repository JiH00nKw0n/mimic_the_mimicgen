"""Stratified evaluation statistics (PLAN.md §2.6).

Pure computations over per-episode results: per-bin success rates along the
d_eval axis (quantile bins of the eval set's own distances), the slope
statistic (SR regressed on bin index — a flatter slope means the policy holds
up in the far bins), and the far-bin gap used as the primary endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genaudit.curation.binning import FrozenBinEdges, assign_bins, compute_quantile_edges


@dataclass(frozen=True)
class StratifiedSuccess:
    k: int
    per_bin_success_rate: tuple[float, ...]
    per_bin_count: tuple[int, ...]
    aggregate_success_rate: float
    slope_per_bin: float  # least-squares slope of SR on bin index
    far_bins_success_rate: float  # mean SR of the two farthest bins


def stratify_success(
    d_eval: np.ndarray,
    successes: np.ndarray,
    k: int = 5,
    edges: FrozenBinEdges | None = None,
) -> StratifiedSuccess:
    """Per-bin success along d_eval; bins are the eval set's own quantiles
    unless frozen edges are supplied (paired comparisons must pass the same
    edges for every arm)."""
    d_eval = np.asarray(d_eval, dtype=float)
    successes = np.asarray(successes, dtype=bool)
    if d_eval.shape != successes.shape:
        raise ValueError(f"shape mismatch: {d_eval.shape} vs {successes.shape}")
    if edges is None:
        edges = compute_quantile_edges(d_eval, k)
    elif edges.k != k:
        raise ValueError(f"edges.k={edges.k} != k={k}")
    bins = assign_bins(d_eval, edges)
    rates = []
    counts = []
    for bin_index in range(k):
        mask = bins == bin_index
        count = int(mask.sum())
        if count == 0:
            raise ValueError(
                f"bin {bin_index} is empty — frozen edges do not match this eval set"
            )
        rates.append(float(successes[mask].mean()))
        counts.append(count)
    slope = float(np.polyfit(np.arange(k), np.asarray(rates), deg=1)[0])
    return StratifiedSuccess(
        k=k,
        per_bin_success_rate=tuple(rates),
        per_bin_count=tuple(counts),
        aggregate_success_rate=float(successes.mean()),
        slope_per_bin=slope,
        far_bins_success_rate=float(np.mean(rates[-2:])),
    )


def paired_success_difference(
    successes_a: np.ndarray, successes_b: np.ndarray
) -> dict[str, float]:
    """McNemar-style paired summary over a shared frozen eval set."""
    a = np.asarray(successes_a, dtype=bool)
    b = np.asarray(successes_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in shape: {a.shape} vs {b.shape}")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    return {
        "n": int(len(a)),
        "sr_a": float(a.mean()),
        "sr_b": float(b.mean()),
        "wins_a_only": only_a,
        "wins_b_only": only_b,
        "discordant": only_a + only_b,
    }
