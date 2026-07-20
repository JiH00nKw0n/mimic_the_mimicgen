"""Ancestry-skew analyses over AttemptRecords (Phase-0's dominant bias)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from genaudit.records.schema import AttemptRecord


@dataclass(frozen=True)
class AncestryStats:
    num_sources: int
    attempted_counts: tuple[int, ...]
    retained_counts: tuple[int, ...]
    per_source_success_rate: tuple[float, ...]
    # Primary skew metric: total-variation distance from the uniform ancestry
    # distribution — parameter-free (no arbitrary "top-k"), bounded in
    # [0, 1 - 1/N], read as "the fraction of demos that would have to change
    # source to make the ancestry uniform". Reported for attempts and
    # survivors; skew_tv isolates the filtering-induced increase.
    tv_uniform_attempted: float
    tv_uniform_retained: float
    skew_tv: float  # TV_retained - TV_attempted
    n_eff_retained: float  # inverse-Simpson effective number of ancestors
    # kept for continuity with the earlier report / cross-checking only
    top3_share_attempted: float
    top3_share_retained: float
    skew_pp: float


def _top_k_share(counts: np.ndarray, k: int = 3) -> float:
    total = counts.sum()
    if total == 0:
        raise ValueError("empty counts")
    return float(np.sort(counts)[::-1][:k].sum() / total)


def _tv_to_uniform(counts: np.ndarray) -> float:
    """Total-variation distance of a count distribution from uniform."""
    total = counts.sum()
    if total == 0:
        raise ValueError("empty counts")
    shares = counts / total
    n = len(counts)
    return float(0.5 * np.abs(shares - 1.0 / n).sum())


def ancestry_stats(records: Sequence[AttemptRecord], num_sources: int) -> AncestryStats:
    attempted = np.zeros(num_sources, dtype=int)
    retained = np.zeros(num_sources, dtype=int)
    for record in records:
        if not 0 <= record.source_demo_id < num_sources:
            raise IndexError(
                f"source_demo_id {record.source_demo_id} out of range ({num_sources})"
            )
        attempted[record.source_demo_id] += 1
        if record.success:
            retained[record.source_demo_id] += 1
    if retained.sum() == 0:
        raise ValueError("no retained attempts — ancestry of survivors undefined")
    success_rates = np.divide(
        retained, attempted, out=np.zeros(num_sources), where=attempted > 0
    )
    shares = retained / retained.sum()
    n_eff = 1.0 / float(np.square(shares).sum())
    tv_att = _tv_to_uniform(attempted)
    tv_ret = _tv_to_uniform(retained)
    top3_attempted = _top_k_share(attempted)
    top3_retained = _top_k_share(retained)
    return AncestryStats(
        num_sources=num_sources,
        attempted_counts=tuple(int(count) for count in attempted),
        retained_counts=tuple(int(count) for count in retained),
        per_source_success_rate=tuple(float(rate) for rate in success_rates),
        tv_uniform_attempted=tv_att,
        tv_uniform_retained=tv_ret,
        skew_tv=tv_ret - tv_att,
        n_eff_retained=n_eff,
        top3_share_attempted=top3_attempted,
        top3_share_retained=top3_retained,
        skew_pp=(top3_retained - top3_attempted) * 100.0,
    )


def arm_ancestry_report(
    records: Sequence[AttemptRecord], selected_ids: Sequence[str], num_sources: int
) -> AncestryStats:
    """Mandatory per-arm ancestry reporting (PLAN.md §2.3): stats of a subset."""
    by_id = {record.attempt_id: record for record in records}
    subset = [by_id[attempt_id] for attempt_id in selected_ids]
    return ancestry_stats(subset, num_sources)
