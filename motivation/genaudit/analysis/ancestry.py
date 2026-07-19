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
    top3_share_attempted: float
    top3_share_retained: float
    skew_pp: float  # (retained top-3 share - attempted top-3 share) * 100
    n_eff_retained: float  # 1 / sum q_i^2 over retained shares


def _top_k_share(counts: np.ndarray, k: int = 3) -> float:
    total = counts.sum()
    if total == 0:
        raise ValueError("empty counts")
    return float(np.sort(counts)[::-1][:k].sum() / total)


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
    top3_attempted = _top_k_share(attempted)
    top3_retained = _top_k_share(retained)
    return AncestryStats(
        num_sources=num_sources,
        attempted_counts=tuple(int(count) for count in attempted),
        retained_counts=tuple(int(count) for count in retained),
        per_source_success_rate=tuple(float(rate) for rate in success_rates),
        top3_share_attempted=top3_attempted,
        top3_share_retained=top3_retained,
        skew_pp=(top3_retained - top3_attempted) * 100.0,
        n_eff_retained=n_eff,
    )


def arm_ancestry_report(
    records: Sequence[AttemptRecord], selected_ids: Sequence[str], num_sources: int
) -> AncestryStats:
    """Mandatory per-arm ancestry reporting (PLAN.md §2.3): stats of a subset."""
    by_id = {record.attempt_id: record for record in records}
    subset = [by_id[attempt_id] for attempt_id in selected_ids]
    return ancestry_stats(subset, num_sources)
