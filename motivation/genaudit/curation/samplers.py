"""Arm construction — post-hoc subsampling of one frozen attempt pool.

Every E2 arm is drawn from the SAME pool so the generator, source demos, and
simulator state are identical between arms; the only difference is the
sampling rule implemented here (PLAN.md §2.3). All sampling is driven by a
caller-provided seeded Generator so arms are reproducible from the config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genaudit.curation.uniformity import CertificationReport, certify_uniform


def _effective_count(counts: np.ndarray) -> float:
    """Inverse-Simpson effective number of ancestors (n_eff)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    shares = counts / total
    return float(1.0 / np.square(shares).sum())


class InsufficientPoolError(RuntimeError):
    """The pool cannot satisfy the requested arm within certification limits."""


@dataclass(frozen=True)
class SampleResult:
    arm: str
    selected: tuple[int, ...]  # indices into the caller's candidate array
    per_stratum_counts: tuple[int, ...]
    certification: CertificationReport | None
    info: dict = field(default_factory=dict)  # extras (excluded sources, n_eff)

    @property
    def size(self) -> int:
        return len(self.selected)


def sample_baseline(
    num_candidates: int, size: int, rng: np.random.Generator
) -> SampleResult:
    """Uniform random subset of the retained pool — the standard-pipeline arm."""
    if num_candidates < size:
        raise InsufficientPoolError(
            f"baseline arm needs {size} retained demos, pool has {num_candidates}"
        )
    selected = rng.choice(num_candidates, size=size, replace=False)
    return SampleResult(
        arm="baseline",
        selected=tuple(int(i) for i in np.sort(selected)),
        per_stratum_counts=(size,),
        certification=None,
    )


def sample_stratified_uniform(
    strata_labels: np.ndarray,
    num_strata: int,
    size: int,
    rng: np.random.Generator,
    arm: str = "transform_uniform",
    tv_threshold: float = 0.02,
    min_bin_fraction: float = 0.9,
) -> SampleResult:
    """Equal quota per stratum, uniform at random within a stratum.

    Used for both the transform-uniform arm (strata = frozen distance bins)
    and the ancestry-balanced arm (strata = source demo ids). If a stratum
    cannot fill its quota, the deficit is redistributed to the strata with the
    most remaining candidates; the final histogram must still pass
    certification, otherwise the pool is declared insufficient (loudly).
    """
    strata_labels = np.asarray(strata_labels)
    if strata_labels.ndim != 1:
        raise ValueError(f"strata_labels must be 1-D, got shape {strata_labels.shape}")
    if size % num_strata != 0:
        raise ValueError(
            f"arm size {size} is not divisible by {num_strata} strata; "
            "choose size and K so the quota is exact (PLAN.md §2.3)"
        )
    quota = size // num_strata

    members = [np.flatnonzero(strata_labels == stratum) for stratum in range(num_strata)]
    picked: list[np.ndarray] = []
    counts = np.zeros(num_strata, dtype=int)
    remaining: list[np.ndarray] = []
    for stratum, candidates in enumerate(members):
        take = min(quota, len(candidates))
        chosen = rng.choice(candidates, size=take, replace=False)
        picked.append(chosen)
        counts[stratum] = take
        leftover = np.setdiff1d(candidates, chosen, assume_unique=True)
        remaining.append(leftover)

    deficit = size - int(counts.sum())
    while deficit > 0:
        donor = int(np.argmax([len(r) for r in remaining]))
        if len(remaining[donor]) == 0:
            raise InsufficientPoolError(
                f"{arm}: pool exhausted with {deficit} demos still missing; "
                f"per-stratum available={[len(m) for m in members]}"
            )
        index = rng.integers(len(remaining[donor]))
        chosen = remaining[donor][index]
        remaining[donor] = np.delete(remaining[donor], index)
        picked.append(np.array([chosen]))
        counts[donor] += 1
        deficit -= 1

    certification = certify_uniform(
        counts, quota=quota, tv_threshold=tv_threshold, min_bin_fraction=min_bin_fraction
    )
    if not certification.passed:
        shortfalls = {
            stratum: quota - len(members[stratum])
            for stratum in range(num_strata)
            if len(members[stratum]) < quota
        }
        raise InsufficientPoolError(
            f"{arm}: certification failed ({certification.summary()}); "
            f"per-stratum quota shortfalls: {shortfalls}. "
            "Grow the pool (PLAN.md §2.4) or fall back to K-1 bins."
        )
    selected = np.sort(np.concatenate(picked))
    return SampleResult(
        arm=arm,
        selected=tuple(int(i) for i in selected),
        per_stratum_counts=tuple(int(count) for count in counts),
        certification=certification,
    )


def sample_ancestry_balanced(
    source_ids: np.ndarray,
    num_sources: int,
    size: int,
    rng: np.random.Generator,
    arm: str = "ancestry_balanced",
) -> SampleResult:
    """Ancestry-balanced arm with the pre-registered source-exclusion fallback
    (PLAN.md §2.4).

    Equal quota per KEPT source. When a source cannot fill the equal quota
    (too few survivors), it is excluded and the quota is recomputed over the
    remaining sources — water-filling until every kept source can meet the
    quota — then the leftover (size - quota*n_kept) is distributed +1 to the
    kept sources with the most spare. This trades exact ancestry uniformity
    for feasibility exactly where the survivor pool is thin, and reports the
    excluded sources and achieved effective-ancestor count instead of gating
    on the transform-axis TV threshold (which does not apply to this axis).
    """
    source_ids = np.asarray(source_ids)
    available = np.array(
        [int(np.sum(source_ids == s)) for s in range(num_sources)], dtype=int
    )
    kept = [s for s in range(num_sources) if available[s] > 0]
    if not kept:
        raise InsufficientPoolError(f"{arm}: no retained demos in any source")

    # water-fill: drop the most-starved source until an equal quota fits
    while True:
        quota = size // len(kept)
        if quota == 0:
            raise InsufficientPoolError(
                f"{arm}: {len(kept)} usable sources cannot supply {size} demos"
            )
        starved = [s for s in kept if available[s] < quota]
        if not starved:
            break
        worst = min(kept, key=lambda s: available[s])
        kept.remove(worst)

    counts = {s: quota for s in kept}
    remainder = size - quota * len(kept)
    # give the remainder to kept sources with the most spare capacity
    spare = sorted(kept, key=lambda s: available[s] - counts[s], reverse=True)
    for s in spare[:remainder]:
        counts[s] += 1

    picked = []
    for s in kept:
        members = np.flatnonzero(source_ids == s)
        picked.append(rng.choice(members, size=counts[s], replace=False))
    selected = np.sort(np.concatenate(picked))

    excluded = sorted(set(range(num_sources)) - set(kept))
    final_counts = np.array([counts.get(s, 0) for s in range(num_sources)])
    return SampleResult(
        arm=arm,
        selected=tuple(int(i) for i in selected),
        per_stratum_counts=tuple(int(final_counts[s]) for s in range(num_sources)),
        certification=None,
        info={
            "kept_sources": kept,
            "excluded_sources": excluded,
            "quota_per_kept_source": quota,
            "n_eff": _effective_count(final_counts),
            "available_per_source": [int(v) for v in available],
        },
    )
