"""Per-task quantile binning with frozen, auditable edges (PLAN.md §2.1).

The rule is identical for every task — K equal-attempt-mass quantile bins of
the attempted distance distribution — while the edge values are computed per
task-variant pool, once, then frozen to a JSON artifact that every later step
(sampling, certification, figures) must load instead of recomputing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrozenBinEdges:
    k: int
    interior_edges: tuple[float, ...]  # k-1 strictly increasing values
    n_fit: int  # pool size the edges were computed from
    metadata: dict

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")
        if len(self.interior_edges) != self.k - 1:
            raise ValueError(
                f"expected {self.k - 1} interior edges, got {len(self.interior_edges)}"
            )
        edges = np.asarray(self.interior_edges)
        if not np.all(np.diff(edges) > 0):
            raise ValueError(
                "interior edges are not strictly increasing — the attempted "
                f"distribution has heavy ties; inspect the pool: {self.interior_edges}"
            )


def compute_quantile_edges(
    values: np.ndarray, k: int, metadata: dict | None = None
) -> FrozenBinEdges:
    """Equal-attempt-mass edges from the attempted distribution of one pool."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")
    if len(values) < 10 * k:
        raise ValueError(
            f"pool too small to freeze {k} quantile bins: n={len(values)} < {10 * k}"
        )
    if np.any(~np.isfinite(values)):
        raise ValueError("values contain NaN/inf")
    quantiles = np.arange(1, k) / k
    interior = np.quantile(values, quantiles, method="linear")
    return FrozenBinEdges(
        k=k,
        interior_edges=tuple(float(edge) for edge in interior),
        n_fit=len(values),
        metadata=dict(metadata or {}),
    )


def assign_bins(values: np.ndarray, edges: FrozenBinEdges) -> np.ndarray:
    """Map values to bin indices 0..k-1 (outermost bins are open-ended)."""
    values = np.asarray(values, dtype=float)
    return np.searchsorted(np.asarray(edges.interior_edges), values, side="right")


def save_edges(edges: FrozenBinEdges, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(edges), indent=2, sort_keys=True))


def load_edges(path: str | Path) -> FrozenBinEdges:
    payload = json.loads(Path(path).read_text())
    payload["interior_edges"] = tuple(payload["interior_edges"])
    return FrozenBinEdges(**payload)
