import numpy as np
import pytest

from genaudit.curation.binning import (
    FrozenBinEdges,
    assign_bins,
    compute_quantile_edges,
    load_edges,
    save_edges,
)


def test_equal_attempt_mass_and_determinism():
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=0.1, size=5000)
    edges_a = compute_quantile_edges(values, k=5)
    edges_b = compute_quantile_edges(values, k=5)
    assert edges_a == edges_b  # deterministic

    bins = assign_bins(values, edges_a)
    counts = np.bincount(bins, minlength=5)
    assert counts.min() >= 990 and counts.max() <= 1010  # ~equal mass


def test_assignment_is_open_ended_beyond_fitted_range():
    values = np.linspace(0.0, 1.0, 1000)
    edges = compute_quantile_edges(values, k=4)
    assigned = assign_bins(np.array([-10.0, 10.0]), edges)
    assert list(assigned) == [0, 3]


def test_pool_too_small_fails_loudly():
    with pytest.raises(ValueError, match="pool too small"):
        compute_quantile_edges(np.arange(30), k=5)


def test_ties_produce_informative_error():
    values = np.zeros(1000)
    with pytest.raises(ValueError, match="heavy ties"):
        compute_quantile_edges(values, k=5)


def test_nan_rejected():
    values = np.linspace(0, 1, 100)
    values[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_quantile_edges(values, k=2)


def test_save_load_round_trip(tmp_path):
    values = np.random.default_rng(0).uniform(size=1000)
    edges = compute_quantile_edges(values, k=5, metadata={"task": "threading", "variant": "D2E"})
    path = tmp_path / "edges.json"
    save_edges(edges, path)
    loaded = load_edges(path)
    assert loaded == edges
    assert loaded.metadata["task"] == "threading"


def test_frozen_edges_validation():
    with pytest.raises(ValueError, match="interior edges"):
        FrozenBinEdges(k=5, interior_edges=(0.1, 0.2), n_fit=100, metadata={})
    with pytest.raises(ValueError, match="strictly increasing"):
        FrozenBinEdges(k=3, interior_edges=(0.2, 0.2), n_fit=100, metadata={})
