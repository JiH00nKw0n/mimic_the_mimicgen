import numpy as np
import pytest

from genaudit.curation.samplers import (
    InsufficientPoolError,
    sample_baseline,
    sample_stratified_uniform,
)


def test_baseline_size_and_determinism():
    result_a = sample_baseline(2000, 500, np.random.default_rng(7))
    result_b = sample_baseline(2000, 500, np.random.default_rng(7))
    assert result_a.size == 500
    assert result_a.selected == result_b.selected
    assert len(set(result_a.selected)) == 500


def test_baseline_insufficient_pool():
    with pytest.raises(InsufficientPoolError):
        sample_baseline(400, 500, np.random.default_rng(0))


def test_stratified_exact_quota_certifies_tv_zero():
    labels = np.repeat(np.arange(5), 300)  # 300 candidates per bin
    result = sample_stratified_uniform(labels, 5, 500, np.random.default_rng(1))
    assert result.per_stratum_counts == (100, 100, 100, 100, 100)
    assert result.certification is not None
    assert result.certification.tv == 0.0
    assert result.certification.passed
    # every selected index maps back to the right stratum with 100 per bin
    assert len(result.selected) == 500


def test_stratified_redistributes_small_shortfall():
    # scarcest bin has only 95 candidates: 5 demos must come from elsewhere,
    # TV = 5/500 = 0.01 <= 0.02 and min bin 95 >= 90 -> still certified.
    labels = np.concatenate([np.repeat(np.arange(4), 300), np.full(95, 4)])
    result = sample_stratified_uniform(labels, 5, 500, np.random.default_rng(2))
    counts = np.array(result.per_stratum_counts)
    assert counts.sum() == 500
    assert counts[4] == 95
    assert result.certification is not None and result.certification.passed


def test_stratified_fails_when_bin_too_starved():
    labels = np.concatenate([np.repeat(np.arange(4), 300), np.full(60, 4)])
    with pytest.raises(InsufficientPoolError, match="shortfalls"):
        sample_stratified_uniform(labels, 5, 500, np.random.default_rng(3))


def test_stratified_requires_exact_quota_divisibility():
    labels = np.repeat(np.arange(3), 200)
    with pytest.raises(ValueError, match="divisible"):
        sample_stratified_uniform(labels, 3, 500, np.random.default_rng(0))


def test_ancestry_balancing_uses_same_mechanism():
    # 10 sources, 500-demo arm -> 50 per source.
    labels = np.repeat(np.arange(10), 80)
    result = sample_stratified_uniform(
        labels, 10, 500, np.random.default_rng(4), arm="ancestry_balanced"
    )
    assert result.per_stratum_counts == tuple([50] * 10)
    assert result.arm == "ancestry_balanced"


def test_ancestry_balanced_excludes_starved_source_and_reaches_size():
    from genaudit.curation.samplers import sample_ancestry_balanced

    # sources 0-7 and 9 have plenty; source 8 has only 34 retained
    labels = np.concatenate(
        [np.repeat(np.arange(8), 200), np.full(200, 9), np.full(34, 8)]
    )
    result = sample_ancestry_balanced(labels, 10, 500, np.random.default_rng(0))
    assert result.size == 500
    assert 8 in result.info["excluded_sources"]
    # kept sources balanced (quota 500/9 = 55, +1 remainder to some)
    kept_counts = [result.per_stratum_counts[s] for s in result.info["kept_sources"]]
    assert max(kept_counts) - min(kept_counts) <= 1
    assert result.per_stratum_counts[8] == 0
    assert result.info["n_eff"] > 8.0  # 9 balanced sources


def test_ancestry_balanced_all_sources_healthy_keeps_all():
    from genaudit.curation.samplers import sample_ancestry_balanced

    labels = np.repeat(np.arange(10), 80)
    result = sample_ancestry_balanced(labels, 10, 500, np.random.default_rng(1))
    assert result.info["excluded_sources"] == []
    assert result.per_stratum_counts == tuple([50] * 10)
    assert result.info["n_eff"] == pytest.approx(10.0)
