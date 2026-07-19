import numpy as np
import pytest

from genaudit.analysis.ancestry import ancestry_stats
from genaudit.analysis.dgr import definition_comparison, dgr, dgr_vs_distance, trend_stats
from genaudit.curation.binning import compute_quantile_edges
from genaudit.evaluation.stratify import paired_success_difference, stratify_success
from genaudit.records.schema import AttemptRecord


def _pool(n=2000, seed=0):
    """Synthetic pool with a genuine monotone DGR-vs-distance decline and a
    source-quality spread (ancestry skew emerges via retention)."""
    rng = np.random.default_rng(seed)
    records = []
    source_quality = np.linspace(0.9, 0.2, 10)
    for index in range(n):
        source = int(rng.integers(10))
        d_pos = float(rng.uniform(0, 1))
        p_success = source_quality[source] * (1.0 - 0.8 * d_pos)
        success = bool(rng.uniform() < p_success)
        records.append(
            AttemptRecord(
                task="toy",
                variant="D2E",
                attempt_id=f"demo_{index}@pool",
                source_demo_id=source,
                success=success,
                episode_length=100,
                displacements=(),
                d_raw=d_pos * 0.8,  # monotone transform of d_pos
                d_pos=d_pos,
                d_rot=0.0,
            )
        )
    return records


def test_dgr_curve_declines_and_trend_stats_agree():
    records = _pool()
    assert 0.2 < dgr(records) < 0.6
    curve = dgr_vs_distance(records, "d_pos", k=5)
    assert curve.per_bin_dgr[0] > curve.per_bin_dgr[-1]
    assert sum(curve.per_bin_attempts) == len(records)

    stats = trend_stats(records, "d_pos", k=5)
    assert stats.point_biserial_r < -0.2
    assert stats.spearman_rho < -0.2


def test_definition_comparison_sees_equivalent_trends_for_monotone_transforms():
    records = _pool()
    comparison = definition_comparison(records, keys=("d_raw", "d_pos"), k=5)
    # d_raw is a monotone rescaling of d_pos here: Spearman must match exactly.
    assert comparison["d_raw"].spearman_rho == pytest.approx(
        comparison["d_pos"].spearman_rho
    )


def test_ancestry_stats_capture_retention_skew():
    records = _pool()
    stats = ancestry_stats(records, num_sources=10)
    assert sum(stats.attempted_counts) == len(records)
    # good sources over-represented among survivors
    assert stats.top3_share_retained > stats.top3_share_attempted
    assert stats.skew_pp > 5.0
    assert 1.0 <= stats.n_eff_retained <= 10.0


def test_wilson_lower_bound_and_pool_plan():
    from genaudit.analysis.dgr import plan_pool_size, wilson_lower_bound

    # textbook check: 10/100 -> LB ~= 0.0552 at z=1.96
    assert wilson_lower_bound(10, 100) == pytest.approx(0.0552, abs=1e-3)
    assert wilson_lower_bound(0, 100) == 0.0

    plan = plan_pool_size(_pool(), "d_pos", k=5, target_retained_total=500)
    assert plan.scarcest_bin == 4  # monotone decline -> farthest bin scarcest
    assert plan.planned_total_attempts > 500 / plan.per_bin_dgr[4]  # LB < point est
    assert sum(plan.per_bin_attempts) == 2000


def test_stratify_success_flags_far_bin_gap():
    rng = np.random.default_rng(1)
    d_eval = rng.uniform(0, 1, size=1000)
    successes = rng.uniform(size=1000) < (0.9 - 0.6 * d_eval)
    result = stratify_success(d_eval, successes, k=5)
    assert result.per_bin_success_rate[0] > result.per_bin_success_rate[-1]
    assert result.slope_per_bin < 0
    assert sum(result.per_bin_count) == 1000
    assert 0 < result.far_bins_success_rate < result.aggregate_success_rate


def test_stratify_with_frozen_edges_is_paired():
    rng = np.random.default_rng(2)
    d_eval = rng.uniform(0, 1, size=500)
    edges = compute_quantile_edges(d_eval, k=5)
    successes_a = rng.uniform(size=500) < 0.5
    successes_b = rng.uniform(size=500) < 0.5
    result_a = stratify_success(d_eval, successes_a, k=5, edges=edges)
    result_b = stratify_success(d_eval, successes_b, k=5, edges=edges)
    assert result_a.per_bin_count == result_b.per_bin_count  # same episodes per bin


def test_paired_difference_counts_discordant_pairs():
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, True, False, True])
    summary = paired_success_difference(a, b)
    assert summary["wins_a_only"] == 1
    assert summary["wins_b_only"] == 1
    assert summary["discordant"] == 2
