import numpy as np
import pytest

from genaudit.curation.uniformity import certify_uniform, tv_distance_to_uniform


def test_tv_zero_for_exact_uniform():
    assert tv_distance_to_uniform(np.array([100, 100, 100, 100, 100])) == 0.0


def test_tv_reads_as_fraction_out_of_place():
    # 10 demos moved from the last bin to the first: TV = 10/500 = 0.02.
    counts = np.array([110, 100, 100, 100, 90])
    assert tv_distance_to_uniform(counts) == pytest.approx(0.02)


def test_tv_of_fully_concentrated_histogram():
    counts = np.array([500, 0, 0, 0, 0])
    assert tv_distance_to_uniform(counts) == pytest.approx(0.8)  # 1 - 1/K


def test_certification_pass_and_fail():
    passing = certify_uniform(np.array([100, 100, 100, 100, 100]), quota=100)
    assert passing.passed

    # Boundary case: total deficit of exactly 10 -> TV = 0.02 and min bin = 90.
    boundary = certify_uniform(np.array([110, 100, 100, 100, 90]), quota=100)
    assert boundary.tv == pytest.approx(0.02)
    assert boundary.passed

    # One demo beyond the boundary fails both criteria together: at
    # N = K * quota the min-bin constraint is implied by the TV threshold
    # (a deficit of 11 forces TV >= 0.022); we keep it as an explicit,
    # redundant safety statement for the paper.
    starved = certify_uniform(np.array([111, 100, 100, 100, 89]), quota=100)
    assert starved.tv > 0.02
    assert starved.min_count < starved.min_count_required
    assert not starved.passed

    skewed = certify_uniform(np.array([150, 120, 100, 80, 50]), quota=100)
    assert not skewed.passed
    assert "FAIL" in skewed.summary()


def test_empty_histogram_rejected():
    with pytest.raises(ValueError, match="empty"):
        tv_distance_to_uniform(np.array([0, 0, 0]))
