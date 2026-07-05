import numpy as np
import pytest

from matrix2d.core.gap import compute_gap
from matrix2d.core.models import GapResult


def test_gap_basic_min_zero():
    top = np.array([[5.0, 6.0], [7.0, 8.0]])
    btm = np.array([[1.0, 1.0], [1.0, 1.0]])
    res = compute_gap(top, btm)
    assert isinstance(res, GapResult)
    # diff = [[4,5],[6,7]], offset = 4, gap = [[0,1],[2,3]]
    assert res.offset == pytest.approx(4.0)
    assert np.nanmin(res.gap) == pytest.approx(0.0)
    np.testing.assert_allclose(res.gap, [[0, 1], [2, 3]])


def test_gap_contact_index():
    top = np.array([[10.0, 2.0], [3.0, 4.0]])
    btm = np.array([[0.0, 0.0], [0.0, 0.0]])
    res = compute_gap(top, btm)
    # diff = top; min is at [0,1] (value 2)
    assert res.contact_index == (0, 1)
    assert res.gap[0, 1] == pytest.approx(0.0)


def test_gap_offset_negative_diff():
    top = np.array([[-5.0, -2.0]])
    btm = np.array([[0.0, 0.0]])
    res = compute_gap(top, btm)
    assert res.offset == pytest.approx(-5.0)
    np.testing.assert_allclose(res.gap, [[0.0, 3.0]])


def test_gap_nan_preserved():
    top = np.array([[1.0, np.nan], [3.0, 4.0]])
    btm = np.array([[0.0, 0.0], [np.nan, 1.0]])
    res = compute_gap(top, btm)
    # Cells where either is NaN stay NaN.
    assert np.isnan(res.gap[0, 1])
    assert np.isnan(res.gap[1, 0])
    # Valid cells: diff[0,0]=1, diff[1,1]=3 -> offset 1 -> gap 0 and 2
    assert res.gap[0, 0] == pytest.approx(0.0)
    assert res.gap[1, 1] == pytest.approx(2.0)


def test_gap_min_over_valid_exactly_zero():
    rng = np.random.default_rng(0)
    top = rng.normal(size=(8, 8))
    btm = rng.normal(size=(8, 8))
    top[0, 0] = np.nan
    res = compute_gap(top, btm)
    valid = ~np.isnan(res.gap)
    assert res.gap[valid].min() == pytest.approx(0.0, abs=1e-12)


def test_gap_shape_mismatch_raises():
    top = np.ones((2, 3))
    btm = np.ones((3, 2))
    with pytest.raises(ValueError):
        compute_gap(top, btm)


def test_gap_no_overlap_raises():
    top = np.array([[1.0, np.nan]])
    btm = np.array([[np.nan, 2.0]])
    with pytest.raises(ValueError):
        compute_gap(top, btm)


def test_gap_all_nan_raises():
    top = np.full((3, 3), np.nan)
    btm = np.full((3, 3), np.nan)
    with pytest.raises(ValueError):
        compute_gap(top, btm)
