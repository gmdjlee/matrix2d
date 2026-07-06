import numpy as np
import pytest

from matrix2d.core.transform import (
    TransformConfig,
    apply_transform,
    flip_lr_invert,
    rotate90_cw,
    zero_at_cell,
)


# ---- flip_lr_invert ----------------------------------------------------------

def test_flip_lr_invert_mirrors_and_negates():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = flip_lr_invert(a)
    np.testing.assert_allclose(out, [[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]])


def test_flip_lr_invert_nan_stays_nan():
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    out = flip_lr_invert(a)
    # NaN moves with its column (col 1 -> col 0) and stays NaN.
    assert np.isnan(out[0, 0])
    assert out[0, 1] == pytest.approx(-1.0)
    np.testing.assert_allclose(out[1], [-4.0, -3.0])


def test_flip_lr_invert_input_unchanged():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    orig = a.copy()
    out = flip_lr_invert(a)
    assert out is not a
    np.testing.assert_array_equal(a, orig)


# ---- rotate90_cw -------------------------------------------------------------

def test_rotate90_cw_one_step():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = rotate90_cw(a, 1)
    # Clockwise: first row becomes last column.
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, [[4.0, 1.0], [5.0, 2.0], [6.0, 3.0]])


def test_rotate90_cw_two_steps_is_180():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = rotate90_cw(a, 2)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])


def test_rotate90_cw_four_steps_is_identity_copy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = rotate90_cw(a, 4)
    assert out is not a
    np.testing.assert_allclose(out, a)
    # Must be a copy, not a view onto the input.
    out[0, 0] = 99.0
    assert a[0, 0] == pytest.approx(1.0)


def test_rotate90_cw_steps_normalized():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(rotate90_cw(a, 5), rotate90_cw(a, 1))
    np.testing.assert_allclose(rotate90_cw(a, -1), rotate90_cw(a, 3))
    np.testing.assert_allclose(rotate90_cw(a, -4), a)


def test_rotate90_cw_odd_steps_swap_shape():
    a = np.ones((2, 5))
    assert rotate90_cw(a, 1).shape == (5, 2)
    assert rotate90_cw(a, 3).shape == (5, 2)
    assert rotate90_cw(a, 2).shape == (2, 5)


def test_rotate90_cw_output_contiguous():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    for steps in range(4):
        assert rotate90_cw(a, steps).flags["C_CONTIGUOUS"]


# ---- zero_at_cell ------------------------------------------------------------

def test_zero_at_cell_anchor_becomes_zero():
    a = np.array([[5.0, 6.0], [7.0, 8.0]])
    out = zero_at_cell(a, 1, 0)
    assert out[1, 0] == 0.0
    # All values shifted by the same anchor value (7).
    np.testing.assert_allclose(out, [[-2.0, -1.0], [0.0, 1.0]])


def test_zero_at_cell_nan_preserved_elsewhere():
    a = np.array([[2.0, np.nan], [3.0, 4.0]])
    out = zero_at_cell(a, 0, 0)
    assert out[0, 0] == 0.0
    assert np.isnan(out[0, 1])
    np.testing.assert_allclose(out[1], [1.0, 2.0])


def test_zero_at_cell_input_unchanged():
    a = np.array([[5.0, 6.0]])
    out = zero_at_cell(a, 0, 1)
    assert out is not a
    np.testing.assert_allclose(a, [[5.0, 6.0]])


def test_zero_at_cell_out_of_bounds_raises():
    a = np.ones((2, 3))
    with pytest.raises(ValueError):
        zero_at_cell(a, 2, 0)
    with pytest.raises(ValueError):
        zero_at_cell(a, 0, 3)
    with pytest.raises(ValueError):
        zero_at_cell(a, -1, 0)


def test_zero_at_cell_nan_target_raises():
    a = np.array([[np.nan, 1.0]])
    with pytest.raises(ValueError, match="blank"):
        zero_at_cell(a, 0, 0)


# ---- apply_transform ---------------------------------------------------------

def test_apply_transform_order_flip_then_rotate():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    cfg = TransformConfig(flip_lr=True, rot90_cw=1)
    out = apply_transform(a, cfg)
    # flip -> [[-2,-1],[-4,-3]]; then rotate cw -> [[-4,-2],[-3,-1]]
    np.testing.assert_allclose(out, [[-4.0, -2.0], [-3.0, -1.0]])
    # The reverse order (rotate then flip) gives a different matrix.
    wrong = flip_lr_invert(rotate90_cw(a, 1))
    assert not np.allclose(out, wrong)


def test_apply_transform_zero_after_flip_rotate():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    cfg = TransformConfig(flip_lr=True, rot90_cw=1, zero_cell=(0, 0))
    out = apply_transform(a, cfg)
    # After flip+rotate the matrix is [[-4,-2],[-3,-1]]; anchor (0,0) -> 0.0.
    np.testing.assert_allclose(out, [[0.0, 2.0], [1.0, 3.0]])


def test_apply_transform_none_config_returns_copy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = apply_transform(a, None)
    assert out is not a
    np.testing.assert_allclose(out, a)
    out[0, 0] = 99.0
    assert a[0, 0] == pytest.approx(1.0)


def test_apply_transform_default_config_returns_copy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = apply_transform(a, TransformConfig())
    assert out is not a
    np.testing.assert_allclose(out, a)


def test_apply_transform_non_2d_raises():
    with pytest.raises(ValueError):
        apply_transform(np.ones((3,)), None)
    with pytest.raises(ValueError):
        apply_transform(np.ones((2, 2, 2)), TransformConfig())


def test_apply_transform_input_never_mutated():
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    orig = a.copy()
    cfg = TransformConfig(flip_lr=True, rot90_cw=3, zero_cell=(1, 1))
    apply_transform(a, cfg)
    np.testing.assert_array_equal(a, orig)
