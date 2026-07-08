import numpy as np
import pytest

from matrix2d.core.resize import (
    resize_crop_blank,
    resize_pair,
    resize_values,
)


# ---- resize_values (warpage-preserving value interpolation) ------------------

def test_resize_values_shape():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = resize_values(a, (6, 8))
    assert out.shape == (6, 8)


def test_resize_values_flat_plane_preserved():
    a = np.full((5, 7), 3.7, dtype=np.float64)
    out = resize_values(a, (11, 13))
    assert np.max(np.abs(out - 3.7)) < 1e-6


def test_resize_values_linear_ramp_upscale_preserved():
    rows, cols = 5, 9
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    a = 2.0 * (xx / (cols - 1)) + 3.0 * (yy / (rows - 1)) + 1.0
    tgt = (10, 17)
    out = resize_values(a, tgt)
    tr, tc = tgt
    yy2, xx2 = np.mgrid[0:tr, 0:tc].astype(np.float64)
    expected = 2.0 * (xx2 / (tc - 1)) + 3.0 * (yy2 / (tr - 1)) + 1.0
    assert np.max(np.abs(out - expected)) < 1e-6


def test_resize_values_linear_ramp_downscale_preserved():
    rows, cols = 20, 30
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    a = 5.0 * (xx / (cols - 1)) - 4.0 * (yy / (rows - 1))
    tgt = (7, 11)
    out = resize_values(a, tgt)
    tr, tc = tgt
    yy2, xx2 = np.mgrid[0:tr, 0:tc].astype(np.float64)
    expected = 5.0 * (xx2 / (tc - 1)) - 4.0 * (yy2 / (tr - 1))
    assert np.max(np.abs(out - expected)) < 1e-6


def test_resize_values_is_finite_even_with_blank():
    a = np.ones((6, 6), dtype=np.float64)
    a[2:4, 2:4] = np.nan
    out = resize_values(a, (12, 12))
    # Values carry no blank -- that is applied separately.
    assert np.isfinite(out).all()


def test_resize_values_all_blank_raises():
    a = np.full((4, 4), np.nan)
    with pytest.raises(ValueError):
        resize_values(a, (8, 8))


def test_resize_values_invalid_target_raises():
    a = np.ones((4, 4))
    with pytest.raises(ValueError):
        resize_values(a, (0, 5))


def test_resize_values_non_2d_raises():
    a = np.ones((4,))
    with pytest.raises(ValueError):
        resize_values(a, (2, 2))


# ---- resize_crop_blank (blank cropped, not scaled) ---------------------------

def test_crop_blank_downscale_keeps_absolute_size_centered():
    a = np.ones((12, 12), dtype=np.float64)
    a[4:8, 5:7] = np.nan  # central 4x2 blank
    out = resize_crop_blank(a, (10, 10))
    assert out.shape == (10, 10)
    # Blank keeps its 4x2 cell extent (cropping only trims the valid border).
    assert int(np.isnan(out).sum()) == 8
    rows = np.where(np.isnan(out).any(axis=1))[0]
    cols = np.where(np.isnan(out).any(axis=0))[0]
    assert rows.max() - rows.min() + 1 == 4
    assert cols.max() - cols.min() + 1 == 2


def test_crop_blank_upscale_pads_valid_and_keeps_size():
    a = np.ones((6, 6), dtype=np.float64)
    a[2:4, 2:4] = np.nan  # 2x2 blank
    out = resize_crop_blank(a, (10, 10))
    assert out.shape == (10, 10)
    # Blank is NOT stretched with the grid; it stays 2x2, centered.
    assert int(np.isnan(out).sum()) == 4
    assert np.isnan(out[4:6, 4:6]).all()
    # Padded border stays valid.
    assert not np.isnan(out[0, 0])


def test_crop_blank_fully_valid_stays_valid():
    a = np.ones((8, 8), dtype=np.float64)
    out = resize_crop_blank(a, (5, 5))
    assert not np.isnan(out).any()


# ---- resize_pair (match both sides to the larger blank) ----------------------

def test_resize_pair_shape_reference_top():
    top = np.ones((8, 10), dtype=np.float64)
    btm = np.ones((6, 8), dtype=np.float64)
    top_out, btm_out = resize_pair(top, btm, "TOP")
    assert top_out.shape == (8, 10)
    assert btm_out.shape == (8, 10)


def test_resize_pair_shape_reference_btm():
    top = np.ones((8, 10), dtype=np.float64)
    btm = np.ones((6, 8), dtype=np.float64)
    top_out, btm_out = resize_pair(top, btm, "BTM")
    assert top_out.shape == (6, 8)
    assert btm_out.shape == (6, 8)


def test_resize_pair_both_get_larger_blank_union():
    # TOP has a bigger central blank, BTM a smaller one; both end identical.
    top = np.ones((10, 10), dtype=np.float64)
    top[3:7, 3:7] = np.nan  # 4x4
    btm = np.ones((10, 10), dtype=np.float64)
    btm[4:6, 4:6] = np.nan  # 2x2 (contained in TOP's)
    top_out, btm_out = resize_pair(top, btm, "TOP")
    # Shared blank equals the larger (TOP) blank.
    assert np.array_equal(np.isnan(top_out), np.isnan(btm_out))
    assert np.array_equal(np.isnan(top_out), np.isnan(top))


def test_resize_pair_union_is_per_dimension():
    # Tall-narrow blank on TOP, short-wide on BTM -> union spans both.
    top = np.ones((10, 10), dtype=np.float64)
    top[2:8, 4:6] = np.nan  # 6 rows x 2 cols
    btm = np.ones((10, 10), dtype=np.float64)
    btm[4:6, 2:8] = np.nan  # 2 rows x 6 cols
    top_out, btm_out = resize_pair(top, btm, "TOP")
    blank = np.isnan(top_out)
    assert np.array_equal(blank, np.isnan(btm_out))
    rows = np.where(blank.any(axis=1))[0]
    cols = np.where(blank.any(axis=0))[0]
    # Bounding box covers the larger extent on each axis (6 rows, 6 cols).
    assert rows.max() - rows.min() + 1 == 6
    assert cols.max() - cols.min() + 1 == 6


def test_resize_pair_blank_cropped_when_paired():
    # Non-reference (TOP, larger) blank keeps absolute size after crop.
    top = np.ones((12, 12), dtype=np.float64)
    top[4:8, 5:7] = np.nan  # 4x2 central blank
    btm = np.ones((10, 10), dtype=np.float64)
    top_out, btm_out = resize_pair(top, btm, "BTM")
    assert top_out.shape == (10, 10)
    assert int(np.isnan(top_out).sum()) == 8  # 4x2 preserved
    assert np.array_equal(np.isnan(top_out), np.isnan(btm_out))


def test_resize_pair_bad_reference_raises():
    a = np.ones((4, 4))
    with pytest.raises(ValueError):
        resize_pair(a, a, "SIDE")


def test_resize_pair_all_blank_side_raises():
    top = np.ones((8, 8), dtype=np.float64)
    btm = np.full((6, 6), np.nan)  # must be resized -> all-blank fails
    with pytest.raises(ValueError):
        resize_pair(top, btm, "TOP")
