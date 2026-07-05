import numpy as np
import pytest

from matrix2d.core.resize import resize_matrix, resize_to_reference


def test_resize_shape():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = resize_matrix(a, (6, 8))
    assert out.shape == (6, 8)


def test_resize_flat_plane_preserved():
    a = np.full((5, 7), 3.7, dtype=np.float64)
    out = resize_matrix(a, (11, 13))
    assert np.nanmax(np.abs(out - 3.7)) < 1e-6


def test_resize_linear_ramp_row_preserved():
    rows, cols = 5, 9
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    # Linear ramp in both directions, normalized.
    a = 2.0 * (xx / (cols - 1)) + 3.0 * (yy / (rows - 1)) + 1.0
    tgt = (10, 17)
    out = resize_matrix(a, tgt)
    tr, tc = tgt
    yy2, xx2 = np.mgrid[0:tr, 0:tc].astype(np.float64)
    expected = 2.0 * (xx2 / (tc - 1)) + 3.0 * (yy2 / (tr - 1)) + 1.0
    assert np.nanmax(np.abs(out - expected)) < 1e-6


def test_resize_downscale_linear_ramp():
    rows, cols = 20, 30
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    a = 5.0 * (xx / (cols - 1)) - 4.0 * (yy / (rows - 1))
    tgt = (7, 11)
    out = resize_matrix(a, tgt)
    tr, tc = tgt
    yy2, xx2 = np.mgrid[0:tr, 0:tc].astype(np.float64)
    expected = 5.0 * (xx2 / (tc - 1)) - 4.0 * (yy2 / (tr - 1))
    assert np.nanmax(np.abs(out - expected)) < 1e-6


def test_resize_mask_scales_proportionally():
    a = np.ones((40, 40), dtype=np.float64)
    # Central 20x20 hole -> blank fraction = 400/1600 = 0.25.
    a[10:30, 10:30] = np.nan
    frac_in = np.isnan(a).mean()
    out = resize_matrix(a, (80, 80))
    frac_out = np.isnan(out).mean()
    assert abs(frac_out - frac_in) < 0.03


def test_resize_preserves_blank_region():
    a = np.ones((10, 10), dtype=np.float64)
    a[4:6, 4:6] = np.nan
    out = resize_matrix(a, (20, 20))
    # Center should still be blank.
    assert np.isnan(out[9:11, 9:11]).all()
    # Corners (valid) should still be valid.
    assert not np.isnan(out[0, 0])


def test_resize_all_blank_raises():
    a = np.full((4, 4), np.nan)
    with pytest.raises(ValueError):
        resize_matrix(a, (8, 8))


def test_resize_invalid_target_raises():
    a = np.ones((4, 4))
    with pytest.raises(ValueError):
        resize_matrix(a, (0, 5))


def test_resize_non_2d_raises():
    a = np.ones((4,))
    with pytest.raises(ValueError):
        resize_matrix(a, (2, 2))


# ---- resize_to_reference -----------------------------------------------------

def test_resize_to_reference_shape():
    data = np.ones((5, 5), dtype=np.float64)
    ref = np.ones((8, 10), dtype=np.float64)
    out = resize_to_reference(data, ref)
    assert out.shape == (8, 10)


def test_resize_to_reference_mask_exact_union():
    # Reference has a blank corner; data blank in the middle.
    data = np.ones((6, 6), dtype=np.float64)
    data[2:4, 2:4] = np.nan
    ref = np.ones((6, 6), dtype=np.float64)
    ref[0, 0] = np.nan
    out = resize_to_reference(data, ref, mask_mode="reference")
    # Reference blank must be blank in output.
    assert np.isnan(out[0, 0])
    # Data blank (middle) must also be blank.
    assert np.isnan(out[2:4, 2:4]).all()


def test_resize_to_reference_mode_reference_covers_ref_mask():
    data = np.ones((10, 10), dtype=np.float64)
    ref = np.ones((10, 10), dtype=np.float64)
    ref[5:8, 5:8] = np.nan
    out = resize_to_reference(data, ref, mask_mode="reference")
    # Every reference-blank cell is blank in output.
    assert np.isnan(out[np.isnan(ref)]).all()


def test_resize_to_reference_mode_own_ignores_ref_mask():
    data = np.ones((10, 10), dtype=np.float64)
    ref = np.ones((10, 10), dtype=np.float64)
    ref[5:8, 5:8] = np.nan
    out = resize_to_reference(data, ref, mask_mode="own")
    # Reference-only blanks are NOT applied in own mode; data is all valid.
    assert not np.isnan(out).any()


def test_resize_to_reference_bad_mode():
    data = np.ones((4, 4))
    ref = np.ones((4, 4))
    with pytest.raises(ValueError):
        resize_to_reference(data, ref, mask_mode="bogus")
