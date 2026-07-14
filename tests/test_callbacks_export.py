"""Unit tests for _export_image_kwargs / _downsample_for_export in
matrix2d.ui.callbacks.

Importing callbacks does not require a running Dash app (the module only
defines callback factories), so the helpers can be exercised in isolation.
Run with:  python -m pytest tests/test_callbacks_export.py
"""

import numpy as np

from matrix2d.ui.callbacks import _downsample_for_export, _export_image_kwargs


def test_valid_ints():
    assert _export_image_kwargs(800, 600, None) == {"width": 800, "height": 600}


def test_valid_scale_float():
    assert _export_image_kwargs(None, None, 2.5) == {"scale": 2.5}


def test_all_three_valid():
    assert _export_image_kwargs(1024, 768, 2) == {
        "width": 1024, "height": 768, "scale": 2.0}


def test_numeric_strings_coerced():
    assert _export_image_kwargs("640", "480", "1.5") == {
        "width": 640, "height": 480, "scale": 1.5}


def test_float_width_truncates_to_int():
    out = _export_image_kwargs(800.9, 600.2, None)
    assert out == {"width": 800, "height": 600}
    assert isinstance(out["width"], int)


def test_none_omitted():
    assert _export_image_kwargs(None, None, None) == {}


def test_blank_strings_omitted():
    assert _export_image_kwargs("", "", "") == {}


def test_junk_strings_omitted():
    assert _export_image_kwargs("abc", "wide", "big") == {}


def test_zero_omitted():
    assert _export_image_kwargs(0, 0, 0) == {}


def test_negative_omitted():
    assert _export_image_kwargs(-100, -50, -1.0) == {}


def test_mixed_valid_and_invalid():
    # width valid, height blank, scale junk
    assert _export_image_kwargs(800, "", "oops") == {"width": 800}


def test_mixed_zero_and_valid():
    # width zero (omitted), height valid, scale valid
    assert _export_image_kwargs(0, 600, 1.5) == {"height": 600, "scale": 1.5}


def test_never_raises_on_odd_types():
    # objects that are neither None/str/number must be swallowed, not raised
    assert _export_image_kwargs([1], {"x": 1}, object()) == {}


# --- _downsample_for_export -------------------------------------------------

def test_downsample_cap_zero_returns_same_array():
    a = np.arange(100).reshape(10, 10)
    assert _downsample_for_export(a, 0) is a


def test_downsample_cap_none_returns_same_array():
    a = np.arange(100).reshape(10, 10)
    assert _downsample_for_export(a, None) is a


def test_downsample_negative_cap_returns_same_array():
    a = np.arange(100).reshape(10, 10)
    assert _downsample_for_export(a, -5) is a


def test_downsample_cap_ge_max_shape_unchanged():
    a = np.arange(100).reshape(10, 10)
    assert _downsample_for_export(a, 10) is a
    assert _downsample_for_export(a, 50) is a


def test_downsample_exact_division_stride():
    # shape (10, 10), cap 5 -> k = ceil(10/5) = 2 -> shape (5, 5)
    a = np.arange(100).reshape(10, 10)
    out = _downsample_for_export(a, 5)
    assert out.shape == (5, 5)
    # stride-2 selection, not interpolation
    np.testing.assert_array_equal(out, a[::2, ::2])


def test_downsample_ceil_case():
    # shape (10, 4), cap 3 -> k = ceil(10/3) = 4 -> rows [0,4,8], cols [0]
    a = np.arange(40).reshape(10, 4)
    out = _downsample_for_export(a, 3)
    assert out.shape == (3, 1)
    np.testing.assert_array_equal(out, a[::4, ::4])


def test_downsample_preserves_nan_cells():
    a = np.arange(64, dtype="float64").reshape(8, 8)
    a[0, 0] = np.nan
    a[4, 4] = np.nan
    out = _downsample_for_export(a, 4)  # k = ceil(8/4) = 2
    assert out.shape == (4, 4)
    assert np.isnan(out[0, 0])   # a[0, 0]
    assert np.isnan(out[2, 2])   # a[4, 4]
    assert not np.isnan(out[1, 1])


def test_downsample_non_square_only_one_dim_exceeds():
    # shape (20, 5), cap 10 -> longest=20 -> k = ceil(20/10) = 2
    a = np.arange(100).reshape(20, 5)
    out = _downsample_for_export(a, 10)
    assert out.shape == (10, 3)  # rows 20->10, cols 5->ceil(5/2)=3
    np.testing.assert_array_equal(out, a[::2, ::2])


def test_downsample_non_square_within_cap_unchanged():
    # longest dim (8) already <= cap -> unchanged even though non-square
    a = np.arange(40).reshape(8, 5)
    assert _downsample_for_export(a, 8) is a
