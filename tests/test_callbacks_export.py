"""Unit tests for _export_image_kwargs in matrix2d.ui.callbacks.

Importing callbacks does not require a running Dash app (the module only
defines callback factories), so the helper can be exercised in isolation.
Run with:  python -m pytest tests/test_callbacks_export.py
"""

from matrix2d.ui.callbacks import _export_image_kwargs


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
