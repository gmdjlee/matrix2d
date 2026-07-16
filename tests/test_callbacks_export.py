"""Unit tests for _export_image_kwargs / _downsample_for_export in
matrix2d.ui.callbacks.

Importing callbacks does not require a running Dash app (the module only
defines callback factories), so the helpers can be exercised in isolation.
Run with:  python -m pytest tests/test_callbacks_export.py
"""

import numpy as np

from matrix2d.ui.callbacks import (
    _downsample_for_export,
    _export_image_kwargs,
    _grouped_3d_items,
    _stem_for_key,
)


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


# --- _stem_for_key ----------------------------------------------------------

def test_stem_for_key_gap_strips_extension():
    assert _stem_for_key("gap::TEST-C25_TOP3-BTM8.txt") == "TEST-C25_TOP3-BTM8"


def test_stem_for_key_meta_windows_path_basename_stem():
    key = "meta::C:\\data\\TOP\\part_PT0001_00192s(240C).dat"
    assert _stem_for_key(key) == "part_PT0001_00192s(240C)"


def test_stem_for_key_unknown_passthrough():
    assert _stem_for_key("weird-key") == "weird-key"


# --- _grouped_3d_items ------------------------------------------------------
# Options whose key can't be resolved to a (sample, phase, temp) point (unknown
# meta path / non-gap-named value) fall back to one single-surface image each
# with the legacy {KIND}_{stem}_3D.png name — exercised with store_metas={}.

def test_grouped_3d_items_ungroupable_kind_prefix_in_filename():
    items = _grouped_3d_items(
        [("TOP", [{"label": "top a", "value": "meta::/x/a.dat"}])], {})
    assert len(items) == 1
    assert items[0]["filename"] == "TOP_a_3D.png"
    assert [m["key"] for m in items[0]["members"]] == ["meta::/x/a.dat"]
    assert items[0]["members"][0]["label"] == "top a"


def test_grouped_3d_items_ungroupable_order_preserved_across_kinds():
    items = _grouped_3d_items([
        ("TOP", [{"label": "t", "value": "meta::/x/t.dat"}]),
        ("BTM", [{"label": "b", "value": "meta::/x/b.dat"}]),
        ("GAP", [{"label": "g", "value": "gap::g.txt"}]),
        ("OUT", [{"label": "o", "value": "meta::/x/o.dat"}]),
    ], {})
    assert [it["filename"] for it in items] == [
        "TOP_t_3D.png", "BTM_b_3D.png", "GAP_g_3D.png", "OUT_o_3D.png"]


def test_grouped_3d_items_duplicate_stems_get_numeric_suffix():
    # same kind + same stem across two folders -> _2 on the second
    items = _grouped_3d_items([("TOP", [
        {"label": "a", "value": "meta::/x/a.dat"},
        {"label": "a2", "value": "meta::/y/a.dat"},
        {"label": "a3", "value": "meta::/z/a.dat"},
    ])], {})
    assert [it["filename"] for it in items] == [
        "TOP_a_3D.png", "TOP_a_3D_2.png", "TOP_a_3D_3.png"]


def test_grouped_3d_items_none_and_empty_options_skipped():
    items = _grouped_3d_items([
        ("TOP", None),
        ("BTM", []),
        ("GAP", [{"label": "g", "value": "gap::g.txt"}]),
    ], {})
    assert [it["filename"] for it in items] == ["GAP_g_3D.png"]


def test_grouped_3d_items_option_missing_value_skipped():
    items = _grouped_3d_items([("TOP", [
        {"label": "no value"},
        {"label": "ok", "value": "meta::/x/ok.dat"},
    ])], {})
    assert [it["filename"] for it in items] == ["TOP_ok_3D.png"]


def test_grouped_3d_items_label_falls_back_to_key():
    items = _grouped_3d_items(
        [("GAP", [{"value": "gap::g.txt"}])], {})
    assert items[0]["members"][0]["label"] == "gap::g.txt"


def test_grouped_3d_items_computed_gaps_same_point_combine():
    # Two computed gaps sharing (top_no=3, C, 25) -> ONE combined image.
    items = _grouped_3d_items([("GAP", [
        {"label": "g8", "value": "gap::TEST-C25_TOP3-BTM8.txt"},
        {"label": "g9", "value": "gap::TEST-C25_TOP3-BTM9.txt"},
    ])], {})
    assert len(items) == 1
    assert items[0]["filename"] == "PT0003-C25C_3D.png"
    assert [m["key"] for m in items[0]["members"]] == [
        "gap::TEST-C25_TOP3-BTM8.txt", "gap::TEST-C25_TOP3-BTM9.txt"]


def test_grouped_3d_items_different_points_stay_separate():
    items = _grouped_3d_items([("GAP", [
        {"label": "h", "value": "gap::T-H25_TOP1-BTM2.txt"},
        {"label": "c", "value": "gap::T-C25_TOP1-BTM2.txt"},
    ])], {})
    assert [it["filename"] for it in items] == [
        "PT0001-H25C_3D.png", "PT0001-C25C_3D.png"]


def _meta(kind, sample, temp, time_s, path):
    return {"title": "T", "sample_no": sample, "time_s": time_s,
            "temp_c": temp, "kind": kind, "path": path}


def test_grouped_3d_items_meta_same_point_combines_across_kinds():
    # A TOP and a BTM at the same sample/temp/phase (single measurement each
    # -> peak == itself -> phase 'H') combine into one PT0001-H25C image.
    store_metas = {
        "TOP": [_meta("TOP", 1, 25, 100, "/x/top.dat")],
        "BTM": [_meta("BTM", 1, 25, 100, "/x/btm.dat")],
    }
    items = _grouped_3d_items([
        ("TOP", [{"label": "t", "value": "meta::/x/top.dat"}]),
        ("BTM", [{"label": "b", "value": "meta::/x/btm.dat"}]),
    ], store_metas)
    assert len(items) == 1
    assert items[0]["filename"] == "PT0001-H25C_3D.png"
    assert {m["kind"] for m in items[0]["members"]} == {"TOP", "BTM"}
