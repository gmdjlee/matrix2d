"""Unit tests for _export_image_kwargs / _downsample_for_export in
matrix2d.ui.callbacks.

Importing callbacks does not require a running Dash app (the module only
defines callback factories), so the helpers can be exercised in isolation.
Run with:  python -m pytest tests/test_callbacks_export.py
"""

import os

import numpy as np

from matrix2d.ui import charts, helpers, layout
from matrix2d.ui.callbacks import (
    _EXPORT3D,
    _build_options,
    _composite_3d_groups,
    _downsample_for_export,
    _export_3d_all_worker,
    _export_image_kwargs,
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


# --- _composite_3d_groups ---------------------------------------------------
#
# Fixtures: one TOP and one BTM sample measured at 25C -> 240C -> 25C, so the
# peak-time phase rule tags the first 25C reading H and the last one C. OUT
# metas carry an explicit phase (gap output naming), like a scanned OUT folder.

def _input_meta(kind, sample_no, time_s, temp_c):
    return {"title": "part", "sample_no": sample_no, "time_s": time_s,
            "temp_c": temp_c, "kind": kind,
            "path": "/{0}/PT{1:04d}_{2:05d}s({3}C).dat".format(
                kind, sample_no, time_s, temp_c),
            "btm_no": None, "phase": None}


def _out_meta(top_no, btm_no, phase, temp_c):
    name = "PFX-{0}{1}_TOP{2}-BTM{3}.txt".format(phase, temp_c, top_no, btm_no)
    return {"title": "PFX", "sample_no": top_no, "time_s": 0,
            "temp_c": temp_c, "kind": "GAP", "path": "/OUT/" + name,
            "btm_no": btm_no, "phase": phase}


_TOP_METAS = [_input_meta("TOP", 1, 0, 25), _input_meta("TOP", 1, 100, 240),
              _input_meta("TOP", 1, 200, 25)]
_BTM_METAS = [_input_meta("BTM", 2, 0, 25), _input_meta("BTM", 2, 100, 240),
              _input_meta("BTM", 2, 200, 25)]
_OUT_METAS = [_out_meta(1, 2, "H", 25), _out_meta(1, 2, "C", 25)]

_STORE_METAS = {"TOP": _TOP_METAS, "BTM": _BTM_METAS,
                "GAP": [], "OUT": _OUT_METAS}


def _opt(meta):
    return {"label": meta["kind"] + " " + meta["path"],
            "value": "meta::" + meta["path"]}


def _options_by_kind(top=None, btm=None, gap=None, out=None):
    return [("TOP", top or []), ("BTM", btm or []),
            ("GAP", gap or []), ("OUT", out or [])]


def _all_options():
    return _options_by_kind(
        top=[_opt(m) for m in _TOP_METAS],
        btm=[_opt(m) for m in _BTM_METAS],
        gap=[{"label": "GAP PFX-H240_TOP1-BTM2.txt",
              "value": "gap::PFX-H240_TOP1-BTM2.txt"}],
        out=[_opt(m) for m in _OUT_METAS])


def test_composite_groups_by_phase_and_temp_across_kinds():
    groups, skipped = _composite_3d_groups(
        _all_options(), {}, _STORE_METAS, [], [])
    assert skipped == []
    assert [g["filename"] for g in groups] == [
        "H25_3D.png", "H240_3D.png", "C25_3D.png"]
    by_name = {g["filename"]: g for g in groups}
    # H25: TOP t=0, BTM t=0, OUT H25
    assert [m["kind"] for m in by_name["H25_3D.png"]["members"]] == [
        "TOP", "BTM", "OUT"]
    # H240: TOP t=100, BTM t=100 and the gap:: option (parsed from its name)
    h240 = by_name["H240_3D.png"]["members"]
    assert [m["kind"] for m in h240] == ["TOP", "BTM", "GAP"]
    assert h240[2]["key"] == "gap::PFX-H240_TOP1-BTM2.txt"
    assert by_name["C25_3D.png"]["phase"] == "C"
    assert by_name["C25_3D.png"]["temp"] == 25


def test_composite_groups_members_ordered_top_btm_gap_out():
    # options handed over in a scrambled kind order still come back TOP-first
    scrambled = [("OUT", [_opt(_OUT_METAS[0])]),
                 ("GAP", [{"label": "g", "value": "gap::PFX-H25_TOP1-BTM2.txt"}]),
                 ("BTM", [_opt(_BTM_METAS[0])]),
                 ("TOP", [_opt(_TOP_METAS[0])])]
    groups, _skipped = _composite_3d_groups(scrambled, {}, _STORE_METAS, [], [])
    assert len(groups) == 1
    assert [m["kind"] for m in groups[0]["members"]] == [
        "TOP", "BTM", "GAP", "OUT"]


def test_composite_groups_selection_picks_which_kinds_take_part():
    selections = {"TOP": ["meta::" + _TOP_METAS[0]["path"]],
                  "BTM": ["meta::" + _BTM_METAS[0]["path"]],
                  "GAP": [], "OUT": None}
    groups, skipped = _composite_3d_groups(
        _all_options(), selections, _STORE_METAS, [], [])
    assert skipped == []   # excluded kinds are ignored, not "skipped"
    kinds = {m["kind"] for g in groups for m in g["members"]}
    assert kinds == {"TOP", "BTM"}


def test_composite_groups_no_selection_anywhere_includes_all_kinds():
    for selections in ({}, {"TOP": [], "BTM": None, "GAP": [], "OUT": []},
                       [[], None, [], []]):
        groups, _skipped = _composite_3d_groups(
            _all_options(), selections, _STORE_METAS, [], [])
        kinds = {m["kind"] for g in groups for m in g["members"]}
        assert kinds == {"TOP", "BTM", "GAP", "OUT"}


def test_composite_groups_offsets_selected_own_unselected_inherit():
    sel_top = "meta::" + _TOP_METAS[0]["path"]      # H25 TOP, offset 5.0
    other_top = "meta::" + _TOP_METAS[1]["path"]    # H240 TOP, not selected
    sel_btm = "meta::" + _BTM_METAS[0]["path"]      # H25 BTM, offset 1.5
    other_btm = "meta::" + _BTM_METAS[1]["path"]    # H240 BTM, not selected
    ids = [{"type": "z-offset", "key": sel_top},
           {"type": "z-offset", "key": sel_btm},
           {"type": "z-offset", "key": other_top}]
    values = [5.0, 1.5, -2.0]
    groups, _skipped = _composite_3d_groups(
        _all_options(),
        {"TOP": [sel_top], "BTM": [sel_btm]},
        _STORE_METAS, values, ids)
    offsets = {m["key"]: m["offset"] for g in groups for m in g["members"]}
    assert offsets[sel_top] == 5.0        # selected -> own value
    assert offsets[sel_btm] == 1.5
    assert offsets[other_top] == -2.0     # has its own input -> keeps it
    assert offsets[other_btm] == 1.5      # inherits BTM's first selected offset


def test_composite_groups_offset_defaults_zero_without_selection_or_value():
    # nothing selected anywhere -> every kind's default offset is 0.0
    groups, _skipped = _composite_3d_groups(
        _all_options(), {}, _STORE_METAS, [], [])
    assert all(m["offset"] == 0.0 for g in groups for m in g["members"])


def test_composite_groups_none_offset_value_is_zero():
    sel_top = "meta::" + _TOP_METAS[0]["path"]
    other_top = "meta::" + _TOP_METAS[1]["path"]
    groups, _skipped = _composite_3d_groups(
        _options_by_kind(top=[_opt(m) for m in _TOP_METAS]),
        {"TOP": [sel_top]}, _STORE_METAS,
        [None], [{"type": "z-offset", "key": sel_top}])
    offsets = {m["key"]: m["offset"] for g in groups for m in g["members"]}
    assert offsets[sel_top] == 0.0
    assert offsets[other_top] == 0.0      # inherits the 0.0 kind default


def test_composite_groups_order_heating_up_then_cooling_down():
    metas = [_input_meta("TOP", 1, t, c) for t, c in
             ((0, 25), (100, 150), (200, 240), (300, 150), (400, 25))]
    store = {"TOP": metas}
    groups, _skipped = _composite_3d_groups(
        _options_by_kind(top=[_opt(m) for m in metas]), {}, store, [], [])
    assert [g["filename"] for g in groups] == [
        "H25_3D.png", "H150_3D.png", "H240_3D.png",
        "C150_3D.png", "C25_3D.png"]


def test_composite_groups_unparseable_gap_name_skipped_by_label():
    groups, skipped = _composite_3d_groups(
        _options_by_kind(gap=[
            {"label": "GAP junk.txt", "value": "gap::junk.txt"},
            {"label": "GAP ok", "value": "gap::PFX-H25_TOP1-BTM2.txt"},
        ]), {}, _STORE_METAS, [], [])
    assert skipped == ["GAP junk.txt"]
    assert [g["filename"] for g in groups] == ["H25_3D.png"]


def test_composite_groups_meta_key_absent_from_store_skipped():
    groups, skipped = _composite_3d_groups(
        _options_by_kind(top=[{"label": "ghost", "value": "meta::/TOP/gone.dat"}]),
        {}, _STORE_METAS, [], [])
    assert groups == []
    assert skipped == ["ghost"]


def test_composite_groups_option_without_value_ignored():
    groups, skipped = _composite_3d_groups(
        _options_by_kind(top=[{"label": "no value"}, _opt(_TOP_METAS[0])]),
        {}, _STORE_METAS, [], [])
    assert skipped == []
    assert [g["filename"] for g in groups] == ["H25_3D.png"]
    assert len(groups[0]["members"]) == 1


# --- _export_3d_all_worker (end-to-end, no Dash) ----------------------------

def test_export_3d_all_worker_writes_one_png_per_group(tmp_path):
    names = ["PFX-H25_TOP1-BTM2.txt", "PFX-C40_TOP1-BTM2.txt"]
    for i, name in enumerate(names):
        path = tmp_path / name
        np.savetxt(str(path), np.arange(9.0).reshape(3, 3) + i, delimiter="\t")
        helpers.register_gap(name, str(path))
    groups, skipped = _composite_3d_groups(
        [("GAP", [{"label": "GAP " + n, "value": "gap::" + n} for n in names])],
        {}, {}, [], [])
    assert [g["filename"] for g in groups] == ["H25_3D.png", "C40_3D.png"]
    assert skipped == []

    try:
        _export_3d_all_worker(groups, [], {}, str(tmp_path),
                              charts.ChartOptions(), {}, None, None,
                              "original", "AUTO")
        for group in groups:
            out = tmp_path / group["filename"]
            assert out.exists()
            assert out.stat().st_size > 0
        assert _EXPORT3D["result"].startswith("Saved 2 image(s)")
        assert _EXPORT3D["running"] is False
        assert _EXPORT3D["error"] is None
        assert _EXPORT3D["total"] == 2 and _EXPORT3D["done"] == 2
    finally:
        helpers.clear_gaps()
        _EXPORT3D.update(running=False, done=0, total=0,
                         result=None, error=None)


def test_export_3d_all_worker_composites_meta_datasets_resized(tmp_path):
    # TOP 9x9 + BTM 4x6 in one group: "resized" must bring the non-reference
    # side onto the reference grid instead of failing on the shape mismatch.
    metas = {"TOP": [], "BTM": []}
    for kind, shape in (("TOP", (9, 9)), ("BTM", (4, 6))):
        for time_s, temp_c in ((0, 25), (100, 240)):
            path = tmp_path / "{0}_PT0001_{1:05d}s({2}C).dat".format(
                kind, time_s, temp_c)
            np.savetxt(str(path), np.ones(shape), delimiter="\t")
            metas[kind].append(
                {"title": "part", "sample_no": 1 if kind == "TOP" else 2,
                 "time_s": time_s, "temp_c": temp_c, "kind": kind,
                 "path": str(path), "btm_no": None, "phase": None})

    def _o(m):
        return {"label": m["kind"] + str(m["temp_c"]),
                "value": "meta::" + m["path"]}

    groups, skipped = _composite_3d_groups(
        [("TOP", [_o(m) for m in metas["TOP"]]),
         ("BTM", [_o(m) for m in metas["BTM"]])],
        {}, metas, [], [])
    assert [g["filename"] for g in groups] == ["H25_3D.png", "H240_3D.png"]
    assert all(len(g["members"]) == 2 for g in groups)  # TOP + BTM per point

    dest = tmp_path / "png"
    try:
        _export_3d_all_worker(groups, skipped, metas, str(dest),
                              charts.ChartOptions(), {}, None, None,
                              "resized", "AUTO")
        assert _EXPORT3D["result"].startswith("Saved 2 image(s)")
        assert "Failed:" not in _EXPORT3D["result"]
        assert "Skipped:" not in _EXPORT3D["result"]
        assert sorted(os.listdir(str(dest))) == ["H240_3D.png", "H25_3D.png"]
    finally:
        _EXPORT3D.update(running=False, done=0, total=0,
                         result=None, error=None)


def test_export_3d_all_worker_reports_skips_and_failures(tmp_path):
    groups, _skipped = _composite_3d_groups(
        [("GAP", [{"label": "GAP missing",
                   "value": "gap::PFX-H25_TOP1-BTM2.txt"}])],
        {}, {}, [], [])
    try:
        _export_3d_all_worker(groups, ["opt without a temperature point"], {},
                              str(tmp_path), charts.ChartOptions(), {},
                              None, None, "original", "AUTO")
        msg = _EXPORT3D["result"]
        assert msg.startswith("Saved 0 image(s)")
        assert "Failed: H25 (no loadable datasets)" in msg
        assert "Skipped: opt without a temperature point" in msg
        assert not os.listdir(str(tmp_path))
        assert _EXPORT3D["running"] is False
    finally:
        helpers.clear_gaps()
        _EXPORT3D.update(running=False, done=0, total=0,
                         result=None, error=None)


# --- colorbar-mode control wiring (_build_options round-trip) ---------------

def _opt_values(prefix, overrides=None):
    """Positional control values for one tab; unset fields default to None."""
    overrides = overrides or {}
    return [overrides.get(k) for k in layout.tab_option_suffixes(prefix)]


def test_colorbar_mode_only_rendered_on_3d_tab():
    assert "colorbar-mode" in layout.tab_option_suffixes("opt3d")
    for prefix in ("opt2d", "optgap", "opteff"):
        assert "colorbar-mode" not in layout.tab_option_suffixes(prefix)


def test_build_options_colorbar_mode_per_item():
    opts = _build_options("opt3d", _opt_values("opt3d",
                                               {"colorbar-mode": "per-item"}))
    assert opts.shared_colorbar is False


def test_build_options_colorbar_mode_shared():
    opts = _build_options("opt3d", _opt_values("opt3d",
                                               {"colorbar-mode": "shared"}))
    assert opts.shared_colorbar is True


def test_build_options_colorbar_mode_missing_defaults_to_shared():
    # control present but empty (None), and tabs that never render it
    assert _build_options("opt3d", _opt_values("opt3d")).shared_colorbar is True
    assert _build_options("opt2d", _opt_values("opt2d")).shared_colorbar is True
