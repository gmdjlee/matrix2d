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
    _downsample_for_export,
    _export_3d_all_worker,
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
#
# Pairing first: every GAP/OUT dataset anchors ONE image with its matching
# TOP/BTM; TOP/BTM no gap claimed group per temperature point; whatever is
# still unplaced exports as a single surface.
#
# Fixtures: one TOP (sample 1) and one BTM (sample 2) measured at
# 25C -> 240C -> 25C, so the peak-time phase rule tags the first 25C reading H
# and the last one C. OUT metas carry an explicit phase (gap output naming),
# like a scanned OUT folder.

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


def _meta(kind, sample, temp, time_s, path, **extra):
    d = {"title": "T", "sample_no": sample, "time_s": time_s,
         "temp_c": temp, "kind": kind, "path": path}
    d.update(extra)
    return d


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


# --- pairing pass -----------------------------------------------------------

def test_grouped_3d_items_each_gap_anchors_its_own_image():
    # Two computed gaps of the same TOP (TOP3) -> ONE image PER gap, named the
    # GAP way. No TOP/BTM options here, so each image is just the gap surface.
    items = _grouped_3d_items([("GAP", [
        {"label": "g8", "value": "gap::TEST-C25_TOP3-BTM8.txt"},
        {"label": "g9", "value": "gap::TEST-C25_TOP3-BTM9.txt"},
    ])], {}, {}, [], [])
    assert [it["filename"] for it in items] == [
        "GAP-C25-TOP3-BTM8.png", "GAP-C25-TOP3-BTM9.png"]
    assert [m["key"] for m in items[0]["members"]] == [
        "gap::TEST-C25_TOP3-BTM8.txt"]
    assert items[0]["title"] == "GAP-C25-TOP3-BTM8"


def test_grouped_3d_items_gap_filename_uses_phase_and_temp():
    items = _grouped_3d_items([("GAP", [
        {"label": "h", "value": "gap::T-H25_TOP1-BTM2.txt"},
        {"label": "c", "value": "gap::T-C240_TOP1-BTM2.txt"},
    ])], {}, {}, [], [])
    assert [it["filename"] for it in items] == [
        "GAP-H25-TOP1-BTM2.png", "GAP-C240-TOP1-BTM2.png"]


def test_grouped_3d_items_gap_pulls_matching_top_and_btm():
    # The user's case: TOP1 shared by two gaps -> two images, each TOP1 + its
    # own GAP + the paired BTM. TOP1 appears in both (not lumped into one).
    store_metas = {
        "TOP": [_meta("TOP", 1, 25, 100, "/x/top1.dat")],
        "BTM": [_meta("BTM", 1, 25, 100, "/x/btm1.dat"),
                _meta("BTM", 2, 25, 100, "/x/btm2.dat")],
    }
    items = _grouped_3d_items([
        ("TOP", [{"label": "TOP1", "value": "meta::/x/top1.dat"}]),
        ("BTM", [{"label": "BTM1", "value": "meta::/x/btm1.dat"},
                 {"label": "BTM2", "value": "meta::/x/btm2.dat"}]),
        ("GAP", [{"label": "g1", "value": "gap::T-H25_TOP1-BTM1.txt"},
                 {"label": "g2", "value": "gap::T-H25_TOP1-BTM2.txt"}]),
    ], {}, store_metas, [], [])
    assert [it["filename"] for it in items] == [
        "GAP-H25-TOP1-BTM1.png", "GAP-H25-TOP1-BTM2.png"]
    # image 1: TOP1 + GAP(TOP1-BTM1) + BTM1
    assert [(m["kind"], m["key"]) for m in items[0]["members"]] == [
        ("TOP", "meta::/x/top1.dat"),
        ("GAP", "gap::T-H25_TOP1-BTM1.txt"),
        ("BTM", "meta::/x/btm1.dat")]
    # image 2 reuses TOP1, pairs BTM2 — no leftover composite, no fallbacks
    assert [m["kind"] for m in items[1]["members"]] == ["TOP", "GAP", "BTM"]
    assert len(items) == 2


def test_grouped_3d_items_btm_matched_within_temp_tolerance():
    # BTM temp (24) lags the gap/TOP temp (25) by <= TEMP_TOLERANCE_C -> matched.
    store_metas = {
        "TOP": [_meta("TOP", 1, 25, 100, "/x/top1.dat")],
        "BTM": [_meta("BTM", 1, 24, 100, "/x/btm1.dat")],
    }
    items = _grouped_3d_items([
        ("TOP", [{"label": "TOP1", "value": "meta::/x/top1.dat"}]),
        ("BTM", [{"label": "BTM1", "value": "meta::/x/btm1.dat"}]),
        ("GAP", [{"label": "g1", "value": "gap::T-H25_TOP1-BTM1.txt"}]),
    ], {}, store_metas, [], [])
    assert len(items) == 1
    assert [m["kind"] for m in items[0]["members"]] == ["TOP", "GAP", "BTM"]


def test_grouped_3d_items_nearest_temp_wins_when_no_exact_match():
    # Two heating BTM readings in range (242 and 239, peak at t=200) -> the
    # nearer one (239) is picked even though 242 comes first in the options.
    store_metas = {
        "TOP": [_meta("TOP", 1, 240, 300, "/x/top1.dat")],
        "BTM": [_meta("BTM", 1, 242, 200, "/x/btm242.dat"),
                _meta("BTM", 1, 239, 100, "/x/btm239.dat")],
    }
    items = _grouped_3d_items([
        ("TOP", [{"label": "TOP1", "value": "meta::/x/top1.dat"}]),
        ("BTM", [{"label": "BTM242", "value": "meta::/x/btm242.dat"},
                 {"label": "BTM239", "value": "meta::/x/btm239.dat"}]),
        ("GAP", [{"label": "g1", "value": "gap::T-H240_TOP1-BTM1.txt"}]),
    ], {}, store_metas, [], [])
    assert items[0]["filename"] == "GAP-H240-TOP1-BTM1.png"
    assert items[0]["members"][2]["key"] == "meta::/x/btm239.dat"


def test_grouped_3d_items_tolerance_matched_btm_not_repeated_in_leftovers():
    # BTM at 239C joins the H240-anchored gap image and must NOT also surface
    # in an H239 leftover composite (consumed exclusion).
    store_metas = {
        "TOP": [_meta("TOP", 1, 240, 100, "/x/top1.dat")],
        "BTM": [_meta("BTM", 1, 239, 100, "/x/btm1.dat")],
    }
    items = _grouped_3d_items([
        ("TOP", [{"label": "TOP1", "value": "meta::/x/top1.dat"}]),
        ("BTM", [{"label": "BTM1", "value": "meta::/x/btm1.dat"}]),
        ("GAP", [{"label": "g1", "value": "gap::T-H240_TOP1-BTM1.txt"}]),
    ], {}, store_metas, [], [])
    assert [it["filename"] for it in items] == ["GAP-H240-TOP1-BTM1.png"]
    assert [m["kind"] for m in items[0]["members"]] == ["TOP", "GAP", "BTM"]


def test_grouped_3d_items_out_datasets_anchor_their_own_images():
    # Scanned OUT files anchor exactly like computed gaps; the C25 OUT file has
    # no TOP/BTM option at its point, so it renders alone in its own image.
    items = _grouped_3d_items(
        _options_by_kind(top=[_opt(_TOP_METAS[0])], btm=[_opt(_BTM_METAS[0])],
                         out=[_opt(m) for m in _OUT_METAS]),
        {}, _STORE_METAS, [], [])
    assert [it["filename"] for it in items] == [
        "OUT-H25-TOP1-BTM2.png", "OUT-C25-TOP1-BTM2.png"]
    assert [m["kind"] for m in items[0]["members"]] == ["TOP", "OUT", "BTM"]
    assert [m["kind"] for m in items[1]["members"]] == ["OUT"]


def test_grouped_3d_items_all_options_pair_up_gap_then_out():
    # Pairing consumes every TOP/BTM here: GAP first, then OUT in option order.
    items = _grouped_3d_items(_all_options(), {}, _STORE_METAS, [], [])
    assert [it["filename"] for it in items] == [
        "GAP-H240-TOP1-BTM2.png",
        "OUT-H25-TOP1-BTM2.png",
        "OUT-C25-TOP1-BTM2.png"]
    assert all([m["kind"] for m in it["members"]][1] in ("GAP", "OUT")
               for it in items)


def test_grouped_3d_items_duplicate_pairing_names_get_numeric_suffix():
    # Two gaps for the same pairing (different prefixes) -> _2 on the second.
    items = _grouped_3d_items([("GAP", [
        {"label": "a", "value": "gap::A-H25_TOP1-BTM2.txt"},
        {"label": "b", "value": "gap::B-H25_TOP1-BTM2.txt"},
    ])], {}, {}, [], [])
    assert [it["filename"] for it in items] == [
        "GAP-H25-TOP1-BTM2.png", "GAP-H25-TOP1-BTM2_2.png"]


# --- leftover temperature-point composites ----------------------------------

def test_grouped_3d_items_leftovers_group_into_temperature_composites():
    # TOP1/BTM1 are paired by a gap; TOP7/BTM9 have none -> one H25 composite.
    store_metas = {
        "TOP": [_meta("TOP", 1, 25, 100, "/x/top1.dat"),
                _meta("TOP", 7, 25, 100, "/x/top7.dat")],
        "BTM": [_meta("BTM", 1, 25, 100, "/x/btm1.dat"),
                _meta("BTM", 9, 25, 100, "/x/btm9.dat")],
    }
    items = _grouped_3d_items([
        ("TOP", [{"label": "TOP1", "value": "meta::/x/top1.dat"},
                 {"label": "TOP7", "value": "meta::/x/top7.dat"}]),
        ("BTM", [{"label": "BTM1", "value": "meta::/x/btm1.dat"},
                 {"label": "BTM9", "value": "meta::/x/btm9.dat"}]),
        ("GAP", [{"label": "g1", "value": "gap::T-H25_TOP1-BTM1.txt"}]),
    ], {}, store_metas, [], [])
    assert [it["filename"] for it in items] == [
        "GAP-H25-TOP1-BTM1.png", "H25_3D.png"]
    assert items[1]["title"] == "H25"
    # only the unpaired datasets, TOP before BTM
    assert [m["key"] for m in items[1]["members"]] == [
        "meta::/x/top7.dat", "meta::/x/btm9.dat"]


def test_grouped_3d_items_leftover_members_ordered_top_before_btm():
    # options handed over BTM-first still come back TOP-first in the composite
    scrambled = [("BTM", [_opt(_BTM_METAS[0])]), ("TOP", [_opt(_TOP_METAS[0])])]
    items = _grouped_3d_items(scrambled, {}, _STORE_METAS, [], [])
    assert [it["filename"] for it in items] == ["H25_3D.png"]
    assert [m["kind"] for m in items[0]["members"]] == ["TOP", "BTM"]


def test_grouped_3d_items_leftover_order_heating_up_then_cooling_down():
    metas = [_input_meta("TOP", 1, t, c) for t, c in
             ((0, 25), (100, 150), (200, 240), (300, 150), (400, 25))]
    store = {"TOP": metas}
    items = _grouped_3d_items(
        _options_by_kind(top=[_opt(m) for m in metas]), {}, store, [], [])
    assert [it["filename"] for it in items] == [
        "H25_3D.png", "H150_3D.png", "H240_3D.png",
        "C150_3D.png", "C25_3D.png"]


# --- single-surface fallback ------------------------------------------------

def test_grouped_3d_items_unparseable_gap_name_falls_back_to_single_surface():
    items = _grouped_3d_items(
        _options_by_kind(gap=[
            {"label": "GAP junk.txt", "value": "gap::junk.txt"},
            {"label": "GAP ok", "value": "gap::PFX-H25_TOP1-BTM2.txt"},
        ]), {}, {}, [], [])
    # pairing images come first, the unplaceable option exports alone last
    assert [it["filename"] for it in items] == [
        "GAP-H25-TOP1-BTM2.png", "GAP_junk_3D.png"]
    assert items[1]["title"] == "GAP junk.txt"


def test_grouped_3d_items_meta_absent_from_store_falls_back_to_single_surface():
    items = _grouped_3d_items(
        _options_by_kind(top=[{"label": "ghost", "value": "meta::/TOP/gone.dat"}]),
        {}, _STORE_METAS, [], [])
    assert [it["filename"] for it in items] == ["TOP_gone_3D.png"]
    assert items[0]["title"] == "ghost"
    assert [m["key"] for m in items[0]["members"]] == ["meta::/TOP/gone.dat"]


def test_grouped_3d_items_fallback_kind_prefix_in_filename():
    items = _grouped_3d_items(
        [("TOP", [{"label": "top a", "value": "meta::/x/a.dat"}])],
        {}, {}, [], [])
    assert len(items) == 1
    assert items[0]["filename"] == "TOP_a_3D.png"
    assert [m["key"] for m in items[0]["members"]] == ["meta::/x/a.dat"]
    assert items[0]["members"][0]["label"] == "top a"


def test_grouped_3d_items_fallback_order_preserved_across_kinds():
    items = _grouped_3d_items([
        ("TOP", [{"label": "t", "value": "meta::/x/t.dat"}]),
        ("BTM", [{"label": "b", "value": "meta::/x/b.dat"}]),
        ("GAP", [{"label": "g", "value": "gap::g.txt"}]),
        ("OUT", [{"label": "o", "value": "meta::/x/o.dat"}]),
    ], {}, {}, [], [])
    assert [it["filename"] for it in items] == [
        "TOP_t_3D.png", "BTM_b_3D.png", "GAP_g_3D.png", "OUT_o_3D.png"]


def test_grouped_3d_items_duplicate_stems_get_numeric_suffix():
    # same kind + same stem across two folders -> _2 on the second
    items = _grouped_3d_items([("TOP", [
        {"label": "a", "value": "meta::/x/a.dat"},
        {"label": "a2", "value": "meta::/y/a.dat"},
        {"label": "a3", "value": "meta::/z/a.dat"},
    ])], {}, {}, [], [])
    assert [it["filename"] for it in items] == [
        "TOP_a_3D.png", "TOP_a_3D_2.png", "TOP_a_3D_3.png"]


def test_grouped_3d_items_none_and_empty_options_skipped():
    items = _grouped_3d_items([
        ("TOP", None),
        ("BTM", []),
        ("GAP", [{"label": "g", "value": "gap::g.txt"}]),
    ], {}, {}, [], [])
    assert [it["filename"] for it in items] == ["GAP_g_3D.png"]


def test_grouped_3d_items_option_without_value_ignored():
    items = _grouped_3d_items(
        _options_by_kind(top=[{"label": "no value"}, _opt(_TOP_METAS[0])]),
        {}, _STORE_METAS, [], [])
    assert [it["filename"] for it in items] == ["H25_3D.png"]
    assert len(items[0]["members"]) == 1


def test_grouped_3d_items_label_falls_back_to_key():
    items = _grouped_3d_items([("GAP", [{"value": "gap::g.txt"}])],
                              {}, {}, [], [])
    assert items[0]["members"][0]["label"] == "gap::g.txt"


# --- kinds rule (dropdown selection picks which kinds take part) ------------

def test_grouped_3d_items_selection_picks_which_kinds_take_part():
    selections = {"TOP": ["meta::" + _TOP_METAS[0]["path"]],
                  "BTM": ["meta::" + _BTM_METAS[0]["path"]],
                  "GAP": [], "OUT": None}
    items = _grouped_3d_items(_all_options(), selections, _STORE_METAS, [], [])
    # GAP/OUT options exist but their kinds are excluded -> no pairing images,
    # so every TOP/BTM option falls into its temperature-point composite
    assert [it["filename"] for it in items] == [
        "H25_3D.png", "H240_3D.png", "C25_3D.png"]
    assert {m["kind"] for it in items for m in it["members"]} == {"TOP", "BTM"}


def test_grouped_3d_items_no_selection_anywhere_includes_all_kinds():
    for selections in ({}, {"TOP": [], "BTM": None, "GAP": [], "OUT": []},
                       [[], None, [], []]):
        items = _grouped_3d_items(_all_options(), selections, _STORE_METAS,
                                  [], [])
        kinds = {m["kind"] for it in items for m in it["members"]}
        assert kinds == {"TOP", "BTM", "GAP", "OUT"}


# --- z offsets --------------------------------------------------------------

def test_grouped_3d_items_offsets_selected_own_unselected_inherit():
    sel_top = "meta::" + _TOP_METAS[0]["path"]      # H25 TOP, offset 5.0
    other_top = "meta::" + _TOP_METAS[1]["path"]    # H240 TOP, not selected
    sel_btm = "meta::" + _BTM_METAS[0]["path"]      # H25 BTM, offset 1.5
    other_btm = "meta::" + _BTM_METAS[1]["path"]    # H240 BTM, not selected
    ids = [{"type": "z-offset", "key": sel_top},
           {"type": "z-offset", "key": sel_btm},
           {"type": "z-offset", "key": other_top}]
    values = [5.0, 1.5, -2.0]
    items = _grouped_3d_items(_all_options(),
                              {"TOP": [sel_top], "BTM": [sel_btm]},
                              _STORE_METAS, values, ids)
    offsets = {m["key"]: m["offset"] for it in items for m in it["members"]}
    assert offsets[sel_top] == 5.0        # selected -> own value
    assert offsets[sel_btm] == 1.5
    assert offsets[other_top] == -2.0     # has its own input -> keeps it
    assert offsets[other_btm] == 1.5      # inherits BTM's first selected offset


def test_grouped_3d_items_offsets_flow_into_pairing_members():
    store_metas = {
        "TOP": [_meta("TOP", 1, 25, 100, "/x/top1.dat"),
                _meta("TOP", 3, 25, 100, "/x/top3.dat")],
        "BTM": [_meta("BTM", 2, 25, 100, "/x/btm2.dat")],
    }
    top1, top3 = "meta::/x/top1.dat", "meta::/x/top3.dat"
    btm2 = "meta::/x/btm2.dat"
    g1, g3 = "gap::T-H25_TOP1-BTM2.txt", "gap::T-H25_TOP3-BTM2.txt"
    items = _grouped_3d_items(
        [("TOP", [{"label": "TOP1", "value": top1},
                  {"label": "TOP3", "value": top3}]),
         ("BTM", [{"label": "BTM2", "value": btm2}]),
         ("GAP", [{"label": "g1", "value": g1},
                  {"label": "g3", "value": g3}])],
        {"TOP": [top1], "BTM": [btm2], "GAP": [g1]},
        store_metas,
        [5.0, 2.0, 7.0],
        [{"type": "z-offset", "key": top1},
         {"type": "z-offset", "key": btm2},
         {"type": "z-offset", "key": g1}])
    by_name = {it["filename"]: it for it in items}
    assert sorted(by_name) == ["GAP-H25-TOP1-BTM2.png", "GAP-H25-TOP3-BTM2.png"]
    own = {m["key"]: m["offset"]
           for m in by_name["GAP-H25-TOP1-BTM2.png"]["members"]}
    assert own[top1] == 5.0     # selected TOP keeps its own offset in the image
    assert own[g1] == 7.0
    assert own[btm2] == 2.0
    inherited = {m["key"]: m["offset"]
                 for m in by_name["GAP-H25-TOP3-BTM2.png"]["members"]}
    assert inherited[top3] == 5.0   # unselected TOP -> TOP kind default
    assert inherited[g3] == 7.0     # unselected gap -> GAP kind default


def test_grouped_3d_items_offset_defaults_zero_without_selection_or_value():
    # nothing selected anywhere -> every kind's default offset is 0.0
    items = _grouped_3d_items(_all_options(), {}, _STORE_METAS, [], [])
    assert all(m["offset"] == 0.0 for it in items for m in it["members"])


def test_grouped_3d_items_none_offset_value_is_zero():
    sel_top = "meta::" + _TOP_METAS[0]["path"]
    other_top = "meta::" + _TOP_METAS[1]["path"]
    items = _grouped_3d_items(
        _options_by_kind(top=[_opt(m) for m in _TOP_METAS]),
        {"TOP": [sel_top]}, _STORE_METAS,
        [None], [{"type": "z-offset", "key": sel_top}])
    offsets = {m["key"]: m["offset"] for it in items for m in it["members"]}
    assert offsets[sel_top] == 0.0
    assert offsets[other_top] == 0.0      # inherits the 0.0 kind default


# --- _export_3d_all_worker (end-to-end, no Dash) ----------------------------

def test_export_3d_all_worker_writes_one_png_per_item(tmp_path):
    names = ["PFX-H25_TOP1-BTM2.txt", "PFX-C40_TOP1-BTM2.txt"]
    for i, name in enumerate(names):
        path = tmp_path / name
        np.savetxt(str(path), np.arange(9.0).reshape(3, 3) + i, delimiter="\t")
        helpers.register_gap(name, str(path))
    items = _grouped_3d_items(
        [("GAP", [{"label": "GAP " + n, "value": "gap::" + n} for n in names])],
        {}, {}, [], [])
    # single-member items too go through multi_surface_3d
    assert [it["filename"] for it in items] == [
        "GAP-H25-TOP1-BTM2.png", "GAP-C40-TOP1-BTM2.png"]

    try:
        _export_3d_all_worker(items, {}, str(tmp_path), charts.ChartOptions(),
                              {}, None, None, "original", "AUTO")
        for item in items:
            out = tmp_path / item["filename"]
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
    # TOP 9x9 + BTM 4x6 in one leftover composite: "resized" must bring the
    # non-reference side onto the reference grid instead of failing on the
    # shape mismatch.
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

    items = _grouped_3d_items(
        [("TOP", [_o(m) for m in metas["TOP"]]),
         ("BTM", [_o(m) for m in metas["BTM"]])],
        {}, metas, [], [])
    assert [it["filename"] for it in items] == ["H25_3D.png", "H240_3D.png"]
    assert all(len(it["members"]) == 2 for it in items)  # TOP + BTM per point

    dest = tmp_path / "png"
    try:
        _export_3d_all_worker(items, metas, str(dest), charts.ChartOptions(),
                              {}, None, None, "resized", "AUTO")
        assert _EXPORT3D["result"].startswith("Saved 2 image(s)")
        assert "Failed:" not in _EXPORT3D["result"]
        assert "Skipped:" not in _EXPORT3D["result"]
        assert sorted(os.listdir(str(dest))) == ["H240_3D.png", "H25_3D.png"]
    finally:
        _EXPORT3D.update(running=False, done=0, total=0,
                         result=None, error=None)


def test_export_3d_all_worker_reports_failures_and_member_notes(tmp_path):
    items = _grouped_3d_items(
        [("GAP", [{"label": "GAP missing",
                   "value": "gap::PFX-H25_TOP1-BTM2.txt"}])],
        {}, {}, [], [])
    assert [it["filename"] for it in items] == ["GAP-H25-TOP1-BTM2.png"]
    try:
        _export_3d_all_worker(items, {}, str(tmp_path), charts.ChartOptions(),
                              {}, None, None, "original", "AUTO")
        msg = _EXPORT3D["result"]
        assert msg.startswith("Saved 0 image(s)")
        assert "Failed: GAP-H25-TOP1-BTM2 (no loadable datasets)" in msg
        assert ("Skipped: GAP-H25-TOP1-BTM2.png: GAP missing (not loadable)"
                in msg)
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
