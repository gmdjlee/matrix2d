"""Tests for the pure backend paging/sort/filter helper (ui.table_paging)."""

from matrix2d.ui.table_paging import page_view


def _row(out_name, top="", btm="", phase="H", offset="0", out_path=""):
    return {
        "out_name": out_name,
        "top": top,
        "btm": btm,
        "phase": phase,
        "offset": offset,
        "out_path": out_path,
    }


def test_pagination_slices():
    rows = [_row("r{0}".format(i)) for i in range(120)]

    page0, count0 = page_view(rows, 0, 50, None, None)
    assert count0 == 3
    assert len(page0) == 50
    assert [r["out_name"] for r in page0] == ["r{0}".format(i) for i in range(0, 50)]

    page1, count1 = page_view(rows, 1, 50, None, None)
    assert count1 == 3
    assert len(page1) == 50
    assert [r["out_name"] for r in page1] == ["r{0}".format(i) for i in range(50, 100)]

    page2, count2 = page_view(rows, 2, 50, None, None)
    assert count2 == 3
    assert len(page2) == 20
    assert [r["out_name"] for r in page2] == ["r{0}".format(i) for i in range(100, 120)]


def test_out_of_range_page_is_empty_but_page_count_correct():
    rows = [_row("r{0}".format(i)) for i in range(120)]
    page, count = page_view(rows, 10, 50, None, None)
    assert page == []
    assert count == 3


def test_empty_rows():
    assert page_view([], 0, 50, None, None) == ([], 1)


def test_text_sort_asc_desc():
    rows = [_row("b"), _row("a"), _row("c")]

    asc, _ = page_view(rows, 0, 50, [{"column_id": "out_name", "direction": "asc"}], None)
    assert [r["out_name"] for r in asc] == ["a", "b", "c"]

    desc, _ = page_view(rows, 0, 50, [{"column_id": "out_name", "direction": "desc"}], None)
    assert [r["out_name"] for r in desc] == ["c", "b", "a"]


def test_numeric_sort_on_offset_not_lexical():
    rows = [_row("x", offset="9"), _row("y", offset="10"), _row("z", offset="100")]
    asc, _ = page_view(rows, 0, 50, [{"column_id": "offset", "direction": "asc"}], None)
    assert [r["out_name"] for r in asc] == ["x", "y", "z"]

    desc, _ = page_view(rows, 0, 50, [{"column_id": "offset", "direction": "desc"}], None)
    assert [r["out_name"] for r in desc] == ["z", "y", "x"]


def test_missing_offset_sorts_last_both_directions():
    rows = [_row("a", offset="5"), _row("missing", offset=""), _row("b", offset="1")]

    asc, _ = page_view(rows, 0, 50, [{"column_id": "offset", "direction": "asc"}], None)
    assert [r["out_name"] for r in asc] == ["b", "a", "missing"]

    desc, _ = page_view(rows, 0, 50, [{"column_id": "offset", "direction": "desc"}], None)
    assert [r["out_name"] for r in desc] == ["a", "b", "missing"]


def test_filter_contains():
    rows = [
        _row("TEST-H25_TOP1-BTM3"),
        _row("TEST-H25_TOP3-BTM8"),
        _row("TEST-C25_TOP2-BTM8"),
    ]
    page, count = page_view(rows, 0, 50, None, '{out_name} contains "TOP3"')
    assert count == 1
    assert [r["out_name"] for r in page] == ["TEST-H25_TOP3-BTM8"]


def test_filter_equality_on_phase():
    rows = [_row("a", phase="H"), _row("b", phase="C"), _row("c", phase="H")]
    page, count = page_view(rows, 0, 50, None, '{phase} = "H"')
    assert count == 1
    assert sorted(r["out_name"] for r in page) == ["a", "c"]


def test_numeric_filter_greater_and_less_equal():
    rows = [_row("a", offset="1"), _row("b", offset="5"), _row("c", offset="9")]

    gt, count_gt = page_view(rows, 0, 50, None, "{offset} > 5")
    assert count_gt == 1
    assert [r["out_name"] for r in gt] == ["c"]

    lte, count_lte = page_view(rows, 0, 50, None, "{offset} <= 5")
    assert count_lte == 1
    assert sorted(r["out_name"] for r in lte) == ["a", "b"]


def test_combined_filter():
    rows = [
        _row("TEST-H25_TOP3-BTM8", phase="H"),
        _row("TEST-C25_TOP3-BTM8", phase="C"),
        _row("TEST-H25_TOP1-BTM8", phase="H"),
    ]
    page, count = page_view(
        rows, 0, 50, None, '{phase} = "H" && {out_name} contains "BTM8"')
    assert count == 1
    assert sorted(r["out_name"] for r in page) == [
        "TEST-H25_TOP1-BTM8", "TEST-H25_TOP3-BTM8"]


def test_filter_then_page_count_reflects_filtered_count():
    rows = [_row("keep{0}".format(i)) for i in range(75)] + \
        [_row("drop{0}".format(i)) for i in range(10)]
    page, count = page_view(rows, 0, 50, None, "{out_name} contains keep")
    assert count == 2
    assert len(page) == 50
    page2, count2 = page_view(rows, 1, 50, None, "{out_name} contains keep")
    assert count2 == 2
    assert len(page2) == 25
