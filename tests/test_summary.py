"""Tests for the max-gap summary table builder (core.summary)."""

import math

import pytest

from matrix2d.core.summary import build_summary, combo_label, temp_point_label


_STAT_LABELS = ("MIN", "MAX", "AVG", "STD")


def _parse(text):
    """Parse summary text into (columns, {row_label: {col: cell}}).

    Includes the MIN/MAX/AVG/STD statistic rows alongside the combo rows.
    """
    lines = text.rstrip("\n").split("\n")
    header = lines[0].split("\t")
    cols = header[1:]
    rows = {}
    for line in lines[1:]:
        parts = line.split("\t")
        rows[parts[0]] = dict(zip(cols, parts[1:]))
    return cols, rows


def _combo_rows(rows):
    return [k for k in rows if k not in _STAT_LABELS]


def test_labels():
    assert temp_point_label("H", 25) == "H25"
    assert combo_label(1, 8) == "TOP1-BTM8"


def test_empty_records():
    assert build_summary([]) == ""


def test_columns_heating_before_cooling_then_temp_ascending():
    records = [
        (1, 1, "C", 25, 1.0),
        (1, 1, "H", 75, 2.0),
        (1, 1, "H", 25, 3.0),
        (1, 1, "C", 50, 4.0),
    ]
    cols, _ = _parse(build_summary(records))
    assert cols == ["H25", "H75", "C25", "C50"]


def test_rows_sorted_by_top_then_btm():
    records = [
        (2, 1, "H", 25, 1.0),
        (1, 3, "H", 25, 1.0),
        (1, 1, "H", 25, 1.0),
    ]
    _, rows = _parse(build_summary(records))
    assert _combo_rows(rows) == ["TOP1-BTM1", "TOP1-BTM3", "TOP2-BTM1"]


def test_value_is_maximum_and_placed_at_combo_x_point():
    records = [
        (1, 1, "H", 25, 3.5),
        (2, 8, "C", 50, 9.25),
    ]
    _, rows = _parse(build_summary(records))
    assert rows["TOP1-BTM1"]["H25"] == "3.5"
    assert rows["TOP2-BTM8"]["C50"] == "9.25"


def test_duplicate_combo_point_aggregates_by_max():
    records = [
        (1, 1, "H", 25, 3.0),
        (1, 1, "H", 25, 7.0),
        (1, 1, "H", 25, 5.0),
    ]
    _, rows = _parse(build_summary(records))
    assert rows["TOP1-BTM1"]["H25"] == "7"


def test_missing_cell_is_blank():
    records = [
        (1, 1, "H", 25, 1.0),
        (1, 2, "H", 50, 2.0),
    ]
    _, rows = _parse(build_summary(records))
    # TOP1-BTM1 has no H50 value.
    assert rows["TOP1-BTM1"]["H50"] == ""
    assert rows["TOP1-BTM2"]["H25"] == ""


def test_nan_registers_point_but_leaves_blank():
    records = [(1, 1, "H", 25, float("nan"))]
    cols, rows = _parse(build_summary(records))
    assert cols == ["H25"]
    assert rows["TOP1-BTM1"]["H25"] == ""


def test_finite_value_wins_over_nan_for_same_cell():
    records = [
        (1, 1, "H", 25, float("nan")),
        (1, 1, "H", 25, 4.0),
    ]
    _, rows = _parse(build_summary(records))
    assert rows["TOP1-BTM1"]["H25"] == "4"


def test_stat_rows_directly_under_header_in_order():
    text = build_summary([(1, 1, "H", 25, 1.0)])
    labels = [ln.split("\t")[0] for ln in text.rstrip("\n").split("\n")]
    assert labels[:5] == ["", "MIN", "MAX", "AVG", "STD"]
    assert labels[5] == "TOP1-BTM1"


def test_stat_values_per_column():
    # H25 column has combo maxes 2, 4, 6; C25 has a single 10.
    records = [
        (1, 1, "H", 25, 2.0),
        (1, 2, "H", 25, 4.0),
        (1, 3, "H", 25, 6.0),
        (1, 1, "C", 25, 10.0),
    ]
    _, rows = _parse(build_summary(records))
    assert rows["MIN"]["H25"] == "2"
    assert rows["MAX"]["H25"] == "6"
    assert rows["AVG"]["H25"] == "4"
    # sample stdev of {2,4,6} == 2.0
    assert float(rows["STD"]["H25"]) == pytest.approx(2.0)
    # single value: MIN=MAX=AVG=10, STD blank (needs >=2)
    assert rows["MIN"]["C25"] == "10"
    assert rows["MAX"]["C25"] == "10"
    assert rows["AVG"]["C25"] == "10"
    assert rows["STD"]["C25"] == ""


def test_stat_cells_blank_for_all_nan_column():
    records = [(1, 1, "H", 25, float("nan"))]
    _, rows = _parse(build_summary(records))
    for lbl in ("MIN", "MAX", "AVG", "STD"):
        assert rows[lbl]["H25"] == ""
