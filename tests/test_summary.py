"""Tests for the max-gap summary table builder (core.summary)."""

import math

from matrix2d.core.summary import build_summary, combo_label, temp_point_label


def _parse(text):
    """Parse summary text into (columns, {row_label: {col: cell}})."""
    lines = text.rstrip("\n").split("\n")
    header = lines[0].split("\t")
    cols = header[1:]
    rows = {}
    for line in lines[1:]:
        parts = line.split("\t")
        rows[parts[0]] = dict(zip(cols, parts[1:]))
    return cols, rows


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
    assert list(rows.keys()) == ["TOP1-BTM1", "TOP1-BTM3", "TOP2-BTM1"]


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
