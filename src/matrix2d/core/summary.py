"""Build the max-gap summary table (temperature points x TOP-BTM combos).

Pure logic. No I/O.

The summary is a pivot: one row per TOP-BTM sample combination
(``TOP{n}-BTM{m}``), one column per temperature point (``{H|C}{temp}``,
e.g. ``H25``), each cell the MAXIMUM gap value of that combination at that
temperature point. Columns are ordered heating-before-cooling then by
ascending temperature; rows by (top_no, btm_no). Missing combination/point
cells are blank.

Directly under the header, four statistic rows (``MIN``, ``MAX``, ``AVG``,
``STD``) give per-column aggregates over the combo cells at that
temperature point (blanks ignored). ``STD`` is the sample standard
deviation (Excel ``STDEV``, ddof=1); a column with fewer than two finite
values leaves it blank.
"""

import math
import statistics
from typing import Iterable, List, Optional, Tuple


def temp_point_label(phase: str, temp_c: int) -> str:
    """Column label for a temperature point, e.g. ``H25``."""
    return "{0}{1}".format(phase, temp_c)


def combo_label(top_no: int, btm_no: int) -> str:
    """Row label for a TOP-BTM sample combination, e.g. ``TOP1-BTM1``."""
    return "TOP{0}-BTM{1}".format(top_no, btm_no)


def _col_sort_key(col: "Tuple[str, int]") -> "Tuple[int, int]":
    phase, temp = col
    # Heating ('H') columns come before cooling ('C'); ascending temp within.
    return (0 if phase == "H" else 1, temp)


def _is_missing(val: "Optional[float]") -> bool:
    return val is None or (isinstance(val, float) and math.isnan(val))


def build_summary(
    records: "Iterable[Tuple[int, int, str, int, Optional[float]]]",
    value_fmt: str = "{:.4g}",
    delimiter: str = "\t",
    corner: str = "",
) -> str:
    """Build the max-gap summary as delimited text.

    Args:
        records: Iterable of ``(top_no, btm_no, phase, temp_c, max_gap)``.
            Duplicate (combo, point) records are aggregated by MAX. A
            None/NaN ``max_gap`` still registers the row/column but leaves
            the cell blank unless a finite value also appears for it.
        value_fmt: ``str.format`` spec for finite cell values.
        delimiter: Field delimiter (default tab).
        corner: Top-left header cell text (default empty).

    Returns:
        The table as a newline-terminated string. Header row first
        (corner + temperature-point columns), then one row per combo.
        Returns an empty string when ``records`` yields nothing.
    """
    cells = {}  # type: dict
    rows = set()
    cols = set()
    for top_no, btm_no, phase, temp_c, val in records:
        rk = (top_no, btm_no)
        ck = (phase, temp_c)
        rows.add(rk)
        cols.add(ck)
        if _is_missing(val):
            continue
        cur = cells.get((rk, ck))
        cells[(rk, ck)] = val if cur is None else max(cur, val)

    if not rows:
        return ""

    row_keys = sorted(rows)
    col_keys = sorted(cols, key=_col_sort_key)

    # Finite combo values per column, for the MIN/MAX/AVG/STD rows.
    col_values = {}  # type: dict
    for ck in col_keys:
        vals = [cells[(rk, ck)] for rk in row_keys
                if cells.get((rk, ck)) is not None]
        col_values[ck] = vals

    def _fmt(v):
        # type: (Optional[float]) -> str
        return value_fmt.format(v) if v is not None else ""

    def _stat_row(label, fn):
        cells_out = [label]
        for ck in col_keys:
            vals = col_values[ck]
            cells_out.append(_fmt(fn(vals) if vals else None))
        return delimiter.join(cells_out)

    def _std(vals):
        # type: (List[float]) -> Optional[float]
        return statistics.stdev(vals) if len(vals) >= 2 else None

    lines = []
    header = [corner] + [temp_point_label(p, t) for (p, t) in col_keys]
    lines.append(delimiter.join(header))

    # Statistic rows sit directly under the header, above the combos.
    lines.append(_stat_row("MIN", min))
    lines.append(_stat_row("MAX", max))
    lines.append(_stat_row("AVG", lambda v: sum(v) / len(v)))
    lines.append(_stat_row("STD", _std))

    for rk in row_keys:
        cell_row = [combo_label(*rk)]
        for ck in col_keys:
            v = cells.get((rk, ck))
            cell_row.append(_fmt(v))
        lines.append(delimiter.join(cell_row))
    return "\n".join(lines) + "\n"
