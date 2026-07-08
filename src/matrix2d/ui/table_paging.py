"""Pure backend paging/sort/filter for the Gap result DataTable.

Kept Dash-free and side-effect-free so it is unit-testable and portable to
the planned React migration. Operates on a list of plain row dicts.
"""

from typing import Dict, List, Optional, Tuple

# Column ids shown in the result table, in display order. "offset" is sorted
# numerically (its cell is a formatted string); all others sort as text.
COLUMN_IDS = ["out_name", "top", "btm", "phase", "offset", "out_path"]
_NUMERIC_COLS = {"offset"}


def _num(value):
    """Best-effort float for numeric sort/compare; None if not parseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_filter(rows, filter_query):
    # Parse Dash's custom filter_query grammar for the operators the table can
    # emit, one clause per column, joined by " && ". Supported operators:
    #   contains, =, !=, >, >=, <, <=
    # Unknown/garbled clauses are ignored (row kept). Numeric compares fall
    # back to text equality when a side is non-numeric.
    if not filter_query:
        return rows
    clauses = [c.strip() for c in filter_query.split("&&") if c.strip()]
    parsed = []
    for clause in clauses:
        p = _parse_clause(clause)
        if p is not None:
            parsed.append(p)
    if not parsed:
        return rows
    out = []
    for row in rows:
        if all(_match(row, col, op, val) for col, op, val in parsed):
            out.append(row)
    return out


def _parse_clause(clause):
    # Expected shape: {col} <op> value   (col wrapped in braces)
    if "{" not in clause or "}" not in clause:
        return None
    col = clause[clause.index("{") + 1:clause.index("}")]
    rest = clause[clause.index("}") + 1:].strip()
    for op in ("contains", ">=", "<=", "!=", "=", ">", "<"):
        if rest.startswith(op):
            val = rest[len(op):].strip()
            if len(val) >= 2 and val[0] in "\"'" and val[-1] == val[0]:
                val = val[1:-1]
            return (col, op, val)
    return None


def _match(row, col, op, val):
    cell = row.get(col)
    cell_s = "" if cell is None else str(cell)
    if op == "contains":
        return val in cell_s
    if op in (">", ">=", "<", "<="):
        a, b = _num(cell), _num(val)
        if a is None or b is None:
            return False
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
        if op == "<":
            return a < b
        return a <= b
    if op == "=":
        b = _num(val)
        a = _num(cell)
        if a is not None and b is not None:
            return a == b
        return cell_s == val
    if op == "!=":
        b = _num(val)
        a = _num(cell)
        if a is not None and b is not None:
            return a != b
        return cell_s != val
    return True


def _sort_key(row, col, numeric):
    """(missing, value) key: missing cells sort last regardless of direction."""
    v = row.get(col)
    if numeric:
        n = _num(v)
        return (n is None, n if n is not None else 0.0)
    return (v is None, "" if v is None else str(v))


def _apply_sort(rows, sort_by):
    # sort_by is Dash's list: [{"column_id": <id>, "direction": "asc"|"desc"}].
    # Single-column (sort_mode="single"). Numeric-aware for _NUMERIC_COLS;
    # missing/unparseable values sort last in both directions.
    if not sort_by:
        return rows
    spec = sort_by[0]
    col = spec.get("column_id")
    desc = spec.get("direction") == "desc"
    numeric = col in _NUMERIC_COLS

    present = [r for r in rows if not _sort_key(r, col, numeric)[0]]
    missing = [r for r in rows if _sort_key(r, col, numeric)[0]]
    present.sort(key=lambda r: _sort_key(r, col, numeric), reverse=desc)
    return present + missing


def page_view(rows, page_current, page_size, sort_by, filter_query):
    # type: (List[Dict], int, int, Optional[list], Optional[str]) -> Tuple[List[Dict], int]
    """Return (rows_for_page, page_count) after filter -> sort -> slice."""
    if page_size is None or page_size <= 0:
        page_size = 50
    if page_current is None or page_current < 0:
        page_current = 0
    filtered = _apply_filter(rows, filter_query)
    ordered = _apply_sort(filtered, sort_by)
    n = len(ordered)
    page_count = (n + page_size - 1) // page_size if n else 1
    start = page_current * page_size
    return ordered[start:start + page_size], page_count
