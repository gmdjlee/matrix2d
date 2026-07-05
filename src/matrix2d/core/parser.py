"""Filename parsing and matrix file loading.

I/O lives here (and in services). Core computation modules stay pure.
"""

import os
import re
from typing import List, Optional

import numpy as np

from .models import SampleMeta, WarpageData

# Filename format: AAAAA_PTXXXX_YYYYYYs(ZZZC).ext
# Applied to the file stem (extension removed).
_FILENAME_RE = re.compile(
    r"^(?P<title>.*)_PT(?P<sample>\d{4})_(?P<time>\d{6})s\((?P<temp>\d{1,3})C\)$"
)

# Values at or above this sentinel threshold are treated as blank.
BLANK_THRESHOLD = 2000.0

_VALID_EXTS = (".dat", ".csv", ".txt")


def parse_filename(filename: str, kind: str, path: str = "") -> SampleMeta:
    """Parse a measurement filename into a SampleMeta.

    Args:
        filename: The filename (with or without directory, with or without extension).
        kind: "TOP" | "BTM" | "GAP".
        path: Optional path stored on the result.

    Returns:
        A SampleMeta with parsed fields.

    Raises:
        ValueError: If the filename does not match the expected format.
    """
    base = os.path.basename(filename)
    stem, _ext = os.path.splitext(base)
    match = _FILENAME_RE.match(stem)
    if match is None:
        raise ValueError(
            "Filename '{0}' does not match expected format "
            "'AAAAA_PTXXXX_YYYYYYs(ZZZC).ext' (stem checked: '{1}')".format(
                filename, stem
            )
        )
    title = match.group("title")
    sample_no = int(match.group("sample"))
    time_s = int(match.group("time"))
    temp_c = int(match.group("temp"))
    return SampleMeta(
        title=title,
        sample_no=sample_no,
        time_s=time_s,
        temp_c=temp_c,
        kind=kind,
        path=path if path else filename,
    )


def _parse_cell(token: str) -> float:
    """Parse a single text cell into a float, mapping blanks/sentinels to NaN."""
    tok = token.strip()
    if tok == "":
        return np.nan
    low = tok.lower()
    if low == "nan":
        return np.nan
    try:
        val = float(tok)
    except ValueError:
        raise ValueError("Cannot parse cell value: {0!r}".format(token))
    if val >= BLANK_THRESHOLD:
        return np.nan
    return val


def load_matrix(path: str) -> np.ndarray:
    """Load a 2D numeric matrix from a text file.

    Delimiter auto-detected: comma if any content line contains a comma,
    otherwise whitespace/tab. Blank cells (empty string, "nan"/"NaN" any case,
    or numeric value >= 2000) become np.nan. Ragged rows are padded with NaN.

    Args:
        path: Path to the matrix text file (no header/index).

    Returns:
        A 2D float64 ndarray.

    Raises:
        ValueError: If the file has no numeric content or a cell is unparseable.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    # Keep non-empty content lines (a line with only whitespace/newline is skipped).
    content_lines = [ln.rstrip("\n").rstrip("\r") for ln in raw_lines]
    content_lines = [ln for ln in content_lines if ln.strip() != ""]
    if not content_lines:
        raise ValueError("No numeric content found in matrix file: {0}".format(path))

    use_comma = any("," in ln for ln in content_lines)

    rows: List[List[float]] = []
    for ln in content_lines:
        if use_comma:
            tokens = ln.split(",")
        else:
            tokens = ln.split()
        rows.append([_parse_cell(tok) for tok in tokens])

    max_len = max(len(r) for r in rows)
    for r in rows:
        if len(r) < max_len:
            r.extend([np.nan] * (max_len - len(r)))

    return np.asarray(rows, dtype=np.float64)


def load_warpage(path: str, kind: str) -> WarpageData:
    """Load a matrix file and its parsed metadata into a WarpageData.

    Args:
        path: Path to the matrix file.
        kind: "TOP" | "BTM" | "GAP".

    Returns:
        A WarpageData bundling metadata and values.

    Raises:
        ValueError: On filename or matrix parse failure.
    """
    meta = parse_filename(path, kind, path=path)
    values = load_matrix(path)
    return WarpageData(meta=meta, values=values)
