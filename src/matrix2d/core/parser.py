"""Filename parsing and matrix file loading.

I/O lives here (and in services). Core computation modules stay pure.
"""

import os
import re
from typing import List, Optional

import numpy as np

from .models import SampleMeta, WarpageData

# Filename format: AAAAA_PTXXXX_YYYYYs(ZZZC).ext (5-digit seconds).
# Applied to the file stem (extension removed).
_FILENAME_RE = re.compile(
    r"^(?P<title>.*)_PT(?P<sample>\d{4})_(?P<time>\d{5})s\((?P<temp>\d{1,3})C\)$"
)

# Gap filename format: {prefix}-{H|C}{temp}_TOP{n}-BTM{m}[_k].ext where prefix
# is a free user-entered phrase and the optional _k suffix disambiguates
# duplicate output names (_2, _3, ...). Example: TEST-C25_TOP3-BTM8.txt
_GAP_FILENAME_RE = re.compile(
    r"^(?P<title>.+)-(?P<phase>[HC])(?P<temp>\d{1,3})"
    r"_TOP(?P<top>\d+)-BTM(?P<btm>\d+)(?:_\d+)?$"
)

# Legacy gap filename format: TOP{n}-BTM{m}_{H|C}{temp}[_k].ext (old OUT files).
_GAP_FILENAME_LEGACY_RE = re.compile(
    r"^TOP(?P<top>\d+)-BTM(?P<btm>\d+)_(?P<phase>[HC])(?P<temp>\d{1,3})(?:_\d+)?$"
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
            "'AAAAA_PTXXXX_YYYYYs(ZZZC).ext' (stem checked: '{1}')".format(
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


def parse_gap_filename(filename: str, path: str = "") -> SampleMeta:
    """Parse a gap filename like ``TEST-C25_TOP3-BTM8.txt`` into a SampleMeta.

    The primary format is ``{prefix}-{H|C}{temp}_TOP{n}-BTM{m}[_k].ext``;
    the legacy output format ``TOP{n}-BTM{m}_{H|C}{temp}[_k].ext`` is still
    accepted. The result has ``kind="GAP"``, ``sample_no`` = TOP sample
    number, ``btm_no`` = BTM sample number, an explicit ``phase`` ("H"/"C")
    and ``time_s=0`` (gap files carry no measurement time). A duplicate
    suffix (``_2``, ``_3``, ...) is accepted and ignored.

    Args:
        filename: The filename (with or without directory/extension).
        path: Optional path stored on the result.

    Returns:
        A SampleMeta with parsed fields.

    Raises:
        ValueError: If the filename matches neither gap naming format.
    """
    base = os.path.basename(filename)
    stem, _ext = os.path.splitext(base)
    match = _GAP_FILENAME_RE.match(stem)
    if match is None:
        match = _GAP_FILENAME_LEGACY_RE.match(stem)
    if match is None:
        raise ValueError(
            "Filename '{0}' does not match gap format "
            "'{{prefix}}-{{H|C}}{{temp}}_TOP{{n}}-BTM{{m}}.ext' (or the "
            "legacy 'TOP{{n}}-BTM{{m}}_{{H|C}}{{temp}}.ext'; stem checked: "
            "'{1}')".format(filename, stem)
        )
    return SampleMeta(
        title=stem,
        sample_no=int(match.group("top")),
        time_s=0,
        temp_c=int(match.group("temp")),
        kind="GAP",
        path=path if path else filename,
        btm_no=int(match.group("btm")),
        phase=match.group("phase"),
    )


def parse_data_filename(filename: str, kind: str, path: str = "") -> SampleMeta:
    """Parse a data filename for a kind, dispatching on the naming format.

    GAP files use the gap output format ``{prefix}-{H|C}{temp}_TOP{n}-BTM{m}``
    (or the legacy ``TOP{n}-BTM{m}_{H|C}{temp}``), falling back to the
    standard measurement format for legacy files. TOP/BTM always use the
    measurement format.

    Raises:
        ValueError: If no applicable format matches.
    """
    if kind == "GAP":
        try:
            return parse_gap_filename(filename, path=path)
        except ValueError:
            return parse_filename(filename, kind, path=path)
    return parse_filename(filename, kind, path=path)


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

    # Convert each line to a compact float64 row array as soon as it's parsed
    # so only one row's worth of python floats is alive at a time (the
    # python list of floats is discarded per-row instead of accumulating
    # R*C python floats before the final np.asarray).
    row_arrays: List[np.ndarray] = []
    for ln in content_lines:
        if use_comma:
            tokens = ln.split(",")
        else:
            tokens = ln.split()
        row_list = [_parse_cell(tok) for tok in tokens]
        row_arrays.append(np.asarray(row_list, dtype=np.float64))

    max_len = max(a.shape[0] for a in row_arrays)
    if all(a.shape[0] == max_len for a in row_arrays):
        return np.vstack(row_arrays)

    padded = []
    for a in row_arrays:
        if a.shape[0] < max_len:
            full = np.full(max_len, np.nan, dtype=np.float64)
            full[: a.shape[0]] = a
            padded.append(full)
        else:
            padded.append(a)
    return np.vstack(padded)


def load_warpage(path: str, kind: str) -> WarpageData:
    """Load a matrix file and its parsed metadata into a WarpageData.

    Args:
        path: Path to the matrix file.
        kind: "TOP" | "BTM" | "GAP" (GAP accepts the gap naming format,
            see parse_data_filename).

    Returns:
        A WarpageData bundling metadata and values.

    Raises:
        ValueError: On filename or matrix parse failure.
    """
    meta = parse_data_filename(path, kind, path=path)
    values = load_matrix(path)
    return WarpageData(meta=meta, values=values)
