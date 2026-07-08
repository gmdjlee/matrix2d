"""Filesystem repository: scanning folders, loading and saving matrices."""

import glob
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from ..core.models import SampleMeta, WarpageData
from ..core.parser import load_matrix, load_warpage, parse_data_filename

logger = logging.getLogger(__name__)

_SCAN_EXTS = ("*.dat", "*.csv", "*.txt")


def list_data_files(folder: str) -> "List[str]":
    """Return the sorted, deduped list of data-file paths in a folder.

    Matches *.dat/*.csv/*.txt (case as globbed by the OS). Used by
    :func:`scan_folder` and by callers that need a file count up front (e.g.
    to size a scan progress bar).

    Args:
        folder: Directory to list.

    Returns:
        Sorted list of matching file paths.
    """
    paths: List[str] = []
    for pattern in _SCAN_EXTS:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(set(paths))


def scan_folder(
    folder: str,
    kind: str,
    progress_cb=None,
    matrix_cache: "Optional[Dict[str, np.ndarray]]" = None,
) -> "List[SampleMeta]":
    """Scan a folder for parseable measurement files.

    Files matching *.dat/*.csv/*.txt are parsed; files whose NAME or CONTENT
    does not match the expected format are skipped with a logged warning
    (content is validated by loading the matrix once). Results are sorted by
    (sample_no, time_s).

    Args:
        folder: Directory to scan.
        kind: "TOP" | "BTM" | "GAP".
        progress_cb: Optional callable ``progress_cb(done, total)`` invoked
            after each file is processed (done = files processed so far,
            including skipped ones; total = number of candidate files).
        matrix_cache: Optional dict to populate with the raw matrix already
            loaded for content validation (path -> ndarray), so callers can
            reuse it instead of re-reading the file. Only files that pass
            validation are added; skipped/invalid files are never added.

    Returns:
        Sorted list of SampleMeta.
    """
    paths = list_data_files(folder)
    total = len(paths)

    metas: List[SampleMeta] = []
    for done, path in enumerate(paths, start=1):
        try:
            meta = parse_data_filename(path, kind, path=path)
            mat = load_matrix(path)  # content validation; also cached below
            if matrix_cache is not None:
                matrix_cache[path] = mat
            metas.append(meta)
        except (ValueError, OSError) as exc:
            logger.warning("Skipping invalid data file '%s': %s", path, exc)
        if progress_cb is not None:
            progress_cb(done, total)
    metas.sort(key=lambda m: (m.sample_no, m.time_s))
    return metas


def load_data(meta: SampleMeta) -> WarpageData:
    """Load the WarpageData for a given SampleMeta from its path.

    Args:
        meta: SampleMeta whose ``path`` points to the matrix file.

    Returns:
        The loaded WarpageData.
    """
    return load_warpage(meta.path, meta.kind)


def save_matrix(
    path: str,
    values: np.ndarray,
    delimiter: str = "\t",
    fmt: str = "%.2f",
) -> None:
    """Save a 2D matrix to a text file, writing NaN as the literal 'nan'.

    Args:
        path: Output file path.
        values: 2D array to save.
        delimiter: Field delimiter (default tab).
        fmt: printf-style format for finite values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must be 2D, got shape {0}".format(arr.shape))

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # np.savetxt formats at C speed (~3x faster than a per-cell Python loop,
    # which dominates the gap pipeline at scale). NaN renders as the literal
    # "nan" because ``fmt % np.nan`` == "nan"; newline is pinned to "\n" and
    # encoding to utf-8 so output is byte-identical to the previous loop.
    np.savetxt(
        path, arr, fmt=fmt, delimiter=delimiter, newline="\n", encoding="utf-8"
    )


def save_text(path: str, text: str) -> None:
    """Write a text file (utf-8, LF newlines), creating parent dirs.

    Args:
        path: Output file path.
        text: Full file contents to write verbatim.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
