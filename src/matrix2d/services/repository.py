"""Filesystem repository: scanning folders, loading and saving matrices."""

import glob
import logging
import os
from typing import List

import numpy as np

from ..core.models import SampleMeta, WarpageData
from ..core.parser import load_warpage, parse_filename

logger = logging.getLogger(__name__)

_SCAN_EXTS = ("*.dat", "*.csv", "*.txt")


def scan_folder(folder: str, kind: str) -> "List[SampleMeta]":
    """Scan a folder for parseable measurement files.

    Files matching *.dat/*.csv/*.txt are parsed; unparseable names are skipped
    with a logged warning. Results are sorted by (sample_no, time_s).

    Args:
        folder: Directory to scan.
        kind: "TOP" | "BTM" | "GAP".

    Returns:
        Sorted list of SampleMeta.
    """
    paths: List[str] = []
    for pattern in _SCAN_EXTS:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
    paths = sorted(set(paths))

    metas: List[SampleMeta] = []
    for path in paths:
        try:
            metas.append(parse_filename(path, kind, path=path))
        except ValueError as exc:
            logger.warning("Skipping unparseable file '%s': %s", path, exc)
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

    with open(path, "w", encoding="utf-8", newline="") as fh:
        for row in arr:
            cells = []
            for v in row:
                if np.isnan(v):
                    cells.append("nan")
                else:
                    cells.append(fmt % v)
            fh.write(delimiter.join(cells))
            fh.write("\n")
