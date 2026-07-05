"""Core data models for warpage analysis.

Pure data containers with no I/O or service dependencies.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SampleMeta:
    """Metadata parsed from a measurement filename.

    Attributes:
        title: Free-form title (letters/digits/spaces).
        sample_no: Sample number (from PTXXXX, leading zeros stripped).
        time_s: Measurement time in seconds.
        temp_c: Temperature in Celsius.
        kind: One of "TOP", "BTM", "GAP".
        path: Source file path (optional).
    """

    title: str
    sample_no: int
    time_s: int
    temp_c: int
    kind: str
    path: str = ""


@dataclass
class WarpageData:
    """A parsed warpage matrix plus its metadata.

    Attributes:
        meta: Associated SampleMeta.
        values: 2D float64 array; NaN marks blank/masked cells.
    """

    meta: SampleMeta
    values: np.ndarray


@dataclass
class GapResult:
    """Result of a gap computation between a TOP and BTM surface.

    Attributes:
        gap: 2D float64 array; NaN marks blank cells. Minimum valid value == 0.0.
        offset: nanmin(top - btm) that was subtracted to bring surfaces to contact.
        contact_index: (row, col) index of the first-contact point.
    """

    gap: np.ndarray
    offset: float
    contact_index: Tuple[int, int]
