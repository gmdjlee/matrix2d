"""Core data models for warpage analysis.

Pure data containers with no I/O or service dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SampleMeta:
    """Metadata parsed from a measurement filename.

    Attributes:
        title: Free-form title (letters/digits/spaces).
        sample_no: Sample number (from PTXXXX, leading zeros stripped).
            For GAP files named ``TOP{n}-BTM{m}_...`` this is the TOP number.
        time_s: Measurement time in seconds (0 for gap-named files, which
            carry no time).
        temp_c: Temperature in Celsius.
        kind: One of "TOP", "BTM", "GAP".
        path: Source file path (optional).
        btm_no: BTM sample number for gap-named files, else None.
        phase: Explicit "H"/"C" phase for gap-named files, else None
            (phase is then derived from the sample's peak time).
    """

    title: str
    sample_no: int
    time_s: int
    temp_c: int
    kind: str
    path: str = ""
    btm_no: Optional[int] = None
    phase: Optional[str] = None


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
