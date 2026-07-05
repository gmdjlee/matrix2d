"""Gap computation between a TOP and a BTM warpage surface.

Pure numpy. No I/O.

The two surfaces are brought together until first contact (subtracting the
minimum of top-btm over overlapping valid cells), then the gap at every point
is reported. The minimum valid gap is exactly 0.0 at the contact point.
"""

from typing import Tuple

import numpy as np

from .models import GapResult


def compute_gap(top: np.ndarray, btm: np.ndarray) -> GapResult:
    """Compute the contact gap between two aligned warpage surfaces.

    Args:
        top: 2D float array (NaN = blank).
        btm: 2D float array (NaN = blank), same shape as ``top``.

    Returns:
        A GapResult whose ``gap`` has minimum valid value 0.0, ``offset`` is the
        subtracted first-contact offset, and ``contact_index`` is the (row, col)
        of the first-contact point.

    Raises:
        ValueError: If shapes differ or there are no overlapping valid cells.
    """
    top_arr = np.asarray(top, dtype=np.float64)
    btm_arr = np.asarray(btm, dtype=np.float64)
    if top_arr.shape != btm_arr.shape:
        raise ValueError(
            "Shape mismatch: top {0} != btm {1}".format(top_arr.shape, btm_arr.shape)
        )

    diff = top_arr - btm_arr  # NaN where either input is NaN
    valid = ~np.isnan(diff)
    if not valid.any():
        raise ValueError("No overlapping valid cells between top and btm.")

    offset = float(np.nanmin(diff))

    # Locate first-contact point (the minimum of diff over valid cells).
    masked = np.where(valid, diff, np.inf)
    flat_idx = int(np.argmin(masked))
    contact_index = np.unravel_index(flat_idx, diff.shape)
    contact_index = (int(contact_index[0]), int(contact_index[1]))

    gap = diff - offset  # NaN preserved

    return GapResult(gap=gap, offset=offset, contact_index=contact_index)
