"""Matrix resizing with warpage-preserving interpolation.

Pure numpy/scipy. No I/O.

Approach:
- Values are interpolated bilinearly on a normalized [0,1] x [0,1] grid so that
  a flat plane or linear ramp is preserved to machine precision.
- Before interpolating, NaN (blank) cells are filled by nearest-valid values so
  interpolation near a blank edge does not bleed NaN into valid neighbours.
- The blank region is NOT scaled with the values. It is CROPPED: the source
  blank keeps its absolute cell extent and is center-aligned onto the target
  grid (cropped when the target is smaller, padded with valid cells when the
  target is larger). This preserves the blank's real size and shape instead of
  stretching it with the grid.
- When bringing a TOP/BTM pair onto a common grid, both surfaces receive the
  SAME blank: the UNION of each side's center-fit blank ("match to the larger
  blank"). Per dimension the union spans at least the larger of the two blanks
  while keeping each blank's actual shape.
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


def _nearest_fill(values: np.ndarray, mask_valid: np.ndarray) -> np.ndarray:
    """Fill invalid (NaN) cells with their nearest valid neighbour's value.

    Args:
        values: 2D array (may contain NaN).
        mask_valid: 2D bool, True where value is valid.

    Returns:
        A 2D array with all cells finite (assuming at least one valid cell).
    """
    if mask_valid.all():
        return values.astype(np.float64, copy=True)
    # distance_transform_edt on the *invalid* region returns, for each invalid
    # cell, the indices of the nearest valid (zero-distance) cell.
    indices = distance_transform_edt(
        ~mask_valid, return_distances=False, return_indices=True
    )
    filled = values[tuple(indices)]
    return filled.astype(np.float64, copy=True)


def _resize_filled(filled: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a fully-finite array onto a normalized target grid."""
    src_rows, src_cols = filled.shape
    tgt_rows, tgt_cols = target_shape

    # Normalized source coordinates in [0, 1].
    if src_rows > 1:
        src_r = np.linspace(0.0, 1.0, src_rows)
    else:
        src_r = np.array([0.0])
    if src_cols > 1:
        src_c = np.linspace(0.0, 1.0, src_cols)
    else:
        src_c = np.array([0.0])

    interp = RegularGridInterpolator(
        (src_r, src_c), filled, method="linear", bounds_error=False, fill_value=None
    )

    tgt_r = np.linspace(0.0, 1.0, tgt_rows) if tgt_rows > 1 else np.array([0.0])
    tgt_c = np.linspace(0.0, 1.0, tgt_cols) if tgt_cols > 1 else np.array([0.0])
    grid_r, grid_c = np.meshgrid(tgt_r, tgt_c, indexing="ij")
    pts = np.stack([grid_r.ravel(), grid_c.ravel()], axis=-1)
    out = interp(pts).reshape(tgt_rows, tgt_cols)
    return out.astype(np.float64, copy=False)


def _center_span(src_n: int, tgt_n: int) -> Tuple[int, int, int]:
    """Overlap length and start offsets for center-aligning a 1D axis.

    Returns ``(n, src_start, tgt_start)`` such that
    ``dst[tgt_start:tgt_start+n] = src[src_start:src_start+n]`` centers the
    source span within the target (cropping when ``src_n > tgt_n``, padding
    when ``src_n < tgt_n``). Any odd leftover cell goes to the trailing edge.
    """
    n = min(src_n, tgt_n)
    src_start = (src_n - n) // 2
    tgt_start = (tgt_n - n) // 2
    return n, src_start, tgt_start


def _center_fit_mask(
    blank: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """Center-align a boolean blank mask into ``target_shape`` by crop/pad.

    No scaling: the blank keeps its absolute cell extent. Rows/cols beyond the
    target are cropped (centered); missing rows/cols are padded as valid
    (``False``, i.e. not blank), also centered.

    Args:
        blank: 2D bool, True where the source cell is blank.
        target_shape: (rows, cols) of the output mask.

    Returns:
        A 2D bool array of ``target_shape``, True where blank.
    """
    src_rows, src_cols = blank.shape
    tgt_rows, tgt_cols = target_shape
    out = np.zeros((tgt_rows, tgt_cols), dtype=bool)
    n_r, sr0, tr0 = _center_span(src_rows, tgt_rows)
    n_c, sc0, tc0 = _center_span(src_cols, tgt_cols)
    out[tr0:tr0 + n_r, tc0:tc0 + n_c] = blank[sr0:sr0 + n_r, sc0:sc0 + n_c]
    return out


def _validate_resizable(arr: np.ndarray, target_shape: Tuple[int, int]) -> None:
    """Shared input checks for value resizing. Raises ValueError on failure."""
    if arr.ndim != 2:
        raise ValueError("values must be 2D, got shape {0}".format(arr.shape))
    tgt_rows, tgt_cols = target_shape
    if tgt_rows < 1 or tgt_cols < 1:
        raise ValueError("target_shape must be positive, got {0}".format(target_shape))


def resize_values(values: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinearly resize warpage VALUES to ``target_shape`` (no blank applied).

    NaN cells are nearest-filled before interpolation so blank edges do not
    bleed NaN, and the result is fully finite. The caller is responsible for
    applying a blank mask (see ``resize_crop_blank`` / ``resize_pair``).

    Args:
        values: 2D float array; NaN marks blank cells.
        target_shape: (rows, cols) of the output.

    Returns:
        A finite 2D float64 array of ``target_shape``.

    Raises:
        ValueError: If input is not 2D, target_shape is invalid, or all-blank.
    """
    arr = np.asarray(values, dtype=np.float64)
    _validate_resizable(arr, target_shape)
    mask_valid = ~np.isnan(arr)
    if not mask_valid.any():
        raise ValueError("Cannot resize an all-blank (all-NaN) matrix.")
    filled = _nearest_fill(arr, mask_valid)
    return _resize_filled(filled, target_shape)


def resize_crop_blank(
    values: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """Resize VALUES to ``target_shape``; blank is CROPPED, not scaled.

    Values are bilinearly resized (warpage geometry preserved). The source's
    own blank keeps its absolute cell extent and is center-fit (crop/pad) onto
    the target grid, then re-applied.

    Args:
        values: 2D float array; NaN marks blank cells.
        target_shape: (rows, cols) of the output.

    Returns:
        A resized 2D float64 array with NaN in the center-fit blank region.

    Raises:
        ValueError: If input is not 2D, target_shape is invalid, or all-blank.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = resize_values(arr, target_shape)
    blank_fit = _center_fit_mask(np.isnan(arr), target_shape)
    out[blank_fit] = np.nan
    return out


def resize_pair(
    top: np.ndarray, btm: np.ndarray, reference: str
) -> "Tuple[np.ndarray, np.ndarray]":
    """Bring a TOP/BTM pair onto a common grid, matching to the larger blank.

    The reference side's grid is authoritative; the other side's VALUES are
    bilinearly resized onto it. Both sides then receive the SAME blank: the
    UNION of each side's center-fit (cropped, not scaled) blank. This matches
    both surfaces to the larger blank while keeping each blank's actual shape.

    Args:
        top: 2D float array (NaN = blank).
        btm: 2D float array (NaN = blank).
        reference: "TOP" or "BTM" -- whose grid is authoritative.

    Returns:
        ``(top_out, btm_out)``, both on the reference grid, sharing one blank.

    Raises:
        ValueError: On bad ``reference``, non-2D input, or an all-blank side
            that must be resized.
    """
    if reference not in ("TOP", "BTM"):
        raise ValueError(
            "reference must be 'TOP' or 'BTM', got {0!r}".format(reference)
        )
    top_arr = np.asarray(top, dtype=np.float64)
    btm_arr = np.asarray(btm, dtype=np.float64)
    if top_arr.ndim != 2 or btm_arr.ndim != 2:
        raise ValueError(
            "top/btm must be 2D, got {0} and {1}".format(
                top_arr.shape, btm_arr.shape
            )
        )

    if reference == "TOP":
        common = top_arr.shape
        top_vals = top_arr.copy()
        btm_vals = resize_values(btm_arr, common)
        top_blank = np.isnan(top_arr)
        btm_blank = _center_fit_mask(np.isnan(btm_arr), common)
    else:
        common = btm_arr.shape
        btm_vals = btm_arr.copy()
        top_vals = resize_values(top_arr, common)
        btm_blank = np.isnan(btm_arr)
        top_blank = _center_fit_mask(np.isnan(top_arr), common)

    union_blank = top_blank | btm_blank
    top_vals[union_blank] = np.nan
    btm_vals[union_blank] = np.nan
    return top_vals, btm_vals
