"""Matrix resizing with warpage-preserving interpolation.

Pure numpy/scipy. No I/O.

Approach:
- Values are interpolated bilinearly on a normalized [0,1] x [0,1] grid so that
  a flat plane or linear ramp is preserved to machine precision.
- Before interpolating, NaN (blank) cells are filled by nearest-valid values so
  interpolation near a blank edge does not bleed NaN into valid neighbours.
- The blank mask is resized independently by block: a target cell is blank if
  ANY source cell in its footprint is blank, so blanks never shrink; it is
  then re-applied to the interpolated values.
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


def _blank_bins(src_n, tgt_n):
    """Per target-index inclusive source-index range ``[lo, hi]``.

    Each target cell pools every source cell whose center falls in its
    footprint (always >= 1 cell). Downscale -> wide ranges so a blank grows to
    its bigger extent; upscale -> a single nearest source cell (blank
    replicated). Returns a list of (lo, hi) inclusive index pairs.
    """
    if tgt_n <= 1:
        return [(0, src_n - 1)]
    if src_n <= 1:
        return [(0, 0)] * tgt_n
    centers = np.linspace(0.0, src_n - 1, tgt_n)
    mids = (centers[:-1] + centers[1:]) / 2.0
    los = [0]
    his = []
    for m in mids:
        cut = int(np.floor(m))
        his.append(cut)
        los.append(cut + 1)
    his.append(src_n - 1)
    ranges = []
    for i in range(tgt_n):
        lo, hi = los[i], his[i]
        if hi < lo:  # upscale: empty bin -> nearest source cell
            near = int(round(centers[i]))
            lo = hi = near
        ranges.append((max(0, lo), min(src_n - 1, hi)))
    return ranges


def _resize_mask(mask_valid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a boolean valid-mask, preserving blanks at their bigger extent.

    A target cell is blank (invalid) if ANY source cell in its footprint is
    blank ("bigger blank wins"): blanks never shrink away, and their shape is
    kept by block/crop mapping rather than proportional point-sampling.
    """
    src_rows, src_cols = mask_valid.shape
    tgt_rows, tgt_cols = target_shape
    invalid = ~mask_valid
    r_ranges = _blank_bins(src_rows, tgt_rows)
    c_ranges = _blank_bins(src_cols, tgt_cols)
    out_valid = np.ones((tgt_rows, tgt_cols), dtype=bool)
    for i, (rlo, rhi) in enumerate(r_ranges):
        row_block = invalid[rlo:rhi + 1, :]
        if not row_block.any():
            continue
        for j, (clo, chi) in enumerate(c_ranges):
            if row_block[:, clo:chi + 1].any():
                out_valid[i, j] = False
    return out_valid


def resize_matrix(values: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a warpage matrix to ``target_shape`` preserving warpage geometry.

    NaN cells are treated as blank: they are filled by nearest-valid before
    interpolation, and the blank mask is resized by block (any source blank
    -> blank) and re-applied so blanks stay blank in the output.

    Args:
        values: 2D float array; NaN marks blank cells.
        target_shape: (rows, cols) of the output.

    Returns:
        A resized 2D float64 array with NaN in the resized blank region.

    Raises:
        ValueError: If input is not 2D, target_shape is invalid, or all-blank.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must be 2D, got shape {0}".format(arr.shape))
    tgt_rows, tgt_cols = target_shape
    if tgt_rows < 1 or tgt_cols < 1:
        raise ValueError("target_shape must be positive, got {0}".format(target_shape))

    mask_valid = ~np.isnan(arr)
    if not mask_valid.any():
        raise ValueError("Cannot resize an all-blank (all-NaN) matrix.")

    filled = _nearest_fill(arr, mask_valid)
    resized_vals = _resize_filled(filled, (tgt_rows, tgt_cols))
    resized_mask = _resize_mask(mask_valid, (tgt_rows, tgt_cols))

    out = resized_vals
    out[~resized_mask] = np.nan
    return out


def resize_to_reference(
    data: np.ndarray, reference: np.ndarray, mask_mode: str = "reference"
) -> np.ndarray:
    """Resize ``data`` to ``reference.shape`` and reconcile blank masks.

    Args:
        data: 2D array to resize (NaN = blank).
        reference: 2D array whose shape (and, in "reference" mode, blank mask)
            is authoritative.
        mask_mode:
            "reference" -> final blank mask = reference_mask | resized_data_mask
                           (cells blank in either become blank).
            "own"       -> keep only the resized data mask.

    Returns:
        The resized data with the reconciled blank mask applied.

    Raises:
        ValueError: On invalid dimensions or unknown mask_mode.
    """
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError("reference must be 2D, got shape {0}".format(ref.shape))
    if mask_mode not in ("reference", "own"):
        raise ValueError(
            "mask_mode must be 'reference' or 'own', got {0!r}".format(mask_mode)
        )

    resized = resize_matrix(data, ref.shape)

    if mask_mode == "own":
        return resized

    # "reference": blank where reference is blank OR resized data is blank.
    ref_blank = np.isnan(ref)
    resized_blank = np.isnan(resized)
    final_blank = ref_blank | resized_blank
    resized[final_blank] = np.nan
    return resized
