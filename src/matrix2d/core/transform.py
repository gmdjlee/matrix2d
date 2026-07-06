"""Orientation transforms applied to a warpage matrix before gap computation.

Pure numpy. No I/O.

Transforms:
- Left-right flip mirrors the columns AND inverts value signs, modelling the
  same surface viewed from the opposite side (horizontal axis reverses and
  up/down warpage inverts together).
- Rotation is in clockwise 90-degree steps; any integer is accepted and
  normalized modulo 4.
- Zero-cell re-anchoring shifts the whole matrix so one chosen cell becomes
  exactly 0.0; its (row, col) coordinates refer to the POST-flip/rotate
  orientation.

``apply_transform`` applies them in fixed order: flip -> rotate -> zero.
All functions return new arrays and never mutate their input (cache safety).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TransformConfig:
    """Orientation adjustments to apply to one dataset before pairing.

    Attributes:
        flip_lr: Mirror columns left-right AND invert value signs (view the
            surface from the other side).
        rot90_cw: Clockwise 90-degree rotation steps; any integer is accepted
            and normalized modulo 4.
        zero_cell: Optional (row, col) of the cell whose value becomes 0.0;
            coordinates refer to the post-flip/rotate orientation.
    """

    flip_lr: bool = False
    rot90_cw: int = 0
    zero_cell: Optional[Tuple[int, int]] = None


def _as_2d(values: np.ndarray) -> np.ndarray:
    """Coerce input to a float64 array and validate that it is 2D."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must be 2D, got shape {0}".format(arr.shape))
    return arr


def flip_lr_invert(values: np.ndarray) -> np.ndarray:
    """Mirror columns left-right and invert value signs.

    Models viewing the same surface from the opposite side: the horizontal
    axis reverses and the warpage sign inverts.

    Args:
        values: 2D float array (NaN = blank).

    Returns:
        A new float64 array; NaN cells stay NaN. Input is never mutated.

    Raises:
        ValueError: If input is not 2D.
    """
    arr = _as_2d(values)
    # Negating the flipped view allocates a fresh array; NaN survives negation.
    return -np.fliplr(arr)


def rotate90_cw(values: np.ndarray, steps: int) -> np.ndarray:
    """Rotate the matrix by clockwise 90-degree steps.

    Args:
        values: 2D float array (NaN = blank).
        steps: Number of clockwise quarter turns; any integer is accepted and
            normalized modulo 4.

    Returns:
        A new contiguous float64 array (a copy even when steps % 4 == 0).

    Raises:
        ValueError: If input is not 2D.
    """
    arr = _as_2d(values)
    # np.rot90 rotates counter-clockwise for positive k, hence the negation.
    rotated = np.rot90(arr, k=-(steps % 4))
    # np.array(copy=True) guarantees a fresh C-contiguous array even when the
    # rotation is a no-op (np.rot90 returns a view in that case).
    return np.array(rotated, dtype=np.float64, order="C")


def zero_at_cell(values: np.ndarray, row: int, col: int) -> np.ndarray:
    """Shift the whole matrix so the cell at (row, col) becomes exactly 0.0.

    Args:
        values: 2D float array (NaN = blank).
        row: Row index of the anchor cell.
        col: Column index of the anchor cell.

    Returns:
        A new float64 array equal to ``values - values[row, col]``; NaN cells
        elsewhere stay NaN.

    Raises:
        ValueError: If input is not 2D, (row, col) is out of bounds, or the
            anchor cell is blank (NaN).
    """
    arr = _as_2d(values)
    rows, cols = arr.shape
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(
            "zero cell ({0}, {1}) out of bounds for shape {2}".format(
                row, col, arr.shape
            )
        )
    anchor = arr[row, col]
    if np.isnan(anchor):
        raise ValueError(
            "zero cell ({0}, {1}) is blank (NaN); choose a non-blank cell.".format(
                row, col
            )
        )
    return arr - anchor


def apply_transform(
    values: np.ndarray, config: Optional[TransformConfig]
) -> np.ndarray:
    """Apply a TransformConfig in fixed order: flip -> rotate -> zero.

    Args:
        values: 2D float array (NaN = blank).
        config: Transform to apply; None (or an all-default config) means no
            transform.

    Returns:
        A new float64 array. The input is never mutated and the result is
        never the same object as the input, even for a no-op config.

    Raises:
        ValueError: If input is not 2D, or the zero cell is invalid.
    """
    arr = _as_2d(values)
    if config is None:
        return arr.copy()

    out = arr
    if config.flip_lr:
        out = flip_lr_invert(out)
    # rotate90_cw always returns a fresh copy, so the result never aliases the
    # input even when the whole config is a no-op.
    out = rotate90_cw(out, config.rot90_cw)
    if config.zero_cell is not None:
        out = zero_at_cell(out, config.zero_cell[0], config.zero_cell[1])
    return out
