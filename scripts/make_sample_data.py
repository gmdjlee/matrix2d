"""Generate deterministic demo warpage data.

Creates demo_data/{TOP,BTM,GAP,OUT}. TOP and BTM samples differ in grid size
(TOP 60x80, BTM 50x70). Two TOP samples x two BTM samples, temperatures 25 and
240, with heating (H) and cooling (C) times around a peak at 260C. Surfaces are
bowl/saddle shapes with amplitude ~ +/-50, a rectangular blank hole in the
middle, and blank corners. Written as comma-separated .dat with a few 2000+
sentinel values marking blanks.

The GAP folder is populated by running the real gap pipeline over the demo
TOP/BTM data, so its files use the gap naming format
``TOP{n}-BTM{m}_{H|C}{temp}.txt``.

Run:  python scripts/make_sample_data.py
"""

import os

import numpy as np

# Ensure the package is importable when run directly from repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in os.sys.path:
    os.sys.path.insert(0, _SRC)

SEED = 20260705


def _surface(rows, cols, kind, sample_no, temp_c, rng):
    """Build a bowl/saddle warpage surface with amplitude ~ +/-50."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    y = (yy / (rows - 1)) * 2.0 - 1.0  # [-1, 1]
    x = (xx / (cols - 1)) * 2.0 - 1.0  # [-1, 1]

    # Bowl for TOP, saddle for BTM, with sample/temp-dependent variation.
    if kind == "TOP":
        base = 45.0 * (x ** 2 + y ** 2) - 30.0
    else:
        base = 40.0 * (x ** 2 - y ** 2)

    base = base + 5.0 * sample_no + 0.05 * temp_c
    noise = rng.normal(0.0, 1.5, size=(rows, cols))
    surf = base + noise
    # Round to 1-2 decimals.
    return np.round(surf, 2)


def _apply_blanks(surf, rng):
    """Punch a rectangular hole in the middle and blank the four corners.

    Blanks are encoded either as empty cells or 2000+ sentinel values.
    Returns (values_with_nan, sentinel_mask) where sentinel_mask marks cells to
    be written as a 2000+ number rather than an empty string.
    """
    rows, cols = surf.shape
    vals = surf.astype(np.float64).copy()

    # Middle rectangular hole (~20% of each dimension, centered).
    rh = max(2, rows // 5)
    ch = max(2, cols // 5)
    r0 = (rows - rh) // 2
    c0 = (cols - ch) // 2
    vals[r0 : r0 + rh, c0 : c0 + ch] = np.nan

    # Blank corners (small squares).
    cs = max(2, min(rows, cols) // 10)
    vals[0:cs, 0:cs] = np.nan
    vals[0:cs, cols - cs : cols] = np.nan
    vals[rows - cs : rows, 0:cs] = np.nan
    vals[rows - cs : rows, cols - cs : cols] = np.nan

    # Choose a few blank cells to encode as 2000+ sentinels instead of empty.
    blank_idx = np.argwhere(np.isnan(vals))
    sentinel_mask = np.zeros(surf.shape, dtype=bool)
    if len(blank_idx) > 0:
        n_sent = min(5, len(blank_idx))
        pick = rng.choice(len(blank_idx), size=n_sent, replace=False)
        for p in pick:
            r, c = blank_idx[p]
            sentinel_mask[r, c] = True
    return vals, sentinel_mask


def _write_dat(path, vals, sentinel_mask, rng):
    """Write a matrix as comma-separated .dat; blanks empty or 2000+."""
    rows, cols = vals.shape
    lines = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            v = vals[r, c]
            if np.isnan(v):
                if sentinel_mask[r, c]:
                    cells.append("{0:.1f}".format(2000.0 + rng.uniform(0, 50)))
                else:
                    cells.append("")  # empty blank cell
            else:
                cells.append("{0:.2f}".format(v))
        lines.append(",".join(cells))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def _filename(title, sample_no, time_s, temp_c):
    return "{0}_PT{1:04d}_{2:06d}s({3}C).dat".format(title, sample_no, time_s, temp_c)


def main():
    rng = np.random.default_rng(SEED)

    root = os.path.join(os.path.dirname(_HERE), "demo_data")
    dirs = {k: os.path.join(root, k) for k in ("TOP", "BTM", "GAP", "OUT")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Temperature/time schedule: heat up to peak (260C at 100s) then cool.
    # Files at temps 25 and 240 exist both during heating (H) and cooling (C).
    #   H: 25C at 11s, 240C at 60s
    #   peak: 260C at 100s
    #   C: 240C at 150s, 25C at 192s
    schedule = [
        (11, 25),
        (60, 240),
        (100, 260),
        (150, 240),
        (192, 25),
    ]

    top_shape = (60, 80)
    btm_shape = (50, 70)

    for kind, shape, title in (
        ("TOP", top_shape, "WAFER TOP"),
        ("BTM", btm_shape, "WAFER BTM"),
    ):
        for sample_no in (1, 2):
            for time_s, temp_c in schedule:
                surf = _surface(shape[0], shape[1], kind, sample_no, temp_c, rng)
                vals, sent = _apply_blanks(surf, rng)
                fname = _filename(title, sample_no, time_s, temp_c)
                path = os.path.join(dirs[kind], fname)
                _write_dat(path, vals, sent, rng)

    # Populate GAP with real pipeline output (gap naming: TOPn-BTMm_H|Ctemp).
    from matrix2d.services.pipeline import run_pipeline

    gap_results = run_pipeline(dirs["TOP"], dirs["BTM"], dirs["GAP"])

    print("Demo data written under: {0}".format(root))
    for k in ("TOP", "BTM"):
        n = len([f for f in os.listdir(dirs[k]) if f.endswith(".dat")])
        print("  {0}: {1} files".format(k, n))
    print("  GAP: {0} files (pipeline output)".format(len(gap_results)))


if __name__ == "__main__":
    main()
