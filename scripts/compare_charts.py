"""Render the same data through plotly (charts) and matplotlib (charts_mpl).

Writes side-by-side PNGs into scratch/compare_charts/ so the two backends can be
eyeballed. Prototype tooling for the migration evaluation — not shipped.

    python scripts/compare_charts.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matrix2d.ui import charts, charts_mpl  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "scratch", "compare_charts")
os.makedirs(OUT, exist_ok=True)


def _sample(rows=30, cols=40, blanks=True):
    ys, xs = np.mgrid[0:rows, 0:cols]
    z = (np.sin(xs / 6.0) * np.cos(ys / 5.0) * 5.0
         + (xs / cols) * 3.0).astype("float64")
    if blanks:
        z[0:5, 0:8] = np.nan          # corner blank
        z[rows - 4:, cols - 6:] = np.nan
    return z


def _save_plotly(fig, path):
    try:
        fig.write_image(path)
        return True
    except Exception as e:  # noqa: BLE001
        print("  plotly export FAILED (%s): %s" % (path, e))
        return False


def _save_mpl(fig, path):
    fig.savefig(path)
    return True


def main():
    opts = charts.ChartOptions(title="Sample", colorscale="Jet",
                               contour_levels=10, width=600, height=500)
    z = _sample()
    z2 = _sample(24, 32) + 6.0

    cases = [
        ("contour_2d", lambda m: m.contour_2d(z, opts)),
        ("heatmap_2d", lambda m: m.heatmap_2d(z, opts)),
        ("surface_3d", lambda m: m.surface_3d(z, opts, name="TOP")),
        ("multi_surface_3d",
         lambda m: m.multi_surface_3d([("TOP", z, 0.0), ("BTM", z2, 0.0)], opts)),
        ("effective_gap_chart",
         lambda m: m.effective_gap_chart(
             [{"label": "H25", "avg": 1.2, "std": 0.3},
              {"label": "H85", "avg": 2.4, "std": 0.5},
              {"label": "C85", "avg": 2.1, "std": None},
              {"label": "C25", "avg": 1.0, "std": 0.2}], opts)),
    ]

    for name, build in cases:
        print(name)
        _save_mpl(build(charts_mpl), os.path.join(OUT, name + "_mpl.png"))
        _save_plotly(build(charts), os.path.join(OUT, name + "_plotly.png"))
    print("\nWrote to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
