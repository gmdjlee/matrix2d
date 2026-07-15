"""Unit tests for the matplotlib export builders / reconstruction in
matrix2d.ui.charts_mpl.

These cover the hybrid PNG-export path: the app displays plotly figures but
writes files through matplotlib, so both the direct builders and the
``figure_from_plotly_dict`` reconstruction must produce a saveable Figure.
Run with:  python -m pytest tests/test_charts_mpl.py
"""

import os

import numpy as np
import pytest
from matplotlib.figure import Figure

from matrix2d.ui import charts, charts_mpl


def _z(rows=6, cols=8, blank=True):
    a = np.arange(rows * cols, dtype="float64").reshape(rows, cols)
    if blank:
        a[0, 0] = np.nan
    return a


def _series():
    return [{"label": "H25", "avg": 1.0, "std": 0.2},
            {"label": "C25", "avg": 2.0, "std": None}]


# --- direct builders return a Figure ---------------------------------------

def test_builders_return_figures():
    o = charts_mpl.ChartOptions(title="T", colorscale="Jet")
    z = _z()
    assert isinstance(charts_mpl.contour_2d(z, o), Figure)
    assert isinstance(charts_mpl.heatmap_2d(z, o), Figure)
    assert isinstance(charts_mpl.surface_3d(z, o, name="TOP"), Figure)
    assert isinstance(
        charts_mpl.multi_surface_3d([("TOP", z, 0.0), ("BTM", z + 3, 0.0)], o),
        Figure)
    assert isinstance(charts_mpl.effective_gap_chart(_series(), o), Figure)


# --- save_figure honours plotly-style size kwargs --------------------------

def test_save_figure_writes_png(tmp_path):
    o = charts_mpl.ChartOptions()
    fig = charts_mpl.heatmap_2d(_z(), o)
    p = os.path.join(str(tmp_path), "h.png")
    charts_mpl.save_figure(fig, p, {"width": 400, "height": 300, "scale": 2.0})
    assert os.path.getsize(p) > 0


def test_save_figure_scale_only(tmp_path):
    fig = charts_mpl.contour_2d(_z(), charts_mpl.ChartOptions())
    p = os.path.join(str(tmp_path), "c.png")
    charts_mpl.save_figure(fig, p, {"scale": 1.5})
    assert os.path.getsize(p) > 0


def test_save_figure_empty_kwargs(tmp_path):
    fig = charts_mpl.surface_3d(_z(), charts_mpl.ChartOptions())
    p = os.path.join(str(tmp_path), "s.png")
    charts_mpl.save_figure(fig, p, {})
    assert os.path.getsize(p) > 0


# --- colorscale parity -----------------------------------------------------

def test_cmap_from_rgb_stops():
    stops = [[0.0, "rgb(0,0,131)"], [1.0, "rgb(128,0,0)"]]
    cmap = charts_mpl._cmap_from_stops(stops, reverse=False)
    # low end is the first stop colour
    r, g, b, _ = cmap(0.0)
    assert (round(r, 3), round(g, 3), round(b, 3)) == (0.0, 0.0, round(131 / 255, 3))


def test_cmap_from_stops_reverse_flips_ends():
    stops = [[0.0, "rgb(0,0,131)"], [1.0, "rgb(128,0,0)"]]
    fwd = charts_mpl._cmap_from_stops(stops, reverse=False)
    rev = charts_mpl._cmap_from_stops(stops, reverse=True)
    assert fwd(0.0)[:3] == pytest.approx(rev(1.0)[:3], abs=1e-3)


def test_parse_color_hex_and_rgb():
    assert charts_mpl._parse_color("rgb(255,0,0)") == pytest.approx((1.0, 0.0, 0.0))
    assert charts_mpl._parse_color("#00ff00") == pytest.approx((0.0, 1.0, 0.0))


# --- reconstruction from plotly dicts (each builder round-trips) -----------

@pytest.mark.parametrize("make", [
    lambda o: charts.contour_2d(_z(), o),
    lambda o: charts.heatmap_2d(_z(), o),
    lambda o: charts.surface_3d(_z(), o, name="TOP"),
    lambda o: charts.multi_surface_3d([("TOP", _z(), 0.0), ("BTM", _z() + 3, 0.0)], o),
    lambda o: charts.effective_gap_chart(_series(), o),
])
def test_reconstruct_from_plotly_dict(make, tmp_path):
    o = charts.ChartOptions(title="R", colorscale="Turbo",
                            reverse_colorscale=True, zmin=0.0, zmax=10.0)
    d = make(o).to_dict()
    fig = charts_mpl.figure_from_plotly_dict(d)
    assert isinstance(fig, Figure)
    p = os.path.join(str(tmp_path), "r.png")
    charts_mpl.save_figure(fig, p, {})
    assert os.path.getsize(p) > 0


def test_reconstruct_preserves_title_without_double_suffix():
    # plotly title already carries the "rows×cols" suffix; reconstruction must
    # not append it a second time.
    o = charts.ChartOptions(title="Sample", show_shape=True)
    d = charts.heatmap_2d(_z(6, 8), o).to_dict()
    assert d["layout"]["title"]["text"] == "Sample (6×8)"
    opts = charts_mpl._recon_options(d)
    assert opts.title == "Sample (6×8)"
    assert opts.show_shape is False


def test_reconstruct_empty_figure_raises():
    with pytest.raises(ValueError):
        charts_mpl.figure_from_plotly_dict({"data": []})


def test_reconstruct_unknown_type_raises():
    with pytest.raises(ValueError):
        charts_mpl.figure_from_plotly_dict({"data": [{"type": "pie"}]})


def test_z_array_maps_none_to_nan():
    arr = charts_mpl._z_array([[1.0, None], [None, 4.0]])
    assert np.isnan(arr[0, 1]) and np.isnan(arr[1, 0])
    assert arr[0, 0] == 1.0 and arr[1, 1] == 4.0
