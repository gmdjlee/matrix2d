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


def _multi_items():
    """Three surfaces with distinct value ranges, z offsets and a blank cell."""
    return [("TOP PT0001 H240C (6×8)", _z() / 10.0, 0.0),
            ("BTM PT0008 H240C (6×8)", _z() / 5.0 + 100.0, 40.0),
            ("GAP TOP1-BTM8 (6×8)", _z() + 3.0, 90.0)]


def _cbar_axes(fig):
    """Colorbar axes of *fig* (everything that is not the 3D plot axes)."""
    return [a for a in fig.axes if a.name != "3d"]


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


# --- multi-surface colorbar modes + slot layout ----------------------------

def test_multi_surface_shared_has_one_colorbar_and_legend():
    fig = charts_mpl.multi_surface_3d(_multi_items(),
                                      charts_mpl.ChartOptions(title="M"))
    # main 3D axes + exactly one shared colorbar
    assert len(fig.axes) == 2
    assert len(_cbar_axes(fig)) == 1
    legend = fig.axes[0].get_legend()
    assert legend is not None
    names = [t.get_text() for t in legend.get_texts()]
    assert names == [nm for nm, _v, _o in _multi_items()]
    # the colorbar slot is added to the figure, not carved out of the plot
    assert fig.get_size_inches()[0] > 7.0


def test_multi_surface_per_item_colorbars_do_not_overlap():
    # regression: fraction/pad colorbars used to pile up on one another
    items = _multi_items()
    fig = charts_mpl.multi_surface_3d(
        items, charts_mpl.ChartOptions(shared_colorbar=False))
    caxes = _cbar_axes(fig)
    assert len(fig.axes) == 4 and len(caxes) == 3

    boxes = sorted((a.get_position() for a in caxes), key=lambda b: b.x0)
    for prev, nxt in zip(boxes, boxes[1:]):
        assert prev.x1 <= nxt.x0 + 1e-9

    labels = sorted(a.get_ylabel() for a in caxes)
    assert labels == sorted(nm for nm, _v, _o in items)


def test_multi_surface_explicit_width_is_not_grown():
    fig = charts_mpl.multi_surface_3d(
        _multi_items(),
        charts_mpl.ChartOptions(shared_colorbar=False, width=900))
    assert fig.get_size_inches()[0] == pytest.approx(9.0)


def test_multi_surface_no_colorbar_keeps_default_width():
    fig = charts_mpl.multi_surface_3d(
        _multi_items(), charts_mpl.ChartOptions(show_colorbar=False))
    assert _cbar_axes(fig) == []
    assert fig.get_size_inches()[0] == pytest.approx(7.0)
    assert fig.axes[0].get_legend() is not None  # names still identifiable


def test_multi_surface_shared_all_nan_item(tmp_path):
    fig = charts_mpl.multi_surface_3d(
        [("dead", np.full((4, 5), np.nan), 0.0)], charts_mpl.ChartOptions())
    p = os.path.join(str(tmp_path), "nan.png")
    charts_mpl.save_figure(fig, p, {})
    assert os.path.getsize(p) > 0


# --- 3D tick parity (plotly scene dticks) ----------------------------------

def test_style_ticks_3d_applies_tick_steps():
    fig = charts_mpl.surface_3d(
        _z(6, 8), charts_mpl.ChartOptions(x_tick_step=2, y_tick_step=3))
    ax = fig.axes[0]

    def visible(ticks, lim):
        return [t for t in ticks if lim[0] <= t <= lim[1]]

    xs = visible(ax.get_xticks(), ax.get_xlim())
    ys = visible(ax.get_yticks(), ax.get_ylim())
    assert len(xs) >= 2 and np.allclose(np.diff(xs), 2.0)
    assert len(ys) >= 2 and np.allclose(np.diff(ys), 3.0)


# --- reconstruction of multi-surface plotly figures ------------------------

def test_reconstruct_shared_keeps_offset_surfaces_in_view(tmp_path):
    # cmin/cmax span the RAW values in shared mode; using them as the z range
    # clipped every offset surface out of the plot.
    items = [("TOP", _z() / 20.0, 0.0), ("BTM", _z() / 20.0, 50.0)]
    d = charts.multi_surface_3d(items, charts.ChartOptions(title="S")).to_dict()
    fig = charts_mpl.figure_from_plotly_dict(d)

    top = max(float(np.nanmax(v)) + off for _n, v, off in items)
    assert fig.axes[0].get_zlim()[1] >= top - 1e-6
    assert len(_cbar_axes(fig)) == 1
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["TOP", "BTM"]

    p = os.path.join(str(tmp_path), "shared.png")
    charts_mpl.save_figure(fig, p, {})
    assert os.path.getsize(p) > 0


def test_reconstruct_shared_honours_scene_z_range():
    items = [("TOP", _z() / 20.0, 0.0), ("BTM", _z() / 20.0, 50.0)]
    d = charts.multi_surface_3d(
        items, charts.ChartOptions(zmin=-5.0, zmax=60.0)).to_dict()
    fig = charts_mpl.figure_from_plotly_dict(d)
    assert fig.axes[0].get_zlim() == pytest.approx((-5.0, 60.0))


def test_reconstruct_per_item_colorbars_do_not_overlap(tmp_path):
    items = [("TOP", _z(), 0.0), ("BTM", _z() + 3, 5.0), ("GAP", _z() + 9, 10.0)]
    d = charts.multi_surface_3d(
        items, charts.ChartOptions(shared_colorbar=False)).to_dict()
    fig = charts_mpl.figure_from_plotly_dict(d)

    caxes = _cbar_axes(fig)
    assert len(caxes) == len(items)
    boxes = sorted((a.get_position() for a in caxes), key=lambda b: b.x0)
    for prev, nxt in zip(boxes, boxes[1:]):
        assert prev.x1 <= nxt.x0 + 1e-9
    assert sorted(a.get_ylabel() for a in caxes) == ["BTM", "GAP", "TOP"]

    p = os.path.join(str(tmp_path), "per.png")
    charts_mpl.save_figure(fig, p, {})
    assert os.path.getsize(p) > 0
