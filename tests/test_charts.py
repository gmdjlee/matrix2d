"""Unit tests for the pure plotly figure builders in matrix2d.ui.charts.

No browser / Dash needed. Run with:  python -m pytest tests/test_charts.py
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from matrix2d.ui.charts import (
    ChartOptions,
    contour_2d,
    heatmap_2d,
    surface_3d,
    multi_surface_3d,
)


@pytest.fixture
def sample():
    return np.array(
        [[1.0, 2.0, 3.0],
         [4.0, np.nan, 6.0],
         [7.0, 8.0, 9.0]],
        dtype="float64",
    )


# ---------------------------------------------------------------------------
# return types
# ---------------------------------------------------------------------------

def test_contour_returns_figure(sample):
    fig = contour_2d(sample, ChartOptions())
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Contour)


def test_heatmap_returns_figure(sample):
    fig = heatmap_2d(sample, ChartOptions())
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Heatmap)


def test_surface_returns_figure(sample):
    fig = surface_3d(sample, ChartOptions())
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Surface)


def test_multi_surface_returns_figure(sample):
    items = [("a", sample, 0.0), ("b", sample, 5.0)]
    fig = multi_surface_3d(items, ChartOptions())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    for tr in fig.data:
        assert isinstance(tr, go.Surface)


# ---------------------------------------------------------------------------
# options applied
# ---------------------------------------------------------------------------

def test_font_and_title_applied(sample):
    opts = ChartOptions(title="My Title", font_size=20, title_font_size=28,
                        font_family="Times New Roman")
    fig = contour_2d(sample, opts)
    assert fig.layout.title.text == "My Title"
    assert fig.layout.font.size == 20
    assert fig.layout.font.family == "Times New Roman"
    assert fig.layout.title.font.size == 28


def test_colorscale_and_reverse_applied(sample):
    opts = ChartOptions(colorscale="Viridis", reverse_colorscale=True)
    fig = heatmap_2d(sample, opts)
    tr = fig.data[0]
    # plotly normalizes named scales into (position, color) tuples
    assert tr.colorscale is not None
    assert tr.reversescale is True


def test_dtick_applied_2d(sample):
    opts = ChartOptions(x_tick_step=2.0, y_tick_step=3.0, tick_font_size=9)
    fig = contour_2d(sample, opts)
    assert fig.layout.xaxis.dtick == 2.0
    assert fig.layout.yaxis.dtick == 3.0
    assert fig.layout.xaxis.tickfont.size == 9


def test_dtick_applied_3d(sample):
    opts = ChartOptions(x_tick_step=4.0, y_tick_step=1.0)
    fig = surface_3d(sample, opts)
    assert fig.layout.scene.xaxis.dtick == 4.0
    assert fig.layout.scene.yaxis.dtick == 1.0


def test_zmin_zmax_applied(sample):
    opts = ChartOptions(zmin=-1.0, zmax=10.0)
    fig = contour_2d(sample, opts)
    assert fig.data[0].zmin == -1.0
    assert fig.data[0].zmax == 10.0


def test_colorbar_toggle(sample):
    fig_on = heatmap_2d(sample, ChartOptions(show_colorbar=True))
    fig_off = heatmap_2d(sample, ChartOptions(show_colorbar=False))
    assert fig_on.data[0].showscale is True
    assert fig_off.data[0].showscale is False


def test_size_applied(sample):
    opts = ChartOptions(width=800, height=600)
    fig = contour_2d(sample, opts)
    assert fig.layout.width == 800
    assert fig.layout.height == 600


def test_contour_levels_applied(sample):
    opts = ChartOptions(contour_levels=5, zmin=0.0, zmax=10.0)
    fig = contour_2d(sample, opts)
    contours = fig.data[0].contours
    assert contours.start == 0.0
    assert contours.end == 10.0
    assert contours.size == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_does_not_raise(sample):
    # every builder should accept NaN without error
    contour_2d(sample, ChartOptions())
    heatmap_2d(sample, ChartOptions())
    surface_3d(sample, ChartOptions())
    multi_surface_3d([("x", sample, 0.0)], ChartOptions())
    # NaN preserved in the z data
    assert np.isnan(np.asarray(contour_2d(sample, ChartOptions()).data[0].z, dtype="float64")).any()


def test_all_nan_input(sample):
    nan_arr = np.full((3, 3), np.nan)
    fig = contour_2d(nan_arr, ChartOptions(contour_levels=5))
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# offsets
# ---------------------------------------------------------------------------

def test_surface_offset_applied():
    arr = np.zeros((2, 2))
    fig = surface_3d(arr, ChartOptions(), z_offset=7.0)
    z = np.asarray(fig.data[0].z, dtype="float64")
    assert np.allclose(z, 7.0)


def test_multi_surface_offsets_applied():
    arr = np.zeros((2, 2))
    items = [("a", arr, 1.0), ("b", arr, 2.0), ("c", arr, 3.0)]
    fig = multi_surface_3d(items, ChartOptions())
    z0 = np.asarray(fig.data[0].z, dtype="float64")
    z1 = np.asarray(fig.data[1].z, dtype="float64")
    z2 = np.asarray(fig.data[2].z, dtype="float64")
    assert np.allclose(z0, 1.0)
    assert np.allclose(z1, 2.0)
    assert np.allclose(z2, 3.0)


def test_multi_surface_names_and_legend():
    arr = np.ones((2, 2))
    items = [("top", arr, 0.0), ("btm", arr, 0.0)]
    fig = multi_surface_3d(items, ChartOptions())
    names = [tr.name for tr in fig.data]
    assert names == ["top", "btm"]
    assert all(tr.showlegend for tr in fig.data)
