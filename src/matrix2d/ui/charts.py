"""Pure plotly figure-builder functions.

No Dash imports here on purpose: these functions take numpy arrays plus a
:class:`ChartOptions` and return :class:`plotly.graph_objects.Figure` objects.
That keeps them unit-testable without a browser and easy to migrate to any other
front end later.

NaN values in the input arrays are passed straight through to plotly, which
renders them as blank cells natively (no masking required).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


@dataclass
class ChartOptions:
    """Styling / layout options applied to every figure builder."""

    title: str = ""
    font_family: str = "Arial"
    font_size: int = 12
    title_font_size: int = 16
    colorscale: str = "Jet"
    reverse_colorscale: bool = False
    show_colorbar: bool = True
    zmin: Optional[float] = None
    zmax: Optional[float] = None
    tick_font_size: int = 10
    x_tick_step: Optional[float] = None   # dtick on x axis
    y_tick_step: Optional[float] = None   # dtick on y axis
    contour_levels: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _as_2d_float(values) -> np.ndarray:
    """Coerce input to a 2D float64 ndarray (NaN preserved)."""
    arr = np.asarray(values, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _apply_layout(fig: go.Figure, options: ChartOptions, is_3d: bool = False) -> go.Figure:
    """Apply common ChartOptions (title, fonts, ticks, size) to a figure layout."""
    layout_kwargs = dict(
        title=dict(
            text=options.title,
            font=dict(size=options.title_font_size, family=options.font_family),
        ),
        font=dict(family=options.font_family, size=options.font_size),
    )
    if options.width is not None:
        layout_kwargs["width"] = options.width
    if options.height is not None:
        layout_kwargs["height"] = options.height

    fig.update_layout(**layout_kwargs)

    tick_font = dict(size=options.tick_font_size, family=options.font_family)
    if is_3d:
        scene_kwargs = dict(
            xaxis=dict(tickfont=tick_font),
            yaxis=dict(tickfont=tick_font),
            zaxis=dict(tickfont=tick_font),
        )
        if options.x_tick_step is not None:
            scene_kwargs["xaxis"]["dtick"] = options.x_tick_step
        if options.y_tick_step is not None:
            scene_kwargs["yaxis"]["dtick"] = options.y_tick_step
        fig.update_layout(scene=scene_kwargs)
    else:
        x_axis = dict(tickfont=tick_font)
        y_axis = dict(tickfont=tick_font)
        if options.x_tick_step is not None:
            x_axis["dtick"] = options.x_tick_step
        if options.y_tick_step is not None:
            y_axis["dtick"] = options.y_tick_step
        fig.update_xaxes(**x_axis)
        fig.update_yaxes(**y_axis)
    return fig


def _z_bounds(options: ChartOptions):
    """Return (zmin, zmax) honoring explicit options; None means auto."""
    return options.zmin, options.zmax


# ---------------------------------------------------------------------------
# public figure builders
# ---------------------------------------------------------------------------

def contour_2d(values, options: ChartOptions, name: str = "") -> go.Figure:
    """2D filled contour chart (go.Contour). NaN cells render blank."""
    arr = _as_2d_float(values)
    zmin, zmax = _z_bounds(options)

    contours = None
    if options.contour_levels is not None and options.contour_levels > 0:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            lo = zmin if zmin is not None else float(np.min(finite))
            hi = zmax if zmax is not None else float(np.max(finite))
            if hi > lo:
                size = (hi - lo) / float(options.contour_levels)
                contours = dict(start=lo, end=hi, size=size)

    trace = go.Contour(
        z=arr,
        colorscale=options.colorscale,
        reversescale=options.reverse_colorscale,
        showscale=options.show_colorbar,
        zmin=zmin,
        zmax=zmax,
        connectgaps=False,
        name=name,
    )
    if contours is not None:
        trace.contours = contours

    fig = go.Figure(data=[trace])
    return _apply_layout(fig, options, is_3d=False)


def heatmap_2d(values, options: ChartOptions, name: str = "") -> go.Figure:
    """2D heatmap (go.Heatmap). NaN cells render blank."""
    arr = _as_2d_float(values)
    zmin, zmax = _z_bounds(options)

    trace = go.Heatmap(
        z=arr,
        colorscale=options.colorscale,
        reversescale=options.reverse_colorscale,
        showscale=options.show_colorbar,
        zmin=zmin,
        zmax=zmax,
        name=name,
    )
    fig = go.Figure(data=[trace])
    return _apply_layout(fig, options, is_3d=False)


def surface_3d(values, options: ChartOptions, name: str = "", z_offset: float = 0.0) -> go.Figure:
    """3D surface (go.Surface) with an optional constant z offset."""
    arr = _as_2d_float(values)
    z = arr + z_offset
    zmin, zmax = _z_bounds(options)

    trace = go.Surface(
        z=z,
        colorscale=options.colorscale,
        reversescale=options.reverse_colorscale,
        showscale=options.show_colorbar,
        cmin=zmin,
        cmax=zmax,
        name=name,
        showlegend=bool(name),
    )
    fig = go.Figure(data=[trace])
    return _apply_layout(fig, options, is_3d=True)


def multi_surface_3d(items: "List[Tuple[str, np.ndarray, float]]", options: ChartOptions) -> go.Figure:
    """Several go.Surface traces in one scene.

    ``items`` is a list of ``(name, values, z_offset)``. Each surface uses its
    own z (values + offset). Per-trace colorbars are positioned along the right
    edge so they do not overlap, and legend entries are enabled so surfaces can
    be toggled independently.
    """
    fig = go.Figure()
    n = len(items)
    zmin, zmax = _z_bounds(options)

    for i, item in enumerate(items):
        name, values, z_offset = item
        arr = _as_2d_float(values)
        z = arr + z_offset

        surface = go.Surface(
            z=z,
            colorscale=options.colorscale,
            reversescale=options.reverse_colorscale,
            name=name,
            showlegend=True,
            showscale=options.show_colorbar,
            cmin=zmin,
            cmax=zmax,
        )
        if options.show_colorbar and n > 0:
            # spread colorbars horizontally across the right side so they do
            # not stack on top of one another.
            x_pos = 1.02 + (i * 0.08)
            surface.colorbar = dict(
                title=dict(text=name, side="right"),
                x=x_pos,
                len=max(0.3, 1.0 / max(n, 1)),
                thickness=12,
            )
        fig.add_trace(surface)

    fig = _apply_layout(fig, options, is_3d=True)
    fig.update_layout(legend=dict(font=dict(size=options.font_size, family=options.font_family)))
    return fig
