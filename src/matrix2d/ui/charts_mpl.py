"""Matplotlib figure-builder prototype — API-compatible with :mod:`charts`.

PROTOTYPE (migration evaluation). These functions mirror the plotly builders in
``charts.py`` one-for-one, but return :class:`matplotlib.figure.Figure` objects
instead of ``plotly.graph_objects.Figure``. Nothing in the running app imports
this module yet; it exists so matplotlib output can be compared side-by-side
with the current plotly output before committing to a full swap.

Design notes / deliberate parity choices vs. plotly:

* No pyplot. Figures are built through the object API (``Figure`` +
  ``FigureCanvasAgg``) so builders stay thread-safe for the parallel export
  pool (pyplot's global state is not).
* NaN cells render transparent (``cmap.set_bad(alpha=0)``) to mirror plotly's
  blank-cell behaviour.
* ``origin="lower"`` on 2D images so row 0 sits at the bottom, matching plotly
  heatmap/contour orientation.
* Multi-surface figures lay their colorbars out in fixed-width slots down the
  right edge (:func:`_slot_colorbars`) and GROW the figure to fit them, instead
  of letting each ``fraction``/``pad`` colorbar steal width from the axes —
  which crammed the bars and their vertical labels on top of one another as
  soon as there were two surfaces.
* ``ChartOptions`` is reused unchanged from :mod:`charts`; sizes given in pixels
  are converted to inches at ``_DPI``.
"""

from typing import List, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib import cm as mcm
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from .charts import ChartOptions, _as_2d_float, _with_shape

# plotly colorscale name -> matplotlib colormap name. Reverse handled by "_r".
_CMAP = {
    "Jet": "jet",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "RdBu": "RdBu",
    "Turbo": "turbo",
    "Greys": "Greys",
}

_DPI = 100
_DEFAULT_2D_IN = (7.0, 6.0)
_DEFAULT_3D_IN = (7.0, 6.0)

# Figure inches reserved per colorbar in the slot layout: the slim bar itself
# (_CBAR_BAR_IN) plus room for its tick labels and one vertical name label.
_CBAR_SLOT_IN = 0.78
_CBAR_BAR_IN = 0.14


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _named_cmap(name: str):
    cmap = matplotlib.colormaps[name].copy()
    cmap.set_bad(alpha=0.0)  # NaN -> transparent (plotly blank parity)
    return cmap


def _make_cmap(options: ChartOptions):
    base = _CMAP.get(options.colorscale, "jet")
    return _named_cmap(base + "_r" if options.reverse_colorscale else base)


def _parse_color(c) -> tuple:
    """Parse a plotly colorscale color ('rgb(r,g,b)' or '#hex') to 0-1 RGB."""
    if isinstance(c, (list, tuple)):
        return tuple(float(v) for v in c[:3])
    s = str(c).strip()
    if s.startswith("rgb"):
        nums = s[s.index("(") + 1:s.index(")")].split(",")
        return tuple(float(n) / 255.0 for n in nums[:3])
    return mcolors.to_rgb(s)


def _cmap_from_stops(colorscale, reverse: bool):
    """Build a matplotlib colormap from a plotly colorscale.

    ``colorscale`` is either a registered name (fallback) or plotly's resolved
    ``[[pos, 'rgb(...)'], ...]`` stop list. Using the stop list reproduces
    plotly's exact colors, so exported PNGs match the on-screen figure.
    """
    if isinstance(colorscale, str):
        base = _CMAP.get(colorscale, colorscale.lower())
        try:
            return _named_cmap(base + "_r" if reverse else base)
        except KeyError:
            return _named_cmap("jet_r" if reverse else "jet")
    stops = [[float(p), _parse_color(c)] for p, c in (colorscale or [])]
    if not stops:
        return _named_cmap("jet_r" if reverse else "jet")
    if reverse:
        stops = [[1.0 - p, c] for p, c in stops][::-1]
    cmap = mcolors.LinearSegmentedColormap.from_list("plotly", stops)
    cmap.set_bad(alpha=0.0)
    return cmap


def _figsize(options: ChartOptions, default):
    w = options.width / _DPI if options.width else default[0]
    h = options.height / _DPI if options.height else default[1]
    return (w, h)


def _new_figure(options: ChartOptions, default) -> Figure:
    fig = Figure(figsize=_figsize(options, default), dpi=_DPI)
    FigureCanvasAgg(fig)  # attach an Agg canvas so savefig works off-pyplot
    return fig


def _apply_title(ax, options: ChartOptions, title: Optional[str]):
    text = options.title if title is None else title
    if text:
        ax.set_title(text, fontsize=options.title_font_size,
                     fontfamily=options.font_family)


def _style_ticks_2d(ax, options: ChartOptions):
    ax.tick_params(labelsize=options.tick_font_size)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(options.font_family)
    if options.x_tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(options.x_tick_step))
    if options.y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(options.y_tick_step))


def _style_ticks_3d(ax, options: ChartOptions):
    """Tick styling for a 3D axes: dtick parity + label size/family.

    Mirrors what :func:`charts._apply_layout` puts on the plotly scene axes:
    x = columns, y = rows (z is left on autorange). Locators go on first so the
    Text objects we hand the font family to are the final ones.
    """
    if options.x_tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(options.x_tick_step))
    if options.y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(options.y_tick_step))
    ax.tick_params(labelsize=options.tick_font_size)
    for lbl in (ax.get_xticklabels() + ax.get_yticklabels()
                + ax.get_zticklabels()):
        lbl.set_fontfamily(options.font_family)


def _colorbar(fig, ax, mappable, options: ChartOptions, label: str = ""):
    if not options.show_colorbar:
        return
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=options.tick_font_size)
    if label:
        cb.set_label(label, fontsize=options.font_size,
                     fontfamily=options.font_family)


def _slot_colorbars(fig, ax, specs, options: ChartOptions):
    """Place N colorbars in fixed-width slots along the right edge of *fig*.

    ``specs`` is a list of ``(mappable, label)``; an empty label means no name
    text (the single shared bar). Each slot is ``_CBAR_SLOT_IN`` figure inches
    wide, so bars, tick labels and the vertical name label always clear one
    another no matter how many surfaces there are.

    When the caller did not pin ``options.width`` the figure GROWS by one slot
    per colorbar, keeping the plot area at its normal size; with an explicit
    width the slots are carved out of it instead (the axes shrinks).

    The caller must NOT run ``fig.tight_layout()`` afterwards — it would
    discard this placement.
    """
    n = len(specs)
    w_in, h_in = fig.get_size_inches()
    if options.width is None and n:
        w_in = _DEFAULT_3D_IN[0] + n * _CBAR_SLOT_IN
        fig.set_size_inches(w_in, h_in)

    slot = _CBAR_SLOT_IN / float(w_in)          # slot width, figure fraction
    # y0 low + tall box leaves headroom for the title; Axes3D draws its cube
    # well inside the rect, so a full-bleed x span clips nothing.
    ax.set_position([0.0, 0.02, max(0.15, 1.0 - n * slot), 0.88])

    bar_w = _CBAR_BAR_IN / float(w_in)
    for i, (mappable, label) in enumerate(specs):
        x0 = 1.0 - (n - i) * slot + 0.1 * slot  # small gap between slots
        cax = fig.add_axes([x0, 0.14, bar_w, 0.66])
        cb = fig.colorbar(mappable, cax=cax)
        cb.ax.tick_params(labelsize=options.tick_font_size)
        if label:
            cb.set_label(label, fontsize=options.font_size,
                         fontfamily=options.font_family)


def _color_bounds(arrays, lo, hi):
    """Resolve color bounds: explicit *lo*/*hi* win, else the finite data range.

    Falls back to 0..1 when nothing is finite and pads a degenerate (equal or
    inverted) range so ``Normalize`` stays valid.
    """
    if lo is None or hi is None:
        mins = []
        maxs = []
        for arr in arrays:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                mins.append(float(finite.min()))
                maxs.append(float(finite.max()))
        if lo is None:
            lo = min(mins) if mins else 0.0
        if hi is None:
            hi = max(maxs) if maxs else 1.0
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        lo, hi = lo - 0.5, lo + 0.5
    return lo, hi


def _facecolors(arr, cmap, norm):
    """(rows, cols, 4) RGBA for ``plot_surface(facecolors=...)``.

    Colors come from the RAW values, so a vertical z offset cannot shift them.
    Blank cells land on the colormap's transparent 'bad' color.
    """
    return cmap(norm(np.ma.masked_invalid(arr)))


def _proxy_patch(name, arr, cmap, norm):
    """Legend swatch for a shared-scale surface (colored by its mean value).

    Shared mode has no per-surface colorbar to carry the dataset name, so the
    names move to a proxy legend — plotly keeps them as legend entries.
    """
    if np.isfinite(arr).any():
        color = cmap(norm(float(np.nanmean(arr))))
    else:
        color = "0.6"
    return mpatches.Patch(facecolor=color, label=name)


def _add_legend(ax, handles, options: ChartOptions):
    if handles:
        ax.legend(handles=handles, loc="upper left",
                  fontsize=options.font_size, framealpha=0.7)


# ---------------------------------------------------------------------------
# public figure builders (mirror charts.py signatures)
# ---------------------------------------------------------------------------

def contour_2d(values, options: ChartOptions, name: str = "") -> Figure:
    """2D filled contour. NaN cells transparent."""
    arr = _as_2d_float(values)
    masked = np.ma.masked_invalid(arr)
    zmin, zmax = options.zmin, options.zmax

    finite = arr[np.isfinite(arr)]
    lo = zmin if zmin is not None else (float(np.min(finite)) if finite.size else 0.0)
    hi = zmax if zmax is not None else (float(np.max(finite)) if finite.size else 1.0)

    levels = 12
    if options.contour_levels is not None and options.contour_levels > 0:
        levels = options.contour_levels
    lvl = np.linspace(lo, hi, levels + 1) if hi > lo else levels

    fig = _new_figure(options, _DEFAULT_2D_IN)
    ax = fig.add_subplot(111)
    rows, cols = arr.shape
    xs = np.arange(cols)
    ys = np.arange(rows)
    cs = ax.contourf(xs, ys, masked, levels=lvl, cmap=_make_cmap(options),
                     vmin=zmin, vmax=zmax, extend="neither")
    _colorbar(fig, ax, cs, options)

    title = _with_shape(options.title, arr) if options.show_shape else options.title
    _apply_title(ax, options, title)
    _style_ticks_2d(ax, options)
    if options.match_aspect:
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def heatmap_2d(values, options: ChartOptions, name: str = "") -> Figure:
    """2D heatmap (imshow, nearest). NaN cells transparent."""
    arr = _as_2d_float(values)
    masked = np.ma.masked_invalid(arr)

    fig = _new_figure(options, _DEFAULT_2D_IN)
    ax = fig.add_subplot(111)
    im = ax.imshow(masked, origin="lower", interpolation="nearest",
                   cmap=_make_cmap(options), vmin=options.zmin, vmax=options.zmax,
                   aspect="equal" if options.match_aspect else "auto")
    _colorbar(fig, ax, im, options)

    title = _with_shape(options.title, arr) if options.show_shape else options.title
    _apply_title(ax, options, title)
    _style_ticks_2d(ax, options)
    fig.tight_layout()
    return fig


def _plot_surface(ax, arr, z_offset, options, name=""):
    rows, cols = arr.shape
    xs = np.arange(cols)
    ys = np.arange(rows)
    X, Y = np.meshgrid(xs, ys)
    Z = arr + z_offset
    return ax.plot_surface(
        X, Y, Z, cmap=_make_cmap(options),
        vmin=options.zmin, vmax=options.zmax,
        linewidth=0, antialiased=True, rstride=1, cstride=1,
    )


def surface_3d(values, options: ChartOptions, name: str = "",
               z_offset: float = 0.0) -> Figure:
    """3D surface with optional constant z offset."""
    arr = _as_2d_float(values)
    fig = _new_figure(options, _DEFAULT_3D_IN)
    ax = fig.add_subplot(111, projection="3d")
    surf = _plot_surface(ax, arr, z_offset, options, name)
    _colorbar(fig, ax, surf, options)

    trace_name = _with_shape(name, arr) if options.show_shape else name
    title = trace_name or options.title
    _apply_title(ax, options, title)

    if options.match_aspect:
        rows, cols = arr.shape
        m = float(max(rows, cols, 1))
        ax.set_box_aspect((cols / m, rows / m, 0.6))
    if options.zmin is not None and options.zmax is not None:
        ax.set_zlim(options.zmin, options.zmax)
    fig.tight_layout()
    # after tight_layout: materialising 3D tick labels (which _style_ticks_3d
    # must do to set their font) blows up the pre-projection bbox estimate and
    # tight_layout then bails out with "margins cannot be made large enough".
    _style_ticks_3d(ax, options)
    return fig


def multi_surface_3d(items: "List[Tuple[str, np.ndarray, float]]",
                     options: ChartOptions) -> Figure:
    """Several surfaces in one 3D axes.

    ``options.shared_colorbar`` (default) puts every surface on ONE common
    color scale spanning the RAW values of all items and draws a single
    colorbar; colors come from ``facecolors`` so the vertical z offsets cannot
    shift them, and the dataset names move to a proxy legend. With it False
    each surface maps its own z and keeps its own name-labelled colorbar.

    Either way the colorbars go through :func:`_slot_colorbars`, so they never
    overlap and the figure widens to make room for them (unless the caller
    pinned ``options.width``).
    """
    fig = _new_figure(options, _DEFAULT_3D_IN)
    ax = fig.add_subplot(111, projection="3d")
    arrays = [_as_2d_float(values) for _name, values, _offset in items]
    max_rows = max([a.shape[0] for a in arrays] + [1])
    max_cols = max([a.shape[1] for a in arrays] + [1])

    specs = []
    if options.shared_colorbar:
        cmap = _make_cmap(options)
        norm = mcolors.Normalize(*_color_bounds(arrays, options.zmin,
                                                options.zmax))
        handles = []
        for (name, _values, z_offset), arr in zip(items, arrays):
            rows, cols = arr.shape
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            ax.plot_surface(X, Y, arr + z_offset,
                            facecolors=_facecolors(arr, cmap, norm),
                            shade=False, linewidth=0, antialiased=True,
                            rstride=1, cstride=1)
            if name:
                handles.append(_proxy_patch(name, arr, cmap, norm))
        if options.show_colorbar:
            sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            specs.append((sm, ""))       # one bar for every surface -> no name
        _add_legend(ax, handles, options)
    else:
        for (name, _values, z_offset), arr in zip(items, arrays):
            surf = _plot_surface(ax, arr, z_offset, options, name)
            if options.show_colorbar:
                specs.append((surf, name))

    _apply_title(ax, options, options.title)
    if options.match_aspect:
        m = float(max(max_rows, max_cols, 1))
        ax.set_box_aspect((max_cols / m, max_rows / m, 0.6))
    if options.zmin is not None and options.zmax is not None:
        ax.set_zlim(options.zmin, options.zmax)
    _style_ticks_3d(ax, options)
    _slot_colorbars(fig, ax, specs, options)   # no tight_layout after this
    return fig


def effective_gap_chart(series: "List[dict]", options: ChartOptions) -> Figure:
    """Line/marker chart of per-temperature AVG with STD 'T' error bars.

    Fixed black/grey publication styling matching the plotly builder: solid
    black line/markers/error bars, dashed light-grey grid, white background.
    """
    labels = [p["label"] for p in series]
    avg = [p["avg"] for p in series]
    std = [p["std"] if p.get("std") is not None else 0.0 for p in series]
    x = np.arange(len(labels))

    fig = _new_figure(options, _DEFAULT_2D_IN)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.errorbar(x, avg, yerr=std, fmt="-o", color="black",
                ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5,
                markersize=6, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Effective Gap", fontsize=options.font_size,
                  fontfamily=options.font_family)
    ax.set_xlabel("Temperature point", fontsize=options.font_size,
                  fontfamily=options.font_family)
    _apply_title(ax, options, options.title)
    _style_ticks_2d(ax, options)

    ax.grid(True, linestyle="--", color="lightgrey")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1)

    if options.zmin is not None or options.zmax is not None:
        ax.set_ylim(options.zmin, options.zmax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# export: save + reconstruct-from-plotly-dict (hybrid path — matplotlib PNGs
# from the plotly figures the app displays, so kaleido/Chromium is not needed)
# ---------------------------------------------------------------------------

def save_figure(fig: Figure, path: str, img_kwargs: Optional[dict] = None) -> None:
    """Write *fig* to *path* honoring plotly-style ``{width, height, scale}``.

    ``img_kwargs`` is what ``callbacks._export_image_kwargs`` produces: width /
    height in pixels, scale a dpi multiplier. Missing keys leave the figure's
    own size / 1x dpi. width+height set the size directly; a lone width or
    height rescales the other axis proportionally (mirrors plotly's behaviour
    of preserving aspect when only one dimension is given).
    """
    img_kwargs = img_kwargs or {}
    scale = float(img_kwargs.get("scale") or 1.0)
    w = img_kwargs.get("width")
    h = img_kwargs.get("height")
    cur_w, cur_h = fig.get_size_inches()
    if w and h:
        fig.set_size_inches(w / _DPI, h / _DPI)
    elif w:
        fig.set_size_inches(w / _DPI, cur_h * (w / _DPI) / cur_w)
    elif h:
        fig.set_size_inches(cur_w * (h / _DPI) / cur_h, h / _DPI)
    fig.savefig(path, dpi=_DPI * scale)


def _z_array(z) -> np.ndarray:
    """Plotly dict z (nested lists, None for blanks) -> float64 ndarray w/ NaN."""
    rows = []
    for row in z:
        rows.append([np.nan if v is None else float(v) for v in row])
    return np.asarray(rows, dtype="float64")


def _recon_options(d: dict) -> ChartOptions:
    """Rebuild a ChartOptions carrying title/fonts from a plotly layout dict.

    ``show_shape`` is forced False: the plotly title/trace-name already has the
    ``rows×cols`` suffix baked in, so builders must not append it again.
    """
    layout = d.get("layout") or {}
    ttl = layout.get("title") or {}
    tfont = ttl.get("font") or {}
    lfont = layout.get("font") or {}
    return ChartOptions(
        title=ttl.get("text") or "",
        show_shape=False,
        font_family=lfont.get("family") or tfont.get("family") or "Arial",
        font_size=int(lfont.get("size") or 12),
        title_font_size=int(tfont.get("size") or 16),
    )


def _recon_2d(trace: dict, opts: ChartOptions, typ: str) -> Figure:
    arr = _z_array(trace["z"])
    masked = np.ma.masked_invalid(arr)
    cmap = _cmap_from_stops(trace.get("colorscale"), trace.get("reversescale", False))
    zmin = trace.get("zmin")
    zmax = trace.get("zmax")
    show_cb = trace.get("showscale", True)

    fig = _new_figure(opts, _DEFAULT_2D_IN)
    ax = fig.add_subplot(111)
    if typ == "contour":
        rows, cols = arr.shape
        levels = 12
        contours = trace.get("contours") or {}
        start, end, size = (contours.get("start"), contours.get("end"),
                            contours.get("size"))
        if start is not None and end is not None and size:
            levels = np.arange(start, end + size / 2.0, size)
        mappable = ax.contourf(np.arange(cols), np.arange(rows), masked,
                               levels=levels, cmap=cmap, vmin=zmin, vmax=zmax,
                               extend="neither")
    else:
        mappable = ax.imshow(masked, origin="lower", interpolation="nearest",
                             cmap=cmap, vmin=zmin, vmax=zmax,
                             aspect="equal")
    if show_cb:
        cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=opts.tick_font_size)
    _apply_title(ax, opts, opts.title)
    _style_ticks_2d(ax, opts)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def _scene_zrange(layout: dict):
    """``layout.scene.zaxis.range`` as a (lo, hi) pair, or None if unset.

    :mod:`charts` only writes it when the user pinned BOTH zmin and zmax, so
    its absence means "autorange" — the surfaces' own extent decides.
    """
    scene = (layout or {}).get("scene") or {}
    rng = (scene.get("zaxis") or {}).get("range")
    if rng is None or len(rng) != 2 or rng[0] is None or rng[1] is None:
        return None
    return float(rng[0]), float(rng[1])


def _recon_surface(traces: "List[dict]", opts: ChartOptions,
                   layout: dict) -> Figure:
    """Rebuild a (multi-)surface scene from plotly trace dicts.

    A trace carrying ``surfacecolor`` comes from a shared-colorbar figure: it
    is colored by those raw values over its own cmin/cmax, never by z (which
    holds the stacking offset). Traces without it keep the z-mapped colormap.

    The z-axis range is read from ``layout.scene.zaxis`` — NOT from cmin/cmax,
    which in shared mode span the raw values and would clip offset surfaces.
    """
    fig = _new_figure(opts, _DEFAULT_3D_IN)
    ax = fig.add_subplot(111, projection="3d")
    max_rows = max_cols = 1
    n_bars = sum(1 for tr in traces if tr.get("showscale", True))
    specs = []
    handles = []
    for tr in traces:
        arr = _z_array(tr["z"])
        max_rows = max(max_rows, arr.shape[0])
        max_cols = max(max_cols, arr.shape[1])
        cmap = _cmap_from_stops(tr.get("colorscale"), tr.get("reversescale", False))
        cmin, cmax = tr.get("cmin"), tr.get("cmax")
        rows, cols = arr.shape
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        name = tr.get("name") or ""
        scolor = tr.get("surfacecolor")
        if scolor is not None:
            cvals = _z_array(scolor)
            norm = mcolors.Normalize(*_color_bounds([cvals], cmin, cmax))
            ax.plot_surface(X, Y, arr, facecolors=_facecolors(cvals, cmap, norm),
                            shade=False, linewidth=0, antialiased=True,
                            rstride=1, cstride=1)
            mappable = mcm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            if name:
                handles.append(_proxy_patch(name, cvals, cmap, norm))
        else:
            mappable = ax.plot_surface(X, Y, arr, cmap=cmap, vmin=cmin,
                                       vmax=cmax, linewidth=0,
                                       antialiased=True, rstride=1, cstride=1)
        if tr.get("showscale", True):
            # a lone bar describes the whole figure, so it stays unlabelled
            specs.append((mappable, name if n_bars > 1 else ""))

    # single-surface figures carry their label in the trace name (title empty)
    title = opts.title or (traces[0].get("name") if len(traces) == 1 else "")
    _apply_title(ax, opts, title)
    _add_legend(ax, handles, opts)
    m = float(max(max_rows, max_cols, 1))
    ax.set_box_aspect((max_cols / m, max_rows / m, 0.6))
    zrange = _scene_zrange(layout)
    if zrange is not None:
        ax.set_zlim(*zrange)
    _style_ticks_3d(ax, opts)
    _slot_colorbars(fig, ax, specs, opts)      # no tight_layout after this
    return fig


def _recon_scatter(trace: dict, opts: ChartOptions) -> Figure:
    series = []
    xs = trace.get("x") or []
    ys = trace.get("y") or []
    err = ((trace.get("error_y") or {}).get("array")) or [0.0] * len(xs)
    for i, label in enumerate(xs):
        series.append({"label": label,
                       "avg": ys[i] if i < len(ys) else None,
                       "std": err[i] if i < len(err) else None})
    return effective_gap_chart(series, opts)


def figure_from_plotly_dict(d: dict) -> Figure:
    """Reconstruct a matplotlib Figure from a plotly figure dict.

    Dispatches on the first trace's ``type`` (contour/heatmap/surface/scatter)
    — the exact types produced by :mod:`charts`. Colors come from the dict's
    resolved colorscale stops, so the PNG matches the displayed figure. Surface
    figures also need the layout (scene z range), so it is passed along. Raises
    ValueError on an empty or unrecognised figure.
    """
    data = (d or {}).get("data") or []
    if not data:
        raise ValueError("figure has no data traces")
    opts = _recon_options(d)
    typ = data[0].get("type")
    if typ in ("contour", "heatmap"):
        return _recon_2d(data[0], opts, typ)
    if typ == "surface":
        return _recon_surface(data, opts, (d.get("layout") or {}))
    if typ == "scatter":
        return _recon_scatter(data[0], opts)
    raise ValueError("unsupported trace type: {0!r}".format(typ))
