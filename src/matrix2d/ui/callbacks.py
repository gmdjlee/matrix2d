"""Dash callbacks wiring the layout to core/services and the chart builders.

All folder / pipeline calls are wrapped in try/except so a bad path or a
core error surfaces as a message in an html.Div instead of crashing the app.
"""

import os
import traceback
from typing import List, Optional

import numpy as np
from dash import ALL, MATCH, Input, Output, State, callback_context, dcc, html, no_update

from matrix2d.ui import charts, helpers


# ---------------------------------------------------------------------------
# ChartOptions assembled from the sidebar controls (shared by all chart tabs).
# ---------------------------------------------------------------------------

def _build_options(title, font_family, font_size, title_size, tick_size,
                   x_dtick, y_dtick, colorscale, toggles, zmin, zmax,
                   contour_levels, width, height) -> charts.ChartOptions:
    toggles = toggles or []

    def _int(v):
        return int(v) if v is not None and v != "" else None

    def _float(v):
        return float(v) if v is not None and v != "" else None

    return charts.ChartOptions(
        title=title or "",
        font_family=font_family or "Arial",
        font_size=int(font_size) if font_size else 12,
        title_font_size=int(title_size) if title_size else 16,
        tick_font_size=int(tick_size) if tick_size else 10,
        x_tick_step=_float(x_dtick),
        y_tick_step=_float(y_dtick),
        colorscale=colorscale or "Jet",
        reverse_colorscale="reverse" in toggles,
        show_colorbar="colorbar" in toggles,
        zmin=_float(zmin),
        zmax=_float(zmax),
        contour_levels=_int(contour_levels),
        width=_int(width),
        height=_int(height),
    )


_OPTION_STATES = [
    State("opt-title", "value"),
    State("opt-font-family", "value"),
    State("opt-font-size", "value"),
    State("opt-title-size", "value"),
    State("opt-tick-size", "value"),
    State("opt-x-dtick", "value"),
    State("opt-y-dtick", "value"),
    State("opt-colorscale", "value"),
    State("opt-toggles", "value"),
    State("opt-zmin", "value"),
    State("opt-zmax", "value"),
    State("opt-contour-levels", "value"),
    State("opt-width", "value"),
    State("opt-height", "value"),
]

# Same set but as Inputs, so charts re-render live when options change.
_OPTION_INPUTS = [Input(s.component_id, s.component_property) for s in _OPTION_STATES]


def _empty_fig(message: str = ""):
    fig = charts.go.Figure()
    fig.update_layout(
        annotations=[dict(text=message, showarrow=False,
                          xref="paper", yref="paper", x=0.5, y=0.5)],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# dataset key helpers: a selected 3D/2D dataset is identified by a string key.
#   input datasets:   "meta::<path>"
#   computed gaps:    "gap::<out_name>"
# ---------------------------------------------------------------------------

def _meta_key(path: str) -> str:
    return "meta::" + path


def _gap_key(out_name: str) -> str:
    return "gap::" + out_name


def _all_meta_dicts(store_metas) -> List[dict]:
    out = []
    for kind in ("TOP", "BTM", "GAP"):
        out.extend(store_metas.get(kind, []))
    return out


def _find_meta(store_metas, path) -> Optional[dict]:
    for d in _all_meta_dicts(store_metas):
        if d["path"] == path:
            return d
    return None


def _resolve_values(key: str, store_metas):
    """Return ndarray for a dataset key, loading/caching as needed."""
    if key.startswith("gap::"):
        out_name = key[len("gap::"):]
        return helpers.get_gap(out_name)
    if key.startswith("meta::"):
        path = key[len("meta::"):]
        md = _find_meta(store_metas, path)
        if md is None:
            return None
        return helpers.load_matrix(md)
    return None


def _key_label(key: str, store_metas) -> str:
    if key.startswith("gap::"):
        return "GAP " + key[len("gap::"):]
    if key.startswith("meta::"):
        md = _find_meta(store_metas, key[len("meta::"):])
        if md:
            return helpers.meta_label_from_dict(md)
    return key


def register_callbacks(app):
    # -------------------------------------------------------------------
    # 1. Scan folders -> store metas, populate dropdowns, show counts.
    # -------------------------------------------------------------------
    @app.callback(
        Output("store-metas", "data"),
        Output("scan-status", "children"),
        Output("view2d-dataset", "options"),
        Output("view3d-datasets", "options"),
        Input("btn-scan", "n_clicks"),
        State("folder-top", "value"),
        State("folder-btm", "value"),
        State("folder-gap", "value"),
        prevent_initial_call=True,
    )
    def scan_folders(_n, top_dir, btm_dir, gap_dir):
        from matrix2d.services.repository import scan_folder

        result = {"TOP": [], "BTM": [], "GAP": []}
        errors = []
        specs = [("TOP", top_dir), ("BTM", btm_dir), ("GAP", gap_dir)]
        for kind, folder in specs:
            if not folder:
                continue
            try:
                metas = scan_folder(folder, kind)
                result[kind] = [helpers.meta_to_dict(m) for m in metas]
            except Exception as exc:  # noqa: BLE001 - surface any core error
                errors.append("{kind}: {exc}".format(kind=kind, exc=exc))

        counts = "  ".join(
            "{k}={n}".format(k=k, n=len(result[k])) for k in ("TOP", "BTM", "GAP")
        )
        status_children = [html.Span("Scanned: " + counts)]
        if errors:
            status_children.append(
                html.Div("Errors: " + " | ".join(errors), className="error"))

        # 2D options: every input dataset
        opts_2d = [
            {"label": helpers.meta_label_from_dict(d), "value": _meta_key(d["path"])}
            for d in _all_meta_dicts(result)
        ]
        # 3D options: input datasets + any already-computed gaps
        opts_3d = list(opts_2d)
        for name in helpers.gap_names():
            opts_3d.append({"label": "GAP " + name, "value": _gap_key(name)})

        return result, status_children, opts_2d, opts_3d

    # -------------------------------------------------------------------
    # 2. 2D view render.
    # -------------------------------------------------------------------
    @app.callback(
        Output("view2d-graph", "figure"),
        Output("view2d-error", "children"),
        [Input("view2d-dataset", "value"), Input("view2d-type", "value")]
        + _OPTION_INPUTS,
        State("store-metas", "data"),
        prevent_initial_call=True,
    )
    def render_2d(dataset_key, chart_type, *rest):
        option_values = rest[:-1]
        store_metas = rest[-1]
        if not dataset_key:
            return _empty_fig("Select a dataset"), ""
        try:
            values = _resolve_values(dataset_key, store_metas)
            if values is None:
                return _empty_fig(), "Dataset not found / not loaded."
            opts = _build_options(*option_values)
            if not opts.title:
                opts.title = _key_label(dataset_key, store_metas)
            if chart_type == "heatmap":
                fig = charts.heatmap_2d(values, opts)
            else:
                fig = charts.contour_2d(values, opts)
            return fig, ""
        except Exception:  # noqa: BLE001
            return _empty_fig(), "Render error: " + traceback.format_exc(limit=2)

    # -------------------------------------------------------------------
    # 3. 3D view: build per-dataset z-offset inputs (pattern-matching).
    # -------------------------------------------------------------------
    @app.callback(
        Output("view3d-offsets", "children"),
        Input("view3d-datasets", "value"),
        State("store-metas", "data"),
        prevent_initial_call=True,
    )
    def build_offset_inputs(selected_keys, store_metas):
        selected_keys = selected_keys or []
        rows = []
        for key in selected_keys:
            label = _key_label(key, store_metas)
            rows.append(html.Div(className="offset-row", children=[
                html.Span(label, className="offset-label"),
                dcc.Input(
                    id={"type": "z-offset", "key": key},
                    type="number", value=0.0, step=0.1,
                    className="offset-input",
                ),
            ]))
        return rows

    # 3b. 3D view render (reacts to selection, offsets, options).
    @app.callback(
        Output("view3d-graph", "figure"),
        Output("view3d-error", "children"),
        [Input("view3d-datasets", "value"),
         Input({"type": "z-offset", "key": ALL}, "value"),
         Input({"type": "z-offset", "key": ALL}, "id")]
        + _OPTION_INPUTS,
        State("store-metas", "data"),
        prevent_initial_call=True,
    )
    def render_3d(selected_keys, offset_values, offset_ids, *rest):
        option_values = rest[:-1]
        store_metas = rest[-1]
        selected_keys = selected_keys or []
        if not selected_keys:
            return _empty_fig("Select datasets"), ""
        try:
            # map key -> offset from the pattern-matched inputs
            offset_map = {}
            for oid, oval in zip(offset_ids or [], offset_values or []):
                offset_map[oid["key"]] = float(oval) if oval is not None else 0.0

            items = []
            missing = []
            for key in selected_keys:
                values = _resolve_values(key, store_metas)
                if values is None:
                    missing.append(key)
                    continue
                items.append((_key_label(key, store_metas), values,
                              offset_map.get(key, 0.0)))
            if not items:
                return _empty_fig(), "No loadable datasets selected."
            opts = _build_options(*option_values)
            fig = charts.multi_surface_3d(items, opts)
            err = ""
            if missing:
                err = "Skipped (not loaded): " + ", ".join(missing)
            return fig, err
        except Exception:  # noqa: BLE001
            return _empty_fig(), "Render error: " + traceback.format_exc(limit=2)

    # -------------------------------------------------------------------
    # 4. Gap compute: run pipeline, build table + result dropdowns.
    # -------------------------------------------------------------------
    @app.callback(
        Output("store-gaps", "data"),
        Output("gap-table", "children"),
        Output("gap-error", "children"),
        Output("gap-result-select", "options"),
        Output("view3d-datasets", "options", allow_duplicate=True),
        Input("btn-compute-gaps", "n_clicks"),
        State("folder-top", "value"),
        State("folder-btm", "value"),
        State("folder-out", "value"),
        State("gap-reference", "value"),
        State("store-metas", "data"),
        prevent_initial_call=True,
    )
    def compute_gaps(_n, top_dir, btm_dir, out_dir, reference, store_metas):
        from matrix2d.services.pipeline import run_pipeline

        if not top_dir or not btm_dir or not out_dir:
            return no_update, no_update, "TOP, BTM and OUT folders are required.", \
                no_update, no_update
        try:
            helpers.clear_gaps()
            results = run_pipeline(top_dir, btm_dir, out_dir, reference=reference)
        except Exception:  # noqa: BLE001
            return no_update, no_update, \
                "Pipeline error: " + traceback.format_exc(limit=3), no_update, no_update

        rows = []
        summaries = []
        for r in results:
            job = r.job
            gap_arr = np.asarray(r.result.gap, dtype="float64")
            helpers.cache_gap(job.out_name, gap_arr)
            summaries.append({
                "out_name": job.out_name,
                "top": os.path.basename(getattr(job.top, "path", "")),
                "btm": os.path.basename(getattr(job.btm, "path", "")),
                "phase": job.phase,
                "offset": r.result.offset,
                "out_path": r.out_path,
            })

        header = html.Tr([html.Th(c) for c in
                          ["out_name", "top", "btm", "phase", "offset", "saved path"]])
        for s in summaries:
            rows.append(html.Tr([
                html.Td(s["out_name"]),
                html.Td(s["top"]),
                html.Td(s["btm"]),
                html.Td(s["phase"]),
                html.Td("{:.4g}".format(s["offset"]) if s["offset"] is not None else ""),
                html.Td(s["out_path"]),
            ]))
        table = html.Table([header] + rows, className="result-table")

        result_opts = [{"label": s["out_name"], "value": _gap_key(s["out_name"])}
                       for s in summaries]

        # refresh 3D options: inputs + freshly computed gaps
        opts_3d = [
            {"label": helpers.meta_label_from_dict(d), "value": _meta_key(d["path"])}
            for d in _all_meta_dicts(store_metas or {})
        ]
        opts_3d.extend(
            {"label": "GAP " + s["out_name"], "value": _gap_key(s["out_name"])}
            for s in summaries
        )

        msg = "Computed {n} gap(s).".format(n=len(summaries))
        return summaries, table, html.Span(msg, className="ok"), result_opts, opts_3d

    # 4b. Inspect a chosen computed gap: 2D contour + 3D surface.
    @app.callback(
        Output("gap-graph-2d", "figure"),
        Output("gap-graph-3d", "figure"),
        Output("gap-inspect-error", "children"),
        [Input("gap-result-select", "value")] + _OPTION_INPUTS,
        prevent_initial_call=True,
    )
    def inspect_gap(gap_key, *option_values):
        if not gap_key:
            return _empty_fig("Select a result"), _empty_fig("Select a result"), ""
        try:
            values = _resolve_values(gap_key, {})
            if values is None:
                return _empty_fig(), _empty_fig(), "Gap result not in cache."
            opts = _build_options(*option_values)
            name = gap_key[len("gap::"):]
            if not opts.title:
                opts.title = "GAP " + name
            fig2d = charts.contour_2d(values, opts)
            fig3d = charts.surface_3d(values, opts, name=name)
            return fig2d, fig3d, ""
        except Exception:  # noqa: BLE001
            return _empty_fig(), _empty_fig(), \
                "Render error: " + traceback.format_exc(limit=2)

    # -------------------------------------------------------------------
    # 5. Export current 2D / 3D figure to OUT as PNG (kaleido).
    # -------------------------------------------------------------------
    def _export(fig_dict, out_dir, default_name):
        if not fig_dict:
            return "Nothing to export."
        if not out_dir:
            return "Set an OUT folder first."
        try:
            os.makedirs(out_dir, exist_ok=True)
            fig = charts.go.Figure(fig_dict)
            path = os.path.join(out_dir, default_name)
            fig.write_image(path)  # kaleido backend
            return "Saved: " + path
        except Exception:  # noqa: BLE001
            return "Export error: " + traceback.format_exc(limit=2)

    @app.callback(
        Output("export2d-status", "children"),
        Input("btn-export-2d", "n_clicks"),
        State("view2d-graph", "figure"),
        State("folder-out", "value"),
        prevent_initial_call=True,
    )
    def export_2d(_n, fig_dict, out_dir):
        return _export(fig_dict, out_dir, "chart_2d.png")

    @app.callback(
        Output("export3d-status", "children"),
        Input("btn-export-3d", "n_clicks"),
        State("view3d-graph", "figure"),
        State("folder-out", "value"),
        prevent_initial_call=True,
    )
    def export_3d(_n, fig_dict, out_dir):
        return _export(fig_dict, out_dir, "chart_3d.png")
