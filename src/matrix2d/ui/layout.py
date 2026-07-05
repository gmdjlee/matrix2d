"""Dash layout for the single-page warpage-analysis app.

Left sidebar (folders panel + chart options) and a main content area driven by
dcc.Tabs (2D view / 3D view / Gap compute). No multi-page routing.
"""

import os
from typing import Optional

from dash import dcc, html

# Default folder prefill: use ./demo_data/<KIND> if present, else blank.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEMO = os.path.join(_REPO_ROOT, "demo_data")


def _demo_default(kind: str) -> str:
    candidate = os.path.join(_DEMO, kind)
    return candidate if os.path.isdir(candidate) else ""


FONT_FAMILIES = ["Arial", "Times New Roman", "Courier New", "Malgun Gothic"]
COLORSCALES = ["Jet", "Viridis", "Plasma", "RdBu", "Turbo", "Greys"]


def _folders_panel() -> html.Div:
    return html.Div(
        className="panel",
        children=[
            html.H3("Folders"),
            html.Div(className="field", children=[
                html.Label("TOP"),
                dcc.Input(id="folder-top", type="text", value=_demo_default("TOP"),
                          placeholder="path to TOP folder", className="input-full"),
            ]),
            html.Div(className="field", children=[
                html.Label("BTM"),
                dcc.Input(id="folder-btm", type="text", value=_demo_default("BTM"),
                          placeholder="path to BTM folder", className="input-full"),
            ]),
            html.Div(className="field", children=[
                html.Label("GAP"),
                dcc.Input(id="folder-gap", type="text", value=_demo_default("GAP"),
                          placeholder="path to GAP folder", className="input-full"),
            ]),
            html.Div(className="field", children=[
                html.Label("OUT"),
                dcc.Input(id="folder-out", type="text", value=_demo_default("OUT"),
                          placeholder="path to OUT folder", className="input-full"),
            ]),
            html.Button("Scan", id="btn-scan", n_clicks=0, className="btn btn-primary"),
            html.Div(id="scan-status", className="status"),
        ],
    )


def _num(id_, value, step=None, placeholder=""):
    return dcc.Input(id=id_, type="number", value=value, step=step,
                     placeholder=placeholder, className="input-full")


def _chart_options_panel() -> html.Div:
    return html.Div(
        className="panel",
        children=[
            html.H3("Chart Options"),
            html.Div(className="field", children=[
                html.Label("Title"),
                dcc.Input(id="opt-title", type="text", value="", className="input-full"),
            ]),
            html.Div(className="field", children=[
                html.Label("Font family"),
                dcc.Dropdown(id="opt-font-family",
                             options=[{"label": f, "value": f} for f in FONT_FAMILIES],
                             value="Arial", clearable=False),
            ]),
            html.Div(className="row", children=[
                html.Div(className="field half", children=[
                    html.Label("Font size"), _num("opt-font-size", 12, step=1)]),
                html.Div(className="field half", children=[
                    html.Label("Title size"), _num("opt-title-size", 16, step=1)]),
            ]),
            html.Div(className="field", children=[
                html.Label("Tick font size"), _num("opt-tick-size", 10, step=1)]),
            html.Div(className="row", children=[
                html.Div(className="field half", children=[
                    html.Label("X dtick"), _num("opt-x-dtick", None, step=1, placeholder="auto")]),
                html.Div(className="field half", children=[
                    html.Label("Y dtick"), _num("opt-y-dtick", None, step=1, placeholder="auto")]),
            ]),
            html.Div(className="field", children=[
                html.Label("Colorscale"),
                dcc.Dropdown(id="opt-colorscale",
                             options=[{"label": c, "value": c} for c in COLORSCALES],
                             value="Jet", clearable=False),
            ]),
            html.Div(className="field", children=[
                dcc.Checklist(id="opt-toggles",
                              options=[
                                  {"label": " Reverse colorscale", "value": "reverse"},
                                  {"label": " Show colorbar", "value": "colorbar"},
                              ],
                              value=["colorbar"], className="checklist"),
            ]),
            html.Div(className="row", children=[
                html.Div(className="field half", children=[
                    html.Label("zmin"), _num("opt-zmin", None, placeholder="auto")]),
                html.Div(className="field half", children=[
                    html.Label("zmax"), _num("opt-zmax", None, placeholder="auto")]),
            ]),
            html.Div(className="field", children=[
                html.Label("Contour levels"), _num("opt-contour-levels", None, step=1, placeholder="auto")]),
            html.Div(className="row", children=[
                html.Div(className="field half", children=[
                    html.Label("Width"), _num("opt-width", None, step=10, placeholder="auto")]),
                html.Div(className="field half", children=[
                    html.Label("Height"), _num("opt-height", 500, step=10)]),
            ]),
        ],
    )


def _tab_2d() -> html.Div:
    return html.Div(className="tab-body", children=[
        html.Div(className="controls-row", children=[
            html.Div(className="field grow", children=[
                html.Label("Dataset"),
                dcc.Dropdown(id="view2d-dataset", options=[], placeholder="scan a folder first"),
            ]),
            html.Div(className="field", children=[
                html.Label("Chart type"),
                dcc.RadioItems(id="view2d-type",
                               options=[{"label": " Contour", "value": "contour"},
                                        {"label": " Heatmap", "value": "heatmap"}],
                               value="contour", className="radio-inline"),
            ]),
        ]),
        html.Div(id="view2d-error", className="error"),
        dcc.Graph(id="view2d-graph", className="graph"),
        html.Button("Save current figure to OUT as PNG", id="btn-export-2d",
                    n_clicks=0, className="btn"),
        html.Div(id="export2d-status", className="status"),
    ])


def _tab_3d() -> html.Div:
    return html.Div(className="tab-body", children=[
        html.Div(className="field", children=[
            html.Label("Datasets (TOP / BTM / GAP inputs and computed gaps)"),
            dcc.Dropdown(id="view3d-datasets", options=[], multi=True,
                         placeholder="scan / compute first"),
        ]),
        # per-dataset z-offset inputs are injected here by a pattern-matching callback
        html.Div(id="view3d-offsets", className="offsets"),
        html.Div(id="view3d-error", className="error"),
        dcc.Graph(id="view3d-graph", className="graph"),
        html.Button("Save current figure to OUT as PNG", id="btn-export-3d",
                    n_clicks=0, className="btn"),
        html.Div(id="export3d-status", className="status"),
    ])


def _tab_gap() -> html.Div:
    return html.Div(className="tab-body", children=[
        html.Div(className="controls-row", children=[
            html.Div(className="field", children=[
                html.Label("Reference size"),
                dcc.RadioItems(id="gap-reference",
                               options=[{"label": " TOP", "value": "TOP"},
                                        {"label": " BTM", "value": "BTM"}],
                               value="TOP", className="radio-inline"),
            ]),
            html.Button("Compute All Gaps", id="btn-compute-gaps",
                        n_clicks=0, className="btn btn-primary"),
        ]),
        html.Div(id="gap-error", className="error"),
        html.Div(id="gap-table", className="table-wrap"),
        html.Hr(),
        html.Div(className="field", children=[
            html.Label("Inspect a computed result"),
            dcc.Dropdown(id="gap-result-select", options=[],
                         placeholder="compute gaps first"),
        ]),
        html.Div(id="gap-inspect-error", className="error"),
        html.Div(className="graph-pair", children=[
            dcc.Graph(id="gap-graph-2d", className="graph half"),
            dcc.Graph(id="gap-graph-3d", className="graph half"),
        ]),
    ])


def build_layout() -> html.Div:
    return html.Div(className="app-root", children=[
        # ---- stores ----
        dcc.Store(id="store-metas", data={"TOP": [], "BTM": [], "GAP": []}),
        dcc.Store(id="store-gaps", data=[]),  # list of result summary dicts

        # ---- sidebar ----
        html.Div(className="sidebar", children=[
            html.H1("Warpage Analysis"),
            _folders_panel(),
            _chart_options_panel(),
        ]),

        # ---- main ----
        html.Div(className="main", children=[
            dcc.Tabs(id="tabs", value="tab-2d", children=[
                dcc.Tab(label="2D View", value="tab-2d", children=[_tab_2d()]),
                dcc.Tab(label="3D View", value="tab-3d", children=[_tab_3d()]),
                dcc.Tab(label="Gap Compute", value="tab-gap", children=[_tab_gap()]),
            ]),
        ]),
    ])
