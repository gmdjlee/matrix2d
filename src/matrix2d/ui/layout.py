"""Dash layout for the single-page warpage-analysis app.

Left sidebar (folders panel + data options + chart options) and a main content
area driven by dcc.Tabs (2D view / 3D view / Gap compute). No multi-page
routing.
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


def _folder_field(kind: str) -> html.Div:
    """One folder row: label, path input and a Browse... dialog button."""
    lower = kind.lower()
    return html.Div(className="field", children=[
        html.Label(kind),
        html.Div(className="row", children=[
            dcc.Input(id="folder-" + lower, type="text",
                      value=_demo_default(kind),
                      placeholder="path to {0} folder".format(kind),
                      className="input-full grow"),
            html.Button("Browse...", id="btn-browse-" + lower, n_clicks=0,
                        className="btn", title="Select {0} folder".format(kind)),
        ]),
    ])


def _folders_panel() -> html.Div:
    return html.Div(
        className="panel",
        children=[
            html.H3("Folders"),
            _folder_field("TOP"),
            _folder_field("BTM"),
            _folder_field("GAP"),
            _folder_field("OUT"),
            html.Button("Scan", id="btn-scan", n_clicks=0, className="btn btn-primary"),
            html.Div(id="scan-status", className="status"),
        ],
    )


def _num(id_, value, step=None, placeholder=""):
    return dcc.Input(id=id_, type="number", value=value, step=step,
                     placeholder=placeholder, className="input-full")


ROTATE_OPTIONS = [
    {"label": "0°", "value": 0},
    {"label": "90° CW", "value": 90},
    {"label": "180° CW", "value": 180},
    {"label": "270° CW", "value": 270},
]


def _data_options_panel() -> html.Div:
    return html.Div(
        className="panel",
        children=[
            html.H3("Data Options"),
            html.Div(className="field", children=[
                dcc.Checklist(id="data-top-flip",
                              options=[
                                  {"label": " Flip TOP (L-R + invert values)",
                                   "value": "flip"},
                              ],
                              value=[], className="checklist"),
            ]),
            html.Div(className="field", children=[
                html.Label("TOP Rotate"),
                dcc.Dropdown(id="data-top-rotate", options=ROTATE_OPTIONS,
                             value=0, clearable=False),
            ]),
            html.Div(className="field", children=[
                html.Label("TOP Zero cell (row, col)")]),
            html.Div(className="row", children=[
                html.Div(className="field half", children=[
                    _num("data-top-zero-row", None, step=1, placeholder="row")]),
                html.Div(className="field half", children=[
                    _num("data-top-zero-col", None, step=1, placeholder="col")]),
            ]),
            html.Div(className="field", children=[
                html.Label("BTM Zero cell (row, col)")]),
            html.Div(className="row", children=[
                html.Div(className="field half", children=[
                    _num("data-btm-zero-row", None, step=1, placeholder="row")]),
                html.Div(className="field half", children=[
                    _num("data-btm-zero-col", None, step=1, placeholder="col")]),
            ]),
            html.Div("Zero cell coordinates apply after flip/rotate.",
                     className="status"),
            html.Div(className="field", children=[
                html.Label("Reference size (resize target)"),
                dcc.RadioItems(id="gap-reference",
                               options=[{"label": " Auto (smaller → larger)",
                                         "value": "AUTO"},
                                        {"label": " TOP", "value": "TOP"},
                                        {"label": " BTM", "value": "BTM"}],
                               value="AUTO", className="radio-inline"),
            ]),
            html.Div(className="field", children=[
                html.Label("Display data (2D/3D view)"),
                dcc.RadioItems(id="data-show-resized",
                               options=[{"label": " Original", "value": "original"},
                                        {"label": " Resized", "value": "resized"}],
                               value="original", className="radio-inline"),
            ]),
            html.Div("Resized preview mirrors the gap pipeline: the "
                     "non-reference dataset is interpolated onto the "
                     "reference grid.", className="status"),
        ],
    )


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
                                  {"label": " Show shape", "value": "shape"},
                              ],
                              value=["colorbar", "shape"], className="checklist"),
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
                html.Label("TOP sample"),
                dcc.Dropdown(id="view2d-top-sample", options=[],
                             placeholder="scan a folder first"),
            ]),
            html.Div(className="field grow", children=[
                html.Label("BTM sample"),
                dcc.Dropdown(id="view2d-btm-sample", options=[],
                             placeholder="scan a folder first"),
            ]),
            html.Div(className="field grow", children=[
                html.Label("Temperature (common)"),
                dcc.Dropdown(id="view2d-temp", options=[],
                             placeholder="select sample(s) first"),
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
        html.Div(className="graph-pair", children=[
            dcc.Graph(id="view2d-graph-top", className="graph half"),
            dcc.Graph(id="view2d-graph-btm", className="graph half"),
        ]),
        html.Button("Save current figures to OUT as PNG", id="btn-export-2d",
                    n_clicks=0, className="btn"),
        html.Div(id="export2d-status", className="status"),
    ])


def _tab_3d() -> html.Div:
    return html.Div(className="tab-body", children=[
        html.Div(className="controls-row", children=[
            html.Div(className="field grow", children=[
                html.Label("Filter: sample no."),
                dcc.Dropdown(id="view3d-filter-sample", options=[], multi=True,
                             placeholder="all samples"),
            ]),
            html.Div(className="field grow", children=[
                html.Label("Filter: temperature"),
                dcc.Dropdown(id="view3d-filter-temp", options=[], multi=True,
                             placeholder="all temperatures"),
            ]),
        ]),
        html.Div(className="controls-row", children=[
            html.Div(className="field grow", children=[
                html.Label("TOP datasets"),
                dcc.Dropdown(id="view3d-top", options=[], multi=True,
                             placeholder="scan first"),
            ]),
            html.Div(className="field grow", children=[
                html.Label("BTM datasets"),
                dcc.Dropdown(id="view3d-btm", options=[], multi=True,
                             placeholder="scan first"),
            ]),
            html.Div(className="field grow", children=[
                html.Label("GAP datasets (scanned + computed)"),
                dcc.Dropdown(id="view3d-gap", options=[], multi=True,
                             placeholder="scan / compute first"),
            ]),
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
                html.Label("Output name prefix"),
                dcc.Input(id="gap-out-prefix", type="text", value="GAP",
                          placeholder="e.g. TEST", className="input-full"),
            ]),
            html.Div("Output: {prefix}-{H|C}{temp}_TOP{n}-BTM{m}.txt — "
                     "reference size is set in Data Options (sidebar).",
                     className="status"),
            html.Button("Compute All Gaps", id="btn-compute-gaps",
                        n_clicks=0, className="btn btn-primary"),
            html.Button("Save All Images (2D+3D)", id="btn-export-all-gaps",
                        n_clicks=0, className="btn"),
        ]),
        # progress bar for the background compute; polled by the interval
        html.Div(className="progress-outer", children=[
            html.Div(id="gap-progress-bar", className="progress-inner"),
        ]),
        html.Div(id="gap-progress-label", className="status"),
        dcc.Interval(id="gap-progress-interval", interval=400, disabled=True),
        html.Div(id="export-all-status", className="status"),
        html.Div(id="gap-error", className="error"),
        # charts on the left, result list on the right (table scrolls itself)
        html.Div(className="gap-split", children=[
            html.Div(className="gap-left", children=[
                html.Div(className="field", children=[
                    html.Label("Inspect a computed result"),
                    dcc.Dropdown(id="gap-result-select", options=[],
                                 placeholder="compute gaps first"),
                ]),
                html.Div(id="gap-inspect-error", className="error"),
                dcc.Graph(id="gap-graph-2d", className="graph"),
                dcc.Graph(id="gap-graph-3d", className="graph"),
            ]),
            html.Div(className="gap-right", children=[
                html.Label("Results"),
                html.Div(id="gap-table", className="table-wrap"),
            ]),
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
            _data_options_panel(),
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
