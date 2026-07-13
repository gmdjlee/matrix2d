"""Dash layout for the single-page warpage-analysis app.

Left sidebar (folders panel + data options + chart options) and a main content
area driven by dcc.Tabs (2D view / 3D view / Gap compute). No multi-page
routing.
"""

import os
from typing import Optional

from dash import dash_table, dcc, html

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
            # progress bar for the background scan; polled by the interval
            html.Div(className="progress-outer", children=[
                html.Div(id="scan-progress-bar", className="progress-inner"),
            ]),
            html.Div(id="scan-progress-label", className="status"),
            dcc.Interval(id="scan-progress-interval", interval=300, disabled=True),
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
                               options=[{"label": " Auto (larger → smaller)",
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


def _image_export_panel() -> html.Div:
    """PNG export size controls (shared by every Save-as-PNG path).

    Always visible in the sidebar (not per-tab). Blank width/height/scale keep
    plotly's default behaviour (figure layout size, scale 1).
    """
    return html.Div(
        className="panel",
        children=[
            html.H3("Image Export"),
            html.Div(className="row", children=[
                html.Div(className="field half", children=[
                    html.Label("Width (px)"),
                    _num("export-img-width", None, step=10, placeholder="auto"),
                ]),
                html.Div(className="field half", children=[
                    html.Label("Height (px)"),
                    _num("export-img-height", None, step=10, placeholder="auto"),
                ]),
            ]),
            html.Div(className="field", children=[
                html.Label("Scale"),
                dcc.Input(id="export-img-scale", type="number", value=None,
                          step=0.5, min=0.1, placeholder="1",
                          className="input-full"),
            ]),
            html.Div("Blank fields keep the on-screen figure size.",
                     className="status"),
        ],
    )


# Chart Options are per-tab and per-chart-type: 2D View, 3D View, Gap Compute
# and Effective Gap each own an independent control set (own id prefix
# "opt2d"/"opt3d"/"optgap"/"opteff"). The controls shown are tailored to that
# tab's chart type — a 3D surface has no contour levels, a line chart has no
# colorscale/contour/aspect — so users only see options that actually affect
# the figure.
#
# Every control's id is "{prefix}-{suffix}"; ``_OPTION_ROWS`` lays them out
# (paired suffixes share a sidebar row) and ``TAB_OPTION_FIELDS`` selects the
# subset each tab renders. Callbacks derive the matching Input/State lists from
# ``tab_option_suffixes`` so panel and callback never drift.
_OPTION_ROWS = [
    ("title",),
    ("font-family",),
    ("font-size", "title-size"),
    ("tick-size",),
    ("x-dtick", "y-dtick"),
    ("colorscale",),
    ("toggles",),
    ("zmin", "zmax"),
    ("contour-levels",),
    ("width", "height"),
]

# Flat suffix order (used to align callback values with control ids).
_OPTION_ORDER = [k for row in _OPTION_ROWS for k in row]

# Common to every chart type: title, fonts, output size.
_COMMON_FIELDS = {"title", "font-family", "font-size", "title-size",
                  "tick-size", "width", "height"}
# Color-mapped 2D charts (contour + heatmap): full color + contour controls.
_2D_FIELDS = _COMMON_FIELDS | {"x-dtick", "y-dtick", "colorscale", "toggles",
                               "zmin", "zmax", "contour-levels"}
# 3D surface: same as 2D minus contour levels (surfaces are not contoured).
_3D_FIELDS = _2D_FIELDS - {"contour-levels"}
# Line chart (Effective Gap): no colorscale/contour/aspect; zmin/zmax act as
# the y-axis range, y-dtick as its tick step.
_LINE_FIELDS = _COMMON_FIELDS | {"y-dtick", "zmin", "zmax"}

TAB_OPTION_FIELDS = {
    "opt2d": _2D_FIELDS,
    "opt3d": _3D_FIELDS,
    "optgap": _2D_FIELDS,   # Gap tab shows 2D (contour/heatmap) + 3D surface
    "opteff": _LINE_FIELDS,
}

# Default control label per suffix; ``toggles`` is a checklist (no label).
_OPTION_LABELS = {
    "title": "Title",
    "font-family": "Font family",
    "font-size": "Font size",
    "title-size": "Title size",
    "tick-size": "Tick font size",
    "x-dtick": "X dtick",
    "y-dtick": "Y dtick",
    "colorscale": "Colorscale",
    "toggles": None,
    "zmin": "zmin",
    "zmax": "zmax",
    "contour-levels": "Contour levels",
    "width": "Width",
    "height": "Height",
}

# Per-tab label overrides: on the line chart zmin/zmax bound the y-axis.
_TAB_OPTION_LABELS = {
    "opteff": {"zmin": "Y-axis min", "zmax": "Y-axis max"},
}

# (default value, step) for the numeric controls; placeholder is "auto" when
# the default is None (leaves the figure builder to pick the value).
_OPTION_NUM = {
    "font-size": (12, 1),
    "title-size": (16, 1),
    "tick-size": (10, 1),
    "x-dtick": (None, 1),
    "y-dtick": (None, 1),
    "zmin": (None, None),
    "zmax": (None, None),
    "contour-levels": (None, 1),
    "width": (None, 10),
    "height": (500, 10),
}


def tab_option_suffixes(prefix: str):
    """Ordered suffix list for one tab's chart-option controls.

    Callbacks map their positional option values onto these suffixes, so the
    order here is the contract between the panel and ``_build_options``.
    """
    allowed = TAB_OPTION_FIELDS[prefix]
    return [k for k in _OPTION_ORDER if k in allowed]


def _option_control(cid, key: str):
    """Build the input control for one option ``key`` (id ``cid(key)``)."""
    c = cid(key)
    if key == "title":
        return dcc.Input(id=c, type="text", value="", className="input-full")
    if key == "font-family":
        return dcc.Dropdown(id=c, options=[{"label": f, "value": f}
                                           for f in FONT_FAMILIES],
                            value="Arial", clearable=False)
    if key == "colorscale":
        return dcc.Dropdown(id=c, options=[{"label": s, "value": s}
                                           for s in COLORSCALES],
                            value="Jet", clearable=False)
    if key == "toggles":
        return dcc.Checklist(
            id=c,
            options=[
                {"label": " Reverse colorscale", "value": "reverse"},
                {"label": " Show colorbar", "value": "colorbar"},
                {"label": " Show shape", "value": "shape"},
                {"label": " Match data aspect", "value": "aspect"},
            ],
            value=["colorbar", "shape", "aspect"], className="checklist")
    val, step = _OPTION_NUM[key]
    placeholder = "auto" if val is None else ""
    return _num(c, val, step=step, placeholder=placeholder)


def _option_field(cid, key: str, label_override: dict, half: bool) -> html.Div:
    """One control wrapped in a ``.field`` (or ``.field half``) div."""
    label = label_override.get(key, _OPTION_LABELS[key])
    children = []
    if label is not None:
        children.append(html.Label(label))
    children.append(_option_control(cid, key))
    return html.Div(className="field half" if half else "field",
                    children=children)


def _chart_options_panel(prefix: str, heading: str) -> html.Div:
    """Chart Options panel for one tab, showing only its chart type's fields."""
    def cid(name: str) -> str:
        return prefix + "-" + name

    allowed = TAB_OPTION_FIELDS[prefix]
    label_override = _TAB_OPTION_LABELS.get(prefix, {})
    children = [html.H3(heading)]
    for row in _OPTION_ROWS:
        keys = [k for k in row if k in allowed]
        if not keys:
            continue
        if len(keys) == 1:
            children.append(_option_field(cid, keys[0], label_override, half=False))
        else:
            children.append(html.Div(className="row", children=[
                _option_field(cid, k, label_override, half=True) for k in keys]))
    return html.Div(className="panel", children=children)


def _chart_options_stack() -> html.Div:
    """All three per-tab Chart Options panels; only the active tab's shows.

    The wrapping divs are toggled by a callback keyed on the active tab
    (``chart-options-<tab>`` ids). 2D is visible initially to match the
    default active tab.
    """
    return html.Div(children=[
        html.Div(id="chart-options-tab-2d",
                 children=[_chart_options_panel("opt2d", "Chart Options — 2D View")]),
        html.Div(id="chart-options-tab-3d", style={"display": "none"},
                 children=[_chart_options_panel("opt3d", "Chart Options — 3D View")]),
        html.Div(id="chart-options-tab-gap", style={"display": "none"},
                 children=[_chart_options_panel("optgap", "Chart Options — Gap Compute")]),
        html.Div(id="chart-options-tab-effgap", style={"display": "none"},
                 children=[_chart_options_panel("opteff", "Chart Options — Effective Gap")]),
    ])


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
                               value="heatmap", className="radio-inline"),
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
            html.Div(className="field grow", children=[
                html.Label("OUT datasets"),
                dcc.Dropdown(id="view3d-out", options=[], multi=True,
                             placeholder="scan first"),
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
        # progress bar for the background compute; polled by the root-level
        # gap-progress-interval (kept outside this tab so switching tabs
        # mid-compute does not unmount the poller)
        html.Div(className="progress-outer", children=[
            html.Div(id="gap-progress-bar", className="progress-inner"),
        ]),
        html.Div(id="gap-progress-label", className="status"),
        # progress bar for the background batch image export; polled by the
        # root-level export-all-progress-interval (same reasoning as above)
        html.Div(className="progress-outer", children=[
            html.Div(id="export-all-progress-bar", className="progress-inner"),
        ]),
        html.Div(id="export-all-progress-label", className="status"),
        html.Div(id="export-all-status", className="status"),
        html.Div(id="gap-error", className="error"),
        # charts on the left, result list on the right (table scrolls itself)
        html.Div(className="gap-split", children=[
            html.Div(className="gap-left", children=[
                html.Div(className="field", children=[
                    html.Label("Chart type (2D)"),
                    dcc.RadioItems(id="gap-view-type",
                                   options=[{"label": " Contour", "value": "contour"},
                                            {"label": " Heatmap", "value": "heatmap"}],
                                   value="heatmap", className="radio-inline"),
                ]),
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
                html.Div(className="table-wrap", children=[
                    dash_table.DataTable(
                        id="gap-result-table",
                        columns=[
                            {"name": "out_name", "id": "out_name"},
                            {"name": "top", "id": "top"},
                            {"name": "btm", "id": "btm"},
                            {"name": "phase", "id": "phase"},
                            {"name": "offset", "id": "offset"},
                            {"name": "saved path", "id": "out_path"},
                        ],
                        data=[],
                        page_action="custom",
                        page_current=0,
                        page_size=50,
                        page_count=1,
                        sort_action="custom",
                        sort_mode="single",
                        sort_by=[],
                        filter_action="custom",
                        filter_query="",
                        fixed_rows={"headers": True},
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "6px 8px",
                                    "fontSize": "12px", "whiteSpace": "nowrap"},
                        style_header={"backgroundColor": "#eef2f7",
                                      "fontWeight": "bold"},
                    ),
                ]),
            ]),
        ]),
    ])


def _tab_effgap() -> html.Div:
    return html.Div(className="tab-body", children=[
        html.Div("Average Effective Gap per temperature point over all "
                 "TOP-BTM combinations (from the last Gap Compute run). "
                 "Error bars show the sample standard deviation. Heating "
                 "points ascend, then cooling points descend.",
                 className="status"),
        html.Div(id="effgap-error", className="error"),
        dcc.Graph(id="effgap-graph", className="graph"),
        html.Button("Save current figure to OUT as PNG", id="btn-export-effgap",
                    n_clicks=0, className="btn"),
        html.Div(id="export-effgap-status", className="status"),
    ])


def build_layout() -> html.Div:
    return html.Div(className="app-root", children=[
        # ---- stores ----
        dcc.Store(id="store-metas", data={"TOP": [], "BTM": [], "GAP": [], "OUT": []}),
        dcc.Store(id="store-gaps", data=[]),  # list of result summary dicts

        # poller for the background gap compute. Lives at the root (not in the
        # Gap tab) so that leaving the tab while a compute runs cannot pause
        # or kill the polling that publishes the results.
        dcc.Interval(id="gap-progress-interval", interval=400, disabled=True),

        # poller for the background batch image export, same root-level
        # placement rationale as gap-progress-interval above.
        dcc.Interval(id="export-all-progress-interval", interval=400, disabled=True),

        # ---- sidebar ----
        html.Div(className="sidebar", children=[
            html.H1("Warpage Analysis"),
            _folders_panel(),
            _data_options_panel(),
            _image_export_panel(),
            _chart_options_stack(),
        ]),

        # ---- main ----
        html.Div(className="main", children=[
            dcc.Tabs(id="tabs", value="tab-2d", children=[
                dcc.Tab(label="2D View", value="tab-2d", children=[_tab_2d()]),
                dcc.Tab(label="3D View", value="tab-3d", children=[_tab_3d()]),
                dcc.Tab(label="Gap Compute", value="tab-gap", children=[_tab_gap()]),
                dcc.Tab(label="Effective Gap", value="tab-effgap",
                        children=[_tab_effgap()]),
            ]),
        ]),
    ])
