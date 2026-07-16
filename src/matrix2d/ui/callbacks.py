"""Dash callbacks wiring the layout to core/services and the chart builders.

All folder / pipeline calls are wrapped in try/except so a bad path or a
core error surfaces as a message in an html.Div instead of crashing the app.

Dataset selection model:
  * 2D view — pick a TOP sample, a BTM sample and a common temperature
    (phase-aware: the same Celsius value can occur twice per session, once
    while heating 'H' and once while cooling 'C'). Two charts render side
    by side, one per surface.
  * 3D view — TOP / BTM / GAP datasets are selected in three separate
    dropdowns; a sample-number / temperature filter row narrows the options
    when folders contain many files.
"""

import dataclasses
import logging
import os
import queue
import threading
import traceback
from typing import List, Optional

import numpy as np
from dash import ALL, Input, Output, State, dcc, html, no_update

from matrix2d.core.summary import effective_gap_series
from matrix2d.ui import charts, charts_mpl, helpers, layout, table_paging
from matrix2d.ui.dialogs import pick_folder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background gap-compute state. The Compute button starts a worker thread and
# a dcc.Interval polls this dict to drive the progress bar and, on completion,
# publish the results. Module-level is fine: local single-user app.
# ---------------------------------------------------------------------------

_COMPUTE_LOCK = threading.Lock()
_COMPUTE = {
    "running": False,
    "done": 0,
    "total": 0,
    "results": None,   # List[GapJobResult] on success
    "error": None,     # traceback string on failure
    "failures": None,  # list of per-failed-job dicts from run_pipeline
}

# Background folder-scan state, mirroring the gap-compute pattern above. The
# Scan button starts a worker thread; a dcc.Interval polls this dict to drive
# the scan progress bar and publish the scanned metas once done.
_SCAN_LOCK = threading.Lock()
_SCAN = {
    "running": False,
    "done": 0,
    "total": 0,
    "result": None,    # {"TOP": [...], "BTM": [...], "GAP": [...]} of dicts
    "errors": None,    # list of error strings
    "skipped": None,   # {"TOP": n, ...} count of skipped files per folder
}

# Background batch-image-export state, mirroring the two patterns above. The
# "Save All Images" button starts a worker thread; a dcc.Interval polls this
# dict to drive the export progress bar and publish the final status string.
_EXPORT_LOCK = threading.Lock()
_EXPORT = {
    "running": False,
    "done": 0,
    "total": 0,
    "result": None,    # status string on success
    "error": None,     # traceback string on failure
}

# Background 3D-tab batch-image-export state, same pattern as _EXPORT above.
# The 3D "Save All Filtered Images" button starts a worker thread; a
# dcc.Interval polls this dict to drive the progress bar and publish the final
# status string. "done"/"total" count datasets (one 3D surface PNG each).
_EXPORT3D_LOCK = threading.Lock()
_EXPORT3D = {
    "running": False,
    "done": 0,
    "total": 0,
    "result": None,    # status string on success
    "error": None,     # traceback string on failure
}

# Background Effective-Gap OUT-load state, same pattern again. The "Load from
# OUT files" button starts a worker thread (scan + per-file matrix load can be
# slow on large OUT folders); a dcc.Interval polls this dict to drive the load
# progress bar and publish the records once done.
_EFFLOAD_LOCK = threading.Lock()
_EFFLOAD = {
    "running": False,
    "stage": "",       # "scan" (folder scan) | "load" (per-file max-gap read)
    "done": 0,
    "total": 0,
    "records": None,   # JSON-safe record lists on success (None -> no_update)
    "result": None,    # status string once finished
    "error": None,     # traceback string on failure
}

# Full Gap result-table rows kept server-side so the browser only ever holds
# one page (backend pagination). Mirrors the other module caches — fine for a
# single-user local app. Rows are the table's own columns; store-gaps carries
# only the fields the 3D/Effective-Gap callbacks consume.
_GAP_TABLE_LOCK = threading.Lock()
_GAP_TABLE = {"rows": []}  # List[dict] with keys = table_paging.COLUMN_IDS


def _set_gap_rows(rows):
    with _GAP_TABLE_LOCK:
        _GAP_TABLE["rows"] = rows


def _get_gap_rows():
    with _GAP_TABLE_LOCK:
        return list(_GAP_TABLE["rows"])


# NOTE on the polling contract: the worker's outcome (result/error) is kept in
# the state dict until the NEXT run starts — the poller must NOT clear it on
# read. The interval can tick faster than a callback round-trip, and
# dash-renderer discards a response whose n_intervals input changed while the
# request was in flight. With a destructive read, the one discarded response
# could carry the only copy of the results and the UI would hang on a full
# progress bar forever (this was a real bug). Publishing is idempotent, so
# re-sending the outcome every tick until the interval is switched off is safe.


# ---------------------------------------------------------------------------
# ChartOptions assembled from the sidebar controls. Each tab (2D / 3D / Gap)
# owns an independent control set with its own id prefix, so styling is
# configured per tab rather than shared.
# ---------------------------------------------------------------------------

def _build_options(prefix, values) -> charts.ChartOptions:
    """Assemble ChartOptions from one tab's control ``values``.

    ``values`` are positional in ``layout.tab_option_suffixes(prefix)`` order.
    Fields a tab does not render (e.g. contour levels on 3D, colorscale on the
    line chart) are simply absent and fall back to ChartOptions defaults.
    """
    d = dict(zip(layout.tab_option_suffixes(prefix), values))

    def _int(v):
        return int(v) if v is not None and v != "" else None

    def _float(v):
        return float(v) if v is not None and v != "" else None

    toggles = d.get("toggles") or []
    font_size = d.get("font-size")
    title_size = d.get("title-size")
    tick_size = d.get("tick-size")

    return charts.ChartOptions(
        title=d.get("title") or "",
        font_family=d.get("font-family") or "Arial",
        font_size=int(font_size) if font_size else 12,
        title_font_size=int(title_size) if title_size else 16,
        tick_font_size=int(tick_size) if tick_size else 10,
        x_tick_step=_float(d.get("x-dtick")),
        y_tick_step=_float(d.get("y-dtick")),
        colorscale=d.get("colorscale") or "Jet",
        reverse_colorscale="reverse" in toggles,
        show_colorbar="colorbar" in toggles,
        show_shape="shape" in toggles,
        match_aspect="aspect" in toggles,
        zmin=_float(d.get("zmin")),
        zmax=_float(d.get("zmax")),
        contour_levels=_int(d.get("contour-levels")),
        width=_int(d.get("width")),
        height=_int(d.get("height")),
    )


def _export_image_kwargs(width, height, scale):
    """Sanitize export size inputs into save-figure kwargs (invalid/empty -> omitted).

    Inputs may arrive as None, "", numbers, or junk strings and must never
    raise. width/height coerce to a positive int; scale to a positive float.
    Anything else (blank, zero, negative, non-numeric) is omitted so the
    exporter falls back to the figure's own size / scale 1
    (`charts_mpl.save_figure`).
    """
    kwargs = {}

    def _pos_int(v):
        try:
            n = int(float(v))
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    def _pos_float(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f > 0 else None

    w = _pos_int(width)
    if w is not None:
        kwargs["width"] = w
    h = _pos_int(height)
    if h is not None:
        kwargs["height"] = h
    s = _pos_float(scale)
    if s is not None:
        kwargs["scale"] = s
    return kwargs


def _downsample_for_export(values, cap):
    """Stride-downsample a 2D array so max(shape) <= cap (0/None = off).

    Striding (not interpolation) so the NaN/blank pattern is preserved and no
    core-layer semantics change: cap falsy or <= 0 returns the array unchanged;
    an array already within the cap is unchanged; otherwise a stride
    ``k = ceil(max(shape) / cap)`` is applied to both axes.
    """
    if not cap or cap <= 0:
        return values
    longest = max(values.shape)
    if longest <= cap:
        return values
    k = -(-longest // cap)  # ceil(longest / cap)
    return values[::k, ::k]


def _option_states(prefix):
    """State() list for one tab's chart-option controls, in build order."""
    return [State("{0}-{1}".format(prefix, sfx), "value")
            for sfx in layout.tab_option_suffixes(prefix)]


def _option_inputs(prefix):
    """Input() list (live re-render) for one tab's chart-option controls."""
    return [Input("{0}-{1}".format(prefix, sfx), "value")
            for sfx in layout.tab_option_suffixes(prefix)]


# option-control counts per tab (used to slice *rest in render callbacks)
_N2D = len(layout.tab_option_suffixes("opt2d"))
_N3D = len(layout.tab_option_suffixes("opt3d"))

# Data-transform controls (Data Options panel). Order matters: the first four
# feed the TOP config (flip / rotate / zero cell), the last two the BTM config
# (zero cell only).
_TRANSFORM_STATES = [
    State("data-top-flip", "value"),
    State("data-top-rotate", "value"),
    State("data-top-zero-row", "value"),
    State("data-top-zero-col", "value"),
    State("data-btm-zero-row", "value"),
    State("data-btm-zero-col", "value"),
]

# As Inputs, so previews re-render live when transforms change.
_TRANSFORM_INPUTS = [Input(s.component_id, s.component_property)
                     for s in _TRANSFORM_STATES]


def _transform_configs(transform_values):
    """(top_config, btm_config) from the _TRANSFORM_STATES value tuple."""
    flip, rotate, tz_row, tz_col, bz_row, bz_col = transform_values
    top_cfg = helpers.build_transform_config(flip, rotate, tz_row, tz_col)
    btm_cfg = helpers.build_transform_config(None, 0, bz_row, bz_col)
    return top_cfg, btm_cfg


def _config_for_kind(kind, top_cfg, btm_cfg):
    """Transform config for a dataset kind; GAP data is never transformed."""
    if kind == "TOP":
        return top_cfg
    if kind == "BTM":
        return btm_cfg
    return None


def _empty_fig(message: str = ""):
    fig = charts.go.Figure()
    fig.update_layout(
        annotations=[dict(text=message, showarrow=False,
                          xref="paper", yref="paper", x=0.5, y=0.5)],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# dataset key helpers: a selected 3D dataset is identified by a string key.
#   input datasets:   "meta::<path>"
#   computed gaps:    "gap::<out_name>"
# ---------------------------------------------------------------------------

def _meta_key(path: str) -> str:
    return "meta::" + path


def _gap_key(out_name: str) -> str:
    return "gap::" + out_name


def _stem_for_key(key):
    """Source-file stem for a 3D dataset key ('gap::name' / 'meta::path')."""
    if key.startswith("gap::"):
        return os.path.splitext(key[len("gap::"):])[0]
    if key.startswith("meta::"):
        return os.path.splitext(os.path.basename(key[len("meta::"):]))[0]
    return key


def _phase_lookup(store_metas) -> dict:
    """path -> (sample_no, phase, temp_c) for every scanned meta, per kind.

    Phase is derived per kind so peak-time assignment sees each sample's full
    set of measurements (see :func:`helpers.phase_entries`).
    """
    out = {}
    for kind in ("TOP", "BTM", "GAP", "OUT"):
        for e in helpers.phase_entries(_kind_metas(store_metas, kind)):
            out[e["meta"]["path"]] = (e["sample_no"], e["phase"], e["temp_c"])
    return out


def _group_key_for(key, phase_lookup):
    """(sample_no, phase, temp_c) grouping key for a 3D dataset key, or None.

    Computed gaps (``gap::name``) parse straight from the output name; scanned
    ``meta::path`` datasets read the precomputed phase lookup. None when the
    key cannot be resolved to a sample/phase/temperature (kept ungrouped).
    """
    if key.startswith("gap::"):
        parsed = helpers.parse_gap_name(key[len("gap::"):])
        if parsed:
            return (parsed["top_no"], parsed["phase"], parsed["temp_c"])
        return None
    if key.startswith("meta::"):
        return phase_lookup.get(key[len("meta::"):])
    return None


def _uniquify(base, used):
    """base, or base_2 / base_3 / ... until unused; records the pick."""
    name = base
    n = 2
    while name in used:
        name = "{0}_{1}".format(base, n)
        n += 1
    used.add(name)
    return name


def _grouped_3d_items(options_by_kind, store_metas):
    """Group the 3D tab's (filtered) dropdown options into combined-image jobs.

    ``options_by_kind``: list of ``(kind, options)`` where options are the
    dropdown option dicts ({"label", "value"}). Datasets that share a
    (sample_no, phase, temperature) point are combined into ONE image — the
    same multi-surface overlay the 3D chart shows — instead of one image per
    dataset. Returns a list of
    ``{"filename", "label", "members": [{"key", "kind", "label"}, ...]}``.

    Combined groups are named ``PT{sample:04d}-{phase}{temp}C_3D.png``; options
    that cannot be resolved to a point fall back to their own single-surface
    image (``{KIND}_{stem}_3D.png``). Duplicate names get ``_2``, ``_3``, ...
    """
    phase_lookup = _phase_lookup(store_metas)
    groups = {}          # group_key -> list of member dicts
    order = []           # group_keys in first-seen order
    ungrouped = []       # members without a resolvable group key
    for kind, options in options_by_kind:
        for o in options or []:
            key = o.get("value")
            if not key:
                continue
            member = {"key": key, "kind": kind, "label": o.get("label") or key}
            gk = _group_key_for(key, phase_lookup)
            if gk is None:
                ungrouped.append(member)
            elif gk in groups:
                groups[gk].append(member)
            else:
                groups[gk] = [member]
                order.append(gk)

    used = set()
    items = []
    for gk in order:
        sample_no, phase, temp_c = gk
        label = "PT{0:04d}-{1}{2}C".format(sample_no, phase, temp_c)
        name = _uniquify(label + "_3D", used)
        items.append({"filename": name + ".png", "label": label,
                      "members": groups[gk]})
    for member in ungrouped:
        base = "{0}_{1}_3D".format(member["kind"], _stem_for_key(member["key"]))
        name = _uniquify(base, used)
        items.append({"filename": name + ".png", "label": member["label"],
                      "members": [member]})
    return items


def _all_meta_dicts(store_metas) -> List[dict]:
    out = []
    for kind in ("TOP", "BTM", "GAP", "OUT"):
        out.extend((store_metas or {}).get(kind, []))
    return out


def _find_meta(store_metas, path) -> Optional[dict]:
    for d in _all_meta_dicts(store_metas):
        if d["path"] == path:
            return d
    return None


def _resolve_values(key: str, store_metas, top_cfg=None, btm_cfg=None):
    """Return ndarray for a dataset key, loading/caching as needed.

    Input datasets get the transform for their kind (TOP: flip/rotate/zero,
    BTM: zero); computed gaps are returned as-is. May raise ValueError when a
    zero cell is out of bounds or blank.
    """
    if key.startswith("gap::"):
        out_name = key[len("gap::"):]
        return helpers.get_gap(out_name)
    if key.startswith("meta::"):
        path = key[len("meta::"):]
        md = _find_meta(store_metas, path)
        if md is None:
            return None
        cfg = _config_for_kind(md.get("kind"), top_cfg, btm_cfg)
        return helpers.transformed_matrix(md, cfg)
    return None


def _key_label(key: str, store_metas) -> str:
    if key.startswith("gap::"):
        return "GAP " + key[len("gap::"):]
    if key.startswith("meta::"):
        md = _find_meta(store_metas, key[len("meta::"):])
        if md:
            return helpers.meta_label_from_dict(md)
    return key


# ---------------------------------------------------------------------------
# selection helpers (sample / phase+temperature pickers).
# ---------------------------------------------------------------------------

def _kind_metas(store_metas, kind: str) -> List[dict]:
    return (store_metas or {}).get(kind, [])


def _sample_options(meta_dicts: List[dict]) -> List[dict]:
    """Dropdown options: one entry per distinct sample number."""
    counts = {}
    for d in meta_dicts:
        try:
            counts[int(d["sample_no"])] = counts.get(int(d["sample_no"]), 0) + 1
        except (KeyError, TypeError, ValueError):
            continue
    return [
        {"label": "PT{0:04d} ({1} files)".format(no, counts[no]), "value": no}
        for no in sorted(counts)
    ]


def _sample_phase_temps(meta_dicts: List[dict], sample_no) -> "set":
    """Set of (phase, temp_c) available for one sample within a kind."""
    if sample_no is None:
        return set()
    return {
        (e["phase"], e["temp_c"])
        for e in helpers.phase_entries(meta_dicts)
        if e["sample_no"] == int(sample_no)
    }


def _entry_for(meta_dicts: List[dict], sample_no, phase_temp: str) -> Optional[dict]:
    """First phase entry matching a sample and an encoded 'H240' key."""
    for e in helpers.phase_entries(meta_dicts):
        if e["sample_no"] == int(sample_no) and \
                helpers.phase_temp_key(e["phase"], e["temp_c"]) == phase_temp:
            return e
    return None


def _phase_label(entry: dict) -> str:
    """Dropdown label with the phase folded in, e.g. 'TOP PT0001 H240C 192s'.

    Gap-named files render as 'GAP TOP1-BTM12 H250C'.
    """
    meta = entry["meta"]
    if meta.get("kind") == "GAP" and meta.get("btm_no") is not None:
        return "GAP TOP{top}-BTM{btm} {phase}{temp}C".format(
            top=meta.get("sample_no", "?"), btm=meta.get("btm_no"),
            phase=entry["phase"], temp=entry["temp_c"])
    try:
        sample = "PT{0:04d}".format(int(meta.get("sample_no")))
    except (TypeError, ValueError):
        sample = "PT????"
    return "{kind} {sample} {phase}{temp}C {time}s".format(
        kind=meta.get("kind", "?"),
        sample=sample,
        phase=entry["phase"],
        temp=entry["temp_c"],
        time=entry["time_s"],
    )


# ---------------------------------------------------------------------------
# Resize-preview helpers: mirror the pipeline's reference/resize rules so the
# 2D/3D views can show data exactly as the gap computation will consume it.
# ---------------------------------------------------------------------------

def _resize_pair(top_vals, btm_vals, reference):
    """Resize the non-reference side of a TOP/BTM pair to the reference grid.

    Same rules as run_pipeline: AUTO picks the smaller element count (tie ->
    TOP); both sides are matched to the larger blank (union of each side's
    center-fit/cropped blank).

    Returns (top_vals, btm_vals, error_message).
    """
    from matrix2d.core.resize import resize_pair

    ref = reference if reference in ("TOP", "BTM") else (
        "TOP" if top_vals.size <= btm_vals.size else "BTM")
    try:
        top_vals, btm_vals = resize_pair(top_vals, btm_vals, ref)
    except ValueError as exc:
        return top_vals, btm_vals, "Resize failed: {0}".format(exc)
    return top_vals, btm_vals, ""


def _pick_reference_record(records, reference):
    """Choose the reference dataset among 3D-selected TOP/BTM records.

    Explicit TOP/BTM restricts the pool to that kind (falls back to all
    input records when the kind is not selected); AUTO uses every input
    record. Smallest element count wins, ties prefer TOP.
    """
    pool = records
    if reference in ("TOP", "BTM"):
        of_kind = [r for r in records if r["kind"] == reference]
        if of_kind:
            pool = of_kind
    return min(pool, key=lambda r: (r["values"].size, r["kind"] != "TOP"))


# ---------------------------------------------------------------------------
# Parallel matplotlib batch-export machinery, shared by the Gap-tab and 3D-tab
# "Save All …" workers. Module-level so both workers reuse one implementation.
# build_figs returns matplotlib Figures; each is written with fig.savefig via
# charts_mpl.save_figure (no kaleido / Chromium). matplotlib's Agg draw releases
# the GIL, so the worker pool still parallelises rendering.
# ---------------------------------------------------------------------------

def _export_worker_count(n_items):
    """Render worker count: MATRIX2D_EXPORT_WORKERS (default 4), clamped
    to [1, 8] and capped at the number of items (never spawn idle workers)."""
    try:
        n = int(os.environ.get("MATRIX2D_EXPORT_WORKERS", "4"))
    except (TypeError, ValueError):
        n = 4
    n = max(1, min(8, n))
    return max(1, min(n, n_items))


def _pooled_figure_export(items, build_figs, out_dir, img_kwargs, on_progress):
    """Render+write PNGs for ``items`` on a parallel matplotlib worker pool.

    ``items``: list of ``(label, payload)``. ``build_figs(payload)`` returns
    ``(figs, err)`` — figs a list of ``(filename, matplotlib_figure)``, err a
    failure message (item skipped, no images) or None. Each figure is written
    with :func:`charts_mpl.save_figure` (honouring ``img_kwargs`` width/height/
    scale) and then closed to free memory.
    ``on_progress(done_items)`` fires after each item completes.
    Returns ``(saved_image_count, failed_messages)`` with failures ordered
    by item index.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Shared, lock-guarded progress/result state across render workers.
    pool_lock = threading.Lock()
    counters = {"done": 0, "saved": 0}
    failed = []  # list of (index, message), sorted by index at the end

    def _render_one(i, label, payload):
        """Render+write one item's images. Returns images written for this
        item (partial writes still count)."""
        written = 0
        try:
            figs, err = build_figs(payload)
            if err is not None:
                with pool_lock:
                    failed.append((i, "{0} ({1})".format(label, err)))
                return 0
            for filename, fig in figs:
                path = os.path.join(out_dir, filename)
                try:
                    charts_mpl.save_figure(fig, path, img_kwargs)
                finally:
                    fig.clear()
                written += 1
        except Exception as exc:  # noqa: BLE001 - keep exporting the rest
            logger.exception("Batch image export failed for %r", label)
            with pool_lock:
                failed.append((i, "{0} ({1})".format(label, exc)))
        return written

    def _finish_item(written):
        with pool_lock:
            counters["done"] += 1
            counters["saved"] += written
            done = counters["done"]
        on_progress(done)

    work = queue.Queue()
    for item in enumerate(items):
        work.put(item)
    n_workers = _export_worker_count(len(items))
    logger.info("Batch image export: %d matplotlib worker(s)", n_workers)

    def _pool_worker():
        while True:
            try:
                i, (label, payload) = work.get_nowait()
            except queue.Empty:
                break
            try:
                _finish_item(_render_one(i, label, payload))
            except Exception:  # noqa: BLE001 - never die silently
                logger.exception("Export pool worker item crashed")
                with pool_lock:
                    failed.append(
                        (i, "{0} (unexpected worker error)".format(label)))
                _finish_item(0)

    threads = [threading.Thread(target=_pool_worker,
                                name="export-pool-%d" % w,
                                daemon=True)
               for w in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return (counters["saved"],
            [m for _, m in sorted(failed, key=lambda x: x[0])])


def register_callbacks(app):
    # -------------------------------------------------------------------
    # 0. Folder Browse... buttons -> native directory dialog. One callback
    #    per folder input; cancel / headless -> no_update. The ✕ button next
    #    to each input clears that path (scan skips empty folders). "img" is
    #    the Image Export save folder (blank -> exports fall back to OUT).
    # -------------------------------------------------------------------
    for _kind in ("top", "btm", "gap", "out", "img"):
        @app.callback(
            Output("folder-{0}".format(_kind), "value"),
            Input("btn-browse-{0}".format(_kind), "n_clicks"),
            State("folder-{0}".format(_kind), "value"),
            prevent_initial_call=True,
        )
        def browse_folder(_n, current, _kind=_kind):  # bind loop var
            path = pick_folder(current or "")
            return path if path else no_update

        @app.callback(
            Output("folder-{0}".format(_kind), "value", allow_duplicate=True),
            Input("btn-clear-{0}".format(_kind), "n_clicks"),
            prevent_initial_call=True,
        )
        def clear_folder(_n, _kind=_kind):  # bind loop var
            logger.info("Folder path cleared: %s", _kind.upper())
            return ""

    # -------------------------------------------------------------------
    # 0b. Show only the active tab's Chart Options panel in the sidebar.
    # -------------------------------------------------------------------
    @app.callback(
        Output("chart-options-tab-2d", "style"),
        Output("chart-options-tab-3d", "style"),
        Output("chart-options-tab-gap", "style"),
        Output("chart-options-tab-effgap", "style"),
        Input("tabs", "value"),
    )
    def toggle_chart_options(active_tab):
        shown, hidden = {}, {"display": "none"}
        return (
            shown if active_tab == "tab-2d" else hidden,
            shown if active_tab == "tab-3d" else hidden,
            shown if active_tab == "tab-gap" else hidden,
            shown if active_tab == "tab-effgap" else hidden,
        )

    # -------------------------------------------------------------------
    # 1. Scan folders -> store metas, show counts. The button starts a
    #    background thread; a dcc.Interval polls the shared _SCAN state to
    #    drive the progress bar and, when the scan finishes, publish the
    #    metas. Dropdown options are derived reactively from the store by the
    #    callbacks below. Mirrors the gap-compute pattern.
    # -------------------------------------------------------------------
    def _scan_worker(top_dir, btm_dir, gap_dir, out_dir):
        from matrix2d.services.repository import list_data_files, scan_folder

        result = {"TOP": [], "BTM": [], "GAP": [], "OUT": []}
        errors = []
        skipped = {}  # per-folder count of skipped (invalid) candidate files
        # (store key, parse kind, folder). OUT files use the gap output naming,
        # so they parse with the GAP format while staying in their own bucket.
        specs = [("TOP", "TOP", top_dir), ("BTM", "BTM", btm_dir),
                 ("GAP", "GAP", gap_dir), ("OUT", "GAP", out_dir)]

        logger.info("Scan started: TOP=%r BTM=%r GAP=%r OUT=%r",
                    top_dir, btm_dir, gap_dir, out_dir)

        # pre-count files across all set folders for one grand total
        active = [(k, pk, f) for k, pk, f in specs if f]
        try:
            grand_total = 0
            offsets = {}
            folder_totals = {}  # key -> candidate file count (None if unknown)
            for key, _pk, folder in active:
                offsets[key] = grand_total
                try:
                    n = len(list_data_files(folder))
                except Exception:  # noqa: BLE001 - count failure -> scan reports it
                    logger.warning("Scan pre-count failed for %s folder %r",
                                   key, folder, exc_info=True)
                    folder_totals[key] = None
                    continue
                folder_totals[key] = n
                grand_total += n
            with _SCAN_LOCK:
                _SCAN["total"] = grand_total

            for key, parse_kind, folder in active:
                offset = offsets.get(key, 0)

                def _on_progress(done, _total, _offset=offset):
                    with _SCAN_LOCK:
                        _SCAN["done"] = _offset + done

                try:
                    metas = scan_folder(folder, parse_kind,
                                        progress_cb=_on_progress)
                    result[key] = [helpers.meta_to_dict(m) for m in metas]
                    total_files = folder_totals.get(key)
                    if total_files is not None:
                        skipped[key] = max(0, total_files - len(metas))
                    if not metas:
                        # invalid-format files are skipped during scan, so an
                        # empty result means the folder holds no usable data.
                        errors.append(
                            "{0}: 데이터 없음 (no valid data files)".format(key))
                except Exception as exc:  # noqa: BLE001 - surface any core error
                    logger.exception("Scan failed for %s folder %r", key, folder)
                    errors.append("{key}: {exc}".format(key=key, exc=exc))
        except Exception as exc:  # noqa: BLE001 - never lose the outcome
            # Without this, an unexpected error would kill the thread before
            # "result" is published and the UI would wait forever.
            logger.exception("Scan worker crashed")
            errors.append("scan worker crashed: {0}".format(exc))
        finally:
            with _SCAN_LOCK:
                _SCAN["result"] = result
                _SCAN["errors"] = errors
                _SCAN["skipped"] = skipped
                _SCAN["running"] = False
            logger.info("Scan finished: TOP=%d BTM=%d GAP=%d OUT=%d, %d error(s)",
                        len(result["TOP"]), len(result["BTM"]),
                        len(result["GAP"]), len(result["OUT"]), len(errors))

    @app.callback(
        Output("scan-progress-interval", "disabled"),
        Output("btn-scan", "disabled"),
        Output("scan-progress-bar", "style"),
        Output("scan-progress-label", "children"),
        Input("btn-scan", "n_clicks"),
        State("folder-top", "value"),
        State("folder-btm", "value"),
        State("folder-gap", "value"),
        State("folder-out", "value"),
        prevent_initial_call=True,
    )
    def start_scan(_n, top_dir, btm_dir, gap_dir, out_dir):
        if not top_dir and not btm_dir and not gap_dir and not out_dir:
            return (True, False, {"width": "0%"},
                    "Set at least one of TOP / BTM / GAP / OUT folders.")
        with _SCAN_LOCK:
            if _SCAN["running"]:
                logger.info("Scan request ignored: a scan is already running")
                return no_update, no_update, no_update, no_update
            _SCAN.update(running=True, done=0, total=0,
                         result=None, errors=None, skipped=None)
        threading.Thread(
            target=_scan_worker,
            args=(top_dir, btm_dir, gap_dir, out_dir),
            name="scan-worker",
            daemon=True,
        ).start()
        return False, True, {"width": "0%"}, "Scanning..."

    @app.callback(
        Output("store-metas", "data"),
        Output("scan-status", "children"),
        Output("scan-progress-interval", "disabled", allow_duplicate=True),
        Output("btn-scan", "disabled", allow_duplicate=True),
        Output("scan-progress-bar", "style", allow_duplicate=True),
        Output("scan-progress-label", "children", allow_duplicate=True),
        Input("scan-progress-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_scan(_n):
        with _SCAN_LOCK:
            running = _SCAN["running"]
            done, total = _SCAN["done"], _SCAN["total"]
            result, errors = _SCAN["result"], _SCAN["errors"]
            skipped = _SCAN["skipped"]

        pct = (100.0 * done / total) if total else 0.0
        bar = {"width": "{0:.0f}%".format(pct)}
        label = ("{0} / {1} files".format(done, total) if total
                 else "Scanning...")

        if running:
            return (no_update, no_update, no_update, no_update, bar, label)

        if result is None:
            # no outcome pending (e.g. a tick right after a fresh page load)
            logger.debug("Scan poll: no pending outcome, disabling interval")
            return (no_update, no_update, True, False, no_update, no_update)

        logger.info("Scan poll: publishing result to UI (%d error(s))",
                    len(errors or []))
        skipped = skipped or {}

        def _count(k):
            base = "{k}={n}".format(k=k, n=len(result.get(k, [])))
            n_skip = skipped.get(k) or 0
            return base + (" (skipped {0})".format(n_skip) if n_skip else "")

        counts = "  ".join(_count(k) for k in ("TOP", "BTM", "GAP", "OUT"))
        status_children = [html.Span("Scanned: " + counts)]
        if errors:
            status_children.append(
                html.Div("Errors: " + " | ".join(errors), className="error"))

        return (result, status_children, True, False,
                {"width": "100%"},
                "{0} / {0} files — done".format(total))

    # -------------------------------------------------------------------
    # 2. 2D view: sample pickers -> common temperature picker -> two charts.
    # -------------------------------------------------------------------
    @app.callback(
        Output("view2d-top-sample", "options"),
        Output("view2d-btm-sample", "options"),
        Output("view2d-top-sample", "value"),
        Output("view2d-btm-sample", "value"),
        Input("store-metas", "data"),
        State("view2d-top-sample", "value"),
        State("view2d-btm-sample", "value"),
        prevent_initial_call=True,
    )
    def update_2d_sample_options(store_metas, cur_top, cur_btm):
        top_opts = _sample_options(_kind_metas(store_metas, "TOP"))
        btm_opts = _sample_options(_kind_metas(store_metas, "BTM"))

        def _prune(cur, opts):
            return cur if any(o["value"] == cur for o in opts) else None

        return (top_opts, btm_opts,
                _prune(cur_top, top_opts), _prune(cur_btm, btm_opts))

    @app.callback(
        Output("view2d-temp", "options"),
        Output("view2d-temp", "value"),
        Input("view2d-top-sample", "value"),
        Input("view2d-btm-sample", "value"),
        Input("store-metas", "data"),
        State("view2d-temp", "value"),
        prevent_initial_call=True,
    )
    def update_2d_temp_options(top_sample, btm_sample, store_metas, current):
        top_set = _sample_phase_temps(_kind_metas(store_metas, "TOP"), top_sample)
        btm_set = _sample_phase_temps(_kind_metas(store_metas, "BTM"), btm_sample)

        if top_sample is not None and btm_sample is not None:
            pairs = top_set & btm_set
        else:
            pairs = top_set | btm_set

        options = [
            {"label": "{0} {1}C".format(phase, temp),
             "value": helpers.phase_temp_key(phase, temp)}
            for phase, temp in helpers.sort_phase_temps(pairs)
        ]
        value = current if any(o["value"] == current for o in options) else None
        return options, value

    @app.callback(
        Output("view2d-graph-top", "figure"),
        Output("view2d-graph-btm", "figure"),
        Output("view2d-error", "children"),
        [Input("view2d-top-sample", "value"),
         Input("view2d-btm-sample", "value"),
         Input("view2d-temp", "value"),
         Input("view2d-type", "value"),
         Input("data-show-resized", "value"),
         Input("gap-reference", "value")]
        + _option_inputs("opt2d") + _TRANSFORM_INPUTS,
        State("store-metas", "data"),
        prevent_initial_call=True,
    )
    def render_2d(top_sample, btm_sample, phase_temp, chart_type,
                  show_resized, reference, *rest):
        option_values = rest[:_N2D]
        transform_values = rest[_N2D:-1]
        store_metas = rest[-1]
        if top_sample is None and btm_sample is None:
            return (_empty_fig("Select a TOP sample"),
                    _empty_fig("Select a BTM sample"), "")
        try:
            opts = _build_options("opt2d", option_values)
            top_cfg, btm_cfg = _transform_configs(transform_values)

            def _side_values(kind, sample_no):
                """(values, entry, message): values None -> show message."""
                if sample_no is None:
                    return None, None, "Select a {0} sample".format(kind)
                if not phase_temp:
                    return None, None, "Select a temperature"
                entry = _entry_for(_kind_metas(store_metas, kind),
                                   sample_no, phase_temp)
                if entry is None:
                    return None, None, \
                        "No {0} file at {1}".format(kind, phase_temp)
                cfg = _config_for_kind(kind, top_cfg, btm_cfg)
                try:
                    values = helpers.transformed_matrix(entry["meta"], cfg)
                except ValueError as exc:  # zero cell out of bounds / blank
                    return None, entry, "{0}: {1}".format(kind, exc)
                return values, entry, None

            top_vals, top_entry, top_msg = _side_values("TOP", top_sample)
            btm_vals, btm_entry, btm_msg = _side_values("BTM", btm_sample)

            # Optional resize preview: only meaningful with both sides loaded.
            resize_err = ""
            if show_resized == "resized" \
                    and top_vals is not None and btm_vals is not None:
                top_vals, btm_vals, resize_err = _resize_pair(
                    top_vals, btm_vals, reference)

            def _side_fig(values, entry, msg):
                if values is None:
                    return _empty_fig(msg or "")
                side_opts = dataclasses.replace(
                    opts, title=opts.title or _phase_label(entry))
                if chart_type == "heatmap":
                    return charts.heatmap_2d(values, side_opts)
                return charts.contour_2d(values, side_opts)

            fig_top = _side_fig(top_vals, top_entry, top_msg)
            fig_btm = _side_fig(btm_vals, btm_entry, btm_msg)
            return fig_top, fig_btm, resize_err
        except Exception:  # noqa: BLE001
            logger.exception("2D render failed")
            return (_empty_fig(), _empty_fig(),
                    "Render error: " + traceback.format_exc(limit=2))

    # -------------------------------------------------------------------
    # 3. 3D view: filters -> per-kind dataset options -> offsets -> render.
    # -------------------------------------------------------------------
    @app.callback(
        Output("view3d-filter-sample", "options"),
        Output("view3d-filter-temp", "options"),
        Output("view3d-filter-sample", "value"),
        Output("view3d-filter-temp", "value"),
        Input("store-metas", "data"),
        Input("store-gaps", "data"),
        State("view3d-filter-sample", "value"),
        State("view3d-filter-temp", "value"),
        prevent_initial_call=True,
    )
    def update_3d_filter_options(store_metas, store_gaps, cur_samples, cur_temps):
        samples = set()
        temps = set()
        for d in _all_meta_dicts(store_metas):
            try:
                samples.add(int(d["sample_no"]))
                temps.add(int(d["temp_c"]))
            except (KeyError, TypeError, ValueError):
                continue
            try:  # gap-named files carry a second (BTM) sample number
                if d.get("btm_no") is not None:
                    samples.add(int(d["btm_no"]))
            except (TypeError, ValueError):
                pass
        for s in store_gaps or []:
            parsed = helpers.parse_gap_name(s.get("out_name", ""))
            if parsed:
                samples.add(parsed["top_no"])
                samples.add(parsed["btm_no"])
                temps.add(parsed["temp_c"])
        sample_opts = [{"label": "PT{0:04d}".format(n), "value": n}
                       for n in sorted(samples)]
        temp_opts = [{"label": "{0}C".format(t), "value": t}
                     for t in sorted(temps)]
        new_samples = [v for v in (cur_samples or []) if v in samples]
        new_temps = [v for v in (cur_temps or []) if v in temps]
        return sample_opts, temp_opts, new_samples, new_temps

    @app.callback(
        Output("view3d-top", "options"),
        Output("view3d-btm", "options"),
        Output("view3d-gap", "options"),
        Output("view3d-out", "options"),
        Output("view3d-top", "value"),
        Output("view3d-btm", "value"),
        Output("view3d-gap", "value"),
        Output("view3d-out", "value"),
        Input("store-metas", "data"),
        Input("store-gaps", "data"),
        Input("view3d-filter-sample", "value"),
        Input("view3d-filter-temp", "value"),
        State("view3d-top", "value"),
        State("view3d-btm", "value"),
        State("view3d-gap", "value"),
        State("view3d-out", "value"),
        prevent_initial_call=True,
    )
    def update_3d_dataset_options(store_metas, store_gaps, f_samples, f_temps,
                                  cur_top, cur_btm, cur_gap, cur_out):
        f_samples = set(f_samples or [])
        f_temps = set(f_temps or [])

        def _matches_sample(e):
            if not f_samples:
                return True
            if e["sample_no"] in f_samples:
                return True
            btm = e["meta"].get("btm_no")  # gap-named files match either no.
            try:
                return btm is not None and int(btm) in f_samples
            except (TypeError, ValueError):
                return False

        def _meta_opts(kind):
            opts = []
            for e in helpers.phase_entries(_kind_metas(store_metas, kind)):
                if not _matches_sample(e):
                    continue
                if f_temps and e["temp_c"] not in f_temps:
                    continue
                opts.append({"label": _phase_label(e),
                             "value": _meta_key(e["meta"]["path"])})
            return opts

        gap_opts = _meta_opts("GAP")  # scanned GAP folder files
        for s in store_gaps or []:    # computed gap results
            name = s.get("out_name", "")
            # prefer fields precomputed by poll_compute; fall back to parsing
            # for older stored data that predates the precomputed keys.
            top_no, btm_no, temp_c = s.get("top_no"), s.get("btm_no"), s.get("temp_c")
            if top_no is None or btm_no is None or temp_c is None:
                parsed = helpers.parse_gap_name(name)
                if parsed:
                    top_no, btm_no, temp_c = \
                        parsed["top_no"], parsed["btm_no"], parsed["temp_c"]
            if top_no is not None and btm_no is not None and temp_c is not None:
                if f_samples and not (top_no in f_samples or btm_no in f_samples):
                    continue
                if f_temps and temp_c not in f_temps:
                    continue
            gap_opts.append({"label": "GAP " + name, "value": _gap_key(name)})

        # Prune selections whose underlying data is gone (re-scan/recompute).
        # Selections merely hidden by the filters are kept.
        def _valid_meta_keys(kind):
            return {_meta_key(d["path"]) for d in _kind_metas(store_metas, kind)}

        valid_gap = _valid_meta_keys("GAP") | {
            _gap_key(s.get("out_name", "")) for s in store_gaps or []}

        def _prune(cur, valid):
            return [v for v in (cur or []) if v in valid]

        return (_meta_opts("TOP"), _meta_opts("BTM"), gap_opts, _meta_opts("OUT"),
                _prune(cur_top, _valid_meta_keys("TOP")),
                _prune(cur_btm, _valid_meta_keys("BTM")),
                _prune(cur_gap, valid_gap),
                _prune(cur_out, _valid_meta_keys("OUT")))

    # 3b. per-dataset z-offset inputs (pattern-matching). Existing values are
    # carried over so editing the selection does not wipe user-entered offsets.
    @app.callback(
        Output("view3d-offsets", "children"),
        Input("view3d-top", "value"),
        Input("view3d-btm", "value"),
        Input("view3d-gap", "value"),
        Input("view3d-out", "value"),
        State({"type": "z-offset", "key": ALL}, "value"),
        State({"type": "z-offset", "key": ALL}, "id"),
        State("store-metas", "data"),
        prevent_initial_call=True,
    )
    def build_offset_inputs(top_keys, btm_keys, gap_keys, out_keys,
                            prev_values, prev_ids, store_metas):
        prev = {}
        for oid, oval in zip(prev_ids or [], prev_values or []):
            prev[oid["key"]] = oval
        rows = []
        for key in ((top_keys or []) + (btm_keys or [])
                    + (gap_keys or []) + (out_keys or [])):
            label = _key_label(key, store_metas)
            rows.append(html.Div(className="offset-row", children=[
                html.Span(label, className="offset-label"),
                dcc.Input(
                    id={"type": "z-offset", "key": key},
                    type="number", value=prev.get(key, 0.0), step=0.1,
                    className="offset-input",
                ),
            ]))
        return rows

    # 3c. 3D render (reacts to selections, offsets, options).
    @app.callback(
        Output("view3d-graph", "figure"),
        Output("view3d-error", "children"),
        [Input("view3d-top", "value"),
         Input("view3d-btm", "value"),
         Input("view3d-gap", "value"),
         Input("view3d-out", "value"),
         Input("data-show-resized", "value"),
         Input("gap-reference", "value"),
         Input("store-gaps", "data"),  # recompute -> re-render gap surfaces
         Input({"type": "z-offset", "key": ALL}, "value"),
         Input({"type": "z-offset", "key": ALL}, "id")]
        + _option_inputs("opt3d") + _TRANSFORM_INPUTS,
        State("store-metas", "data"),
        prevent_initial_call=True,
    )
    def render_3d(top_keys, btm_keys, gap_keys, out_keys, show_resized, reference,
                  _store_gaps, offset_values, offset_ids, *rest):
        option_values = rest[:_N3D]
        transform_values = rest[_N3D:-1]
        store_metas = rest[-1]
        selections = [("TOP", top_keys or []), ("BTM", btm_keys or []),
                      ("GAP", gap_keys or []), ("OUT", out_keys or [])]
        if not any(keys for _k, keys in selections):
            return _empty_fig("Select datasets"), ""
        try:
            # map key -> offset from the pattern-matched inputs
            offset_map = {}
            for oid, oval in zip(offset_ids or [], offset_values or []):
                offset_map[oid["key"]] = float(oval) if oval is not None else 0.0

            opts = _build_options("opt3d", option_values)
            top_cfg, btm_cfg = _transform_configs(transform_values)

            records = []
            problems = []
            for kind, keys in selections:
                for key in keys:
                    try:
                        values = _resolve_values(key, store_metas,
                                                 top_cfg, btm_cfg)
                    except ValueError as exc:  # zero cell out of bounds/blank
                        problems.append("{0} ({1})".format(
                            _key_label(key, store_metas), exc))
                        continue
                    if values is None:
                        problems.append(key)
                        continue
                    records.append({"key": key, "kind": kind,
                                    "values": values})
            if not records:
                return _empty_fig(), "No loadable datasets selected."

            # Optional resize preview: bring every selected TOP/BTM dataset
            # onto one reference grid (GAP surfaces are shown as-is).
            if show_resized == "resized":
                from matrix2d.core.resize import resize_crop_blank

                inputs = [r for r in records if r["kind"] in ("TOP", "BTM")]
                if len(inputs) >= 2:
                    ref = _pick_reference_record(inputs, reference)
                    for r in inputs:
                        if r is ref:
                            continue
                        try:
                            r["values"] = resize_crop_blank(
                                r["values"], ref["values"].shape)
                        except ValueError as exc:
                            problems.append("{0} (resize: {1})".format(
                                _key_label(r["key"], store_metas), exc))

            items = []
            for r in records:
                label = _key_label(r["key"], store_metas)
                if opts.show_shape:
                    # multi_surface_3d takes prebuilt names, so the shape
                    # suffix is folded into the label here.
                    label = "{0} ({1}×{2})".format(
                        label, r["values"].shape[0], r["values"].shape[1])
                items.append((label, r["values"],
                              offset_map.get(r["key"], 0.0)))
            fig = charts.multi_surface_3d(items, opts)
            err = ""
            if problems:
                err = "Skipped / degraded: " + ", ".join(problems)
                logger.warning("3D render skipped/degraded datasets: %s", err)
            return fig, err
        except Exception:  # noqa: BLE001
            logger.exception("3D render failed")
            return _empty_fig(), "Render error: " + traceback.format_exc(limit=2)

    # -------------------------------------------------------------------
    # 4. Gap compute: the button starts a background thread; a dcc.Interval
    #    polls the shared _COMPUTE state to drive the progress bar and, when
    #    the run finishes, publishes the table + result dropdown.
    #    (3D options refresh automatically via the store-gaps Input above.)
    # -------------------------------------------------------------------
    def _compute_worker(top_dir, btm_dir, out_dir, reference,
                        top_cfg, btm_cfg, out_prefix):
        from matrix2d.services.pipeline import run_pipeline

        def _on_progress(done, total):
            with _COMPUTE_LOCK:
                _COMPUTE["done"] = done
                _COMPUTE["total"] = total

        logger.info("Gap compute started: TOP=%r BTM=%r OUT=%r reference=%s "
                    "prefix=%r", top_dir, btm_dir, out_dir, reference,
                    out_prefix)
        try:
            failures = []  # per-failed-job dicts collected by run_pipeline
            results = run_pipeline(top_dir, btm_dir, out_dir,
                                   reference=reference,
                                   top_transform=top_cfg,
                                   btm_transform=btm_cfg,
                                   out_prefix=out_prefix,
                                   progress_cb=_on_progress,
                                   retain_gap=False,
                                   failures=failures)
            # Refresh the gap cache here, in the worker, exactly once per run.
            # The poller may publish the same outcome several times (see the
            # polling-contract note above), so it must stay side-effect free.
            # Clearing only after success keeps store-gaps consistent with the
            # cache when a run fails.
            helpers.clear_gaps()
            for r in results:
                helpers.register_gap(r.job.out_name, r.out_path)
            with _COMPUTE_LOCK:
                _COMPUTE["results"] = results
                _COMPUTE["failures"] = failures
            logger.info("Gap compute finished: %d result(s), %d failed",
                        len(results), len(failures))
        except Exception:  # noqa: BLE001
            logger.exception("Gap compute pipeline crashed")
            with _COMPUTE_LOCK:
                _COMPUTE["error"] = traceback.format_exc(limit=3)
        finally:
            with _COMPUTE_LOCK:
                _COMPUTE["running"] = False

    @app.callback(
        Output("gap-progress-interval", "disabled"),
        Output("btn-compute-gaps", "disabled"),
        Output("gap-error", "children"),
        Output("gap-progress-bar", "style"),
        Output("gap-progress-label", "children"),
        Input("btn-compute-gaps", "n_clicks"),
        [State("folder-top", "value"),
         State("folder-btm", "value"),
         State("folder-out", "value"),
         State("gap-reference", "value"),
         State("gap-out-prefix", "value")]
        + _TRANSFORM_STATES,
        prevent_initial_call=True,
    )
    def start_compute(_n, top_dir, btm_dir, out_dir, reference, out_prefix,
                      *transform_values):
        if not top_dir or not btm_dir or not out_dir:
            return (True, False, "TOP, BTM and OUT folders are required.",
                    {"width": "0%"}, "")
        with _COMPUTE_LOCK:
            if _COMPUTE["running"]:
                logger.info("Compute request ignored: already running")
                return no_update, no_update, no_update, no_update, no_update
            _COMPUTE.update(running=True, done=0, total=0,
                            results=None, error=None, failures=None)
        top_cfg, btm_cfg = _transform_configs(transform_values)
        threading.Thread(
            target=_compute_worker,
            args=(top_dir, btm_dir, out_dir, reference,
                  top_cfg, btm_cfg, out_prefix or ""),
            name="compute-worker",
            daemon=True,
        ).start()
        return False, True, "", {"width": "0%"}, "Scanning folders..."

    @app.callback(
        Output("store-gaps", "data"),
        Output("gap-result-select", "options"),
        Output("gap-result-select", "value"),
        Output("gap-progress-interval", "disabled", allow_duplicate=True),
        Output("btn-compute-gaps", "disabled", allow_duplicate=True),
        Output("gap-error", "children", allow_duplicate=True),
        Output("gap-progress-bar", "style", allow_duplicate=True),
        Output("gap-progress-label", "children", allow_duplicate=True),
        Input("gap-progress-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_compute(_n):
        with _COMPUTE_LOCK:
            running = _COMPUTE["running"]
            done, total = _COMPUTE["done"], _COMPUTE["total"]
            results, error = _COMPUTE["results"], _COMPUTE["error"]
            failures = _COMPUTE["failures"] or []

        pct = (100.0 * done / total) if total else 0.0
        bar = {"width": "{0:.0f}%".format(pct)}
        label = ("{0} / {1} jobs".format(done, total) if total
                 else "Scanning folders...")

        if running:
            return (no_update, no_update, no_update,
                    no_update, no_update, no_update, bar, label)

        if error is not None:
            logger.error("Compute poll: publishing pipeline error to UI:\n%s",
                         error)
            return (no_update, no_update, no_update,
                    True, False, "Pipeline error: " + error, bar, "Failed")
        if results is None:
            # no outcome pending (e.g. a tick right after a fresh page load)
            logger.debug("Compute poll: no pending outcome, disabling interval")
            return (no_update, no_update, no_update,
                    True, False, no_update, no_update, no_update)
        logger.info("Compute poll: publishing %d result(s) to UI", len(results))

        # summaries: trimmed to the fields consumed downstream (3D dataset
        # options + Effective Gap chart) so store-gaps stays small in the
        # browser. rows: the FULL result-table data, kept server-side only
        # (see _GAP_TABLE) so the browser never receives more than one page.
        summaries = []
        rows = []
        for r in results:
            job = r.job
            # parsed once here so downstream callbacks (3D dataset options)
            # don't re-run the gap-name regex on every render.
            parsed = helpers.parse_gap_name(job.out_name)
            # Max gap per result feeds the Effective Gap chart's AVG/STD. It was
            # captured by the pipeline while the array was in memory, so this
            # poll never re-reads the saved files (doing so per tick over a
            # large batch blew past the 400ms interval and the publish response
            # was discarded forever -> progress hung).
            summaries.append({
                "out_name": job.out_name,
                "phase": job.phase,
                "top_no": parsed["top_no"] if parsed else None,
                "btm_no": parsed["btm_no"] if parsed else None,
                "temp_c": parsed["temp_c"] if parsed else None,
                "max_gap": r.max_gap,
            })
            rows.append({
                "out_name": job.out_name,
                "top": os.path.basename(getattr(job.top, "path", "")),
                "btm": os.path.basename(getattr(job.btm, "path", "")),
                "phase": job.phase,
                "offset": "{:.4g}".format(r.result.offset)
                          if r.result.offset is not None else "",
                "out_path": r.out_path,
            })
        _set_gap_rows(rows)

        result_opts = [{"label": s["out_name"], "value": _gap_key(s["out_name"])}
                       for s in summaries]

        # run_pipeline returns successes only (failed jobs are logged and
        # skipped), so an empty result set deserves a warning, not an "ok".
        if not summaries:
            msg = ("Computed 0 gap(s) — no pairs found or every job failed "
                   "(bad zero cell?). See the server log for per-job errors.")
            status = [html.Span(msg, className="error")]
        else:
            status = [html.Span(
                "Computed {n} gap(s).".format(n=len(summaries)), className="ok")]
        # Some jobs may have failed even when others succeeded; surface a count
        # (with up to 5 out_names) without marking the whole run as an error.
        if failures:
            names = [f.get("out_name", "?") for f in failures[:5]]
            listed = ", ".join(names)
            if len(failures) > 5:
                listed += " …"
            status.append(html.Div(
                "{0}건 실패 — 로그 확인: {1}".format(len(failures), listed),
                className="error"))
        # reset the inspect selection: out_names can be identical across runs,
        # so keeping the value would leave a chart of the previous run's data.
        return (summaries, result_opts, None,
                True, False, status, {"width": "100%"},
                "{0} / {0} jobs — done".format(total))

    # 4a-bis. Server-side pagination for the result table. The full row set
    # lives in _GAP_TABLE (server); the browser only ever receives one page.
    # Re-runs on page/sort/filter changes and whenever a new compute publishes
    # store-gaps.
    @app.callback(
        Output("gap-result-table", "data"),
        Output("gap-result-table", "page_count"),
        Input("gap-result-table", "page_current"),
        Input("gap-result-table", "page_size"),
        Input("gap-result-table", "sort_by"),
        Input("gap-result-table", "filter_query"),
        Input("store-gaps", "data"),
        prevent_initial_call=True,
    )
    def update_gap_table_page(page_current, page_size, sort_by, filter_query,
                              _store_gaps):
        rows = _get_gap_rows()
        page, page_count = table_paging.page_view(
            rows, page_current, page_size, sort_by, filter_query)
        return page, page_count

    # 4b. Inspect a chosen computed gap: 2D contour + 3D surface.
    @app.callback(
        Output("gap-graph-2d", "figure"),
        Output("gap-graph-3d", "figure"),
        Output("gap-inspect-error", "children"),
        [Input("gap-result-select", "value"),
         Input("gap-view-type", "value")] + _option_inputs("optgap"),
        prevent_initial_call=True,
    )
    def inspect_gap(gap_key, chart_type, *option_values):
        if not gap_key:
            return _empty_fig("Select a result"), _empty_fig("Select a result"), ""
        try:
            values = _resolve_values(gap_key, {})
            if values is None:
                return _empty_fig(), _empty_fig(), "Gap result not in cache."
            opts = _build_options("optgap", option_values)
            name = gap_key[len("gap::"):]
            if not opts.title:
                opts.title = "GAP " + name
            if chart_type == "heatmap":
                fig2d = charts.heatmap_2d(values, opts)
            else:
                fig2d = charts.contour_2d(values, opts)
            fig3d = charts.surface_3d(values, opts, name=name)
            return fig2d, fig3d, ""
        except Exception:  # noqa: BLE001
            logger.exception("Gap inspect render failed for %r", gap_key)
            return _empty_fig(), _empty_fig(), \
                "Render error: " + traceback.format_exc(limit=2)

    # 4c. Effective Gap tab. The chart reads ready-made records from
    #     store-effgap-records, which is written by one of two callbacks:
    #       - a fresh Gap Compute (store-gaps -> records), OR
    #       - the "Load from OUT files" button (scanned OUT metas -> records).
    #     A fresh compute overwrites any loaded data (and vice versa).

    # 4c-i. Fresh compute -> records. Maps store-gaps summary dicts to
    #       (top_no, btm_no, phase, temp_c, max_gap), skipping entries whose
    #       gap name did not parse (no phase/temp/combo to place them). Records
    #       are JSON-safe lists (not tuples) for the store.
    @app.callback(
        Output("store-effgap-records", "data"),
        Input("store-gaps", "data"),
        prevent_initial_call=True,
    )
    def effgap_records_from_gaps(store_gaps):
        records = []
        for s in (store_gaps or []):
            phase, temp = s.get("phase"), s.get("temp_c")
            top_no, btm_no = s.get("top_no"), s.get("btm_no")
            if None in (phase, temp, top_no, btm_no):
                continue
            records.append([top_no, btm_no, phase, temp, s.get("max_gap")])
        return records

    # 4c-ii. Load button -> records. Scans the OUT folder itself (path from
    #        the sidebar input) so loading already-computed gap files never
    #        depends on the Scan button having run. Matrices load via the
    #        shared repository cache. Runs in a background thread (scan +
    #        per-file loads block the UI on large OUT folders); the root-level
    #        effgap-load-interval polls _EFFLOAD to drive the progress bar and
    #        publish the records (same polling contract as scan/compute).
    def _effgap_load_worker(out_dir):
        from matrix2d.services.repository import read_matrix, scan_folder

        def _progress(stage):
            def cb(done, total):
                with _EFFLOAD_LOCK:
                    _EFFLOAD.update(stage=stage, done=done, total=total)
            return cb

        logger.info("Effective Gap OUT load started: %r", out_dir)
        try:
            try:
                out_metas = scan_folder(out_dir, "GAP",
                                        progress_cb=_progress("scan"))
            except Exception as exc:  # noqa: BLE001 - surface any scan error
                logger.exception("Effective Gap load: OUT scan failed for %r",
                                 out_dir)
                with _EFFLOAD_LOCK:
                    _EFFLOAD["result"] = "Scan error: {0}".format(exc)
                return
            if not out_metas:
                with _EFFLOAD_LOCK:
                    _EFFLOAD["result"] = "OUT 데이터 없음 (no valid gap files)"
                return
            records, skipped = helpers.effgap_records_from_metas(
                out_metas, read_matrix, progress_cb=_progress("load"))
            msg = "Loaded {0} OUT file(s)".format(len(records))
            if skipped:
                msg += " (skipped {0})".format(skipped)
            logger.info("Effective Gap OUT load finished: %d record(s), "
                        "%d skipped", len(records), skipped)
            with _EFFLOAD_LOCK:
                # JSON-safe lists for the store
                _EFFLOAD["records"] = [list(r) for r in records]
                _EFFLOAD["result"] = msg
        except Exception:  # noqa: BLE001 - never lose the outcome
            logger.exception("Effective Gap OUT load worker crashed")
            with _EFFLOAD_LOCK:
                _EFFLOAD["error"] = traceback.format_exc(limit=3)
        finally:
            with _EFFLOAD_LOCK:
                _EFFLOAD["running"] = False

    @app.callback(
        Output("effgap-load-interval", "disabled"),
        Output("btn-effgap-load", "disabled"),
        Output("effgap-load-status", "children"),
        Output("effgap-load-progress-bar", "style"),
        Output("effgap-load-progress-label", "children"),
        Input("btn-effgap-load", "n_clicks"),
        State("folder-out", "value"),
        prevent_initial_call=True,
    )
    def start_effgap_load(_n, out_dir):
        if not out_dir:
            return (True, False,
                    "OUT 폴더를 설정하세요 (set the OUT folder first).",
                    {"width": "0%"}, "")
        with _EFFLOAD_LOCK:
            if _EFFLOAD["running"]:
                logger.info("Effective Gap load request ignored: already running")
                return no_update, no_update, no_update, no_update, no_update
            _EFFLOAD.update(running=True, stage="scan", done=0, total=0,
                            records=None, result=None, error=None)
        threading.Thread(
            target=_effgap_load_worker,
            args=(out_dir,),
            name="effgap-load-worker",
            daemon=True,
        ).start()
        return False, True, "", {"width": "0%"}, "Loading..."

    @app.callback(
        Output("store-effgap-records", "data", allow_duplicate=True),
        Output("effgap-load-status", "children", allow_duplicate=True),
        Output("effgap-load-interval", "disabled", allow_duplicate=True),
        Output("btn-effgap-load", "disabled", allow_duplicate=True),
        Output("effgap-load-progress-bar", "style", allow_duplicate=True),
        Output("effgap-load-progress-label", "children", allow_duplicate=True),
        Input("effgap-load-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_effgap_load(_n):
        with _EFFLOAD_LOCK:
            running = _EFFLOAD["running"]
            stage = _EFFLOAD["stage"]
            done, total = _EFFLOAD["done"], _EFFLOAD["total"]
            records = _EFFLOAD["records"]
            result, error = _EFFLOAD["result"], _EFFLOAD["error"]

        pct = (100.0 * done / total) if total else 0.0
        bar = {"width": "{0:.0f}%".format(pct)}
        stage_label = ("Scanning OUT folder" if stage == "scan"
                       else "Loading gap files")
        label = ("{0}... {1} / {2}".format(stage_label, done, total) if total
                 else "Loading...")

        if running:
            return no_update, no_update, no_update, no_update, bar, label

        if error is not None:
            logger.error("Effective Gap load poll: publishing error to UI:\n%s",
                         error)
            return (no_update, "Load error: " + error, True, False,
                    bar, "Failed")
        if result is None:
            # no outcome pending (e.g. a tick right after a fresh page load)
            logger.debug("Effective Gap load poll: no pending outcome, "
                         "disabling interval")
            return no_update, no_update, True, False, no_update, no_update
        logger.info("Effective Gap load poll: publishing result to UI")
        # records is None when the run ended without data (scan error / empty
        # folder) — keep whatever the chart is currently showing.
        return (records if records is not None else no_update,
                result, True, False, {"width": "100%"},
                "{0} / {0} — done".format(total) if records is not None else "")

    # 4c-iii. Render: AVG (± sample STD) of the combo max-gaps per temperature
    #         point, from whatever records are currently in the store.
    @app.callback(
        Output("effgap-graph", "figure"),
        Output("effgap-error", "children"),
        [Input("store-effgap-records", "data")] + _option_inputs("opteff"),
        prevent_initial_call=True,
    )
    def render_effective_gap(records, *option_values):
        if not records:
            return _empty_fig("Compute gaps first"), ""
        try:
            series = effective_gap_series(records)
            if not series:
                return _empty_fig("No valid temperature points"), ""

            opts = _build_options("opteff", option_values)
            if not opts.title:
                opts.title = "Effective Gap"
            fig = charts.effective_gap_chart(series, opts)
            return fig, ""
        except Exception:  # noqa: BLE001
            logger.exception("Effective Gap render failed")
            return _empty_fig(), "Render error: " + traceback.format_exc(limit=2)

    # -------------------------------------------------------------------
    # 5. Export current 2D / 3D figures as PNG (matplotlib, reconstructed from
    #    the displayed plotly figure). Destination is the
    #    Image Export "Save folder" (sidebar); blank falls back to OUT.
    # -------------------------------------------------------------------
    def _export_dir(img_dir, out_dir):
        return (img_dir or "").strip() or out_dir

    def _export(fig_dict, out_dir, default_name, img_kwargs=None):
        # placeholder figures ("Select a sample" etc.) have no data traces
        if not fig_dict or not fig_dict.get("data"):
            return "Nothing to export."
        if not out_dir:
            return "Set an image save folder (or the OUT folder) first."
        try:
            os.makedirs(out_dir, exist_ok=True)
            fig = charts_mpl.figure_from_plotly_dict(fig_dict)  # matplotlib
            path = os.path.join(out_dir, default_name)
            try:
                charts_mpl.save_figure(fig, path, img_kwargs)
            finally:
                fig.clear()
            logger.info("Exported figure -> %s", path)
            return "Saved: " + path
        except Exception:  # noqa: BLE001
            logger.exception("Figure export failed for %r", default_name)
            return "Export error: " + traceback.format_exc(limit=2)

    @app.callback(
        Output("export2d-status", "children"),
        Input("btn-export-2d", "n_clicks"),
        State("view2d-graph-top", "figure"),
        State("view2d-graph-btm", "figure"),
        State("folder-img", "value"),
        State("folder-out", "value"),
        State("export-img-width", "value"),
        State("export-img-height", "value"),
        State("export-img-scale", "value"),
        prevent_initial_call=True,
    )
    def export_2d(_n, fig_top, fig_btm, img_dir, out_dir,
                  img_w, img_h, img_scale):
        dest = _export_dir(img_dir, out_dir)
        img_kwargs = _export_image_kwargs(img_w, img_h, img_scale)
        msgs = [
            "TOP: " + _export(fig_top, dest, "chart_2d_top.png", img_kwargs),
            "BTM: " + _export(fig_btm, dest, "chart_2d_btm.png", img_kwargs),
        ]
        return [html.Div(m) for m in msgs]

    @app.callback(
        Output("export3d-status", "children"),
        Input("btn-export-3d", "n_clicks"),
        State("view3d-graph", "figure"),
        State("folder-img", "value"),
        State("folder-out", "value"),
        State("export-img-width", "value"),
        State("export-img-height", "value"),
        State("export-img-scale", "value"),
        prevent_initial_call=True,
    )
    def export_3d(_n, fig_dict, img_dir, out_dir, img_w, img_h, img_scale):
        return _export(fig_dict, _export_dir(img_dir, out_dir), "chart_3d.png",
                       _export_image_kwargs(img_w, img_h, img_scale))

    @app.callback(
        Output("export-effgap-status", "children"),
        Input("btn-export-effgap", "n_clicks"),
        State("effgap-graph", "figure"),
        State("folder-img", "value"),
        State("folder-out", "value"),
        State("export-img-width", "value"),
        State("export-img-height", "value"),
        State("export-img-scale", "value"),
        prevent_initial_call=True,
    )
    def export_effgap(_n, fig_dict, img_dir, out_dir, img_w, img_h, img_scale):
        return _export(fig_dict, _export_dir(img_dir, out_dir),
                       "effective_gap.png",
                       _export_image_kwargs(img_w, img_h, img_scale))

    # 5b. Batch export: one 2D contour + one 3D surface PNG per computed gap.
    #    Runs in a background thread (2*N matplotlib renders would otherwise
    #    block the UI for large gap counts); a dcc.Interval polls the shared
    #    _EXPORT state to drive the progress bar and publish the final status,
    #    mirroring the scan/compute pattern above (same polling contract).
    def _build_gap_figs(values, name, opts, chart_type, kinds, downsample):
        """Build the requested (kind, path-stem-suffix, figure) list for one gap.

        Pure figure construction (no I/O) so it is safe to run per worker.
        """
        values = _downsample_for_export(values, downsample)
        gap_opts = dataclasses.replace(opts, title=opts.title or "GAP " + name)
        figs = []
        if "2d" in kinds:
            if chart_type == "heatmap":
                fig2d = charts_mpl.heatmap_2d(values, gap_opts)
            else:
                fig2d = charts_mpl.contour_2d(values, gap_opts)
            figs.append(("_2D.png", fig2d))
        if "3d" in kinds:
            figs.append(("_3D.png",
                         charts_mpl.surface_3d(values, gap_opts, name=name)))
        return figs

    def _export_all_worker(store_gaps, out_dir, chart_type, opts,
                           img_kwargs=None, kinds=None, downsample=0):
        img_kwargs = img_kwargs or {}
        kinds = kinds or ["2d", "3d"]
        total = len(store_gaps)
        with _EXPORT_LOCK:
            _EXPORT["total"] = total
        logger.info("Batch image export started: %d gap(s) -> %r (kinds=%s, "
                    "downsample=%s)", total, out_dir, kinds, downsample)
        try:
            def _build_figs(s):
                """(figs, err) for one gap: figs are (filename, figure) pairs."""
                name = s.get("out_name", "")
                values = helpers.get_gap(name)
                if values is None:
                    return [], "not in cache"
                stem = os.path.splitext(name)[0]
                return ([(stem + suffix, fig)
                         for suffix, fig in _build_gap_figs(
                             values, name, opts, chart_type, kinds, downsample)],
                        None)

            def _on_progress(done):
                with _EXPORT_LOCK:
                    _EXPORT["done"] = done

            items = [(s.get("out_name", ""), s) for s in store_gaps]
            saved, failed_msgs = _pooled_figure_export(
                items, _build_figs, out_dir, img_kwargs, _on_progress)

            msg = "Saved {0} image(s) to {1}.".format(saved, out_dir)
            logger.info("Batch image export finished: %d saved, %d failed -> %s",
                        saved, len(failed_msgs), out_dir)
            if failed_msgs:
                msg += " Failed: " + ", ".join(failed_msgs)
            with _EXPORT_LOCK:
                _EXPORT["result"] = msg
        except Exception:  # noqa: BLE001 - never lose the outcome
            logger.exception("Batch image export worker crashed")
            with _EXPORT_LOCK:
                _EXPORT["error"] = traceback.format_exc(limit=3)
        finally:
            with _EXPORT_LOCK:
                _EXPORT["running"] = False

    def _export_3d_all_worker(items, store_metas, out_dir, opts, img_kwargs,
                              top_cfg, btm_cfg, show_resized, reference):
        """Export one combined 3D surface PNG per filtered 3D-tab group.

        ``items`` are the ``_grouped_3d_items`` dicts (filename/label/members).
        Each group's datasets are overlaid in one ``multi_surface_3d`` figure —
        the same combination the 3D chart shows (TOP/BTM transforms + optional
        resize preview applied, matching :func:`render_3d`). Runs on the shared
        parallel matplotlib pool, same as the Gap batch export.
        """
        from matrix2d.core.resize import resize_crop_blank

        total = len(items)
        with _EXPORT3D_LOCK:
            _EXPORT3D["total"] = total
        logger.info("3D batch image export started: %d group(s) -> %r",
                    total, out_dir)
        try:
            def _build_figs(item):
                """(figs, err) for one group: a single combined (filename, fig)."""
                records = []
                for member in item["members"]:
                    try:
                        values = _resolve_values(member["key"], store_metas,
                                                 top_cfg, btm_cfg)
                    except ValueError as exc:  # zero cell out of bounds / blank
                        return [], str(exc)
                    if values is None:
                        return [], "not loadable"
                    records.append({"key": member["key"], "kind": member["kind"],
                                    "values": values})
                if not records:
                    return [], "no loadable datasets"

                # Resize preview: bring every TOP/BTM dataset in the group onto
                # one reference grid (GAP/OUT shown as-is), mirroring render_3d.
                if show_resized == "resized":
                    inputs = [r for r in records if r["kind"] in ("TOP", "BTM")]
                    if len(inputs) >= 2:
                        ref = _pick_reference_record(inputs, reference)
                        for r in inputs:
                            if r is ref:
                                continue
                            try:
                                r["values"] = resize_crop_blank(
                                    r["values"], ref["values"].shape)
                            except ValueError:
                                pass  # keep the un-resized surface

                # multi_surface_3d takes prebuilt names, so the rows×cols suffix
                # is folded into each label here (as render_3d does).
                surfaces = []
                for r in records:
                    label = _key_label(r["key"], store_metas)
                    if opts.show_shape:
                        label = "{0} ({1}×{2})".format(
                            label, r["values"].shape[0], r["values"].shape[1])
                    surfaces.append((label, r["values"], 0.0))
                fig_opts = dataclasses.replace(
                    opts, title=opts.title or item["label"])
                return [(item["filename"],
                         charts_mpl.multi_surface_3d(surfaces, fig_opts))], None

            def _on_progress(done):
                with _EXPORT3D_LOCK:
                    _EXPORT3D["done"] = done

            pairs = [(it["label"], it) for it in items]
            saved, failed_msgs = _pooled_figure_export(
                pairs, _build_figs, out_dir, img_kwargs, _on_progress)

            msg = "Saved {0} image(s) to {1}.".format(saved, out_dir)
            logger.info("3D batch image export finished: %d saved, %d failed "
                        "-> %s", saved, len(failed_msgs), out_dir)
            if failed_msgs:
                msg += " Failed: " + ", ".join(failed_msgs)
            with _EXPORT3D_LOCK:
                _EXPORT3D["result"] = msg
        except Exception:  # noqa: BLE001 - never lose the outcome
            logger.exception("3D batch image export worker crashed")
            with _EXPORT3D_LOCK:
                _EXPORT3D["error"] = traceback.format_exc(limit=3)
        finally:
            with _EXPORT3D_LOCK:
                _EXPORT3D["running"] = False

    @app.callback(
        Output("export-all-progress-interval", "disabled"),
        Output("btn-export-all-gaps", "disabled"),
        Output("export-all-status", "children"),
        Output("export-all-progress-bar", "style"),
        Output("export-all-progress-label", "children"),
        Input("btn-export-all-gaps", "n_clicks"),
        [State("store-gaps", "data"),
         State("folder-img", "value"),
         State("folder-out", "value"),
         State("gap-view-type", "value"),
         State("export-all-kinds", "value"),
         State("export-all-downsample", "value"),
         State("export-img-width", "value"),
         State("export-img-height", "value"),
         State("export-img-scale", "value")]
        + _option_states("optgap"),
        prevent_initial_call=True,
    )
    def start_export_all(_n, store_gaps, img_dir, out_dir, chart_type,
                         kinds, downsample, img_w, img_h, img_scale,
                         *option_values):
        if not store_gaps:
            return (True, False,
                    "No computed gaps to export — run Compute All Gaps first.",
                    {"width": "0%"}, "")
        kinds = [k for k in (kinds or []) if k in ("2d", "3d")]
        if not kinds:
            return (True, False,
                    "Select at least one image type (2D/3D).",
                    {"width": "0%"}, "")
        out_dir = _export_dir(img_dir, out_dir)
        if not out_dir:
            return (True, False,
                    "Set an image save folder (or the OUT folder) first.",
                    {"width": "0%"}, "")
        try:
            downsample = int(downsample)
        except (TypeError, ValueError):
            downsample = 0
        opts = _build_options("optgap", option_values)
        img_kwargs = _export_image_kwargs(img_w, img_h, img_scale)
        with _EXPORT_LOCK:
            if _EXPORT["running"]:
                logger.info("Export-all request ignored: already running")
                return no_update, no_update, no_update, no_update, no_update
            _EXPORT.update(running=True, done=0, total=0,
                           result=None, error=None)
        threading.Thread(
            target=_export_all_worker,
            args=(store_gaps, out_dir, chart_type, opts, img_kwargs, kinds,
                  downsample),
            name="export-all-worker",
            daemon=True,
        ).start()
        return False, True, "", {"width": "0%"}, "Exporting..."

    @app.callback(
        Output("export-all-status", "children", allow_duplicate=True),
        Output("export-all-progress-interval", "disabled", allow_duplicate=True),
        Output("btn-export-all-gaps", "disabled", allow_duplicate=True),
        Output("export-all-progress-bar", "style", allow_duplicate=True),
        Output("export-all-progress-label", "children", allow_duplicate=True),
        Input("export-all-progress-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_export_all(_n):
        with _EXPORT_LOCK:
            running = _EXPORT["running"]
            done, total = _EXPORT["done"], _EXPORT["total"]
            result, error = _EXPORT["result"], _EXPORT["error"]

        pct = (100.0 * done / total) if total else 0.0
        bar = {"width": "{0:.0f}%".format(pct)}
        label = ("{0} / {1} gaps".format(done, total) if total
                 else "Exporting...")

        if running:
            return no_update, no_update, no_update, bar, label

        if error is not None:
            logger.error("Export-all poll: publishing error to UI:\n%s", error)
            return "Export error: " + error, True, False, bar, "Failed"
        if result is None:
            # no outcome pending (e.g. a tick right after a fresh page load)
            logger.debug("Export-all poll: no pending outcome, disabling interval")
            return no_update, True, False, no_update, no_update
        logger.info("Export-all poll: publishing result to UI")
        return (result, True, False, {"width": "100%"},
                "{0} / {0} gaps — done".format(total))

    # 5c. 3D-tab batch export: one 3D surface PNG per dataset currently listed
    #    in the (filter-narrowed) TOP/BTM/GAP/OUT dropdowns (options, not the
    #    selected values). Runs in a background thread on the shared matplotlib
    #    pool; the root-level export3d-all-progress-interval polls _EXPORT3D
    #    (same polling contract as the Gap batch export above).
    @app.callback(
        Output("export3d-all-progress-interval", "disabled"),
        Output("btn-export-3d-all", "disabled"),
        Output("export3d-all-status", "children"),
        Output("export3d-all-progress-bar", "style"),
        Output("export3d-all-progress-label", "children"),
        Input("btn-export-3d-all", "n_clicks"),
        [State("view3d-top", "options"),
         State("view3d-btm", "options"),
         State("view3d-gap", "options"),
         State("view3d-out", "options"),
         State("store-metas", "data"),
         State("folder-img", "value"),
         State("folder-out", "value"),
         State("data-show-resized", "value"),
         State("gap-reference", "value"),
         State("export-img-width", "value"),
         State("export-img-height", "value"),
         State("export-img-scale", "value")]
        + _option_states("opt3d") + _TRANSFORM_STATES,
        prevent_initial_call=True,
    )
    def start_export_3d_all(_n, top_opts, btm_opts, gap_opts, out_opts,
                            store_metas, img_dir, out_dir, show_resized,
                            reference, img_w, img_h, img_scale, *rest):
        option_values = rest[:_N3D]
        transform_values = rest[_N3D:]
        items = _grouped_3d_items([("TOP", top_opts), ("BTM", btm_opts),
                                   ("GAP", gap_opts), ("OUT", out_opts)],
                                  store_metas)
        if not items:
            return (True, False,
                    "No datasets match the current filter — scan folders "
                    "(or compute gaps) first.", {"width": "0%"}, "")
        dest = _export_dir(img_dir, out_dir)
        if not dest:
            return (True, False,
                    "Set an image save folder (or the OUT folder) first.",
                    {"width": "0%"}, "")
        opts = _build_options("opt3d", option_values)
        top_cfg, btm_cfg = _transform_configs(transform_values)
        img_kwargs = _export_image_kwargs(img_w, img_h, img_scale)
        with _EXPORT3D_LOCK:
            if _EXPORT3D["running"]:
                logger.info("3D export-all request ignored: already running")
                return (no_update, no_update, no_update, no_update, no_update)
            _EXPORT3D.update(running=True, done=0, total=0,
                             result=None, error=None)
        threading.Thread(
            target=_export_3d_all_worker,
            args=(items, store_metas, dest, opts, img_kwargs, top_cfg, btm_cfg,
                  show_resized, reference),
            name="export-3d-all-worker",
            daemon=True,
        ).start()
        return False, True, "", {"width": "0%"}, "Exporting..."

    @app.callback(
        Output("export3d-all-status", "children", allow_duplicate=True),
        Output("export3d-all-progress-interval", "disabled", allow_duplicate=True),
        Output("btn-export-3d-all", "disabled", allow_duplicate=True),
        Output("export3d-all-progress-bar", "style", allow_duplicate=True),
        Output("export3d-all-progress-label", "children", allow_duplicate=True),
        Input("export3d-all-progress-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_export_3d_all(_n):
        with _EXPORT3D_LOCK:
            running = _EXPORT3D["running"]
            done, total = _EXPORT3D["done"], _EXPORT3D["total"]
            result, error = _EXPORT3D["result"], _EXPORT3D["error"]

        pct = (100.0 * done / total) if total else 0.0
        bar = {"width": "{0:.0f}%".format(pct)}
        label = ("{0} / {1} images".format(done, total) if total
                 else "Exporting...")

        if running:
            return no_update, no_update, no_update, bar, label

        if error is not None:
            logger.error("3D export-all poll: publishing error to UI:\n%s",
                         error)
            return "Export error: " + error, True, False, bar, "Failed"
        if result is None:
            # no outcome pending (e.g. a tick right after a fresh page load)
            logger.debug("3D export-all poll: no pending outcome, disabling "
                         "interval")
            return no_update, True, False, no_update, no_update
        logger.info("3D export-all poll: publishing result to UI")
        return (result, True, False, {"width": "100%"},
                "{0} / {0} images — done".format(total))
