# matrix2d — Warpage Gap Analysis

2D matrix warpage measurement analysis tool. Loads TOP/BTM surface warpage
data, computes point-wise gap at first contact, renders interactive 2D/3D
contour charts. Dash single-page app; core is pure numpy/scipy for future
Electron+React+TypeScript migration.

## Commands

```bash
python -m pytest                      # run all tests (from repo root)
python run_app.py                     # start Dash SPA at http://127.0.0.1:8050
python scripts/make_sample_data.py    # generate demo_data/{TOP,BTM,GAP,OUT}
```

Runtime log: `logs/matrix2d.log` (rotating 2MB×5; `MATRIX2D_LOG_DIR` /
`MATRIX2D_LOG_LEVEL` override; configured by `matrix2d/logging_setup.py`
from run_app.py). Dash debug mode is opt-in via `MATRIX2D_DEBUG=1` — its
werkzeug reloader/hot-reload restarts the process and refreshes the page
mid-run, killing background scan/compute threads and wiping dcc.Store state.

Python 3.8.10 — do NOT use `X | Y` union syntax or `match`. Use
`typing.Optional/List/Tuple`. Deps pinned in requirements.txt
(dash 2.17.1, plotly 5.24.1, scipy 1.10.1, numpy 1.24.4, kaleido 0.2.1).

## Architecture (clean, layered)

```
src/matrix2d/
  core/        # pure domain logic — numpy/scipy only, NO I/O in resize/gap/naming
    models.py  # SampleMeta, WarpageData, GapResult dataclasses
    parser.py  # filename parsing + matrix file loading
    resize.py  # bilinear value resize + nearest-neighbor mask resize
    transform.py # flip/rotate/zero-point orientation transforms
    gap.py     # contact-offset gap computation
    naming.py  # output filename + H/C phase assignment
  services/    # application layer — file I/O, orchestration
    repository.py  # scan_folder, load_data, save_matrix
    pipeline.py    # plan_jobs, run_pipeline (all TOP×BTM combos)
  ui/          # Dash presentation layer
    charts.py  # pure plotly figure builders (no Dash imports)
    layout.py / callbacks.py / app.py
```

Dependency rule: core ← services ← ui. Never import upward.
charts.py stays Dash-free so it ports directly to the React migration.

## Domain rules

- **Input files**: `.dat/.csv/.txt`, 2D numeric matrix, no header/index.
  Blank cells = empty string, `nan`, or value >= 2000 → stored as np.nan.
  `scan_folder` validates NAME and CONTENT; invalid files are skipped with
  a logged warning (never abort). A set folder that scans to 0 files →
  UI shows a "데이터 없음 (no valid data files)" error.
- **Filename (TOP/BTM)**: `TITLE_PTXXXX_YYYYYs(ZZZC).ext` — PT+4-digit
  sample no, 5-digit seconds + `s`, 1–3 digit Celsius + `C` in parens.
  Regex anchors on the LAST `_PT` so titles may contain underscores.
- **Filename (GAP folder)**: same as OUT files —
  `{prefix}-{H|C}{temp}_TOP{n}-BTM{m}[_k].ext` (e.g. `TEST-C25_TOP3-BTM8.txt`;
  `prefix` = free user phrase, may itself contain `-`/`_`; `_k` = duplicate
  suffix). `parse_gap_filename` → sample_no = TOP no, `btm_no` = BTM no,
  explicit `phase`, time_s = 0. Legacy `TOP{n}-BTM{m}_{H|C}{temp}` and
  legacy `TITLE_PT...` names still parse as fallbacks
  (`parse_data_filename` dispatches by kind).
- **Gap**: `diff = TOP - BTM`; `offset = nanmin(diff)`; `gap = diff - offset`.
  Minimum valid gap is exactly 0.0 (first contact point). NaN propagates.
- **Resize**: values bilinear-interpolated on normalized grid (no warpage
  distortion; linear ramp survives < 1e-6 error) via `resize_values`. Blank is
  NOT scaled with the values — it is CROPPED: the source blank keeps its
  absolute cell extent and is center-aligned onto the target grid (crop when
  target smaller, pad-valid when larger; odd leftover → trailing edge) by
  `_center_fit_mask`. `resize_crop_blank` = value resize + own center-fit
  blank (used by the 3D "Resized" preview per dataset).
- **Blank matching (TOP/BTM pair)**: `resize_pair(top, btm, ref)` resizes the
  non-reference side's values to the reference grid, then gives BOTH sides the
  SAME blank: the UNION of each side's center-fit blank ("match to the larger
  blank" — per dimension the union spans ≥ the larger blank while keeping each
  blank's actual shape). Used by `run_pipeline` and the 2D preview. Note: for
  fully-valid data the gap is unchanged (union = what `compute_gap` already
  intersects); the real change vs. the old block-resize is crop-fit blanks.
- **Reference size**: `reference="AUTO"` (default) picks the dataset with
  the SMALLER element count per job (larger resized to smaller, tie → TOP);
  explicit `"TOP"`/`"BTM"` overrides. Reference dataset's grid is
  authoritative; both sides share the larger (union) blank.
- **Transforms** (before resize, order flip → rotate → zero):
  flip = left-right mirror INCLUDING value sign inversion (`-fliplr`);
  rotation = clockwise 90° steps; zero-point = subtract value at a given
  `(row, col)` cell so it becomes 0.0 — coordinates refer to POST-flip/rotate
  orientation; blank/OOB zero cell → ValueError. UI: flip/rotate apply to
  TOP only, zero-point to TOP and BTM. `apply_transform` always returns a
  new array (never mutates input — matrix cache safety).
- **H/C phase**: per sample-pair, peak time = time of max temperature;
  `time <= peak` → `H` (heating), else `C` (cooling).
- **Output name**: `{prefix}-{H|C}{temp}_TOP{n}-BTM{m}.txt` into OUT folder,
  tab-delimited, NaN written as `nan`. Duplicate names get `_2`, `_3`.
  Prefix = Gap tab "Output name prefix" input; `naming.sanitize_prefix`
  strips filename-illegal chars, blank → `GAP`.
- **Summary**: `run_pipeline` also writes `{prefix}.txt` into OUT — a pivot
  (`core/summary.build_summary`) with temperature-point columns
  (`{H|C}{temp}`, e.g. `H25`; heating-before-cooling then temp-ascending),
  TOP-BTM combo rows (`TOP{n}-BTM{m}`), each cell the MAX gap for that combo
  at that point (blank if none / all-NaN). Four stat rows (`MIN`, `MAX`,
  `AVG`, `STD`) sit directly under the header — per-column aggregates over
  the combo cells (blanks ignored; `STD` = sample stdev, ddof=1, blank when
  <2 values). Tab-delimited; a write failure never aborts the batch.
  `core/summary.effective_gap_series(records)` reuses the same records to
  return per-temperature-point AVG + sample-STD as structured data for the
  Effective Gap tab chart, ordered heating-ascending then cooling-DESCENDING
  (differs from the file's within-phase ascending column order).
- **Pairing**: every TOP-sample × BTM-sample combination; per TOP
  temperature, H pairs with H and C with C. TOP/BTM temps within ±2°C
  (`pipeline.TEMP_TOLERANCE_C`) count as the same temperature point —
  matched to the nearest BTM temp in range (tie → lower BTM temp); output
  name uses the TOP temp. Phase is still per-sample from own peak time.

## Conventions

- Tests mirror module names (`tests/test_resize.py` etc.); pytest.ini sets
  `pythonpath = src`.
- Errors in one pipeline job must not abort the batch — log and continue.
- UI state: scanned metas in dcc.Store as dicts; loaded matrices cached in
  a module-level dict (single-user local app).
- Chart styling flows through `ChartOptions` dataclass only — never set
  fonts/ticks ad hoc in callbacks.
- Chart Options are PER-TAB and PER-CHART-TYPE: the sidebar holds four
  independent control sets (`opt2d`/`opt3d`/`optgap`/`opteff` id prefixes,
  built by `layout._chart_options_panel(prefix, heading)`); only the active
  tab's panel shows (`callbacks.toggle_chart_options` keyed on `tabs`). Each
  tab renders ONLY the fields its chart type uses, selected by
  `layout.TAB_OPTION_FIELDS` (laid out via `_OPTION_ROWS`): 2D/Gap show the
  full color+contour set; 3D (surface) drops `contour-levels`; Effective Gap
  (line) drops colorscale/toggles/contour-levels/x-dtick and relabels
  zmin/zmax as the y-axis range. Each render callback derives its Input/State
  list from `layout.tab_option_suffixes(prefix)` (`_option_inputs` /
  `_option_states`); `_build_options(prefix, values)` zips those suffixes to
  values, so missing fields fall back to `ChartOptions` defaults. The Gap
  Compute batch export and inspect both use `optgap`.
- Tabs: 2D View / 3D View / Gap Compute / Effective Gap. The Effective Gap
  tab (`effgap-graph`) plots `summary.effective_gap_series` from `store-gaps`
  (each result carries `max_gap`, computed from the gap cache at publish
  time): AVG per temperature point with sample-STD as 'T' error bars,
  y-axis "Effective Gap". Re-renders live from `store-gaps` + `opteff`.
- `ChartOptions.show_shape` (default on) appends `rows×cols` to 2D titles
  and 3D trace names.
- Folder paths picked via native tkinter dialog (`ui/dialogs.py`,
  Browse... buttons) — OK because the app is local single-user; dialog
  failures/cancel return None → `no_update`.
- Reference-size radio (`gap-reference`) and the Original/Resized display
  radio (`data-show-resized`) live in the Data Options panel. AUTO now picks
  larger→smaller (smallest element count wins, tie → TOP). "Resized"
  previews data exactly as the pipeline consumes it: 2D pairs the selected
  TOP/BTM via `resize_pair` so both show the larger (union) blank; 3D brings
  all selected TOP/BTM datasets onto one reference grid (GAP/OUT surfaces
  untouched) via `resize_crop_blank`, each keeping its own crop-fit blank.
- Scan reads TOP/BTM/GAP/OUT folders; each folder's metas go in its own
  `store-metas` bucket. OUT files use the gap output naming, so they are
  parsed with the GAP format (`scan_folder(out_dir, "GAP")`) but kept under
  the "OUT" key. The 3D View tab exposes them in a dedicated "OUT datasets"
  dropdown (`view3d-out`) alongside TOP/BTM/GAP; they render as untransformed
  gap surfaces exactly like scanned GAP files.
- Gap compute runs in a background thread (module-level `_COMPUTE` state in
  callbacks.py, guarded by a Lock); a 400ms `dcc.Interval` polls it to drive
  the progress bar and publishes results when done (duplicate outputs use
  `allow_duplicate=True`). `run_pipeline(progress_cb=...)` reports
  (done, total) per job.
- Polling contract (scan, compute, batch export, Effective-Gap OUT load):
  the worker's outcome stays in the
  state dict until the NEXT run starts — pollers must read it
  NON-destructively and republish every tick until the publish response
  disables the interval. dash-renderer discards responses whose
  `n_intervals` changed mid-flight, so a consume-once poller can lose the
  only response carrying the results (progress bar then hangs forever).
  Publishing must therefore be side-effect free: gap-cache refresh happens
  in the worker, not the poller. `gap-progress-interval` lives at the layout
  ROOT (not in the Gap tab) so tab switches never pause/unmount the poller.
- Worker threads must never die silently: wrap the body so any exception is
  logged (`logger.exception`) AND published to the UI error output;
  `logging_setup.py` also hooks `threading.excepthook` as a safety net.
- Gap tab layout: charts (inspect dropdown + 2D + 3D) on the LEFT, result
  table on the RIGHT; `.table-wrap` scrolls internally (max-height, sticky
  header) instead of the page.
- "Save All Images" exports `{out_name}_2D.png` / `_3D.png` per computed gap
  via kaleido, using current ChartOptions. A `export-all-kinds` checklist
  (2D/3D, both default) picks which images; a `export-all-downsample` dropdown
  stride-downsamples the gap grid (max shape 300/200/150/100, "Off" default)
  for export only (no core-layer change, NaN/blank pattern preserved, shows in
  the title's rows×cols). Rendering runs a parallel per-thread kaleido scope
  pool (`kaleido.scopes.plotly.PlotlyScope`, one Chromium subprocess each) —
  worker count `MATRIX2D_EXPORT_WORKERS` (default 4, clamp 1–8, capped at gap
  count); falls back to sequential `fig.write_image` if PlotlyScope is
  unavailable. `_EXPORT["done"]` still counts completed GAPS.
- PNG export destination (all Save-as-PNG paths, incl. batch export): the
  Image Export panel's "Save folder" (`folder-img`, own Browse/✕ buttons);
  blank falls back to the OUT folder (`callbacks._export_dir`).
- Effective Gap "Load from OUT files" runs in a background thread
  (`_EFFLOAD` state, `effgap-load-interval` poller at layout root) with a
  two-stage progress bar: OUT folder scan, then per-file max-gap load
  (`effgap_records_from_metas(progress_cb=...)`). A run ending without data
  (scan error / empty folder) publishes only the status message and keeps
  the current chart records.

## Migration plan (phase 2)

Electron + React + TypeScript front end; Python core stays as local service
(FastAPI or pyodide-style port). Keep core/services free of Dash imports so
only `ui/` is replaced.
