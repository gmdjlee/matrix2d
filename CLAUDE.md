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
  distortion; linear ramp survives < 1e-6 error). Blank mask resized by
  block — any source cell blank in a target cell's footprint → that target
  cell is blank — so blanks never shrink and keep their shape. Each dataset
  keeps its OWN resized blank (`mask_mode="own"`), so a resized surface shows
  only its own blank (no cross); the gap's NaN still comes from both sides
  via `compute_gap`.
- **Reference size**: `reference="AUTO"` (default) picks the dataset with
  the SMALLER element count per job (larger resized to smaller, tie → TOP);
  explicit `"TOP"`/`"BTM"` overrides. Reference dataset's grid is
  authoritative (each side keeps its own resized blank mask).
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
  at that point (blank if none / all-NaN). Tab-delimited; a write failure
  never aborts the batch.
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
- `ChartOptions.show_shape` (default on) appends `rows×cols` to 2D titles
  and 3D trace names.
- Folder paths picked via native tkinter dialog (`ui/dialogs.py`,
  Browse... buttons) — OK because the app is local single-user; dialog
  failures/cancel return None → `no_update`.
- Reference-size radio (`gap-reference`) and the Original/Resized display
  radio (`data-show-resized`) live in the Data Options panel. AUTO now picks
  larger→smaller (smallest element count wins, tie → TOP). "Resized"
  previews data exactly as the pipeline consumes it: 2D resizes the
  non-reference side of the selected pair; 3D brings all selected TOP/BTM
  datasets onto one reference grid (GAP/OUT surfaces untouched); each
  resized dataset keeps its own blank, not the reference's.
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
- Polling contract (scan AND compute): the worker's outcome stays in the
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
- "Save All Images (2D+3D)" exports `{out_name}_2D.png` / `_3D.png` per
  computed gap into OUT via kaleido, using current ChartOptions.

## Migration plan (phase 2)

Electron + React + TypeScript front end; Python core stays as local service
(FastAPI or pyodide-style port). Keep core/services free of Dash imports so
only `ui/` is replaced.
