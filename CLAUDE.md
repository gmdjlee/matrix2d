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
- **Filename (TOP/BTM)**: `TITLE_PTXXXX_YYYYYYs(ZZZC).ext` — PT+4-digit
  sample no, 6-digit seconds + `s`, 1–3 digit Celsius + `C` in parens.
  Regex anchors on the LAST `_PT` so titles may contain underscores.
- **Filename (GAP folder)**: same as OUT files — `TOP{n}-BTM{m}_{H|C}{temp}
  [_k].ext` (`_k` = duplicate suffix). `parse_gap_filename` → sample_no =
  TOP no, `btm_no` = BTM no, explicit `phase`, time_s = 0. Legacy
  `TITLE_PT...` names still parse as fallback (`parse_data_filename`
  dispatches by kind).
- **Gap**: `diff = TOP - BTM`; `offset = nanmin(diff)`; `gap = diff - offset`.
  Minimum valid gap is exactly 0.0 (first contact point). NaN propagates.
- **Resize**: values bilinear-interpolated on normalized grid (no warpage
  distortion; linear ramp survives < 1e-6 error). Blank mask resized
  separately with nearest-neighbor; final mask = reference dataset's mask
  union resized mask (`mask_mode="reference"`).
- **Reference size**: `reference="AUTO"` (default) picks the dataset with
  the larger element count per job (smaller resized to larger, tie → TOP);
  explicit `"TOP"`/`"BTM"` overrides. Reference dataset's grid AND mask are
  authoritative.
- **Transforms** (before resize, order flip → rotate → zero):
  flip = left-right mirror INCLUDING value sign inversion (`-fliplr`);
  rotation = clockwise 90° steps; zero-point = subtract value at a given
  `(row, col)` cell so it becomes 0.0 — coordinates refer to POST-flip/rotate
  orientation; blank/OOB zero cell → ValueError. UI: flip/rotate apply to
  TOP only, zero-point to TOP and BTM. `apply_transform` always returns a
  new array (never mutates input — matrix cache safety).
- **H/C phase**: per sample-pair, peak time = time of max temperature;
  `time <= peak` → `H` (heating), else `C` (cooling).
- **Output name**: `TOP{n}-BTM{m}_{H|C}{temp}.txt` into OUT folder,
  tab-delimited, NaN written as `nan`. Duplicate names get `_2`, `_3`.
- **Pairing**: every TOP-sample × BTM-sample combination; per shared
  temperature, H pairs with H and C with C.

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
  radio (`data-show-resized`) live in the Data Options panel. "Resized"
  previews data exactly as the pipeline consumes it: 2D resizes the
  non-reference side of the selected pair; 3D brings all selected TOP/BTM
  datasets onto one reference grid (GAP surfaces untouched).

## Migration plan (phase 2)

Electron + React + TypeScript front end; Python core stays as local service
(FastAPI or pyodide-style port). Keep core/services free of Dash imports so
only `ui/` is replaced.
