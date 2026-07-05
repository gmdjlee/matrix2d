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
- **Filename**: `TITLE_PTXXXX_YYYYYYs(ZZZC).ext` — PT+4-digit sample no,
  6-digit seconds + `s`, 1–3 digit Celsius + `C` in parens.
  Regex anchors on the LAST `_PT` so titles may contain underscores.
- **Gap**: `diff = TOP - BTM`; `offset = nanmin(diff)`; `gap = diff - offset`.
  Minimum valid gap is exactly 0.0 (first contact point). NaN propagates.
- **Resize**: values bilinear-interpolated on normalized grid (no warpage
  distortion; linear ramp survives < 1e-6 error). Blank mask resized
  separately with nearest-neighbor; final mask = reference dataset's mask
  union resized mask (`mask_mode="reference"`).
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

## Migration plan (phase 2)

Electron + React + TypeScript front end; Python core stays as local service
(FastAPI or pyodide-style port). Keep core/services free of Dash imports so
only `ui/` is replaced.
