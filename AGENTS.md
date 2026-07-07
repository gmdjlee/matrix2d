# AGENTS.md — matrix2d

Guidance for AI coding agents working in this repository.

## Project

Warpage gap analysis: load 2D matrix measurement files (TOP/BTM surfaces),
compute point-wise gap at first contact, render interactive 2D/3D contour
charts. Dash SPA on a pure-numpy core (clean architecture, migration to
Electron+React+TS planned — only `src/matrix2d/ui/` will be replaced).

## Environment

- Python **3.8.10** (Windows). Forbidden: `X | Y` unions, `match`,
  `functools.cache`. Use `typing.Optional/List/Tuple/Dict`.
- Pinned deps: numpy 1.24.4, scipy 1.10.1, dash 2.17.1, plotly 5.24.1,
  kaleido 0.2.1, pytest 8.3.5 (`requirements.txt`).

## Commands

```bash
python -m pytest                      # full test suite — must pass before done
python -m pytest tests/test_gap.py    # single module
python run_app.py                     # Dash app, http://127.0.0.1:8050
python scripts/make_sample_data.py    # regenerate demo_data/
```

## Layering (strict)

```
core/     pure domain: models, parser, resize, gap, naming (numpy/scipy only)
services/ file I/O + orchestration: repository, pipeline
ui/       Dash: charts (plotly-only, Dash-free), layout, callbacks, app
```

- Import direction: `ui → services → core`. Never upward.
- No file I/O inside `core/resize.py`, `core/gap.py`, `core/naming.py`.
- `ui/charts.py` must not import dash — it is the migration boundary.

## Domain invariants (do not break)

1. Blank cells (empty / `nan` / value >= 2000) become `np.nan` at load time;
   everything downstream treats NaN as "no data".
2. `compute_gap`: `gap = (top - btm) - nanmin(top - btm)`; min valid cell
   is exactly `0.0`. Shape mismatch or zero-overlap raises `ValueError`.
3. Resize never distorts warpage: bilinear on values (linear ramp error
   < 1e-6), block resize on the blank mask (any source blank -> blank, so
   blanks never shrink); each dataset keeps its own resized blank
   (`mask_mode="own"`), not the reference's.
4. Filename regex anchors on the **last** `_PT` — titles may contain `_`.
5. Output naming: `TOP{n}-BTM{m}_{H|C}{temp}.txt`; H/C from peak-time rule
   (time <= peak-temperature time → H). Collisions get `_2`, `_3` suffix.
6. Pipeline processes every TOP×BTM sample combination; one job's failure
   is logged, never aborts the batch.

## Working rules

- Tests first-class: any change to `core/` or `services/` needs a matching
  test in `tests/` (module-mirrored names). Run the full suite before
  reporting done; paste the pytest summary line.
- Keep functions pure where the layer allows; pass ndarrays, not paths,
  into core functions.
- Chart styling goes through the `ChartOptions` dataclass — never hardcode
  fonts/ticks in callbacks.
- Do not add dependencies without pinning a py3.8-compatible version.
- Do not commit; leave git operations to the session owner.
- Demo data is regenerable — safe to delete/rebuild via the script.
