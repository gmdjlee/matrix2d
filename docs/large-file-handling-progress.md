# Large-file handling — improvement plan & progress

Branch: `claude/large-file-handling-840941`. Goal: handle **many, large**
input files without OOM and without redundant disk reads. Each step is an
independent, test-backed commit so a fresh conversation can resume from the
status table below.

## How to resume in a new conversation

1. Read this file. Find the first step whose Status is not ✅ DONE.
2. Read that step's Spec + Files. Re-read the referenced source before editing.
3. Implement, run `python -m pytest` (from repo root), commit, update the table.
4. Keep the CLAUDE.md domain rules intact (skip-and-continue on job/file
   errors; core stays I/O-free except parser/services; Python 3.8 syntax —
   no `X|Y` unions, no `match`).

## Status

| Step | Title | Status | Commit |
|------|-------|--------|--------|
| P2 | Shared bounded, mtime-keyed raw-matrix cache (dedupe disk reads) | ✅ DONE | see git log |
| P1 | Bound pipeline memory (drop unbounded seed + memo) | 🔄 IN PROGRESS | — |
| P4 | Stream `load_matrix` row-wise (cut transient python-float peak) | ⬜ TODO | — |
| P6 | Single-file size guard (skip oversize, never abort) | ⬜ TODO | — |

Status legend: ⬜ TODO · 🔄 IN PROGRESS · ✅ DONE · ⏭️ SKIPPED

---

## P2 — Shared bounded, mtime-keyed raw-matrix cache

**Problem.** TOP/BTM files are read from disk at least twice per compute:
once in the UI scan worker (`scan_folder` loads the full matrix only to
validate content, then discards it — `callbacks.py` `_scan_worker`), and again
when `run_pipeline` re-scans and re-loads. Large files make the wasted read
expensive.

**Fix.** Add a module-level, bounded, `stat`-keyed cache in
`services/repository.py` so a repeated read of an unchanged file is a cache
hit — shared by the UI scan and the pipeline.

- `_RAW_CACHE = OrderedDict()  # path -> (stat_key, ndarray)`
- `stat_key = (st.st_mtime_ns, st.st_size)` — a changed file misses (no stale
  data). Bound `_RAW_CACHE_MAX` (default 128) via env `MATRIX2D_RAW_CACHE`;
  evict LRU (`popitem(last=False)`).
- `def read_matrix(path)`: cache-aware wrapper over `parser.load_matrix`.
  Hit with matching stat_key → `move_to_end`, return cached array. Miss →
  load, store, trim.
- **Returned arrays are shared → treat read-only.** All current consumers are
  safe: `scan_folder` only validates; the pipeline loader runs
  `apply_transform` which always returns a fresh array (never mutates input).
  Document this in `read_matrix`'s docstring.
- `scan_folder`: replace `mat = load_matrix(path)` with `mat = read_matrix(path)`.
  Keep the `matrix_cache` param working (populate from the cached array).
- `load_data`/`load_warpage` compute-path: route through the cache too so a
  compute right after a scan hits it (optional but recommended; keep
  `load_warpage` a thin wrapper).

**Files:** `src/matrix2d/services/repository.py` (primary),
possibly `src/matrix2d/core/parser.py` (`load_warpage`).

**Verify:** all existing tests green; add
`tests/test_repository.py` cases: (a) second `read_matrix` of same file
returns cached object (identity or a monkeypatched-load call-count of 1);
(b) touching the file (new mtime) invalidates; (c) cache respects the max
bound. `MATRIX2D_RAW_CACHE` override honored.

---

## P1 — Bound pipeline memory

**Problem.** `run_pipeline` pre-loads **every** TOP and **every** BTM raw
matrix into `top_seed`/`btm_seed` before the first job, and
`_make_matrix_loader`'s `transformed = {}` memo **never evicts**. Peak RAM ≈
(N_top + N_btm) × matrix size, held for the whole run → OOM on large sets.

**Fix.** Remove the giant up-front seeds; load lazily through the P2
`read_matrix` cache and keep a **bounded** transform memo.

- `run_pipeline`: `scan_folder(top_dir, "TOP")` / `(btm_dir, "BTM")` **without**
  `matrix_cache` (metas only).
- `_make_matrix_loader(cfg)`: bounded LRU `transformed = OrderedDict()` keyed
  by path, cap via env (e.g. `MATRIX2D_XFORM_CACHE`, default 64); evict LRU.
  On miss: `raw = repository.read_matrix(meta.path); vals = apply_transform(raw, cfg)`.
  Keep a **separate small error cache** (path -> Exception) so a bad file is
  not retried per job and per-job skip-and-continue is preserved (do NOT evict
  errors with the LRU, or a bad file gets retried — keep them in their own dict).
- Memory ceiling now bounded: `_RAW_CACHE_MAX` + 2×xform-cap matrices. Huge
  sets spill to disk (graceful) instead of OOM. Because P2 caches the raw
  parse, transform-memo eviction only recomputes the cheap numpy transform,
  not the expensive disk parse — preserves most of the [[pipeline-load-caching]]
  35% win.

**Files:** `src/matrix2d/services/pipeline.py`.

**Verify:** all `tests/test_pipeline.py` green (esp. end-to-end, skip-continue
on NaN zero cell, AUTO reference, transforms). Add a test that a batch with
more unique files than the cache cap still computes every job correctly
(correctness under eviction). Optionally assert `read_matrix` called once per
unique file when cache is large enough.

---

## P4 — Stream `load_matrix` row-wise

**Problem.** `load_matrix` builds a full list-of-lists of **python floats**
(~24 B each + list overhead) before `np.asarray`, a large transient peak for
big matrices. (Note: `readlines()` holds line strings but those are shared
refs — the float list is the real cost. Do NOT vectorize per-cell parsing:
[[load-matrix-parse-perf]] — proven slower.)

**Fix.** Parse each row into a compact `np.ndarray` as it is read, so only one
row of python floats is alive at a time; assemble with NaN padding at the end.

- Keep per-cell `_parse_cell` (blank/`nan`/`>=2000` → NaN semantics unchanged).
- Delimiter rule unchanged: comma if **any** content line contains a comma,
  else whitespace. (Requires knowing all lines first — keep the line list, or
  do a cheap first pass for the comma flag; either is fine. The win is
  converting python floats → C floats per row, not avoiding the line list.)
- Preserve ragged-row NaN padding and the "no numeric content" ValueError.

**Files:** `src/matrix2d/core/parser.py`.

**Verify:** all `tests/test_parser.py` green (blanks, nan, sentinel, ragged,
comma vs whitespace, empty). This is the most correctness-sensitive step —
byte-for-byte identical results required.

---

## P6 — Single-file size guard

**Problem.** One pathological file → `readlines()`/parse OOM with no friendly
error.

**Fix.** In `load_matrix`, before reading, check `os.path.getsize(path)`
against `MATRIX2D_MAX_FILE_MB` (default e.g. 200). Over limit → raise
`ValueError` with a clear message. This flows into the existing
skip-and-continue: `scan_folder` catches `ValueError` → logs + skips;
`run_pipeline` job → logs + continues. No abort.

**Files:** `src/matrix2d/core/parser.py`.

**Verify:** add a test: an oversize file (monkeypatch `getsize` or set a tiny
`MATRIX2D_MAX_FILE_MB`) is skipped by `scan_folder` with a warning; a normal
file still loads. Default (unset env) unchanged behavior.

---

## Notes for the implementer

- Env knobs use the existing `MATRIX2D_*` convention (see `logging_setup.py`,
  CLAUDE.md). Read once at call time (not import time) so tests can set them.
- Single-user local app: module-level caches are acceptable (mirrors
  `helpers._MATRIX_CACHE` / `_GAP_LRU`).
- After all four: run full `python -m pytest`, then a manual smoke via
  `python run_app.py` is optional (scan+compute a folder).
