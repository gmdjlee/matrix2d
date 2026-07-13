import logging
import os

import numpy as np
import pytest

from matrix2d.core.models import SampleMeta
from matrix2d.core.parser import load_matrix
from matrix2d.core.transform import TransformConfig
from matrix2d.services.pipeline import GapJob, plan_jobs, run_pipeline


def _meta(sample_no, time_s, temp_c, kind):
    return SampleMeta(
        title="T", sample_no=sample_no, time_s=time_s, temp_c=temp_c, kind=kind
    )


# ---- plan_jobs ---------------------------------------------------------------

def test_plan_jobs_pairs_h_and_c():
    # One sample each, temp 240 appears in both H (60s) and C (150s),
    # peak at 100s (260C).
    tops = [
        _meta(1, 60, 240, "TOP"),
        _meta(1, 100, 260, "TOP"),
        _meta(1, 150, 240, "TOP"),
    ]
    btms = [
        _meta(2, 60, 240, "BTM"),
        _meta(2, 100, 260, "BTM"),
        _meta(2, 150, 240, "BTM"),
    ]
    jobs = plan_jobs(tops, btms)
    names = sorted(j.out_name for j in jobs)
    # 240 H, 240 C, plus 260 H (peak present in both).
    assert "GAP-H240_TOP1-BTM2.txt" in names
    assert "GAP-C240_TOP1-BTM2.txt" in names
    assert "GAP-H260_TOP1-BTM2.txt" in names


def test_plan_jobs_custom_prefix():
    tops = [_meta(3, 11, 25, "TOP")]
    btms = [_meta(8, 11, 25, "BTM")]
    jobs = plan_jobs(tops, btms, out_prefix="TEST")
    # 25C is at/before the single sample's peak -> H phase.
    assert [j.out_name for j in jobs] == ["TEST-H25_TOP3-BTM8.txt"]


def test_plan_jobs_all_combinations():
    tops = [_meta(1, 11, 25, "TOP"), _meta(2, 11, 25, "TOP")]
    btms = [_meta(3, 11, 25, "BTM"), _meta(4, 11, 25, "BTM")]
    jobs = plan_jobs(tops, btms)
    pairs = sorted((j.top.sample_no, j.btm.sample_no) for j in jobs)
    assert pairs == [(1, 3), (1, 4), (2, 3), (2, 4)]


def test_plan_jobs_skips_missing_counterpart():
    # TOP has temp 25 only heating; BTM has temp 25 only cooling relative
    # to their own peaks -> no matching phase -> no job at that temp.
    tops = [_meta(1, 11, 25, "TOP"), _meta(1, 100, 260, "TOP")]  # 25 -> H
    btms = [_meta(2, 100, 260, "BTM"), _meta(2, 150, 25, "BTM")]  # 25 -> C
    jobs = plan_jobs(tops, btms)
    # Only 260 (both peak/H) should pair; 25 mismatched phase.
    names = [j.out_name for j in jobs]
    assert all("25" not in n for n in names)


def test_plan_jobs_dedup_suffix():
    # Force identical filenames: two TOP-BTM pairs producing same base name is
    # not possible across different sample numbers, so simulate same-name via
    # duplicate temp/phase within combos. Use two temps that collide by name?
    # Simplest: same sample pair cannot repeat a (temp,phase); instead check
    # dedup path by constructing colliding jobs across combos with same nos.
    # Here we verify the suffix mechanism directly through two combos that
    # yield the same base filename is impossible; instead ensure uniqueness.
    tops = [_meta(1, 11, 25, "TOP"), _meta(2, 11, 25, "TOP")]
    btms = [_meta(1, 11, 25, "BTM"), _meta(2, 11, 25, "BTM")]
    jobs = plan_jobs(tops, btms)
    names = [j.out_name for j in jobs]
    assert len(names) == len(set(names))  # all unique


def test_plan_jobs_temp_tolerance_pairs_within_2c():
    # TOP 175C (110s, H) pairs with BTM 174C (98s, H): |175-174|=1 <= 2.
    # TOP 176C (1125s, C) pairs with BTM 176C (1320s, C): exact, both cooling.
    # Peak is the 260C file present on both samples.
    tops = [
        _meta(1, 110, 175, "TOP"),
        _meta(1, 500, 260, "TOP"),
        _meta(1, 1125, 176, "TOP"),
    ]
    btms = [
        _meta(2, 98, 174, "BTM"),
        _meta(2, 500, 260, "BTM"),
        _meta(2, 1320, 176, "BTM"),
    ]
    jobs = plan_jobs(tops, btms)
    names = sorted(j.out_name for j in jobs)
    # Output temperature follows the TOP reading.
    assert "GAP-H175_TOP1-BTM2.txt" in names
    assert "GAP-C176_TOP1-BTM2.txt" in names


def test_plan_jobs_temp_tolerance_excludes_beyond_2c():
    # |180 - 177| = 3 > 2 -> no pairing at that temperature.
    tops = [_meta(1, 110, 180, "TOP"), _meta(1, 500, 260, "TOP")]
    btms = [_meta(2, 98, 177, "BTM"), _meta(2, 500, 260, "BTM")]
    jobs = plan_jobs(tops, btms)
    names = [j.out_name for j in jobs]
    assert all("180" not in n and "177" not in n for n in names)


def test_plan_jobs_temp_tolerance_picks_nearest_btm():
    # TOP 176 is within 2 of both BTM 175 (d=1) and BTM 178 (d=2); nearest wins.
    tops = [_meta(1, 110, 176, "TOP")]
    btms = [_meta(2, 108, 175, "BTM"), _meta(2, 112, 178, "BTM")]
    jobs = plan_jobs(tops, btms)
    # 176 vs 175 (d=1) beats 176 vs 178 (d=2) -> pairs with BTM 175 sample 2.
    assert len(jobs) == 1
    assert jobs[0].btm.temp_c == 175
    assert jobs[0].out_name == "GAP-H176_TOP1-BTM2.txt"


# ---- run_pipeline end-to-end -------------------------------------------------

def _make_surface(rows, cols, base, hole=True):
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    surf = base + 10.0 * (xx / (cols - 1)) - 8.0 * (yy / (rows - 1))
    surf = np.round(surf, 2)
    if hole:
        r0 = rows // 3
        c0 = cols // 3
        surf[r0 : r0 + 2, c0 : c0 + 2] = np.nan
    return surf


def _write_dat(path, arr):
    rows, cols = arr.shape
    lines = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            v = arr[r, c]
            cells.append("" if np.isnan(v) else "{0:.2f}".format(v))
        lines.append(",".join(cells))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


def _build_dirs(tmp_path, top_shape=(8, 10), btm_shape=(6, 8)):
    top_dir = tmp_path / "TOP"
    btm_dir = tmp_path / "BTM"
    out_dir = tmp_path / "OUT"
    top_dir.mkdir()
    btm_dir.mkdir()

    # TOP sample 1, BTM sample 2. Temps 25/240 with H and C, peak 260 at 100s.
    schedule = [(11, 25), (60, 240), (100, 260), (150, 240), (192, 25)]
    for time_s, temp_c in schedule:
        top_arr = _make_surface(top_shape[0], top_shape[1], base=5.0 + 0.01 * temp_c)
        _write_dat(
            str(top_dir / "WAFER_PT0001_{0:05d}s({1}C).dat".format(time_s, temp_c)),
            top_arr,
        )
        # BTM smaller grid by default to exercise resize.
        btm_arr = _make_surface(btm_shape[0], btm_shape[1], base=1.0 + 0.005 * temp_c)
        _write_dat(
            str(btm_dir / "WAFER_PT0002_{0:05d}s({1}C).dat".format(time_s, temp_c)),
            btm_arr,
        )
    return top_dir, btm_dir, out_dir


def test_run_pipeline_end_to_end(tmp_path):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir), reference="TOP")
    assert len(results) > 0

    names = sorted(os.path.basename(r.out_path) for r in results)
    # Expect H/C at 240 and H at 260 (peak) among outputs.
    assert "GAP-H240_TOP1-BTM2.txt" in names
    assert "GAP-C240_TOP1-BTM2.txt" in names

    # Output files exist and round-trip through load_matrix.
    for r in results:
        assert os.path.exists(r.out_path)
        loaded = load_matrix(r.out_path)
        # Reference is TOP (8x10) so output has TOP's shape.
        assert loaded.shape == (8, 10)
        # Min valid gap ~ 0.0.
        valid = ~np.isnan(loaded)
        assert loaded[valid].min() == pytest.approx(0.0, abs=1e-2)
        # Gap result min is exactly 0.0 before rounding.
        assert np.nanmin(r.result.gap) == pytest.approx(0.0, abs=1e-9)


def test_run_pipeline_populates_max_gap(tmp_path):
    # max_gap must be carried on each result so the UI never re-reads the
    # saved gap files per poll tick to obtain it.
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir), reference="TOP")
    assert len(results) > 0
    for r in results:
        assert r.max_gap is not None
        # matches the in-memory gap array's finite maximum
        assert r.max_gap == pytest.approx(float(np.nanmax(r.result.gap)))


def test_run_pipeline_creates_out_dir(tmp_path):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    assert not out_dir.exists()
    run_pipeline(str(top_dir), str(btm_dir), str(out_dir))
    assert out_dir.exists()


def test_run_pipeline_reference_btm_shape(tmp_path):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir), reference="BTM")
    assert len(results) > 0
    for r in results:
        loaded = load_matrix(r.out_path)
        assert loaded.shape == (6, 8)  # BTM grid


def test_run_pipeline_bad_reference(tmp_path):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    with pytest.raises(ValueError):
        run_pipeline(str(top_dir), str(btm_dir), str(out_dir), reference="SIDE")


def test_run_pipeline_auto_default_matches_btm_reference(tmp_path):
    # Default reference is AUTO (larger -> smaller); BTM (6x8) is smaller than
    # TOP (8x10), so it must behave exactly like reference="BTM".
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    out_btm = tmp_path / "OUT_BTM"
    auto_results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir))
    btm_results = run_pipeline(
        str(top_dir), str(btm_dir), str(out_btm), reference="BTM"
    )
    assert len(auto_results) == len(btm_results) > 0
    for ra, rt in zip(auto_results, btm_results):
        assert ra.job.out_name == rt.job.out_name
        assert ra.result.gap.shape == (6, 8)  # BTM grid (smaller)
        np.testing.assert_allclose(ra.result.gap, rt.result.gap, equal_nan=True)


def test_run_pipeline_auto_picks_smaller_grid(tmp_path):
    # TOP smaller than BTM -> AUTO makes TOP the reference grid (larger -> smaller).
    top_dir, btm_dir, out_dir = _build_dirs(
        tmp_path, top_shape=(6, 8), btm_shape=(8, 10)
    )
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir))
    assert len(results) > 0
    for r in results:
        loaded = load_matrix(r.out_path)
        assert loaded.shape == (6, 8)  # TOP grid (smaller)


def test_run_pipeline_top_transform_flip_rotate(tmp_path):
    # Flip + one clockwise turn swaps TOP's dims (8x10 -> 10x8); with
    # reference="TOP" the outputs follow the rotated TOP grid.
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(
        str(top_dir),
        str(btm_dir),
        str(out_dir),
        reference="TOP",
        top_transform=TransformConfig(flip_lr=True, rot90_cw=1),
    )
    assert len(results) > 0
    for r in results:
        loaded = load_matrix(r.out_path)
        assert loaded.shape == (10, 8)  # rotated TOP grid
        # First-contact invariant survives the transform.
        assert np.nanmin(r.result.gap) == pytest.approx(0.0, abs=1e-9)


def test_run_pipeline_zero_cell_leaves_gap_unchanged(tmp_path):
    # Zeroing a TOP cell shifts diff = TOP - BTM by a constant, and
    # gap = diff - nanmin(diff) is invariant under constant shifts, so the
    # gap results must match the untransformed run.
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    out_zero = tmp_path / "OUT_ZERO"
    plain = run_pipeline(str(top_dir), str(btm_dir), str(out_dir))
    zeroed = run_pipeline(
        str(top_dir),
        str(btm_dir),
        str(out_zero),
        top_transform=TransformConfig(zero_cell=(0, 0)),
    )
    assert len(plain) == len(zeroed) > 0
    for rp, rz in zip(plain, zeroed):
        assert rp.job.out_name == rz.job.out_name
        np.testing.assert_allclose(
            rz.result.gap, rp.result.gap, atol=1e-9, equal_nan=True
        )


def test_run_pipeline_zero_cell_on_nan_skips_job_continues(tmp_path, caplog):
    # A zero cell landing on a blank (NaN) TOP cell fails only that job; the
    # rest of the batch still runs.
    top_dir = tmp_path / "TOP"
    btm_dir = tmp_path / "BTM"
    out_dir = tmp_path / "OUT"
    top_dir.mkdir()
    btm_dir.mkdir()

    # Two heating temps (peak 260 at 100s). Only the 25C TOP file has a blank
    # hole covering (2, 3); the 260C TOP file is fully valid.
    holed = _make_surface(8, 10, base=5.0, hole=True)
    clean = _make_surface(8, 10, base=7.0, hole=False)
    _write_dat(str(top_dir / "WAFER_PT0001_00011s(25C).dat"), holed)
    _write_dat(str(top_dir / "WAFER_PT0001_00100s(260C).dat"), clean)
    for time_s, temp_c in ((11, 25), (100, 260)):
        btm_arr = _make_surface(6, 8, base=1.0, hole=False)
        _write_dat(
            str(btm_dir / "WAFER_PT0002_{0:05d}s({1}C).dat".format(time_s, temp_c)),
            btm_arr,
        )

    with caplog.at_level(logging.ERROR, logger="matrix2d.services.pipeline"):
        results = run_pipeline(
            str(top_dir),
            str(btm_dir),
            str(out_dir),
            top_transform=TransformConfig(zero_cell=(2, 3)),
        )

    # The 25C job failed (NaN zero cell); the 260C job still succeeded.
    names = [r.job.out_name for r in results]
    assert names == ["GAP-H260_TOP1-BTM2.txt"]
    assert any("blank" in rec.getMessage() for rec in caplog.records)


def test_run_pipeline_custom_prefix_in_out_names(tmp_path):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir),
                           out_prefix="TEST")
    assert len(results) > 0
    for r in results:
        assert os.path.basename(r.out_path).startswith("TEST-")


def test_run_pipeline_writes_summary_file(tmp_path):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir),
                           out_prefix="TEST")
    assert len(results) > 0

    summary_path = out_dir / "TEST.txt"
    assert summary_path.exists()

    lines = summary_path.read_text(encoding="utf-8").rstrip("\n").split("\n")
    header = lines[0].split("\t")
    # First header cell is the (blank) corner; the rest are temp points.
    assert header[0] == ""
    assert "H240" in header and "C240" in header
    cols = header[1:]
    by_label = {ln.split("\t")[0]: dict(zip(cols, ln.split("\t")[1:]))
                for ln in lines[1:]}
    # Statistic rows sit directly under the header, then the combo rows.
    assert [ln.split("\t")[0] for ln in lines[1:5]] == \
        ["MIN", "MAX", "AVG", "STD"]
    # One row per TOP-BTM combo; this fixture has TOP1 x BTM2.
    combo_labels = [ln.split("\t")[0] for ln in lines[5:]]
    assert combo_labels == ["TOP1-BTM2"]

    # The H240 cell holds a finite max gap equal to the saved gap's max.
    r = next(x for x in results if x.job.out_name == "TEST-H240_TOP1-BTM2.txt")
    expected = float(np.nanmax(r.result.gap))
    assert float(by_label["TOP1-BTM2"]["H240"]) == pytest.approx(
        expected, rel=1e-3)
    # Single combo -> MIN/MAX/AVG equal that value, STD blank.
    assert float(by_label["MAX"]["H240"]) == pytest.approx(expected, rel=1e-3)
    assert by_label["STD"]["H240"] == ""


def test_run_pipeline_bounded_xform_cache_still_correct(tmp_path):
    # A tiny transform-cache cap forces constant eviction/thrash across the
    # job loop (P1 bounded-memory change); results must be identical to an
    # unbounded (default-cache) run.
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    out_bounded = tmp_path / "OUT_BOUNDED"

    baseline = run_pipeline(str(top_dir), str(btm_dir), str(out_dir))

    old_val = os.environ.get("MATRIX2D_XFORM_CACHE")
    os.environ["MATRIX2D_XFORM_CACHE"] = "1"
    try:
        bounded = run_pipeline(str(top_dir), str(btm_dir), str(out_bounded))
    finally:
        if old_val is None:
            os.environ.pop("MATRIX2D_XFORM_CACHE", None)
        else:
            os.environ["MATRIX2D_XFORM_CACHE"] = old_val

    assert len(baseline) == len(bounded) > 0
    for rb, rc in zip(baseline, bounded):
        assert rb.job.out_name == rc.job.out_name
        np.testing.assert_allclose(
            rc.result.gap, rb.result.gap, equal_nan=True
        )


def test_run_pipeline_unexpected_error_is_collected(tmp_path, monkeypatch, caplog):
    # An unexpected (non-ValueError/OSError) error in ONE job must not abort the
    # batch; the rest still run and the failure is recorded in the list.
    from matrix2d.services import pipeline as pl

    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    real_resize = pl.resize_pair
    state = {"raised": False}

    def _flaky(top, btm, ref):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("boom")
        return real_resize(top, btm, ref)

    monkeypatch.setattr(pl, "resize_pair", _flaky)

    failures = []
    with caplog.at_level(logging.INFO, logger="matrix2d.services.pipeline"):
        results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir),
                               reference="TOP", failures=failures)

    assert len(results) > 0          # other jobs completed
    assert len(failures) == 1
    f = failures[0]
    assert set(f) == {"out_name", "top_no", "btm_no", "error"}
    assert "boom" in f["error"]
    # Aggregate line emitted with the failed count.
    assert any("Pipeline finished" in r.getMessage()
               and "1 failed" in r.getMessage() for r in caplog.records)


def test_run_pipeline_failures_none_default_unchanged(tmp_path):
    # Passing no failures list (default None) leaves behaviour unchanged.
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir))
    assert len(results) > 0


def test_run_pipeline_aggregate_log_on_success(tmp_path, caplog):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    with caplog.at_level(logging.INFO, logger="matrix2d.services.pipeline"):
        results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir))
    assert len(results) > 0
    assert any("Pipeline finished" in r.getMessage()
               and "0 failed" in r.getMessage() for r in caplog.records)


def test_run_pipeline_progress_callback(tmp_path):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    calls = []
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir),
                           progress_cb=lambda d, t: calls.append((d, t)))
    total = calls[0][1]
    # one initial (0, total) plus one call per job, done counting up to total
    assert calls[0] == (0, total)
    assert calls[-1] == (total, total)
    assert len(calls) == total + 1
    assert [d for d, _t in calls] == list(range(total + 1))
    assert len(results) == total  # every job succeeded here
