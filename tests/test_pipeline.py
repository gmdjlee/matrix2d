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
    assert "TOP1-BTM2_H240.txt" in names
    assert "TOP1-BTM2_C240.txt" in names
    assert "TOP1-BTM2_H260.txt" in names


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
            str(top_dir / "WAFER_PT0001_{0:06d}s({1}C).dat".format(time_s, temp_c)),
            top_arr,
        )
        # BTM smaller grid by default to exercise resize.
        btm_arr = _make_surface(btm_shape[0], btm_shape[1], base=1.0 + 0.005 * temp_c)
        _write_dat(
            str(btm_dir / "WAFER_PT0002_{0:06d}s({1}C).dat".format(time_s, temp_c)),
            btm_arr,
        )
    return top_dir, btm_dir, out_dir


def test_run_pipeline_end_to_end(tmp_path):
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir), reference="TOP")
    assert len(results) > 0

    names = sorted(os.path.basename(r.out_path) for r in results)
    # Expect H/C at 240 and H at 260 (peak) among outputs.
    assert "TOP1-BTM2_H240.txt" in names
    assert "TOP1-BTM2_C240.txt" in names

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


def test_run_pipeline_auto_default_matches_top_reference(tmp_path):
    # Default reference is AUTO; TOP (8x10) outsizes BTM (6x8), so it must
    # behave exactly like the old reference="TOP" default.
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    out_top = tmp_path / "OUT_TOP"
    auto_results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir))
    top_results = run_pipeline(
        str(top_dir), str(btm_dir), str(out_top), reference="TOP"
    )
    assert len(auto_results) == len(top_results) > 0
    for ra, rt in zip(auto_results, top_results):
        assert ra.job.out_name == rt.job.out_name
        assert ra.result.gap.shape == (8, 10)  # TOP grid
        np.testing.assert_allclose(ra.result.gap, rt.result.gap, equal_nan=True)


def test_run_pipeline_auto_picks_larger_grid(tmp_path):
    # BTM larger than TOP -> AUTO makes BTM the reference grid.
    top_dir, btm_dir, out_dir = _build_dirs(
        tmp_path, top_shape=(6, 8), btm_shape=(8, 10)
    )
    results = run_pipeline(str(top_dir), str(btm_dir), str(out_dir))
    assert len(results) > 0
    for r in results:
        loaded = load_matrix(r.out_path)
        assert loaded.shape == (8, 10)  # BTM grid


def test_run_pipeline_top_transform_flip_rotate(tmp_path):
    # Flip + one clockwise turn swaps TOP's dims (8x10 -> 10x8); the element
    # count is unchanged so AUTO still picks TOP, and outputs follow the
    # rotated grid.
    top_dir, btm_dir, out_dir = _build_dirs(tmp_path)
    results = run_pipeline(
        str(top_dir),
        str(btm_dir),
        str(out_dir),
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
    _write_dat(str(top_dir / "WAFER_PT0001_000011s(25C).dat"), holed)
    _write_dat(str(top_dir / "WAFER_PT0001_000100s(260C).dat"), clean)
    for time_s, temp_c in ((11, 25), (100, 260)):
        btm_arr = _make_surface(6, 8, base=1.0, hole=False)
        _write_dat(
            str(btm_dir / "WAFER_PT0002_{0:06d}s({1}C).dat".format(time_s, temp_c)),
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
    assert names == ["TOP1-BTM2_H260.txt"]
    assert any("blank" in rec.getMessage() for rec in caplog.records)
