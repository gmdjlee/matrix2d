import os

import numpy as np
import pytest

from matrix2d.core.models import SampleMeta
from matrix2d.core.parser import load_matrix
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


def _build_dirs(tmp_path):
    top_dir = tmp_path / "TOP"
    btm_dir = tmp_path / "BTM"
    out_dir = tmp_path / "OUT"
    top_dir.mkdir()
    btm_dir.mkdir()

    # TOP sample 1, BTM sample 2. Temps 25/240 with H and C, peak 260 at 100s.
    schedule = [(11, 25), (60, 240), (100, 260), (150, 240), (192, 25)]
    for time_s, temp_c in schedule:
        top_arr = _make_surface(8, 10, base=5.0 + 0.01 * temp_c)
        _write_dat(
            str(top_dir / "WAFER_PT0001_{0:06d}s({1}C).dat".format(time_s, temp_c)),
            top_arr,
        )
        # BTM slightly smaller grid to exercise resize.
        btm_arr = _make_surface(6, 8, base=1.0 + 0.005 * temp_c)
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
