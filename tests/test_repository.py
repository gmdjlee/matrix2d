import os

import numpy as np
import pytest

from matrix2d.core.parser import load_matrix
from matrix2d.services import repository
from matrix2d.services.repository import (
    list_data_files,
    load_data,
    read_matrix,
    save_matrix,
    scan_folder,
)


def _write(p, text):
    p.write_text(text)


def test_scan_folder_sorted_and_parsed(tmp_path):
    (tmp_path / "A_PT0002_00100s(240C).dat").write_text("1,2\n3,4\n")
    (tmp_path / "A_PT0001_00060s(240C).csv").write_text("1,2\n3,4\n")
    (tmp_path / "A_PT0001_00011s(25C).txt").write_text("1,2\n3,4\n")
    metas = scan_folder(str(tmp_path), "TOP")
    # Sorted by (sample_no, time_s).
    assert [(m.sample_no, m.time_s) for m in metas] == [(1, 11), (1, 60), (2, 100)]
    assert all(m.kind == "TOP" for m in metas)


def test_scan_folder_skips_unparseable(tmp_path, caplog):
    (tmp_path / "good_PT0001_00011s(25C).dat").write_text("1\n")
    (tmp_path / "bad_name.dat").write_text("1\n")
    import logging

    with caplog.at_level(logging.WARNING):
        metas = scan_folder(str(tmp_path), "BTM")
    assert len(metas) == 1
    assert metas[0].sample_no == 1


def test_scan_folder_skips_bad_content(tmp_path, caplog):
    # valid name but unparseable matrix content -> skipped, not raised
    (tmp_path / "good_PT0001_00011s(25C).dat").write_text("1,2\n3,4\n")
    (tmp_path / "bad_PT0002_00011s(25C).dat").write_text("1,abc\n")
    (tmp_path / "empty_PT0003_00011s(25C).dat").write_text("\n  \n")
    import logging

    with caplog.at_level(logging.WARNING):
        metas = scan_folder(str(tmp_path), "TOP")
    assert [m.sample_no for m in metas] == [1]


def test_scan_folder_skips_oversize_file(tmp_path, caplog, monkeypatch):
    import logging

    good = tmp_path / "good_PT0001_00011s(25C).dat"
    huge = tmp_path / "huge_PT0002_00011s(25C).dat"
    good.write_text("1,2\n3,4\n")
    huge.write_text("1,2\n3,4\n")

    real_getsize = os.path.getsize

    def _fake_getsize(path):
        if os.path.abspath(path) == os.path.abspath(str(huge)):
            return 10 ** 12
        return real_getsize(path)

    monkeypatch.setattr("matrix2d.core.parser.os.path.getsize", _fake_getsize)

    with caplog.at_level(logging.WARNING):
        metas = scan_folder(str(tmp_path), "TOP")

    assert [m.sample_no for m in metas] == [1]
    assert any("too large" in rec.message for rec in caplog.records)


def test_scan_gap_folder_gap_naming(tmp_path):
    (tmp_path / "TEST-H250_TOP1-BTM12.txt").write_text("1,2\n3,4\n")
    (tmp_path / "TEST-C85_TOP2-BTM1.txt").write_text("1,2\n3,4\n")
    metas = scan_folder(str(tmp_path), "GAP")
    assert len(metas) == 2
    by_top = {m.sample_no: m for m in metas}
    assert by_top[1].btm_no == 12
    assert by_top[1].phase == "H"
    assert by_top[1].temp_c == 250
    assert by_top[2].phase == "C"
    assert all(m.kind == "GAP" for m in metas)


def test_scan_gap_folder_legacy_gap_naming(tmp_path):
    # old OUT-style gap names still parse in a GAP folder
    (tmp_path / "TOP1-BTM12_H250.txt").write_text("1,2\n3,4\n")
    metas = scan_folder(str(tmp_path), "GAP")
    assert len(metas) == 1
    assert metas[0].btm_no == 12
    assert metas[0].phase == "H"


def test_scan_gap_folder_legacy_format_fallback(tmp_path):
    # old measurement-style names still parse in a GAP folder
    (tmp_path / "G_PT0004_00060s(240C).dat").write_text("1\n")
    metas = scan_folder(str(tmp_path), "GAP")
    assert len(metas) == 1
    assert metas[0].sample_no == 4
    assert metas[0].btm_no is None
    assert metas[0].phase is None


def test_scan_folder_all_extensions(tmp_path):
    (tmp_path / "A_PT0001_00011s(25C).dat").write_text("1\n")
    (tmp_path / "A_PT0002_00011s(25C).csv").write_text("1\n")
    (tmp_path / "A_PT0003_00011s(25C).txt").write_text("1\n")
    metas = scan_folder(str(tmp_path), "TOP")
    assert len(metas) == 3


def test_list_data_files_sorted(tmp_path):
    (tmp_path / "b_PT0002_00011s(25C).dat").write_text("1\n")
    (tmp_path / "a_PT0001_00011s(25C).csv").write_text("1\n")
    (tmp_path / "c_PT0003_00011s(25C).txt").write_text("1\n")
    (tmp_path / "ignore.md").write_text("skip\n")
    paths = list_data_files(str(tmp_path))
    assert paths == sorted(paths)
    assert len(paths) == 3
    assert all(p.endswith((".dat", ".csv", ".txt")) for p in paths)


def test_scan_folder_reports_progress(tmp_path):
    # 2 valid + 1 invalid-name file -> 3 candidate files, all reported.
    (tmp_path / "A_PT0001_00011s(25C).dat").write_text("1,2\n3,4\n")
    (tmp_path / "A_PT0002_00060s(240C).csv").write_text("1,2\n3,4\n")
    (tmp_path / "bad_name.txt").write_text("1\n")
    calls = []

    def _cb(done, total):
        calls.append((done, total))

    metas = scan_folder(str(tmp_path), "TOP", progress_cb=_cb)
    assert len(metas) == 2  # invalid-name file skipped
    assert calls[-1] == (3, 3)
    dones = [c[0] for c in calls]
    assert dones == sorted(dones)  # monotonically nondecreasing
    assert all(c[1] == 3 for c in calls)


def test_load_data(tmp_path):
    p = tmp_path / "A_PT0009_00060s(240C).dat"
    p.write_text("1.0,2.0\n3.0,4.0\n")
    metas = scan_folder(str(tmp_path), "TOP")
    wd = load_data(metas[0])
    assert wd.meta.sample_no == 9
    assert wd.values.shape == (2, 2)


def test_save_matrix_roundtrip(tmp_path):
    vals = np.array([[1.25, np.nan], [3.5, 4.0]])
    out = tmp_path / "out.txt"
    save_matrix(str(out), vals)
    loaded = load_matrix(str(out))
    assert np.isnan(loaded[0, 1])
    assert loaded[0, 0] == pytest.approx(1.25)
    assert loaded[1, 1] == pytest.approx(4.0)


def test_save_matrix_writes_nan_literal(tmp_path):
    vals = np.array([[np.nan, 2.0]])
    out = tmp_path / "out.tsv"
    save_matrix(str(out), vals)
    text = out.read_text()
    assert "nan" in text


def test_save_matrix_creates_parent(tmp_path):
    vals = np.array([[1.0]])
    out = tmp_path / "nested" / "dir" / "out.txt"
    save_matrix(str(out), vals)
    assert out.exists()


def test_save_matrix_non_2d_raises(tmp_path):
    with pytest.raises(ValueError):
        save_matrix(str(tmp_path / "x.txt"), np.array([1.0, 2.0]))


def test_read_matrix_caches_unchanged_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1,2\n3,4\n")
    first = read_matrix(str(p))
    second = read_matrix(str(p))
    assert first is second
    assert np.array_equal(first, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_read_matrix_invalidates_on_rewrite(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1,2\n3,4\n")
    first = read_matrix(str(p))
    # Different size so the (mtime_ns, size) cache key changes even on
    # filesystems with coarse mtime resolution.
    p.write_text("10,20,30\n40,50,60\n")
    second = read_matrix(str(p))
    assert second is not first
    assert np.array_equal(second, np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]))


def test_read_matrix_respects_bound(tmp_path):
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("1,2\n3,4\n")
    p2.write_text("5,6\n7,8\n")
    os.environ["MATRIX2D_RAW_CACHE"] = "1"
    try:
        first = read_matrix(str(p1))
        read_matrix(str(p2))  # evicts p1 (bound = 1)
        refetched = read_matrix(str(p1))
        assert refetched is not first
    finally:
        del os.environ["MATRIX2D_RAW_CACHE"]


def test_load_data_returns_owned_copy(tmp_path):
    p = tmp_path / "A_PT0009_00060s(240C).dat"
    p.write_text("1.0,2.0\n3.0,4.0\n")
    metas = scan_folder(str(tmp_path), "TOP")
    wd = load_data(metas[0])
    cached = repository._RAW_CACHE[str(p)][1]
    assert wd.values is not cached
    assert np.array_equal(wd.values, cached)
