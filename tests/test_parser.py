import numpy as np
import pytest

from matrix2d.core.parser import (
    BLANK_THRESHOLD,
    load_matrix,
    load_warpage,
    parse_filename,
    parse_gap_filename,
)
from matrix2d.core.models import SampleMeta, WarpageData


# ---- parse_filename ----------------------------------------------------------

def test_parse_basic():
    m = parse_filename("WAFER_PT0012_00125s(25C).dat", "TOP")
    assert isinstance(m, SampleMeta)
    assert m.title == "WAFER"
    assert m.sample_no == 12
    assert m.time_s == 125
    assert m.temp_c == 25
    assert m.kind == "TOP"


def test_parse_leading_zeros():
    m = parse_filename("A_PT0001_00009s(5C).csv", "BTM")
    assert m.sample_no == 1
    assert m.time_s == 9
    assert m.temp_c == 5


def test_parse_three_digit_temp():
    m = parse_filename("A_PT1234_12345s(260C).txt", "TOP")
    assert m.sample_no == 1234
    assert m.time_s == 12345
    assert m.temp_c == 260


def test_parse_spaces_in_title():
    m = parse_filename("WAFER TOP 2024_PT0003_00060s(240C).dat", "TOP")
    assert m.title == "WAFER TOP 2024"
    assert m.sample_no == 3


def test_parse_title_containing_underscore_and_pt():
    # Title itself contains underscores; split at the LAST _PT.
    m = parse_filename("LOT_A_PART_PT0007_00011s(25C).dat", "TOP")
    assert m.title == "LOT_A_PART"
    assert m.sample_no == 7


def test_parse_all_extensions():
    for ext in (".dat", ".csv", ".txt"):
        m = parse_filename("X_PT0002_00010s(30C)" + ext, "TOP")
        assert m.sample_no == 2


def test_parse_with_directory_prefix():
    m = parse_filename("/some/dir/X_PT0002_00010s(30C).dat", "GAP", path="p")
    assert m.sample_no == 2
    assert m.path == "p"
    assert m.kind == "GAP"


def test_parse_path_defaults_to_filename():
    m = parse_filename("X_PT0002_00010s(30C).dat", "TOP")
    assert m.path == "X_PT0002_00010s(30C).dat"


@pytest.mark.parametrize(
    "bad",
    [
        "no_pattern_here.dat",
        "X_PT12_00010s(30C).dat",       # sample not 4 digits
        "X_PT0002_0010s(30C).dat",      # time only 4 digits
        "X_PT0002_000010s(30C).dat",    # time 6 digits (old format)
        "X_PT0002_00010(30C).dat",      # missing 's'
        "X_PT0002_00010s(30).dat",      # missing 'C'
        "X_PT0002_00010s(3000C).dat",   # temp too many digits
        "X_PT0002_00010s25C.dat",       # missing parens
    ],
)
def test_parse_bad_names_raise(bad):
    with pytest.raises(ValueError):
        parse_filename(bad, "TOP")


# ---- parse_gap_filename --------------------------------------------------------

def test_parse_gap_basic():
    m = parse_gap_filename("TEST-C25_TOP3-BTM8.txt")
    assert m.kind == "GAP"
    assert m.sample_no == 3
    assert m.btm_no == 8
    assert m.phase == "C"
    assert m.temp_c == 25
    assert m.time_s == 0
    assert m.title == "TEST-C25_TOP3-BTM8"


def test_parse_gap_heating_and_duplicate_suffix():
    m = parse_gap_filename("MY RUN-H240_TOP12-BTM3_2.dat")
    assert (m.sample_no, m.btm_no, m.phase, m.temp_c) == (12, 3, "H", 240)


def test_parse_gap_prefix_with_dashes_and_underscores():
    m = parse_gap_filename("LOT-A_2026-H250_TOP1-BTM2.txt")
    assert (m.sample_no, m.btm_no, m.phase, m.temp_c) == (1, 2, "H", 250)


def test_parse_gap_with_directory_and_path():
    m = parse_gap_filename("/x/y/TEST-H240_TOP2-BTM4.txt", path="p")
    assert m.path == "p"
    assert m.sample_no == 2


def test_parse_gap_legacy_format():
    m = parse_gap_filename("TOP1-BTM12_H250.txt")
    assert m.kind == "GAP"
    assert (m.sample_no, m.btm_no, m.phase, m.temp_c) == (1, 12, "H", 250)


def test_parse_gap_legacy_duplicate_suffix():
    m = parse_gap_filename("TOP12-BTM3_C85_2.dat")
    assert (m.sample_no, m.btm_no, m.phase, m.temp_c) == (12, 3, "C", 85)


@pytest.mark.parametrize(
    "bad",
    [
        "random.txt",
        "TEST-X25_TOP3-BTM8.txt",   # phase not H/C
        "TEST-C2500_TOP3-BTM8.txt",  # temp too many digits
        "-C25_TOP3-BTM8.txt",       # empty prefix
        "TEST-C25_TOP3.txt",        # missing BTM
        "TOP1_BTM2_H240.txt",       # legacy: underscore, not dash
        "TOP1-BTM2_X240.txt",       # legacy: phase not H/C
        "TOP1-BTM2_H.txt",          # legacy: missing temp
        "A_PT0001_00011s(25C).dat",  # measurement format, not gap
    ],
)
def test_parse_gap_bad_names_raise(bad):
    with pytest.raises(ValueError):
        parse_gap_filename(bad)


# ---- load_matrix -------------------------------------------------------------

def test_load_matrix_whitespace(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")
    arr = load_matrix(str(p))
    assert arr.shape == (2, 3)
    np.testing.assert_allclose(arr, [[1, 2, 3], [4, 5, 6]])


def test_load_matrix_comma(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n")
    arr = load_matrix(str(p))
    assert arr.shape == (2, 3)
    np.testing.assert_allclose(arr, [[1, 2, 3], [4, 5, 6]])


def test_load_matrix_empty_cells(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("1.2,,3.4\n,5.5,\n")
    arr = load_matrix(str(p))
    assert arr.shape == (2, 3)
    assert np.isnan(arr[0, 1])
    assert np.isnan(arr[1, 0])
    assert np.isnan(arr[1, 2])
    assert arr[0, 0] == pytest.approx(1.2)
    assert arr[1, 1] == pytest.approx(5.5)


def test_load_matrix_nan_strings(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("1.0,nan,3.0\nNaN,5.0,NAN\n")
    arr = load_matrix(str(p))
    assert np.isnan(arr[0, 1])
    assert np.isnan(arr[1, 0])
    assert np.isnan(arr[1, 2])


def test_load_matrix_sentinel_blanks(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("1.0,2000.0,3.0\n2500,5.0,9999.9\n")
    arr = load_matrix(str(p))
    assert np.isnan(arr[0, 1])   # exactly 2000 -> blank
    assert np.isnan(arr[1, 0])
    assert np.isnan(arr[1, 2])
    assert arr[0, 0] == pytest.approx(1.0)


def test_load_matrix_ragged_padded_with_nan(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("1.0,2.0,3.0\n4.0,5.0\n")
    arr = load_matrix(str(p))
    assert arr.shape == (2, 3)
    assert np.isnan(arr[1, 2])


def test_load_matrix_skips_blank_lines(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("1.0 2.0\n\n3.0 4.0\n")
    arr = load_matrix(str(p))
    assert arr.shape == (2, 2)


def test_load_matrix_empty_file_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("\n  \n")
    with pytest.raises(ValueError):
        load_matrix(str(p))


def test_blank_threshold_value():
    assert BLANK_THRESHOLD == 2000.0


# ---- load_matrix size guard ---------------------------------------------------

def test_load_matrix_rejects_oversize_file(tmp_path, monkeypatch):
    p = tmp_path / "m.txt"
    p.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")
    monkeypatch.setattr("matrix2d.core.parser.os.path.getsize", lambda path: 10 ** 12)
    with pytest.raises(ValueError, match="too large"):
        load_matrix(str(p))


def test_load_matrix_size_guard_disabled_with_zero(tmp_path, monkeypatch):
    p = tmp_path / "m.txt"
    p.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")
    monkeypatch.setenv("MATRIX2D_MAX_FILE_MB", "0")
    monkeypatch.setattr("matrix2d.core.parser.os.path.getsize", lambda path: 10 ** 12)
    arr = load_matrix(str(p))
    assert arr.shape == (2, 3)
    np.testing.assert_allclose(arr, [[1, 2, 3], [4, 5, 6]])


def test_load_matrix_normal_file_unaffected(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")
    arr = load_matrix(str(p))
    assert arr.shape == (2, 3)
    np.testing.assert_allclose(arr, [[1, 2, 3], [4, 5, 6]])


# ---- load_warpage ------------------------------------------------------------

def test_load_warpage(tmp_path):
    p = tmp_path / "WAFER_PT0005_00060s(240C).dat"
    p.write_text("1.0,2.0\n3.0,4.0\n")
    wd = load_warpage(str(p), "TOP")
    assert isinstance(wd, WarpageData)
    assert wd.meta.sample_no == 5
    assert wd.meta.temp_c == 240
    assert wd.values.shape == (2, 2)


def test_load_warpage_gap_naming(tmp_path):
    p = tmp_path / "TEST-H250_TOP1-BTM12.txt"
    p.write_text("1.0\t2.0\n3.0\t4.0\n")
    wd = load_warpage(str(p), "GAP")
    assert wd.meta.kind == "GAP"
    assert wd.meta.sample_no == 1
    assert wd.meta.btm_no == 12
    assert wd.meta.phase == "H"
    assert wd.values.shape == (2, 2)


def test_load_warpage_gap_legacy_naming(tmp_path):
    p = tmp_path / "TOP1-BTM12_H250.txt"
    p.write_text("1.0\t2.0\n3.0\t4.0\n")
    wd = load_warpage(str(p), "GAP")
    assert wd.meta.kind == "GAP"
    assert wd.meta.sample_no == 1
    assert wd.meta.btm_no == 12
    assert wd.meta.phase == "H"


def test_load_warpage_gap_legacy_fallback(tmp_path):
    p = tmp_path / "G_PT0004_00060s(240C).dat"
    p.write_text("1.0\n")
    wd = load_warpage(str(p), "GAP")
    assert wd.meta.sample_no == 4
    assert wd.meta.btm_no is None
