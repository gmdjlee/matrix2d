import numpy as np
import pytest

from matrix2d.core.parser import (
    BLANK_THRESHOLD,
    load_matrix,
    load_warpage,
    parse_filename,
)
from matrix2d.core.models import SampleMeta, WarpageData


# ---- parse_filename ----------------------------------------------------------

def test_parse_basic():
    m = parse_filename("WAFER_PT0012_000125s(25C).dat", "TOP")
    assert isinstance(m, SampleMeta)
    assert m.title == "WAFER"
    assert m.sample_no == 12
    assert m.time_s == 125
    assert m.temp_c == 25
    assert m.kind == "TOP"


def test_parse_leading_zeros():
    m = parse_filename("A_PT0001_000009s(5C).csv", "BTM")
    assert m.sample_no == 1
    assert m.time_s == 9
    assert m.temp_c == 5


def test_parse_three_digit_temp():
    m = parse_filename("A_PT1234_123456s(260C).txt", "TOP")
    assert m.sample_no == 1234
    assert m.time_s == 123456
    assert m.temp_c == 260


def test_parse_spaces_in_title():
    m = parse_filename("WAFER TOP 2024_PT0003_000060s(240C).dat", "TOP")
    assert m.title == "WAFER TOP 2024"
    assert m.sample_no == 3


def test_parse_title_containing_underscore_and_pt():
    # Title itself contains underscores; split at the LAST _PT.
    m = parse_filename("LOT_A_PART_PT0007_000011s(25C).dat", "TOP")
    assert m.title == "LOT_A_PART"
    assert m.sample_no == 7


def test_parse_all_extensions():
    for ext in (".dat", ".csv", ".txt"):
        m = parse_filename("X_PT0002_000010s(30C)" + ext, "TOP")
        assert m.sample_no == 2


def test_parse_with_directory_prefix():
    m = parse_filename("/some/dir/X_PT0002_000010s(30C).dat", "GAP", path="p")
    assert m.sample_no == 2
    assert m.path == "p"
    assert m.kind == "GAP"


def test_parse_path_defaults_to_filename():
    m = parse_filename("X_PT0002_000010s(30C).dat", "TOP")
    assert m.path == "X_PT0002_000010s(30C).dat"


@pytest.mark.parametrize(
    "bad",
    [
        "no_pattern_here.dat",
        "X_PT12_000010s(30C).dat",       # sample not 4 digits
        "X_PT0002_00010s(30C).dat",      # time not 6 digits
        "X_PT0002_000010(30C).dat",      # missing 's'
        "X_PT0002_000010s(30).dat",      # missing 'C'
        "X_PT0002_000010s(3000C).dat",   # temp too many digits
        "X_PT0002_000010s25C.dat",       # missing parens
    ],
)
def test_parse_bad_names_raise(bad):
    with pytest.raises(ValueError):
        parse_filename(bad, "TOP")


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


# ---- load_warpage ------------------------------------------------------------

def test_load_warpage(tmp_path):
    p = tmp_path / "WAFER_PT0005_000060s(240C).dat"
    p.write_text("1.0,2.0\n3.0,4.0\n")
    wd = load_warpage(str(p), "TOP")
    assert isinstance(wd, WarpageData)
    assert wd.meta.sample_no == 5
    assert wd.meta.temp_c == 240
    assert wd.values.shape == (2, 2)
