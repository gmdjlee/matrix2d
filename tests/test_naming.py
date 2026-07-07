import pytest

from matrix2d.core.models import SampleMeta
from matrix2d.core.naming import (
    DEFAULT_GAP_PREFIX,
    assign_phase,
    gap_filename,
    peak_time,
    sanitize_prefix,
)


def _meta(sample_no, time_s, temp_c, kind="TOP"):
    return SampleMeta(
        title="T", sample_no=sample_no, time_s=time_s, temp_c=temp_c, kind=kind
    )


# ---- assign_phase ------------------------------------------------------------

def test_assign_phase_heating():
    assert assign_phase(50, 100) == "H"


def test_assign_phase_at_peak_is_heating():
    assert assign_phase(100, 100) == "H"


def test_assign_phase_cooling():
    assert assign_phase(150, 100) == "C"


# ---- peak_time ---------------------------------------------------------------

def test_peak_time_single_max():
    metas = [_meta(1, 11, 25), _meta(1, 60, 240), _meta(1, 100, 260)]
    assert peak_time(metas) == 100


def test_peak_time_tie_earliest():
    # Two files at the max temp; earliest time wins.
    metas = [_meta(1, 100, 260), _meta(1, 50, 260), _meta(1, 30, 240)]
    assert peak_time(metas) == 50


def test_peak_time_empty_raises():
    with pytest.raises(ValueError):
        peak_time([])


# ---- gap_filename ------------------------------------------------------------

def test_gap_filename_format():
    top = _meta(1, 60, 240, "TOP")
    btm = _meta(2, 60, 240, "BTM")
    assert gap_filename(top, btm, "H", prefix="TEST") == "TEST-H240_TOP1-BTM2.txt"


def test_gap_filename_example_from_spec():
    top = _meta(3, 10, 25, "TOP")
    btm = _meta(8, 10, 25, "BTM")
    assert gap_filename(top, btm, "C", prefix="TEST") == "TEST-C25_TOP3-BTM8.txt"


def test_gap_filename_uses_top_temp():
    top = _meta(3, 10, 25, "TOP")
    btm = _meta(4, 10, 30, "BTM")
    assert gap_filename(top, btm, "C", prefix="X") == "X-C25_TOP3-BTM4.txt"


def test_gap_filename_default_prefix():
    top = _meta(1, 60, 240, "TOP")
    btm = _meta(2, 60, 240, "BTM")
    assert gap_filename(top, btm, "H") == \
        "{0}-H240_TOP1-BTM2.txt".format(DEFAULT_GAP_PREFIX)


# ---- sanitize_prefix -----------------------------------------------------------

def test_sanitize_prefix_passthrough():
    assert sanitize_prefix("TEST") == "TEST"


def test_sanitize_prefix_strips_illegal_chars_and_whitespace():
    assert sanitize_prefix('  a/b\\c:d*e?f"g<h>i|j  ') == "abcdefghij"


def test_sanitize_prefix_blank_falls_back_to_default():
    assert sanitize_prefix("") == DEFAULT_GAP_PREFIX
    assert sanitize_prefix(None) == DEFAULT_GAP_PREFIX
    assert sanitize_prefix("  ") == DEFAULT_GAP_PREFIX
    assert sanitize_prefix("///") == DEFAULT_GAP_PREFIX
