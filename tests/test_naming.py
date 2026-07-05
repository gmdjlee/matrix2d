import pytest

from matrix2d.core.models import SampleMeta
from matrix2d.core.naming import assign_phase, gap_filename, peak_time


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
    assert gap_filename(top, btm, "H") == "TOP1-BTM2_H240.txt"


def test_gap_filename_uses_top_temp():
    top = _meta(3, 10, 25, "TOP")
    btm = _meta(4, 10, 25, "BTM")
    assert gap_filename(top, btm, "C") == "TOP3-BTM4_C25.txt"
