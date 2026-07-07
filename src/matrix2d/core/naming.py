"""Phase assignment and gap output filename generation.

Pure logic. No I/O.

A measurement session heats to a peak temperature then cools. The same
(sample-pair, temperature) can therefore occur twice: the earlier occurrence
(during heating) is phase 'H', the later (during cooling) is 'C'. The peak-time
rule handles both single and dual occurrence.
"""

import re
from typing import List

from .models import SampleMeta

# Characters not allowed in filenames on Windows (plus path separators).
_ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')

DEFAULT_GAP_PREFIX = "GAP"


def assign_phase(time_s: int, peak_time_s: int) -> str:
    """Return 'H' if at/before the peak time, else 'C'.

    Args:
        time_s: Measurement time of the sample.
        peak_time_s: Time at which peak temperature occurs.

    Returns:
        "H" if ``time_s <= peak_time_s`` else "C".
    """
    return "H" if time_s <= peak_time_s else "C"


def peak_time(metas: "List[SampleMeta]") -> int:
    """Return the time of the maximum-temperature measurement.

    On ties in temperature, the earliest such time is returned.

    Args:
        metas: List of SampleMeta.

    Returns:
        The time_s of the peak-temperature measurement.

    Raises:
        ValueError: If ``metas`` is empty.
    """
    if not metas:
        raise ValueError("Cannot compute peak_time of an empty meta list.")
    max_temp = max(m.temp_c for m in metas)
    times_at_peak = [m.time_s for m in metas if m.temp_c == max_temp]
    return min(times_at_peak)


def sanitize_prefix(prefix: str) -> str:
    """Clean a user-entered output prefix for use in a filename.

    Strips whitespace and characters that are illegal in filenames; an
    empty/blank result falls back to ``DEFAULT_GAP_PREFIX``.
    """
    cleaned = _ILLEGAL_FILENAME_CHARS.sub("", (prefix or "")).strip()
    return cleaned if cleaned else DEFAULT_GAP_PREFIX


def gap_filename(
    top: SampleMeta, btm: SampleMeta, phase: str,
    prefix: str = DEFAULT_GAP_PREFIX,
) -> str:
    """Build the gap output filename for a TOP/BTM pair at a given phase.

    Format: ``{prefix}-{phase}{temp}_TOP{top_no}-BTM{btm_no}.txt`` where
    temp is the TOP sample's temperature and prefix is a user-entered
    phrase (sanitized; blank falls back to ``DEFAULT_GAP_PREFIX``).
    Example: ``TEST-C25_TOP3-BTM8.txt``.

    Args:
        top: TOP sample metadata.
        btm: BTM sample metadata.
        phase: "H" or "C".
        prefix: User-entered output phrase.

    Returns:
        The output filename string.
    """
    temp = top.temp_c
    return "{0}-{1}{2}_TOP{3}-BTM{4}.txt".format(
        sanitize_prefix(prefix), phase, temp, top.sample_no, btm.sample_no
    )
