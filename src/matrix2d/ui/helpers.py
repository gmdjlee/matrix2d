"""UI-only helpers: SampleMeta <-> dict serialization and module-level caches.

These live in the UI layer on purpose so we never touch core/services. The
matrix cache is a simple module-level dict; this is fine for a local,
single-user Dash app (no concurrency concerns worth engineering for here).
"""

import re
from typing import Dict, List, Optional

import numpy as np

from matrix2d.core import naming
from matrix2d.core.models import SampleMeta
from matrix2d.core.transform import TransformConfig, apply_transform


# ---------------------------------------------------------------------------
# SampleMeta serialization (SampleMeta is a frozen dataclass; we round-trip via
# plain JSON-serializable dicts for dcc.Store).
# ---------------------------------------------------------------------------

_META_FIELDS = ("title", "sample_no", "time_s", "temp_c", "kind", "path",
                "btm_no", "phase")


def meta_to_dict(meta: SampleMeta) -> dict:
    return {f: getattr(meta, f) for f in _META_FIELDS}


def meta_from_dict(d: dict) -> SampleMeta:
    return SampleMeta(
        title=d["title"],
        sample_no=d["sample_no"],
        time_s=d["time_s"],
        temp_c=d["temp_c"],
        kind=d["kind"],
        path=d["path"],
        btm_no=d.get("btm_no"),   # .get: dicts stored before the field existed
        phase=d.get("phase"),
    )


def meta_label(meta: SampleMeta) -> str:
    """Human-readable dropdown label, e.g. ``TOP PT0002 240C 192s``.

    Gap-named files render as ``GAP TOP1-BTM12 H250C``.
    """
    if meta.kind == "GAP" and meta.btm_no is not None:
        return "GAP TOP{top}-BTM{btm} {phase}{temp}C".format(
            top=meta.sample_no, btm=meta.btm_no,
            phase=meta.phase or "", temp=meta.temp_c)
    return "{kind} PT{sample:04d} {temp}C {time}s".format(
        kind=meta.kind,
        sample=meta.sample_no,
        temp=meta.temp_c,
        time=meta.time_s,
    )


def meta_label_from_dict(d: dict) -> str:
    if d.get("kind") == "GAP" and d.get("btm_no") is not None:
        return "GAP TOP{top}-BTM{btm} {phase}{temp}C".format(
            top=d.get("sample_no", "?"), btm=d.get("btm_no"),
            phase=d.get("phase") or "", temp=d.get("temp_c", "?"))
    try:
        sample = "PT{0:04d}".format(int(d.get("sample_no")))
    except (TypeError, ValueError):
        sample = "PT????"
    return "{kind} {sample} {temp}C {time}s".format(
        kind=d.get("kind", "?"),
        sample=sample,
        temp=d.get("temp_c", "?"),
        time=d.get("time_s", "?"),
    )


# ---------------------------------------------------------------------------
# Selection helpers: phase-aware grouping so the UI can offer
# "sample number + temperature (H/C)" pickers instead of raw file lists.
# ---------------------------------------------------------------------------

def phase_entries(meta_dicts: "List[dict]") -> "List[dict]":
    """Assign an H/C phase to every meta dict, per sample.

    Dicts with an explicit ``phase`` ("H"/"C", set by gap-named files) keep
    it as-is. The rest are grouped by sample_no; each sample's peak time
    (time of max temperature) tags each measurement ``H`` (time <= peak)
    or ``C``.

    Returns a list of ``{"phase", "temp_c", "time_s", "sample_no", "meta"}``
    dicts, in the same order as the input. Dicts that fail to convert are
    skipped.
    """
    by_sample = {}  # type: Dict[int, List[dict]]
    for d in meta_dicts:
        if d.get("phase") in ("H", "C"):
            continue  # explicit phase: peak-time derivation not needed
        try:
            by_sample.setdefault(int(d["sample_no"]), []).append(d)
        except (KeyError, TypeError, ValueError):
            continue

    peak_by_sample = {}
    for sample_no, dicts in by_sample.items():
        metas = []
        for d in dicts:
            try:
                metas.append(meta_from_dict(d))
            except (KeyError, TypeError, ValueError):
                continue  # docstring contract: bad dicts are skipped
        if not metas:
            continue
        peak_by_sample[sample_no] = naming.peak_time(metas)

    entries = []
    for d in meta_dicts:
        try:
            meta_from_dict(d)  # full-convertibility check (docstring contract)
            sample_no = int(d["sample_no"])
            time_s = int(d["time_s"])
            temp_c = int(d["temp_c"])
            if d.get("phase") in ("H", "C"):
                phase = d["phase"]
            else:
                phase = naming.assign_phase(time_s, peak_by_sample[sample_no])
        except (KeyError, TypeError, ValueError):
            continue
        entries.append({
            "phase": phase,
            "temp_c": temp_c,
            "time_s": time_s,
            "sample_no": sample_no,
            "meta": d,
        })
    return entries


def phase_temp_key(phase: str, temp_c: int) -> str:
    """Encode a (phase, temperature) pair as a dropdown value, e.g. ``H240``."""
    return "{0}{1}".format(phase, temp_c)


def sort_phase_temps(pairs):
    """Sort (phase, temp) pairs in session order: H by rising temp, then C falling."""
    heating = sorted(p for p in pairs if p[0] == "H")
    cooling = sorted((p for p in pairs if p[0] == "C"), key=lambda p: -p[1])
    return list(heating) + list(cooling)


_GAP_NAME_RE = re.compile(
    r"^.+-(?P<phase>[HC])(?P<temp>\d{1,3})_TOP(?P<top>\d+)-BTM(?P<btm>\d+)")
_GAP_NAME_LEGACY_RE = re.compile(
    r"^TOP(?P<top>\d+)-BTM(?P<btm>\d+)_(?P<phase>[HC])(?P<temp>\d{1,3})")


def parse_gap_name(out_name: str) -> Optional[dict]:
    """Parse a gap output name like ``TEST-C25_TOP3-BTM8.txt`` (``_2`` ok).

    The legacy naming ``TOP1-BTM2_H240.txt`` is also accepted. Returns
    ``{"top_no", "btm_no", "phase", "temp_c"}`` or None if the name matches
    neither naming convention.
    """
    m = _GAP_NAME_RE.match(out_name) or _GAP_NAME_LEGACY_RE.match(out_name)
    if not m:
        return None
    return {
        "top_no": int(m.group("top")),
        "btm_no": int(m.group("btm")),
        "phase": m.group("phase"),
        "temp_c": int(m.group("temp")),
    }


# ---------------------------------------------------------------------------
# Data-transform helpers: translate raw sidebar control values into a
# TransformConfig (or None for identity). Pure functions, no Dash imports.
# ---------------------------------------------------------------------------

def _cell_index(value) -> Optional[int]:
    """Coerce a dcc.Input value to an int index; None/empty/garbage -> None."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_transform_config(flip_value, rotate_value, zero_row, zero_col
                           ) -> Optional[TransformConfig]:
    """Build a TransformConfig from raw sidebar control values.

    Args:
        flip_value: dcc.Checklist value (list or None); flip is on when
            ``"flip"`` is in it.
        rotate_value: Dropdown value in clockwise degrees (0/90/180/270);
            None or "" is treated as 0.
        zero_row: dcc.Input value for the zero-cell row (None/""/number).
        zero_col: dcc.Input value for the zero-cell column (None/""/number).

    The zero cell is used only when BOTH row and col are set; a half-filled
    pair (only row or only col) is treated as "no zero cell".

    Returns:
        None when every control resolves to the identity transform, else a
        TransformConfig.
    """
    flip = "flip" in (flip_value or [])

    try:
        degrees = int(rotate_value) if rotate_value not in (None, "") else 0
    except (TypeError, ValueError):
        degrees = 0
    steps = (degrees // 90) % 4

    row = _cell_index(zero_row)
    col = _cell_index(zero_col)
    zero = (row, col) if row is not None and col is not None else None

    if not flip and steps == 0 and zero is None:
        return None
    return TransformConfig(flip_lr=flip, rot90_cw=steps, zero_cell=zero)


def transformed_matrix(meta_dict: dict, config: Optional[TransformConfig]
                       ) -> np.ndarray:
    """Load (via the cache) the matrix for a meta dict and apply ``config``.

    apply_transform always returns a new array (even for config=None), so the
    cached raw matrix is never mutated and never aliased by the result.
    """
    values = load_matrix(meta_dict)
    return apply_transform(values, config)


# ---------------------------------------------------------------------------
# Matrix caches.
#   _MATRIX_CACHE: keyed by file path -> ndarray (loaded input datasets)
#   _GAP_CACHE:    keyed by out_name  -> ndarray (computed gap results)
# ---------------------------------------------------------------------------

_MATRIX_CACHE = {}  # type: Dict[str, np.ndarray]
_GAP_CACHE = {}     # type: Dict[str, np.ndarray]


def get_matrix(path: str) -> Optional[np.ndarray]:
    return _MATRIX_CACHE.get(path)


def load_matrix(meta_dict: dict) -> np.ndarray:
    """Load (and cache) the matrix for a serialized SampleMeta dict.

    Uses services.load_data so we honor whatever parsing the core layer does.
    """
    path = meta_dict["path"]
    if path in _MATRIX_CACHE:
        return _MATRIX_CACHE[path]
    from matrix2d.services.repository import load_data
    data = load_data(meta_from_dict(meta_dict))
    arr = np.asarray(data.values, dtype="float64")
    _MATRIX_CACHE[path] = arr
    return arr


def cache_gap(out_name: str, gap: np.ndarray) -> None:
    _GAP_CACHE[out_name] = np.asarray(gap, dtype="float64")


def get_gap(out_name: str) -> Optional[np.ndarray]:
    return _GAP_CACHE.get(out_name)


def clear_gaps() -> None:
    _GAP_CACHE.clear()
