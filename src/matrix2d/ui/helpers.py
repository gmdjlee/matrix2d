"""UI-only helpers: SampleMeta <-> dict serialization and module-level caches.

These live in the UI layer on purpose so we never touch core/services. The
matrix cache is a simple module-level dict; this is fine for a local,
single-user Dash app (no concurrency concerns worth engineering for here).
"""

from typing import Dict, List, Optional

import numpy as np

from matrix2d.core.models import SampleMeta


# ---------------------------------------------------------------------------
# SampleMeta serialization (SampleMeta is a frozen dataclass; we round-trip via
# plain JSON-serializable dicts for dcc.Store).
# ---------------------------------------------------------------------------

_META_FIELDS = ("title", "sample_no", "time_s", "temp_c", "kind", "path")


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
    )


def meta_label(meta: SampleMeta) -> str:
    """Human-readable dropdown label, e.g. ``TOP PT0002 240C 192s``."""
    return "{kind} PT{sample:04d} {temp}C {time}s".format(
        kind=meta.kind,
        sample=meta.sample_no,
        temp=meta.temp_c,
        time=meta.time_s,
    )


def meta_label_from_dict(d: dict) -> str:
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


def gap_names() -> List[str]:
    return list(_GAP_CACHE.keys())


def clear_gaps() -> None:
    _GAP_CACHE.clear()
