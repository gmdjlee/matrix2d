"""Gap-computation pipeline: planning jobs and running them end-to-end."""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ..core.gap import compute_gap
from ..core.models import GapResult, SampleMeta
from ..core.naming import (
    DEFAULT_GAP_PREFIX,
    assign_phase,
    gap_filename,
    peak_time,
)
from ..core.resize import resize_to_reference
from ..core.transform import TransformConfig, apply_transform
from .repository import load_data, save_matrix, scan_folder

logger = logging.getLogger(__name__)


def _make_matrix_loader(cfg, seed):
    """Build a memoized (load + transform) accessor for one side (TOP/BTM).

    Args:
        cfg: Optional TransformConfig applied to each loaded matrix.
        seed: Dict[str, np.ndarray] of pre-loaded RAW matrices (from the
            scan-folder validation pass), consumed lazily so each file's
            raw bytes are read from disk at most once.

    Returns:
        A ``load(meta) -> np.ndarray`` callable, memoized per path so each
        unique file is loaded and transformed exactly once across all
        pairings. Load/transform errors are cached and re-raised so a bad
        file is not retried per job (preserves the per-job skip-and-continue
        semantics of run_pipeline).
    """
    transformed = {}  # type: Dict[str, Tuple[Optional[np.ndarray], Optional[Exception]]]

    def load(meta):
        path = meta.path
        cached = transformed.get(path)
        if cached is None:
            try:
                raw = seed.pop(path, None)
                if raw is None:
                    raw = load_data(meta).values
                vals = apply_transform(raw, cfg)
                cached = (vals, None)
            except (ValueError, OSError) as exc:
                cached = (None, exc)
            transformed[path] = cached
        vals, exc = cached
        if exc is not None:
            raise exc
        return vals

    return load


@dataclass
class GapJob:
    """A planned pairing of a TOP and BTM measurement to compute a gap."""

    top: SampleMeta
    btm: SampleMeta
    phase: str
    out_name: str


@dataclass
class GapJobResult:
    """The outcome of running a GapJob."""

    job: GapJob
    result: GapResult
    out_path: str


def _group_by_sample(metas: "List[SampleMeta]") -> "Dict[int, List[SampleMeta]]":
    """Group metadata by sample number."""
    groups = {}  # type: Dict[int, List[SampleMeta]]
    for m in metas:
        groups.setdefault(m.sample_no, []).append(m)
    return groups


def _phase_map(
    metas: "List[SampleMeta]",
) -> "Dict[Tuple[int, str], SampleMeta]":
    """Map (temp_c, phase) -> meta for one sample's file list.

    The phase is derived from the sample's own peak time. If multiple files map
    to the same (temp, phase), the earliest-time file wins (deterministic).
    """
    if not metas:
        return {}
    peak = peak_time(metas)
    result = {}  # type: Dict[Tuple[int, str], SampleMeta]
    # Sort by time so earliest wins on collision.
    for m in sorted(metas, key=lambda x: x.time_s):
        phase = assign_phase(m.time_s, peak)
        key = (m.temp_c, phase)
        if key not in result:
            result[key] = m
    return result


def plan_jobs(
    tops: "List[SampleMeta]", btms: "List[SampleMeta]",
    out_prefix: str = DEFAULT_GAP_PREFIX,
) -> "List[GapJob]":
    """Plan gap jobs for every TOP-sample x BTM-sample combination.

    For each (top_sample_no, btm_sample_no) pair, and for each temperature
    present in both samples' file sets, TOP-H is paired with BTM-H and TOP-C
    with BTM-C (a pairing is skipped when the counterpart phase is missing).

    Phase is computed per sample from that sample's own file list. Output names
    are deduplicated by appending _2, _3, ... on collision.

    Args:
        tops: TOP sample metadata (may span multiple samples).
        btms: BTM sample metadata (may span multiple samples).
        out_prefix: User phrase for output filenames
            (``{prefix}-{H|C}{temp}_TOP{n}-BTM{m}.txt``).

    Returns:
        A list of GapJob, ordered deterministically.
    """
    top_groups = _group_by_sample(tops)
    btm_groups = _group_by_sample(btms)

    jobs: List[GapJob] = []
    used_names = {}  # type: Dict[str, int]

    for top_no in sorted(top_groups.keys()):
        top_pmap = _phase_map(top_groups[top_no])
        for btm_no in sorted(btm_groups.keys()):
            btm_pmap = _phase_map(btm_groups[btm_no])

            temps_top = set(t for (t, _p) in top_pmap.keys())
            temps_btm = set(t for (t, _p) in btm_pmap.keys())
            common_temps = sorted(temps_top & temps_btm)

            for temp in common_temps:
                for phase in ("H", "C"):
                    key = (temp, phase)
                    if key not in top_pmap or key not in btm_pmap:
                        continue
                    top_meta = top_pmap[key]
                    btm_meta = btm_pmap[key]
                    base_name = gap_filename(
                        top_meta, btm_meta, phase, prefix=out_prefix)

                    count = used_names.get(base_name, 0)
                    used_names[base_name] = count + 1
                    if count == 0:
                        out_name = base_name
                    else:
                        stem, ext = os.path.splitext(base_name)
                        out_name = "{0}_{1}{2}".format(stem, count + 1, ext)

                    jobs.append(
                        GapJob(
                            top=top_meta,
                            btm=btm_meta,
                            phase=phase,
                            out_name=out_name,
                        )
                    )
    return jobs


def run_pipeline(
    top_dir: str,
    btm_dir: str,
    out_dir: str,
    reference: str = "AUTO",
    top_transform: "Optional[TransformConfig]" = None,
    btm_transform: "Optional[TransformConfig]" = None,
    out_prefix: str = DEFAULT_GAP_PREFIX,
    progress_cb: "Optional[Callable[[int, int], None]]" = None,
    retain_gap: bool = True,
) -> "List[GapJobResult]":
    """Run the full gap pipeline over two folders of measurements.

    Scans ``top_dir`` and ``btm_dir``, plans jobs, and for each job loads both
    datasets, applies the optional orientation transforms (flip -> rotate ->
    zero, before any resize), resizes the non-reference dataset to the
    reference dataset's shape (using the reference dataset's blank mask as
    authority), computes the gap, and writes it to ``out_dir`` under the job's
    output name.

    Errors in a single job (including transform errors such as a blank/NaN
    zero cell) are logged and collected; they do not abort other jobs.

    Args:
        top_dir: Folder of TOP measurements.
        btm_dir: Folder of BTM measurements.
        out_dir: Output folder (created if missing).
        reference: "AUTO", "TOP" or "BTM" -- which dataset's grid/mask is
            authoritative. With "AUTO", each job picks (after transforms) the
            dataset with the larger element count; ties go to TOP.
        top_transform: Optional TransformConfig applied to each TOP matrix.
        btm_transform: Optional TransformConfig applied to each BTM matrix.
        out_prefix: User phrase for output filenames
            (``{prefix}-{H|C}{temp}_TOP{n}-BTM{m}.txt``).
        progress_cb: Optional ``callback(done, total)`` invoked once with
            ``(0, total)`` after planning and again after every job
            (successful or failed). Exceptions from the callback propagate.
        retain_gap: when True (default) each returned GapJobResult carries
            the full gap array in result.gap; when False the array is
            dropped after it is saved to disk (result.gap is None,
            result.offset/contact_index kept) so a large batch does not
            accumulate every gap in memory. The gap file on disk is written
            regardless.

    Returns:
        A list of GapJobResult for the successful jobs.

    Raises:
        ValueError: If ``reference`` is not "AUTO", "TOP" or "BTM".
    """
    if reference not in ("AUTO", "TOP", "BTM"):
        raise ValueError(
            "reference must be 'AUTO', 'TOP' or 'BTM', got {0!r}".format(reference)
        )

    os.makedirs(out_dir, exist_ok=True)

    top_seed = {}  # type: Dict[str, np.ndarray]
    btm_seed = {}  # type: Dict[str, np.ndarray]
    tops = scan_folder(top_dir, "TOP", matrix_cache=top_seed)
    btms = scan_folder(btm_dir, "BTM", matrix_cache=btm_seed)
    jobs = plan_jobs(tops, btms, out_prefix=out_prefix)

    top_load = _make_matrix_loader(top_transform, top_seed)
    btm_load = _make_matrix_loader(btm_transform, btm_seed)

    if progress_cb is not None:
        progress_cb(0, len(jobs))

    results: List[GapJobResult] = []
    for done, job in enumerate(jobs, start=1):
        try:
            top_vals = top_load(job.top)
            btm_vals = btm_load(job.btm)

            # Resolve the effective reference per job. AUTO picks the dataset
            # with more elements (rotation changes dims but not size); tie -> TOP.
            if reference == "AUTO":
                effective_ref = "TOP" if top_vals.size >= btm_vals.size else "BTM"
            else:
                effective_ref = reference

            if effective_ref == "TOP":
                # Resize BTM to TOP's grid, TOP's mask authoritative.
                btm_vals = resize_to_reference(
                    btm_vals, top_vals, mask_mode="reference"
                )
            else:
                # Resize TOP to BTM's grid, BTM's mask authoritative.
                top_vals = resize_to_reference(
                    top_vals, btm_vals, mask_mode="reference"
                )

            # Defensive post-resize guard: both grids must match exactly.
            if top_vals.shape != btm_vals.shape:
                raise ValueError(
                    "shape mismatch after resize: TOP {0} vs BTM {1}".format(
                        top_vals.shape, btm_vals.shape
                    )
                )

            gap_res = compute_gap(top_vals, btm_vals)

            out_path = os.path.join(out_dir, job.out_name)
            save_matrix(out_path, gap_res.gap)

            if retain_gap:
                kept = gap_res
            else:
                kept = GapResult(
                    gap=None,
                    offset=gap_res.offset,
                    contact_index=gap_res.contact_index,
                )
            results.append(
                GapJobResult(job=job, result=kept, out_path=out_path)
            )
        except (ValueError, OSError) as exc:
            logger.error(
                "Job failed (TOP%s vs BTM%s, %s): %s",
                job.top.sample_no,
                job.btm.sample_no,
                job.out_name,
                exc,
            )
        finally:
            if progress_cb is not None:
                progress_cb(done, len(jobs))
    return results
