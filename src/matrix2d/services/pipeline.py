"""Gap-computation pipeline: planning jobs and running them end-to-end."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..core.gap import compute_gap
from ..core.models import GapResult, SampleMeta
from ..core.naming import assign_phase, gap_filename, peak_time
from ..core.resize import resize_to_reference
from .repository import load_data, save_matrix, scan_folder

logger = logging.getLogger(__name__)


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
    tops: "List[SampleMeta]", btms: "List[SampleMeta]"
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
                    base_name = gap_filename(top_meta, btm_meta, phase)

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
    reference: str = "TOP",
) -> "List[GapJobResult]":
    """Run the full gap pipeline over two folders of measurements.

    Scans ``top_dir`` and ``btm_dir``, plans jobs, and for each job loads both
    datasets, resizes the non-reference dataset to the reference dataset's shape
    (using the reference dataset's blank mask as authority), computes the gap,
    and writes it to ``out_dir`` under the job's output name.

    Errors in a single job are logged and collected; they do not abort other
    jobs.

    Args:
        top_dir: Folder of TOP measurements.
        btm_dir: Folder of BTM measurements.
        out_dir: Output folder (created if missing).
        reference: "TOP" or "BTM" -- which dataset's grid/mask is authoritative.

    Returns:
        A list of GapJobResult for the successful jobs.

    Raises:
        ValueError: If ``reference`` is not "TOP" or "BTM".
    """
    if reference not in ("TOP", "BTM"):
        raise ValueError("reference must be 'TOP' or 'BTM', got {0!r}".format(reference))

    os.makedirs(out_dir, exist_ok=True)

    tops = scan_folder(top_dir, "TOP")
    btms = scan_folder(btm_dir, "BTM")
    jobs = plan_jobs(tops, btms)

    results: List[GapJobResult] = []
    for job in jobs:
        try:
            top_data = load_data(job.top)
            btm_data = load_data(job.btm)

            top_vals = top_data.values
            btm_vals = btm_data.values

            if reference == "TOP":
                # Resize BTM to TOP's grid, TOP's mask authoritative.
                btm_vals = resize_to_reference(
                    btm_vals, top_vals, mask_mode="reference"
                )
                # Ensure TOP carries the same authoritative mask onto BTM.
            else:
                # Resize TOP to BTM's grid, BTM's mask authoritative.
                top_vals = resize_to_reference(
                    top_vals, btm_vals, mask_mode="reference"
                )

            gap_res = compute_gap(top_vals, btm_vals)

            out_path = os.path.join(out_dir, job.out_name)
            save_matrix(out_path, gap_res.gap)

            results.append(
                GapJobResult(job=job, result=gap_res, out_path=out_path)
            )
        except (ValueError, OSError) as exc:
            logger.error(
                "Job failed (TOP%s vs BTM%s, %s): %s",
                job.top.sample_no,
                job.btm.sample_no,
                job.out_name,
                exc,
            )
    return results
