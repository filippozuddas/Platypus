"""
Cross-target veto: a candidate seen at the same frequency toward other targets.

A technosignature comes from one direction on the sky. Within one observing
night, each compute node records the same sub-band for every target observed,
so the other targets' cadences of that night and node are further OFF
pointings for this one: an event present at the same channels in them is
local interference (the direction-of-origin rule of UCLA SETI, Pinchuk &
Margot 2022; the test that resolved BLC1). The veto never removes a candidate
from the tables; it adds the count of other targets and a flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.search.injection import night_groups

__all__ = ["same_session_rows", "count_other_targets", "VETO_COLUMNS"]

VETO_COLUMNS = ["n_other_targets", "cross_target_veto"]


def _node(path: str) -> str:
    return Path(path).name.split("_")[0]


def same_session_rows(cadences: Sequence[Sequence[str]], gap_hours: float = 12.0) -> Dict[int, List[int]]:
    """For each manifest row, the other rows of the same observing night and
    compute node (same frequency grid, other cadences)."""
    out: Dict[int, List[int]] = {}
    for night in night_groups(cadences, gap_hours):
        for r in night:
            out[r] = [o for o in night if o != r and _node(cadences[o][0]) == _node(cadences[r][0])]
    return out


def count_other_targets(lo: np.ndarray, hi: np.ndarray, references: Sequence[pd.DataFrame],
                        fchans: int, tol_channels: int = 0) -> np.ndarray:
    """Number of reference candidate tables with a cluster overlapping each
    channel span ``[lo, hi)`` (widened by ``tol_channels`` on both sides).

    References are the other targets' candidate tables (``f_start_first`` /
    ``f_start_last`` columns); a cluster covers ``[f_start_first, f_start_last + fchans)``.
    """
    lo = np.asarray(lo) - tol_channels
    hi = np.asarray(hi) + tol_channels
    count = np.zeros(len(lo), dtype=int)
    for ref in references:
        if len(ref) == 0:
            continue
        r_lo = np.sort(ref["f_start_first"].to_numpy())
        r_hi = ref["f_start_last"].to_numpy() + fchans
        order = np.argsort(ref["f_start_first"].to_numpy())
        r_hi_sorted = r_hi[order]
        # A cluster overlaps [lo, hi) if it starts before hi and ends after lo:
        # among clusters starting before hi, check the furthest end.
        n_before = np.searchsorted(r_lo, hi, side="left")
        run_max = np.maximum.accumulate(r_hi_sorted)
        hit = np.zeros(len(lo), dtype=bool)
        has = n_before > 0
        hit[has] = run_max[n_before[has] - 1] > lo[has]
        count += hit
    return count
