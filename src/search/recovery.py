"""
Injection recovery through the production chain: association of each injection
with the candidates of its cadence, per-stage outcomes, and the per-cadence
driver that serves both the background scan and the injection sweep from one
load of the data.

Association rule. An injection's *region* is the channel span its rendering
actually changed (``span_lo``..``span_hi`` from :func:`build_patch`); for an
SNR-0 control nothing changes, so the region is the site's window and the
outcome measures what the search reports there without a signal. A candidate
(cluster) covers ``[f_start_first, f_start_last + fchans)``; it is associated
with the injection if that range overlaps the region. When several clusters
overlap, the one whose peak window overlaps the region is preferred
(short-listed first, then higher ON/OFF contrast), otherwise the one with the
highest peak score.

Stages recorded per injection (what counts as "recovered" is decided in the
completeness analysis, not here):

* ``detected``      — a window overlapping the region scores above the
  cadence's FAR threshold, i.e. some candidate covers the region;
* ``peak_in_region``— the associated candidate's peak window overlaps the
  region (otherwise the candidate shown is something else, e.g. RFI it merged
  into);
* ``merged``        — the associated candidate overlaps a candidate of the
  background search (the injection joined an existing event);
* ``in_short_list`` — the candidate passes the ON/OFF short-list rule;
* ``export_rank``   — its position in the review order (short list by ON/OFF
  contrast), ``exported`` if within ``top_k``;
* ``reviewed``      — left empty for the human review stage.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.search.injection import Injection, build_patch
from src.search.pipeline import CadenceSearch, MethodResult, rank_for_export

OUTCOME_COLUMNS = [
    "injection_id", "row", "site", "morphology", "snr", "on_pattern", "control",
    "window_c0", "region_lo", "region_hi", "freq_mhz_at_span_lo", "freq_mhz_at_span_hi",
    "max_window_score", "far_thresh", "detected", "n_clusters_overlapping",
    "cluster_f_start_peak", "peak_in_region", "merged", "peak_score", "on_off_contrast",
    "n_on_hits_full", "n_off_hits_full", "off_leak", "in_short_list", "export_rank",
    "exported", "reviewed", "injection_info",
]


def _covers(clusters: pd.DataFrame, fchans: int, lo: int, hi: int) -> np.ndarray:
    return ((clusters["f_start_first"].to_numpy() < hi)
            & (clusters["f_start_last"].to_numpy() + fchans > lo))


def associate(result: MethodResult, background: MethodResult, scores: np.ndarray,
              f_starts: np.ndarray, fchans: int, lo: int, hi: int, *, top_k: int) -> Dict:
    """Per-stage outcome of the region ``[lo, hi)`` in a search result."""
    clusters = result.clusters
    in_region = (f_starts < hi) & (f_starts + fchans > lo)
    out = {
        "max_window_score": float(scores[in_region].max()) if in_region.any() else None,
        "far_thresh": result.thresholds["far_thresh"],
        "n_clusters_overlapping": 0, "cluster_f_start_peak": None, "peak_in_region": False,
        "merged": False, "peak_score": None, "on_off_contrast": None, "n_on_hits_full": None,
        "n_off_hits_full": None, "off_leak": None, "in_short_list": False,
        "export_rank": None, "exported": False, "reviewed": None,
    }
    if len(clusters) == 0:
        out["detected"] = False
        return out
    overlapping = clusters[_covers(clusters, fchans, lo, hi)]
    out["detected"] = len(overlapping) > 0
    out["n_clusters_overlapping"] = len(overlapping)
    if not len(overlapping):
        return out

    peaks = overlapping["f_start_peak"].to_numpy()
    peak_hits = overlapping[(peaks < hi) & (peaks + fchans > lo)]
    if len(peak_hits):
        has_map = "in_short_list" in peak_hits
        order = (peak_hits.sort_values(["in_short_list", "on_off_contrast"], ascending=False)
                 if has_map else peak_hits.sort_values("peak_score", ascending=False))
        chosen, peak_in_region = order.iloc[0], True
    else:
        chosen, peak_in_region = overlapping.sort_values("peak_score", ascending=False).iloc[0], False

    c_lo, c_hi = int(chosen["f_start_first"]), int(chosen["f_start_last"]) + fchans
    bg = background.clusters
    merged = bool(len(bg) and _covers(bg, fchans, c_lo, c_hi).any())
    out.update(cluster_f_start_peak=int(chosen["f_start_peak"]), peak_in_region=peak_in_region,
               merged=merged, peak_score=float(chosen["peak_score"]))
    if "in_short_list" in chosen:
        out.update(on_off_contrast=float(chosen["on_off_contrast"]),
                   n_on_hits_full=int(chosen["n_on_hits_full"]),
                   n_off_hits_full=int(chosen["n_off_hits_full"]),
                   off_leak=bool(chosen["off_leak"]), in_short_list=bool(chosen["in_short_list"]))
        ranked, _ = rank_for_export(result, log=lambda _: None)
        pos = np.flatnonzero(ranked["f_start_peak"].to_numpy() == out["cluster_f_start_peak"])
        if pos.size:
            out["export_rank"] = int(pos[0])
            out["exported"] = out["export_rank"] < top_k
    return out


def write_background(search: CadenceSearch, out_dir: Path) -> None:
    """Background (no injection) outputs of one cadence: window scores,
    candidate table, cluster anomaly maps and thresholds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = search.background
    np.savez(out_dir / "background_scores.npz", f_start=search.windows.f_starts, score=search.scores)
    bg.clusters.to_csv(out_dir / "background_candidates.csv", index=False)
    maps = [m for m in bg.amaps if m is not None]
    np.savez(out_dir / "background_maps.npz",
             f_start_peak=bg.clusters["f_start_peak"].to_numpy() if len(bg.clusters) else np.array([]),
             amap=np.stack(maps) if maps else np.empty((0,)))
    (out_dir / "background_thresholds.json").write_text(
        json.dumps({k: float(v) for k, v in bg.thresholds.items()}, indent=2))


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {r["injection_id"] for r in csv.DictReader(f)}


def run_injections(search: CadenceSearch, injections: Iterable[Injection], data_cfg: dict, *,
                   fch1_mhz: float, foff_mhz: float, out_csv: Path, top_k: int,
                   log=print) -> int:
    """Search the cadence once per injection and append its outcome to ``out_csv``.

    Resumable: injections already in ``out_csv`` are skipped. Returns the number
    of injections processed in this call.
    """
    done = _done_ids(out_csv)
    new_file = not out_csv.exists()
    fchans = search.windows.fchans
    control_outcomes: dict = {}
    n = 0
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTCOME_COLUMNS)
        if new_file:
            writer.writeheader()
        for inj in injections:
            if inj.injection_id in done:
                continue
            patch, record = build_patch(search.windows, inj, data_cfg,
                                        fch1_mhz=fch1_mhz, foff_mhz=foff_mhz)
            control = record["span_lo"] is None
            lo, hi = ((record["window_c0"], record["window_c1"]) if control
                      else (record["span_lo"], record["span_hi"]))
            if control:
                # Nothing was injected, so the search result is the background
                # itself and the outcome is the same for every morphology and ON
                # pattern at this site: associate it once per region.
                if (lo, hi) not in control_outcomes:
                    control_outcomes[(lo, hi)] = associate(
                        search.background, search.background, search.scores,
                        search.windows.f_starts, fchans, lo, hi, top_k=top_k)
                outcome = control_outcomes[(lo, hi)]
            else:
                result, scores = search.search(patch)
                outcome = associate(result, search.background, scores, search.windows.f_starts,
                                    fchans, lo, hi, top_k=top_k)
            info = {k: v for k, v in record.items()
                    if k not in ("injection_id", "window_c0", "window_c1", "on_pattern",
                                 "span_lo", "span_hi", "freq_mhz_at_span_lo", "freq_mhz_at_span_hi")}
            writer.writerow({
                "injection_id": inj.injection_id, "row": inj.row, "site": inj.site,
                "morphology": inj.morphology, "snr": inj.snr, "on_pattern": inj.on_pattern,
                "control": control, "window_c0": inj.window_c0, "region_lo": lo, "region_hi": hi,
                "freq_mhz_at_span_lo": record.get("freq_mhz_at_span_lo"),
                "freq_mhz_at_span_hi": record.get("freq_mhz_at_span_hi"),
                **outcome,
                "injection_info": json.dumps(info, default=lambda o: o.item() if hasattr(o, "item") else str(o),
                                             sort_keys=True),
            })
            f.flush()
            n += 1
            if n % 200 == 0:
                log(f"  {n} injections done")
    return n
