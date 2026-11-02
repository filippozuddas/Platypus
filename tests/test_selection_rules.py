"""Selection-rule building blocks: local (CFAR) statistics and their incremental
update, mask-driven clustering, the leak reference row, the cross-target veto."""

import numpy as np
import pandas as pd
import pytest

from src.search.candidates import (
    cluster_candidates, full_row_hits, local_stats, update_local_stats,
)
from src.search.pipeline import NOTCH_MHZ, RULES, SelectionRule, usable_windows
from src.search.veto import count_other_targets, same_session_rows


@pytest.mark.parametrize("span", [(0, 2), (500, 503), (1990, 1999), (1000, 1000)])
def test_incremental_local_stats_equal_a_full_computation(span):
    rng = np.random.default_rng(1)
    scores = rng.standard_cauchy(2000).astype(np.float64)
    b, m = local_stats(scores, half=16)
    lo, hi = span
    changed = scores.copy()
    changed[lo:hi + 1] += 40.0
    np.testing.assert_array_equal(update_local_stats(b, m, changed, lo, hi, 16)[0],
                                  local_stats(changed, 16)[0])
    np.testing.assert_array_equal(update_local_stats(b, m, changed, lo, hi, 16)[1],
                                  local_stats(changed, 16)[1])


def test_mask_clustering_matches_the_scalar_threshold():
    rng = np.random.default_rng(2)
    f_starts = np.arange(300) * 512
    scores = rng.random(300)
    scores[[40, 41, 42, 200]] += 5
    a = cluster_candidates(f_starts, scores, 1.0, 512, 1024)
    b = cluster_candidates(f_starts, scores, np.inf, 512, 1024, above=scores > 1.0)
    pd.testing.assert_frame_equal(a, b)
    # A mask may admit windows below the scalar threshold.
    c = cluster_candidates(f_starts, scores, 1.0, 512, 1024, above=(scores > 1.0) | (f_starts == 100 * 512))
    assert len(c) == len(a) + 1


def test_leak_reference_for_a_signal_in_two_on_scans():
    amap = np.full((6, 64), 0.1)
    amap[0, 10] = amap[4, 10] = 5.0          # ON 1 and ON 3 hold the signal, ON 2 is empty
    amap[1, 10] = amap[3, 10] = 0.6          # model leakage into two OFF rows
    old = full_row_hits(amap, threshold=0.5, leak_frac=0.3, leak_ref="weakest_on")
    new = full_row_hits(amap, threshold=0.5, leak_frac=0.3, leak_ref="weakest_hit_on")
    assert old["off_leak"] and not old["in_short_list"]
    assert not new["off_leak"] and new["in_short_list"]
    # Real leakage comparable to the signal is still rejected.
    amap[1, 10] = amap[3, 10] = 2.0
    assert full_row_hits(amap, threshold=0.5, leak_frac=0.3, leak_ref="weakest_hit_on")["off_leak"]
    with pytest.raises(ValueError):
        full_row_hits(amap, threshold=0.5, leak_ref="nope")


def test_named_rules():
    assert RULES["production"] == SelectionRule()
    h = RULES["hybrid_v1"]
    assert h.cfar_half > 0 and np.isfinite(h.cfar_k) and h.far_quantile > 0.99
    assert h.leak_ref == "weakest_hit_on"
    assert RULES["hybrid_v2"].exclude_mhz == NOTCH_MHZ
    assert RULES["hybrid_v2"].cfar_k == h.cfar_k and h.exclude_mhz == ()


def test_usable_windows_follow_the_signed_frequency_axis():
    f_starts = np.arange(0, 100_000, 512)
    # fch1 1350 MHz, negative step of 1 kHz/channel: channel c is at 1350 - c/1000 MHz.
    u = usable_windows(f_starts, 1024, -1000.0, 1350.0, ((1200.0, 1341.2),))
    lo_mhz = 1350.0 - (f_starts + 1024) / 1000.0
    hi_mhz = 1350.0 - f_starts / 1000.0
    np.testing.assert_array_equal(u, ~((hi_mhz > 1200.0) & (lo_mhz < 1341.2)))
    assert u[0] and not u[100] and usable_windows(f_starts, 1024, -1000.0, 1350.0, ()).all()


def test_cross_target_counts_and_sessions():
    ref_a = pd.DataFrame({"f_start_first": [0, 10_240], "f_start_last": [512, 10_240]})
    ref_b = pd.DataFrame({"f_start_first": [10_752], "f_start_last": [11_264]})
    lo = np.array([1_000, 11_000, 50_000])
    n = count_other_targets(lo, lo + 1024, [ref_a, ref_b, ref_a.iloc[:0]], fchans=1024)
    assert n.tolist() == [1, 2, 0]
    assert count_other_targets(np.array([2_100]), np.array([2_200]), [ref_a], 1024).tolist() == [0]
    assert count_other_targets(np.array([2_100]), np.array([2_200]), [ref_a], 1024,
                               tol_channels=600).tolist() == [1]

    def row(node, mjd, sec):
        return [f"/d/{node}_guppi_{mjd}_{sec + 320 * o:05d}_SRC_{o:04d}.rawspec.0000.h5" for o in range(6)]
    cad = [row("blc01", 60000, 10000), row("blc02", 60000, 10000), row("blc01", 60000, 12000),
           row("blc01", 60005, 10000)]
    s = same_session_rows(cad)
    assert s == {0: [2], 2: [0], 1: [], 3: []}
