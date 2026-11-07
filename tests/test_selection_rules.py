"""Selection-rule building blocks: local (CFAR) statistics and their incremental
update, mask-driven clustering, the leak reference row, the one-OFF gate, the
review order, rules recorded in provenance, the cross-target veto."""

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from src.search.candidates import (
    cluster_candidates, full_row_hits, local_stats, short_list_order, update_local_stats,
)
from src.search.pipeline import (
    NOTCH_MHZ, RULES, SelectionRule, rule_from_record, shares_clusters, short_list_verdict,
    usable_windows,
)
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
    v3 = RULES["hybrid_v3"]
    assert v3.leak_frac == 0.5 and v3.exclude_mhz == NOTCH_MHZ and v3.cfar_k == h.cfar_k
    v4 = RULES["hybrid_v4"]
    assert v4 == replace(v3, single_off_frac=1.0, veto_min_targets=2)
    assert shares_clusters(v3, v4) and not shares_clusters(RULES["hybrid_v2"], RULES["production"])


def test_one_off_gate_at_the_weakest_lit_on_peak():
    amap = np.full((6, 64), 0.1)
    amap[0, 10], amap[2, 12] = 8.0, 5.0      # signal in ON 1 and ON 2; ON 3 empty
    amap[3, 40] = 5.0                         # one OFF as bright as the weakest lit ON
    kw = dict(threshold=0.5, leak_frac=0.5, leak_ref="weakest_hit_on")
    off = full_row_hits(amap, **kw)
    assert not off["off_leak"] and not off["off_single_leak"] and off["in_short_list"]
    on = full_row_hits(amap, **kw, single_off_frac=1.0)
    assert on["off_single_leak"] and on["off_leak"] and not on["in_short_list"]
    assert on["n_off_hits_full"] == off["n_off_hits_full"] == 1   # two-OFF gate unchanged
    amap[3, 40] = 4.9
    assert full_row_hits(amap, **kw, single_off_frac=1.0)["in_short_list"]
    # The gate needs the OFF row above threshold even when the reference is low.
    amap[3, 40] = 0.4
    weak = full_row_hits(amap, threshold=0.5, leak_frac=0.5, single_off_frac=0.01)
    assert not weak["off_single_leak"]


def test_short_list_verdict_follows_the_rule():
    amaps = np.full((3, 6, 64), 0.1)
    amaps[:, 0, 5] = amaps[:, 2, 5] = amaps[:, 4, 5] = 4.0   # three lit ON rows
    amaps[1, 1, 9] = 4.0                                      # one bright OFF
    amaps[2, 1, 9] = amaps[2, 3, 9] = 4.0                     # two bright OFF
    np.testing.assert_array_equal(short_list_verdict(amaps, 1.0, RULES["hybrid_v3"]),
                                  [True, True, False])
    np.testing.assert_array_equal(short_list_verdict(amaps, 1.0, RULES["hybrid_v4"]),
                                  [True, False, False])


def test_review_order_survives_a_csv_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    c = pd.DataFrame({"in_short_list": rng.random(200) < 0.4,
                      "on_off_contrast": rng.lognormal(0.3, 1.0, 200)})
    order = short_list_order(c.in_short_list.to_numpy(), c.on_off_contrast.to_numpy())
    assert c.in_short_list.to_numpy()[order].all() and len(order) == c.in_short_list.sum()
    assert np.all(np.diff(c.on_off_contrast.to_numpy()[order]) <= 0)
    c.to_csv(tmp_path / "c.csv", index=False)
    back = pd.read_csv(tmp_path / "c.csv")
    np.testing.assert_array_equal(
        short_list_order(back.in_short_list.to_numpy(), back.on_off_contrast.to_numpy()), order)


def test_rule_recorded_in_provenance():
    for name in ("production", "hybrid_v3", "hybrid_v4"):
        record = json.loads(json.dumps(asdict(RULES[name])))
        assert rule_from_record(record) == RULES[name]
    # A run recorded before the one-OFF gate and the veto existed.
    old = {k: v for k, v in json.loads(json.dumps(asdict(RULES["hybrid_v3"]))).items()
           if k not in ("single_off_frac", "veto_min_targets")}
    assert rule_from_record(old) == RULES["hybrid_v3"]
    with pytest.raises(ValueError):
        rule_from_record({**old, "nope": 1})


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
