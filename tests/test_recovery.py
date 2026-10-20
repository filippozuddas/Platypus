"""Injection association and per-stage outcomes through the production chain
(PLAN 03: near RFI, window overlap, cluster merge, band edges, two weak ONs,
control without injection)."""

import csv
import json

import numpy as np
import pytest
import yaml

from src.data.morphologies import MORPHOLOGIES
from src.search.injection import Catalog, build_catalog
from src.search.pipeline import CadenceSearch, CadenceWindows, Patch, SelectionRule
from src.search.recovery import OUTCOME_COLUMNS, associate, run_injections, write_background
from tests.test_search_pipeline import (
    FCHANS, N_OBS, NCHANS, ON_ONLY_CHAN, PERSISTENT_CHAN, PREPROC, STRIDE, TCHANS, StubUDMA,
)

SECOND_ON_ONLY_CHAN = 12_600          # a second ON-only event, separated by quiet windows
RULE = SelectionRule(far_quantile=0.9, off_ceiling_probe=20)
TOP_K = 30


@pytest.fixture(scope="module")
def cadence():
    rng = np.random.default_rng(0)
    obs = [rng.random((16, NCHANS), dtype=np.float32) + 1.0 for _ in range(N_OBS)]
    for i in range(N_OBS):
        obs[i][:, PERSISTENT_CHAN] += 50.0
        if i in (0, 2, 4):
            obs[i][:, ON_ONLY_CHAN] += 50.0
            obs[i][:, SECOND_ON_ONLY_CHAN] += 50.0
    return obs


@pytest.fixture(scope="module")
def search(cadence, tmp_path_factory):
    with CadenceWindows(cadence, tmp_path_factory.mktemp("w"), "w", fchans=FCHANS, tchans=TCHANS,
                        stride=STRIDE, preproc=PREPROC) as w:
        yield CadenceSearch(StubUDMA(), w, method="topk", rule=RULE, freq_step_hz=-2.79,
                            fch1_mhz=8000.0, probe_seed=1, batch_size=16, device="cpu",
                            num_workers=1, log=lambda _: None)


def _lines(cadence, chans, on=(0, 2, 4), amplitude=50.0):
    """Narrow lines at ``chans`` in the ``on`` observations, as one Patch."""
    c0, c1 = min(chans) - 2, max(chans) + 3
    data = {}
    for i in on:
        block = cadence[i][:, c0:c1].copy()
        for c in chans:
            block[:, c - c0] += amplitude
        data[i] = block
    return Patch(c0=c0, data=data), min(chans), max(chans) + 1


def _outcome(search, patch, lo, hi):
    result, scores = search.search(patch)
    return associate(result, search.background, scores, search.windows.f_starts, FCHANS,
                     lo, hi, top_k=TOP_K), result


def _background_covers(search, lo, hi):
    c = search.background.clusters
    return bool(((c.f_start_first < hi) & (c.f_start_last + FCHANS > lo)).any())


def test_control_without_injection_reports_the_background(search):
    for lo in (3000, 20_000, 36_000):
        out, result = _outcome(search, Patch(c0=0, data={}), lo, lo + FCHANS)
        assert out["detected"] == _background_covers(search, lo, lo + FCHANS)
        assert len(result.clusters) == len(search.background.clusters)


def test_on_only_injection_in_a_quiet_band_is_recovered(search, cadence):
    patch, lo, hi = _lines(cadence, [23_000])
    out, _ = _outcome(search, patch, lo, hi)
    assert out["detected"] and out["peak_in_region"] and out["in_short_list"]
    assert out["n_on_hits_full"] == 3 and out["exported"] and out["export_rank"] is not None
    assert out["max_window_score"] > out["far_thresh"]


def test_injection_next_to_persistent_rfi_merges_and_is_flagged(search, cadence):
    patch, lo, hi = _lines(cadence, [PERSISTENT_CHAN + 400])
    out, _ = _outcome(search, patch, lo, hi)
    assert out["detected"] and out["merged"]
    assert out["n_clusters_overlapping"] >= 1


def test_injection_straddling_two_windows(search, cadence):
    boundary = 25_088                           # f_start of a window; previous window ends at +512
    patch, lo, hi = _lines(cadence, [boundary + 10])
    in_region = (search.windows.f_starts < hi) & (search.windows.f_starts + FCHANS > lo)
    assert in_region.sum() == 2
    out, _ = _outcome(search, patch, lo, hi)
    assert out["detected"] and out["peak_in_region"]


def test_injection_bridging_two_events_merges_their_clusters(search, cadence):
    bg = search.background.clusters
    covers = lambda c, ch: (c.f_start_first <= ch) & (c.f_start_last + FCHANS > ch)
    assert not (covers(bg, ON_ONLY_CHAN) & covers(bg, SECOND_ON_ONLY_CHAN)).any()
    patch, lo, hi = _lines(cadence, list(range(10_400, 12_300, 150)))
    out, result = _outcome(search, patch, lo, hi)
    assert out["detected"] and out["merged"]
    merged = result.clusters[covers(result.clusters, ON_ONLY_CHAN)]
    assert len(merged) == 1 and bool(covers(merged, SECOND_ON_ONLY_CHAN).iloc[0])
    assert len(result.clusters) < len(bg) + 1


@pytest.mark.parametrize("chan", [3, NCHANS - 4])
def test_injection_at_the_band_edges(search, cadence, chan):
    patch, lo, hi = _lines(cadence, [chan])
    out, _ = _outcome(search, patch, lo, hi)
    assert out["detected"] and out["peak_in_region"]


def test_two_on_injection_and_one_on_control(search, cadence):
    two, lo, hi = _lines(cadence, [27_000], on=(0, 2), amplitude=20.0)
    out2, _ = _outcome(search, two, lo, hi)
    assert out2["detected"] and out2["n_on_hits_full"] == 2 and out2["in_short_list"]
    one, lo, hi = _lines(cadence, [27_000], on=(4,), amplitude=20.0)
    out1, _ = _outcome(search, one, lo, hi)
    assert out1["detected"] and out1["n_on_hits_full"] == 1 and not out1["in_short_list"]
    assert out1["export_rank"] is None and not out1["exported"]


def test_driver_writes_resumable_outcomes_and_background(search, tmp_path):
    with open("configs/data/gbt_fine_nightsplit.yaml") as f:
        data_cfg = yaml.safe_load(f)
    data_cfg["raw"]["nchans"] = NCHANS
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(" ".join(f"/data/cad0_obs{o}.h5" for o in range(6)) + "\n")
    build_catalog(tmp_path / "cat", manifest=manifest, data_cfg=data_cfg, seed=3,
                  morphologies=list(MORPHOLOGIES), snr_list=[0, 50], on_patterns=["3on", "1on_2"],
                  n_sites=2)
    catalog = Catalog(tmp_path / "cat")
    out_csv = tmp_path / "injections.csv"
    kw = dict(fch1_mhz=8000.0, foff_mhz=-2.79e-6, out_csv=out_csv, top_k=TOP_K, log=lambda _: None)

    injections = list(catalog.injections(row=0))
    assert run_injections(search, injections[:10], data_cfg, **kw) == 10
    assert run_injections(search, injections, data_cfg, **kw) == len(injections) - 10
    assert run_injections(search, injections, data_cfg, **kw) == 0

    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == OUTCOME_COLUMNS
    assert sorted(r["injection_id"] for r in rows) == sorted(i.injection_id for i in injections)
    controls = [r for r in rows if r["snr"] == "0.0"]
    assert controls and all(r["control"] == "True" for r in controls)
    assert all(r["control"] == "False" for r in rows if r["snr"] != "0.0")
    assert all(json.loads(r["injection_info"])["morphology"] == r["morphology"] for r in rows)

    write_background(search, tmp_path / "bg")
    scores = np.load(tmp_path / "bg" / "background_scores.npz")
    np.testing.assert_array_equal(scores["score"], search.scores)
    maps = np.load(tmp_path / "bg" / "background_maps.npz")
    assert maps["amap"].shape == (len(search.background.clusters), 6, 64)
