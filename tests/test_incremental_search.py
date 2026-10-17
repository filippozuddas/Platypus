"""Incremental re-search (CadenceSearch.search) equals a full search of the
modified cadence, through the production selection code."""

import numpy as np
import pandas as pd
import pytest

from src.search.pipeline import (
    CadenceSearch, CadenceWindows, Patch, PatchedWindows, SelectionRule,
    score_windows, select_candidates,
)
from tests.test_search_pipeline import (
    FCHANS, N_OBS, NCHANS, ON_ONLY_CHAN, PERSISTENT_CHAN, PREPROC, STRIDE, TCHANS, StubUDMA,
)

QUIET = dict(log=lambda _: None)
SEARCH_ARGS = dict(method="topk", freq_step_hz=-2.79, fch1_mhz=8000.0, probe_seed=5,
                   batch_size=16, device="cpu")


@pytest.fixture
def cadence():
    rng = np.random.default_rng(0)
    obs = [rng.random((16, NCHANS), dtype=np.float32) + 1.0 for _ in range(N_OBS)]
    for i in range(N_OBS):
        obs[i][:, PERSISTENT_CHAN] += 50.0
        if i in (0, 2, 4):
            obs[i][:, ON_ONLY_CHAN] += 50.0
    return obs


def _on_only_line(cadence, chan, width=64, amplitude=50.0, on=(0, 2, 4)):
    c0 = chan - width // 2
    data = {}
    for i in on:
        block = cadence[i][:, c0:c0 + width].copy()
        block[:, width // 2] += amplitude
        data[i] = block
    return Patch(c0=c0, data=data)


def _full_search(cadence, patch, rule, tmp_path):
    modified = [o.copy() for o in cadence]
    for i, block in patch.data.items():
        modified[i][:, patch.c0:patch.c1] = block
    with CadenceWindows(modified, tmp_path / "full", "full", fchans=FCHANS, tchans=TCHANS,
                        stride=STRIDE, preproc=PREPROC) as w:
        scores = score_windows(StubUDMA(), w, ["topk"], batch_size=16, num_workers=1,
                               device="cpu", **QUIET)["topk"]
        return select_candidates(scores, w.f_starts, model=StubUDMA(), windows=w, rule=rule,
                                 **SEARCH_ARGS, **QUIET), scores


def _assert_same(a, b):
    pd.testing.assert_frame_equal(a.clusters, b.clusters)
    assert a.thresholds == b.thresholds
    for x, y in zip(a.amaps, b.amaps):
        np.testing.assert_array_equal(x, y)


@pytest.fixture
def make_search(cadence, tmp_path):
    opened = []

    def make(rule):
        w = CadenceWindows(cadence, tmp_path / "bg", f"bg{len(opened)}", fchans=FCHANS,
                           tchans=TCHANS, stride=STRIDE, preproc=PREPROC)
        opened.append(w)
        return CadenceSearch(StubUDMA(), w, rule=rule, num_workers=1, **SEARCH_ARGS, **QUIET)

    yield make
    for w in opened:
        w.close()


RULE = SelectionRule(far_quantile=0.95, off_ceiling_probe=20)


def test_unchanged_patch_reproduces_the_background(cadence, make_search, tmp_path):
    search = make_search(RULE)
    identity = Patch(c0=5000, data={0: cadence[0][:, 5000:5100].copy()})
    result, scores = search.search(identity)
    _assert_same(result, search.background)
    np.testing.assert_array_equal(scores, search.scores)
    full, _ = _full_search(cadence, Patch(c0=0, data={}), RULE, tmp_path)
    _assert_same(search.background, full)


@pytest.mark.parametrize("chan", [20_000,                   # quiet band
                                  PERSISTENT_CHAN + 700,    # next to a persistent RFI cluster
                                  ON_ONLY_CHAN + 600,       # extends an existing cluster
                                  32, NCHANS - 32])         # band edges
def test_incremental_equals_full_search(cadence, make_search, tmp_path, chan):
    search = make_search(RULE)
    patch = _on_only_line(cadence, chan)
    result, scores = search.search(patch)
    full, full_scores = _full_search(cadence, patch, RULE, tmp_path)
    np.testing.assert_array_equal(scores, full_scores)
    _assert_same(result, full)


def test_injection_in_a_quiet_band_becomes_a_short_listed_cluster(cadence, make_search):
    # A looser quota than RULE: with only 79 windows, a 5% quota is already
    # filled by the two existing lines, so a third would compete for it.
    search = make_search(SelectionRule(far_quantile=0.9, off_ceiling_probe=20))
    result, _ = search.search(_on_only_line(cadence, 20_000))

    def at(clusters, chan):
        return clusters[(clusters.f_start_first <= chan) & (clusters.f_start_last + FCHANS > chan)]

    assert len(at(search.background.clusters, 20_000)) == 0
    hit = at(result.clusters, 20_000)
    assert len(hit) == 1 and bool(hit.iloc[0].in_short_list)
    assert bool(at(result.clusters, ON_ONLY_CHAN).iloc[0].in_short_list)


def test_probe_window_modified_forces_recompute(cadence, make_search, tmp_path):
    every_window = SelectionRule(far_quantile=0.95, off_ceiling_probe=10_000)
    search = make_search(every_window)
    patch = _on_only_line(cadence, 20_000)
    result, _ = search.search(patch)
    full, _ = _full_search(cadence, patch, every_window, tmp_path)
    _assert_same(result, full)
    assert search.cache.probe is not None   # the unmodified background probe stays cached


def test_cache_not_corrupted_by_previous_searches(cadence, make_search):
    search = make_search(RULE)
    first, _ = search.search(_on_only_line(cadence, 20_000))
    search.search(_on_only_line(cadence, 12_345, amplitude=80.0))
    again, _ = search.search(_on_only_line(cadence, 20_000))
    _assert_same(first, again)
    _assert_same(search.search(Patch(c0=0, data={}))[0], search.background)


def test_patch_affected_windows_and_validation(cadence, tmp_path):
    with CadenceWindows(cadence, tmp_path, "v", fchans=FCHANS, tchans=TCHANS, stride=STRIDE,
                        preproc=PREPROC) as w:
        patched = PatchedWindows(w, _on_only_line(cadence, 20_000))   # channels [19968, 20032)
        assert list(w.f_starts[patched.affected]) == [19_456, 19_968]
        with pytest.raises(ValueError, match="outside"):
            PatchedWindows(w, Patch(c0=NCHANS - 10, data={0: np.zeros((16, 64), np.float32)}))
        with pytest.raises(ValueError, match="shape"):
            PatchedWindows(w, Patch(c0=0, data={0: np.zeros((8, 64), np.float32)}))
