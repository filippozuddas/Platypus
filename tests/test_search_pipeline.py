"""The production search chain (src/search/pipeline.py) on a toy cadence.

The model is a deterministic stub with UDMA's scoring interface; every stage
after it (windows, preprocessing, thresholds, clustering, ON/OFF, short list,
ranking) is the production code itself.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.search.pipeline import (
    CadenceWindows, SelectionRule, enable_determinism, preprocess_window, rank_for_export,
    score_windows, select_candidates,
)

N_OBS, NTIME, NCHANS, FCHANS, TCHANS, STRIDE = 6, 16, 40960, 1024, 96, 512
PREPROC = {"bandpass_method": "polynomial", "poly_degree": 3, "mad_epsilon": 1e-6}
ON_ONLY_CHAN, PERSISTENT_CHAN = 10_000, 30_000
CSV_COLUMNS = [
    "f_start_peak", "freq_mhz_peak", "peak_score", "mean_score", "n_snippets",
    "f_start_first", "f_start_last", "cluster_width_channels", "on_off_contrast",
    "on_mean", "off_mean", "n_on_hits", "n_off_hits", "n_on_hits_full",
    "n_off_hits_full", "off_leak", "in_short_list", "off_ceiling", "thresh_3", "thresh_5",
]


class StubUDMA(torch.nn.Module):
    """Anomaly map = mean squared input per (16 rows x 16 channels) cell, i.e.
    UDMA's (6, 64) grid over a (96, 1024) window; score = mean of the top 1%."""

    def anomaly_map(self, x):
        return F.avg_pool2d(x ** 2, kernel_size=16)[:, 0]

    def anomaly_score(self, x, method="topk"):
        m = self.anomaly_map(x).flatten(1)
        if method == "max":
            return m.max(dim=1).values
        k = max(1, int(round(0.01 * m.shape[1])))
        return m.topk(k, dim=1).values.mean(dim=1)


@pytest.fixture
def cadence():
    rng = np.random.default_rng(0)
    obs = [rng.random((NTIME, NCHANS), dtype=np.float32) + 1.0 for _ in range(N_OBS)]
    for i in range(N_OBS):
        obs[i][:, PERSISTENT_CHAN] += 50.0          # present in every observation
        if i in (0, 2, 4):
            obs[i][:, ON_ONLY_CHAN] += 50.0         # present in the ON observations only
    return obs


@pytest.fixture
def windows(cadence, tmp_path):
    with CadenceWindows(cadence, tmp_path, "cad00", fchans=FCHANS, tchans=TCHANS,
                        stride=STRIDE, preproc=PREPROC) as w:
        yield w


def _search(windows, rule, probe_seed=0):
    model = StubUDMA()
    scores = {"topk": np.array([
        float(model.anomaly_score(torch.from_numpy(windows.window(int(f)))[None, None])[0])
        for f in windows.f_starts])}
    return select_candidates(scores["topk"], windows.f_starts, model=model, windows=windows,
                             method="topk", rule=rule, freq_step_hz=-2.79, fch1_mhz=8000.0,
                             probe_seed=probe_seed, batch_size=64, device="cpu", log=lambda _: None)


def test_window_grid_and_preprocessing(cadence, windows):
    assert len(windows.f_starts) == (NCHANS - FCHANS) // STRIDE + 1
    np.testing.assert_array_equal(windows.window(4096),
                                  preprocess_window(cadence, 4096, FCHANS, TCHANS, PREPROC))
    assert windows.window(0).shape == (TCHANS, FCHANS)


def test_scratch_files_removed_on_close(cadence, tmp_path):
    w = CadenceWindows(cadence, tmp_path / "s", "cad07", fchans=FCHANS, tchans=TCHANS,
                       stride=STRIDE, preproc=PREPROC)
    assert len(list((tmp_path / "s").iterdir())) == N_OBS
    w.close()
    assert list((tmp_path / "s").iterdir()) == []


def test_pool_scoring_matches_in_process_scoring(windows):
    model = StubUDMA()
    pooled = score_windows(model, windows, ["topk", "max"], batch_size=16, num_workers=2,
                           device="cpu", log=lambda _: None)
    for method in ("topk", "max"):
        direct = [float(model.anomaly_score(torch.from_numpy(windows.window(int(f)))[None, None],
                                            method=method)[0]) for f in windows.f_starts]
        np.testing.assert_allclose(pooled[method], direct, rtol=1e-6)
        assert pooled[method].dtype == np.float64


def test_on_only_line_is_short_listed_and_persistent_line_leaks(windows):
    result = _search(windows, SelectionRule(far_quantile=0.95))
    clusters = result.clusters
    assert list(clusters.columns) == CSV_COLUMNS
    assert len(result.amaps) == len(clusters) and all(a.shape == (6, 64) for a in result.amaps)

    def cluster_at(chan):
        hit = clusters[(clusters.f_start_first <= chan) & (clusters.f_start_last + FCHANS > chan)]
        assert len(hit) == 1
        return hit.iloc[0]

    on_only, persistent = cluster_at(ON_ONLY_CHAN), cluster_at(PERSISTENT_CHAN)
    assert on_only.in_short_list and not on_only.off_leak and on_only.n_on_hits_full == 3
    assert persistent.off_leak and not persistent.in_short_list
    assert on_only.on_off_contrast > persistent.on_off_contrast
    # Signed foff: frequency decreases with channel.
    assert on_only.freq_mhz_peak > persistent.freq_mhz_peak
    assert result.thresholds["off_ceiling"] >= result.thresholds["thresh_5"]


def test_probe_is_deterministic_given_seed(windows):
    rule = SelectionRule(far_quantile=0.95, off_ceiling_probe=20)
    a = _search(windows, rule, probe_seed=3).thresholds["off_ceiling"]
    b = _search(windows, rule, probe_seed=3).thresholds["off_ceiling"]
    assert a == b


def test_export_ranking_short_list_then_all(windows):
    result = _search(windows, SelectionRule(far_quantile=0.95))
    ranked, amaps = rank_for_export(result, log=lambda _: None)
    assert ranked["in_short_list"].all() and len(amaps) == len(ranked)
    assert list(ranked["on_off_contrast"]) == sorted(ranked["on_off_contrast"], reverse=True)

    everything, _ = rank_for_export(result, ignore_short_list=True, log=lambda _: None)
    assert len(everything) == len(result.clusters)
    assert list(everything["on_off_contrast"]) == sorted(everything["on_off_contrast"], reverse=True)


def test_enable_determinism_sets_fp32_deterministic_kernels():
    before = (torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic,
              torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32)
    try:
        settings = enable_determinism()
        assert settings["cudnn_allow_tf32"] is False and settings["matmul_allow_tf32"] is False
        assert settings["cudnn_deterministic"] is True and settings["deterministic_algorithms"] is True
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic = before[:2]
        torch.use_deterministic_algorithms(before[2])
        torch.backends.cuda.matmul.allow_tf32 = before[3]
