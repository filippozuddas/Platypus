"""Map-level ON->OFF leakage probe (``teacher_row_leakage_test.py --architecture
udma_map``, phase 06a, 2026-11-23): the stage reader over a whole UDMA model,
the background level the short-list rule would see, and one end-to-end run.

CPU, small widths, random weights; nothing is downloaded.
"""

import sys

import numpy as np
import pytest
import torch
import yaml

from scripts.debug import teacher_row_leakage_test as leak
from scripts.debug.teacher_row_leakage_test import (
    MAP_STAGES, UDMA_MAP_STAGES, UDMAMapStages, map_background, off_rows_above_background,
    row_displacement, row_level,
)
from src.models.udma import TeacherCNN, build_udma

N_OBS, OBS_PX = 6, 16
FILTERS = [8, 16, 32, 64]
LATENT = 16


def _udma(tmp_path, width, per_observation=True):
    torch.manual_seed(0)
    shape = (N_OBS * OBS_PX, width, 1)
    teacher = TeacherCNN(shape, FILTERS, LATENT, per_observation=per_observation)
    path = tmp_path / f"teacher_{width}_{per_observation}.pt"
    torch.save({"trunk_state_dict": teacher.trunk.state_dict(), "filters": FILTERS,
                "latent_dim": LATENT, "kernel_size": (3, 3), "activation": "relu",
                "use_batchnorm": True, "convs_per_block": 2,
                "per_observation": per_observation, "n_obs": N_OBS}, path)
    cfg = {
        "architecture": "udma",
        "teacher": {"type": "cnn_distilled", "checkpoint": str(path)},
        "student": {"encoder": {"filters": FILTERS}, "bottleneck": {"latent_dim": 8},
                    "head_hidden": 16, "mem_slots": 20},
        "scoring": {"topk_frac": 0.01},
    }
    return build_udma(shape, cfg).eval(), cfg


def _frames(n, width, seed=0):
    g = np.random.default_rng(seed)
    return g.standard_normal((n, N_OBS * OBS_PX, width)).astype(np.float32)


def _on_only(frames, seed=1):
    """Add a bright line to the ON observations only; OFF rows bit-identical."""
    out = frames.copy()
    col = 40
    for k in (0, 2, 4):
        out[:, k * OBS_PX:(k + 1) * OBS_PX, col:col + 2] += 25.0
    return out


def test_map_stages_are_the_components_of_the_anomaly_map(tmp_path):
    model, _ = _udma(tmp_path, width=128)
    stages = UDMAMapStages(model)
    x = torch.from_numpy(_frames(2, 128)).unsqueeze(1)
    comps = model.anomaly_map_components(x)
    for s in MAP_STAGES:
        got = stages.features(x, s)
        assert got.shape == (2, 1, N_OBS, 128 // 16)
        torch.testing.assert_close(got[:, 0], comps[s], rtol=0, atol=0)
    torch.testing.assert_close(stages.features(x, "cob")[:, 0], model.anomaly_map(x),
                               rtol=0, atol=0)
    assert stages.features(x, "teacher").shape == (2, LATENT, N_OBS, 128 // 16)
    for s in ("student_ae", "student_mem"):
        assert stages.features(x, s).shape == (2, LATENT, N_OBS, 128 // 16)
    with pytest.raises(ValueError):
        stages.features(x, "layer3")


def test_confined_teacher_moves_no_off_row_while_whole_frame_students_do(tmp_path):
    model, _ = _udma(tmp_path, width=128)
    stages = UDMAMapStages(model)
    assert stages.teacher_confined
    clean = _frames(4, 128)
    inj = _on_only(clean)
    off = [1, 3, 5]
    teacher_rows = row_displacement(stages, clean, inj, "teacher", "cpu")
    assert teacher_rows[:, off].max() == 0.0
    assert teacher_rows[:, [0, 2, 4]].min() > 0.0
    # whole-frame students see across observations: their OFF rows move, and
    # with a still teacher that motion reaches st1 on the OFF rows
    assert row_displacement(stages, clean, inj, "student_ae", "cpu")[:, off].max() > 0.0
    assert row_displacement(stages, clean, inj, "st1", "cpu")[:, off].max() > 0.0


def test_whole_frame_teacher_is_reported_as_not_confined(tmp_path):
    model, _ = _udma(tmp_path, width=128, per_observation=False)
    assert not UDMAMapStages(model).teacher_confined


def test_row_level_is_the_row_maximum_of_the_map(tmp_path):
    model, _ = _udma(tmp_path, width=128)
    stages = UDMAMapStages(model)
    frames = _frames(3, 128)
    got = row_level(stages, frames, "cob", "cpu", batch=2)
    with torch.no_grad():
        want = model.anomaly_map(torch.from_numpy(frames).unsqueeze(1)).amax(dim=2).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_background_is_the_quantile_of_the_off_rows_only():
    rows = np.zeros((100, N_OBS))
    rows[:, [0, 2, 4]] = 1000.0                       # ON rows must not count
    rows[:, [1, 3, 5]] = np.arange(300).reshape(100, 3)
    assert map_background(rows, quantile=0.5) == pytest.approx(np.quantile(np.arange(300), 0.5))


def test_off_rows_above_background_counts_frames_with_two_off_rows():
    inj = np.array([
        [10.0, 2.0, 10.0, 2.0, 10.0, 0.5],   # two OFF rows above 1 -> counts
        [10.0, 2.0, 10.0, 0.5, 10.0, 0.5],   # one OFF row above 1
        [10.0, 0.5, 10.0, 0.5, 10.0, 0.5],   # none
        [10.0, 0.5, 10.0, 0.5, 10.0, 0.5],
    ])
    out = off_rows_above_background(inj, background=1.0)
    assert out["share_off_ge2"] == pytest.approx(0.25)
    assert out["share_off_ge1"] == pytest.approx(0.50)
    np.testing.assert_allclose(out["row_level"], inj.mean(axis=0))


def test_udma_map_runs_end_to_end_and_passes_the_confined_teacher_check(tmp_path, monkeypatch,
                                                                       capsys):
    width = 1024  # the probe's injector and INPUT_SHAPE are fixed at 1024 channels
    model, cfg = _udma(tmp_path, width=width)
    ckpt = tmp_path / "run" / "checkpoints" / "epoch=000.ckpt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, ckpt)
    model_cfg = tmp_path / "udma.yaml"
    model_cfg.write_text(yaml.safe_dump(cfg))
    cache = tmp_path / "cache"
    cache.mkdir()
    g = np.random.default_rng(0)
    np.save(cache / "train.npy",
            g.chisquare(4, size=(12, N_OBS, OBS_PX, width)).astype(np.float32))
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "teacher_row_leakage_test.py", "--architecture", "udma_map",
        "--checkpoint", str(ckpt), "--model_config", str(model_cfg),
        "--data_config", str(leak.ROOT / "configs/data/gbt_fine_nightsplit.yaml"),
        "--cache", str(cache), "--n_frames", "3", "--snr_list", "40",
        "--out_dir", str(out_dir), "--device", "cpu",
    ])
    leak.main()
    text = capsys.readouterr().out
    assert "MAP ROWS AGAINST THE BACKGROUND" in text
    assert "teacher: max OFF response" in text and "[PASS]" in text
    saved = np.load(out_dir / "teacher_row_leakage_udma_map_run.npz")
    assert list(saved["stages"]) == list(UDMA_MAP_STAGES)
    assert bool(saved["teacher_confined"])
    assert saved["instrument_check"].all()
    for s in MAP_STAGES:
        assert saved[f"map_row_level_{s}"].shape == (1, N_OBS)
        assert float(saved[f"map_background_{s}"]) > 0
    # the confined teacher's OFF rows stayed at zero in the saved profile
    assert saved["row_profile_teacher"][0][[1, 3, 5]].max() == 0.0
