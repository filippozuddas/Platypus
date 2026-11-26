"""UDMA students confined to one observation (``student.per_observation``,
phase 06a, 2026-11-26): with a confined teacher too, no row of the anomaly map
may move when only ON pixels change.

CPU, small widths, random weights; nothing is downloaded. Every check on
confinement runs in eval mode: in train mode BatchNorm couples the strips
through the batch statistics, by design.
"""

import sys

import numpy as np
import pytest
import torch
import yaml

from scripts.debug import teacher_row_leakage_test as leak
from scripts.debug.teacher_row_leakage_test import (
    MAP_STAGES, UDMA_MAP_STAGES, UDMAMapStages, row_displacement,
)
from src.models.observation_fold import fold_observations
from src.models.udma import FeatureStudent, TeacherCNN, build_udma

N_OBS, OBS_PX = 6, 16
FILTERS = [8, 16, 32, 64]
LATENT = 16
ON, OFF = [0, 2, 4], [1, 3, 5]


def _cfg(tmp_path, width, teacher_obs=True, students_obs=True):
    torch.manual_seed(0)
    shape = (N_OBS * OBS_PX, width, 1)
    teacher = TeacherCNN(shape, FILTERS, LATENT, per_observation=teacher_obs)
    path = tmp_path / f"teacher_{width}_{teacher_obs}.pt"
    torch.save({"trunk_state_dict": teacher.trunk.state_dict(), "filters": FILTERS,
                "latent_dim": LATENT, "kernel_size": (3, 3), "activation": "relu",
                "use_batchnorm": True, "convs_per_block": 2,
                "per_observation": teacher_obs, "n_obs": N_OBS}, path)
    student = {"encoder": {"filters": FILTERS}, "bottleneck": {"latent_dim": 8},
               "head_hidden": 16, "mem_slots": 20}
    if students_obs:
        student["per_observation"] = True
    cfg = {
        "architecture": "udma",
        "teacher": {"type": "cnn_distilled", "checkpoint": str(path)},
        "student": student,
        "scoring": {"topk_frac": 0.01},
    }
    return shape, cfg


def _udma(tmp_path, width=128, **kw):
    shape, cfg = _cfg(tmp_path, width, **kw)
    torch.manual_seed(1)
    return build_udma(shape, cfg).eval(), cfg


def _frames(n, width, seed=0):
    g = np.random.default_rng(seed)
    return g.standard_normal((n, N_OBS * OBS_PX, width)).astype(np.float32)


def _on_only(frames):
    """A bright line in the ON observations only; OFF rows bit-identical."""
    out = frames.copy()
    for k in ON:
        out[:, k * OBS_PX:(k + 1) * OBS_PX, 40:42] += 25.0
    return out


# ---- the student on its own ----

@pytest.mark.parametrize("memory", [False, True])
def test_grid_row_k_is_the_student_applied_to_observation_k_alone(memory):
    torch.manual_seed(0)
    st = FeatureStudent((N_OBS * OBS_PX, 128, 1), FILTERS, 8, out_channels=LATENT,
                        memory=memory, mem_slots=20, head_hidden=16,
                        per_observation=True).eval()
    x = torch.randn(2, 1, N_OBS * OBS_PX, 128)
    with torch.no_grad():
        out = st(x)
        out, att = out if memory else (out, None)
        assert out.shape == (2, LATENT, N_OBS, 128 // 16)
        for b in range(2):
            for k in range(N_OBS):
                strip = x[b:b + 1, :, k * OBS_PX:(k + 1) * OBS_PX]
                z = st.trunk(strip)
                if memory:
                    z, _ = st.memory(z)
                torch.testing.assert_close(out[b:b + 1, :, k:k + 1], st.head(z),
                                           rtol=1e-5, atol=1e-6)
    if memory:
        # same (B * nh * nw, slots) layout as a whole-frame student
        assert att.shape == (2 * N_OBS * (128 // 16), 20)


def test_memory_attention_keeps_the_whole_frame_row_order():
    torch.manual_seed(0)
    st = FeatureStudent((N_OBS * OBS_PX, 128, 1), FILTERS, 8, out_channels=LATENT,
                        memory=True, mem_slots=20, head_hidden=16,
                        per_observation=True).eval()
    x = torch.randn(2, 1, N_OBS * OBS_PX, 128)
    with torch.no_grad():
        _, att = st(x)
        _, att_folded = st.memory(st.trunk(fold_observations(x, N_OBS)))
    # row (b, k, w) of the unfolded grid is row (b * n_obs + k, 0, w) of the fold
    torch.testing.assert_close(att, att_folded, rtol=0, atol=0)
    assert att.reshape(2, N_OBS, 128 // 16, 20).shape[1] == N_OBS


def test_bad_strip_height_is_refused():
    with pytest.raises(ValueError, match="per_observation student"):
        FeatureStudent((N_OBS * 8, 128, 1), FILTERS, 8, out_channels=LATENT,
                       per_observation=True)


def test_whole_frame_student_is_unchanged():
    torch.manual_seed(0)
    a = FeatureStudent((N_OBS * OBS_PX, 128, 1), FILTERS, 8, out_channels=LATENT,
                       head_hidden=16)
    assert not a.per_observation
    assert "per_observation_n_obs" not in a.state_dict()
    b = FeatureStudent((N_OBS * OBS_PX, 128, 1), FILTERS, 8, out_channels=LATENT,
                       head_hidden=16, per_observation=True)
    assert set(b.state_dict()) - set(a.state_dict()) == {"per_observation_n_obs"}


# ---- the whole model ----

def test_no_stage_moves_an_off_row_when_only_on_pixels_change(tmp_path):
    model, _ = _udma(tmp_path)
    stages = UDMAMapStages(model)
    assert stages.teacher_confined and stages.students_confined
    assert stages.confined_stages == list(UDMA_MAP_STAGES)
    clean = _frames(4, 128)
    inj = _on_only(clean)
    for s in UDMA_MAP_STAGES:
        rows = row_displacement(stages, clean, inj, s, "cpu")
        assert rows[:, OFF].max() == 0.0, s
        assert rows[:, ON].min() > 0.0, s


def test_students_per_obs_with_a_whole_frame_teacher_confines_only_the_students(tmp_path):
    model, _ = _udma(tmp_path, teacher_obs=False)
    stages = UDMAMapStages(model)
    assert not stages.teacher_confined and stages.students_confined
    assert stages.confined_stages == ["student_ae", "student_mem"]


def test_compute_loss_trains_both_students(tmp_path):
    model, _ = _udma(tmp_path)
    model.train()
    x = torch.from_numpy(_frames(3, 128)).unsqueeze(1)
    total, parts = model.compute_loss(x)
    total.backward()
    assert torch.isfinite(total)
    assert set(parts) == {"st1", "st2", "ss", "entropy", "st_sum"}
    for st in (model.student_ae, model.student_mem):
        grads = [p.grad for p in st.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)
    assert all(p.grad is None for p in model.teacher.parameters())


def test_checkpoint_and_config_must_agree_on_the_students_scope(tmp_path):
    obs, _ = _udma(tmp_path, students_obs=True)
    whole, _ = _udma(tmp_path, students_obs=False)
    with pytest.raises(RuntimeError, match="per_observation_n_obs"):
        whole.load_state_dict(obs.state_dict())
    with pytest.raises(RuntimeError, match="per_observation_n_obs"):
        obs.load_state_dict(whole.state_dict())
    whole.load_state_dict(whole.state_dict())
    obs.load_state_dict(obs.state_dict())


def test_n_obs_comes_from_the_confined_teacher(tmp_path):
    model, _ = _udma(tmp_path)
    assert model.student_ae.n_obs == model.teacher.n_obs == N_OBS
    assert int(model.student_mem.per_observation_n_obs) == N_OBS


def test_config_differs_from_cnn_obs_only_in_the_students_scope():
    root = leak.ROOT
    a = yaml.safe_load((root / "configs/model/udma_cnn_obs.yaml").read_text())
    b = yaml.safe_load((root / "configs/model/udma_cnn_obs_students_obs.yaml").read_text())
    assert set(a) == set(b)
    assert [k for k in a if a[k] != b[k]] == ["student"]
    assert {k: v for k, v in b["student"].items() if k != "per_observation"} == a["student"]
    assert b["student"]["per_observation"] is True

    from src.utils.config import load_config

    ta = load_config(root / "configs/training/udma_gbt_fine_cnn_obs.yaml")
    tb = load_config(root / "configs/training/udma_gbt_fine_cnn_obs_students_obs.yaml")
    assert [k for k in ta if ta[k] != tb[k]] == ["model"]


# ---- the leak probe on a fully confined model ----

def test_map_leak_probe_reports_every_stage_confined_and_no_off_row_above_background(
        tmp_path, monkeypatch, capsys):
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
    assert "students True" in text
    for s in UDMA_MAP_STAGES:
        assert f"  {s}: max OFF response" in text
    assert "[FAIL]" not in text
    saved = np.load(out_dir / "teacher_row_leakage_udma_map_run.npz")
    assert bool(saved["teacher_confined"]) and bool(saved["students_confined"])
    assert saved["instrument_check"].all()
    # the OFF rows of every stage are exactly where the clean frames left them
    # (the background share is not asserted: with 3 frames the top clean OFF
    # row sits above its own q99, injection or not)
    for s in UDMA_MAP_STAGES:
        assert saved[f"row_profile_{s}"][0][OFF].max() == 0.0, s
    for s in MAP_STAGES:
        assert saved[f"map_row_level_{s}"].shape == (1, N_OBS)
