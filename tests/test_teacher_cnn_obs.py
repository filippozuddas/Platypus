"""Distilled CNN teacher confined to one observation (``cnn_obs``, phase 06a):
the per-observation TeacherCNN, its checkpoint round trip, the distillation
loop, and the UDMA model built from it.

CPU, small widths and filters, random weights; nothing is downloaded.
"""

import math

import pytest
import torch
import yaml
from torch import nn

from src.models.observation_fold import PerObservation, fold_observations, unfold_rows
from src.models.udma import TeacherCNN, _load_teacher_cnn, build_udma
from src.training.distill import distill_teacher

N_OBS, OBS_PX = 6, 16
H, W = N_OBS * OBS_PX, 128
FILTERS = [8, 16, 32, 64]
LATENT = 16
INPUT_SHAPE = (H, W, 1)


def _frame(batch=2, seed=0):
    return torch.randn(batch, 1, H, W, generator=torch.Generator().manual_seed(seed))


def _perturb_obs(x, sample, obs, seed=1):
    out = x.clone()
    g = torch.Generator().manual_seed(seed)
    out[sample, :, obs * OBS_PX:(obs + 1) * OBS_PX] += 3.0 * torch.randn(1, OBS_PX, W, generator=g)
    return out


def _changed_rows(a, b):
    return (a - b).abs().amax(dim=(1, 3)) > 0


def _teacher(per_observation=True):
    torch.manual_seed(0)
    return TeacherCNN(INPUT_SHAPE, FILTERS, LATENT, per_observation=per_observation)


# ---- TeacherCNN ----

@pytest.mark.parametrize("obs", range(N_OBS))
def test_per_observation_teacher_moves_only_the_row_of_the_perturbed_observation(obs):
    t = _teacher()
    assert t.grid_size == (N_OBS, W // 16)
    x = _frame()
    with torch.no_grad():
        base, moved = t._raw_tokens(x), t._raw_tokens(_perturb_obs(x, sample=1, obs=obs))
    assert base.shape == (2, LATENT, N_OBS, W // 16)
    expected = torch.zeros(2, N_OBS, dtype=torch.bool)
    expected[1, obs] = True
    assert torch.equal(_changed_rows(base, moved), expected)


def test_whole_frame_teacher_leaks_across_observations():
    """The July 2026 teacher: perturbing one observation moves its neighbours."""
    t = _teacher(per_observation=False)
    x = _frame(batch=1)
    with torch.no_grad():
        changed = _changed_rows(t._raw_tokens(x), t._raw_tokens(_perturb_obs(x, 0, 2)))[0]
    assert changed[1] and changed[2] and changed[3]


def test_per_observation_teacher_equals_its_trunk_on_each_observation():
    t = _teacher()
    x = _frame()
    with torch.no_grad():
        per_obs = t.trunk(fold_observations(x, N_OBS))
        torch.testing.assert_close(t._raw_tokens(x), unfold_rows(per_obs, N_OBS))
        # observation k of sample b is batch entry b * N_OBS + k of the folded trunk
        torch.testing.assert_close(t._raw_tokens(x)[1:2, :, 4:5],
                                   t.trunk(x[1:2, :, 4 * OBS_PX:5 * OBS_PX]))


def test_per_observation_teacher_rejects_heights_that_do_not_split():
    with pytest.raises(ValueError):
        TeacherCNN((90, W, 1), FILTERS, LATENT, per_observation=True)  # 90 / 6 = 15 rows
    with pytest.raises(ValueError):
        TeacherCNN((48, W, 1), FILTERS, LATENT, per_observation=True)  # 8-row obs, 16x blocks


def test_teacher_stays_frozen_and_in_eval_mode():
    t = _teacher()
    t.train()
    assert not t.training and not t.trunk.training
    assert not any(p.requires_grad for p in t.parameters())


# ---- checkpoint round trip ----

def _save_ckpt(path, trunk, per_observation, **extra):
    ckpt = {"trunk_state_dict": trunk.state_dict(), "filters": FILTERS, "latent_dim": LATENT,
            "kernel_size": (3, 3), "activation": "relu", "use_batchnorm": True,
            "convs_per_block": 2, **extra}
    if per_observation is not None:
        ckpt.update(per_observation=per_observation, n_obs=N_OBS)
    torch.save(ckpt, path)


def test_loader_rebuilds_a_per_observation_teacher_from_its_checkpoint(tmp_path):
    t = _teacher()
    path = tmp_path / "t.pt"
    _save_ckpt(path, t.trunk, per_observation=True)
    loaded = _load_teacher_cnn({"checkpoint": str(path)}, INPUT_SHAPE)
    assert loaded.per_observation and loaded.grid_size == t.grid_size
    x = _frame()
    with torch.no_grad():
        torch.testing.assert_close(loaded._raw_tokens(x), t._raw_tokens(x), rtol=0, atol=0)


def test_loader_reads_old_checkpoints_as_whole_frame(tmp_path):
    t = _teacher(per_observation=False)
    path = tmp_path / "old.pt"
    _save_ckpt(path, t.trunk, per_observation=None)  # saved before 2026-11: no key
    loaded = _load_teacher_cnn({"checkpoint": str(path)}, INPUT_SHAPE)
    assert not loaded.per_observation
    x = _frame()
    with torch.no_grad():
        torch.testing.assert_close(loaded._raw_tokens(x), t._raw_tokens(x), rtol=0, atol=0)


# ---- distillation loop ----

class _ConfinedP(nn.Module):
    """Stand-in for ResNetTeacher(confine="obs"): a fixed strided conv applied to
    each observation, on the same (6, W/16) grid."""

    grid_size = (N_OBS, W // 16)
    channels = 12

    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.net = PerObservation(nn.Sequential(nn.Conv2d(1, 12, 16, stride=16), nn.Tanh()), N_OBS)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.net(x)


def _loader(n_batches=6, batch=4):
    return [_frame(batch=batch, seed=10 + i) for i in range(n_batches)]


def test_distill_per_observation_returns_a_strip_trunk_that_learns():
    torch.manual_seed(0)
    P = _ConfinedP()
    T, losses = distill_teacher(P, _loader(), INPUT_SHAPE, "cpu", filters=FILTERS,
                                latent_dim=LATENT, p_channels=P.channels, epochs=3, lr=3e-3,
                                log_every_n_steps=1000, log_fn=lambda _: None,
                                per_observation=True)
    assert len(losses) == 18 and all(math.isfinite(v) for v in losses)
    assert sum(losses[-6:]) < sum(losses[:6])
    # the returned trunk is built for one observation
    with torch.no_grad():
        assert T.eval()(torch.zeros(1, 1, OBS_PX, W)).shape == (1, LATENT, 1, W // 16)


def test_distilled_per_observation_trunk_loads_into_a_confined_teacher(tmp_path):
    torch.manual_seed(0)
    P = _ConfinedP()
    T, _ = distill_teacher(P, _loader(n_batches=2), INPUT_SHAPE, "cpu", filters=FILTERS,
                           latent_dim=LATENT, p_channels=P.channels, epochs=1,
                           log_every_n_steps=1000, log_fn=lambda _: None, per_observation=True)
    path = tmp_path / "cnn_obs.pt"
    _save_ckpt(path, T, per_observation=True)
    teacher = _load_teacher_cnn({"checkpoint": str(path)}, INPUT_SHAPE)
    x = _frame(batch=1)
    with torch.no_grad():
        changed = _changed_rows(teacher._raw_tokens(x), teacher._raw_tokens(_perturb_obs(x, 0, 3)))
    assert changed[0].tolist() == [False, False, False, True, False, False]


def test_distill_rejects_a_height_that_does_not_split_into_observations():
    class P90(_ConfinedP):
        grid_size = (90 // 16, W // 16)

    with pytest.raises(ValueError):
        distill_teacher(P90(), [], (90, W, 1), "cpu", filters=FILTERS, latent_dim=LATENT,
                        p_channels=12, per_observation=True)


def test_distill_whole_frame_path_is_unchanged():
    torch.manual_seed(0)
    P = _ConfinedP()
    T, losses = distill_teacher(P, _loader(n_batches=2), INPUT_SHAPE, "cpu", filters=FILTERS,
                                latent_dim=LATENT, p_channels=P.channels, epochs=1,
                                log_every_n_steps=1000, log_fn=lambda _: None)
    assert len(losses) == 2
    with torch.no_grad():
        assert T.eval()(torch.zeros(1, 1, H, W)).shape == (1, LATENT, N_OBS, W // 16)


# ---- the UDMA model built on it ----

def test_build_udma_with_a_per_observation_teacher(tmp_path):
    t = _teacher()
    path = tmp_path / "cnn_obs.pt"
    _save_ckpt(path, t.trunk, per_observation=True)
    cfg = {
        "architecture": "udma",
        "teacher": {"type": "cnn_distilled", "checkpoint": str(path)},
        "student": {"encoder": {"filters": FILTERS}, "bottleneck": {"latent_dim": 8},
                    "head_hidden": 16, "mem_slots": 20},
        "scoring": {"topk_frac": 0.01},
    }
    model = build_udma(INPUT_SHAPE, cfg).eval()
    assert model.teacher.per_observation and model.teacher.grid_size == (N_OBS, W // 16)
    x = _frame()
    loss, parts = model.compute_loss(x)
    assert torch.isfinite(loss) and set(parts) >= {"st1", "st2", "ss", "st_sum"}
    assert model.anomaly_map(x).shape == (2, N_OBS, W // 16)


def test_cnn_obs_config_differs_from_the_reference_only_in_the_teacher():
    from scripts.debug.teacher_row_leakage_test import ROOT

    ref = yaml.safe_load((ROOT / "configs/model/udma_nightsplit.yaml").read_text())
    new = yaml.safe_load((ROOT / "configs/model/udma_cnn_obs.yaml").read_text())
    assert set(ref) == set(new)
    assert [k for k in ref if ref[k] != new[k]] == ["teacher"]
    assert new["teacher"]["type"] == "cnn_distilled"

    from src.utils.config import load_config

    a = load_config(ROOT / "configs/training/udma_gbt_fine_nightsplit.yaml")
    b = load_config(ROOT / "configs/training/udma_gbt_fine_cnn_obs.yaml")
    assert [k for k in a if a[k] != b[k]] == ["model"]
