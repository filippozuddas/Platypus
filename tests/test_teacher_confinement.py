"""Confinement of a teacher to one observation: the fold, the confined ResNet
target, the geometry the row-leakage probe relies on, and the ON-only gate.

Everything runs on CPU with random weights. Which feature rows can move when a
pixel changes depends on kernels, strides and padding, not on the weights, so
nothing is downloaded.
"""

import numpy as np
import pytest
import torch

from scripts.debug.resnet_teacher import ResNetTeacher
from scripts.debug.teacher_row_leakage_test import (
    N_OBS, ON_OBS, STAGES, ObsConfined, ResNetStages, expected_clean_off_rows, leak_criterion,
    on_off_response, row_displacement, row_on_off_masks, stage_displacement,
)
from scripts.debug.teacher_sensitivity_test import (
    AFFECTED_THR, UNAFFECTED_THR, on_observation_tokens, token_groups,
)
from src.models.encoder import build_encoder
from src.models.observation_fold import fold_observations, unfold_rows

H, OBS_PX, W = 96, 16, 256


def _frame(batch: int = 2, width: int = W, seed: int = 0) -> torch.Tensor:
    return torch.randn(batch, 1, H, width, generator=torch.Generator().manual_seed(seed))


def _perturb_obs(x: torch.Tensor, sample: int, obs: int, seed: int = 1) -> torch.Tensor:
    out = x.clone()
    g = torch.Generator().manual_seed(seed)
    out[sample, :, obs * OBS_PX:(obs + 1) * OBS_PX] += 3.0 * torch.randn(
        1, OBS_PX, x.shape[3], generator=g)
    return out


def _changed_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, nh) bool: grid rows where any channel or column differs."""
    return (a - b).abs().amax(dim=(1, 3)) > 0


# ---- fold / unfold ----

def test_fold_puts_observation_k_of_sample_b_at_index_b_times_n_plus_k():
    x = _frame(batch=3)
    folded = fold_observations(x, N_OBS)
    assert folded.shape == (3 * N_OBS, 1, OBS_PX, W)
    for b in range(3):
        for k in range(N_OBS):
            torch.testing.assert_close(folded[b * N_OBS + k],
                                       x[b, :, k * OBS_PX:(k + 1) * OBS_PX], rtol=0, atol=0)


def test_unfold_rows_inverts_fold():
    x = _frame(batch=3)
    torch.testing.assert_close(unfold_rows(fold_observations(x, N_OBS), N_OBS), x, rtol=0, atol=0)


def test_fold_rejects_a_height_that_is_not_a_multiple_of_the_observations():
    with pytest.raises(ValueError):
        fold_observations(torch.zeros(1, 1, 95, 32), N_OBS)
    with pytest.raises(ValueError):
        unfold_rows(torch.zeros(5, 1, 1, 4), N_OBS)


# ---- confined targets: observation k moves grid row k and nothing else ----

@pytest.fixture(scope="module")
def confined_models():
    torch.manual_seed(0)
    trunk = build_encoder((OBS_PX, W, 1), [32, 64, 128, 256], 128).eval()

    def folded_trunk(x):
        with torch.no_grad():
            return unfold_rows(trunk(fold_observations(x, N_OBS)), N_OBS)

    return {
        "resnet_obs": ResNetTeacher(confine="obs", pretrained=False),
        "resnet_tile": ResNetTeacher(confine="tile", pretrained=False),
        "trunk_obs": folded_trunk,
    }


@pytest.mark.parametrize("name", ["resnet_obs", "resnet_tile", "trunk_obs"])
@pytest.mark.parametrize("obs", range(N_OBS))
def test_confined_target_moves_only_the_row_of_the_perturbed_observation(confined_models, name,
                                                                         obs):
    model = confined_models[name]
    x = _frame(batch=2)
    base, moved = model(x), model(_perturb_obs(x, sample=1, obs=obs))
    assert base.shape[-2:] == (N_OBS, W // 16)
    expected = torch.zeros(2, N_OBS, dtype=torch.bool)
    expected[1, obs] = True
    assert torch.equal(_changed_rows(base, moved), expected)


def test_unconfined_resnet_target_leaks_into_the_neighbouring_rows():
    """The July 2026 target: an ON observation moves OFF rows as well."""
    model = ResNetTeacher(confine="none", pretrained=False)
    x = _frame(batch=1)
    changed = _changed_rows(model(x), model(_perturb_obs(x, sample=0, obs=2)))[0]
    assert changed[2] and changed[1] and changed[3]


def test_tile_token_depends_only_on_its_own_crop():
    """Tile of cell c spans channels [16c - 24, 16c + 40): changing the pixels
    of cell c0 can move tokens c0-2 .. c0+2 of the same observation only."""
    model = ResNetTeacher(confine="tile", pretrained=False)
    x = _frame(batch=1)
    obs, c0 = 3, 7
    xp = x.clone()
    xp[0, :, obs * OBS_PX:(obs + 1) * OBS_PX, 16 * c0:16 * c0 + 16] += 3.0
    changed = ((model(x) - model(xp)).abs().amax(dim=1) > 0)[0]  # (nh, nw)
    expected = torch.zeros_like(changed)
    expected[obs, c0 - 2:c0 + 3] = True
    assert torch.equal(changed, expected)


def test_resnet_teacher_rejects_unknown_confinement_and_odd_tiles():
    with pytest.raises(ValueError):
        ResNetTeacher(confine="row", pretrained=False)
    with pytest.raises(ValueError):
        ResNetTeacher(confine="tile", tile_width=40, pretrained=False)


# ---- geometry behind the row-leakage probe ----

def test_expected_clean_off_rows_of_the_stem_are_the_four_interior_ones():
    assert expected_clean_off_rows("stem").tolist() == [6, 14, 22, 23]
    for stage in ("layer1", "layer2", "layer3"):
        assert expected_clean_off_rows(stage).tolist() == []
    assert expected_clean_off_rows("layer3", confine="obs").tolist() == [1, 3, 5]


@pytest.mark.parametrize("stage", STAGES)
def test_off_rows_that_move_are_exactly_those_whose_field_holds_on_pixels(stage):
    """ON-only perturbation through the real ResNet stages (random weights):
    the OFF rows left untouched are the ones the geometry calls clean."""
    torch.manual_seed(0)
    model = ResNetStages(pretrained=False)
    x = _frame(batch=1)
    xp = x.clone()
    for obs in ON_OBS:
        xp = _perturb_obs(xp, sample=0, obs=obs, seed=10 + obs)
    changed = _changed_rows(model.features(x, stage), model.features(xp, stage))[0].numpy()
    on_mask, off_mask = row_on_off_masks(len(changed))
    assert changed[on_mask].all()
    assert np.flatnonzero(off_mask & ~changed).tolist() == expected_clean_off_rows(stage).tolist()


@pytest.mark.parametrize("stage", STAGES)
def test_obs_confined_resnet_moves_no_off_row(stage):
    torch.manual_seed(0)
    model = ObsConfined(ResNetStages(pretrained=False))
    x = _frame(batch=1)
    xp = x.clone()
    for obs in ON_OBS:
        xp = _perturb_obs(xp, sample=0, obs=obs, seed=10 + obs)
    changed = _changed_rows(model.features(x, stage), model.features(xp, stage))[0].numpy()
    on_mask, off_mask = row_on_off_masks(len(changed))
    assert changed[on_mask].all() and not changed[off_mask].any()
    assert np.flatnonzero(off_mask).tolist() == expected_clean_off_rows(stage, "obs").tolist()


# ---- per-row profile and the instrument check ----

class _FakeStages:
    """features(x, stage) = the input itself as a 1-channel (24-row) grid."""

    def features(self, x, stage):
        return x[:, :, ::4, :]


def test_row_profile_averages_back_to_the_on_off_response():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(5, H, 32)).astype(np.float32)
    inj = clean + rng.normal(size=(5, H, 32)).astype(np.float32)
    model = _FakeStages()
    rows = row_displacement(model, clean, inj, "any", "cpu", batch=2)
    assert rows.shape == (5, 24)
    on_mask, off_mask = row_on_off_masks(24)
    on_resp, off_resp = on_off_response(rows)
    np.testing.assert_array_equal(on_resp, rows[:, on_mask].mean(axis=1))
    np.testing.assert_array_equal(off_resp, rows[:, off_mask].mean(axis=1))
    legacy_on, legacy_off = stage_displacement(model, clean, inj, "any", "cpu", batch=2)
    np.testing.assert_array_equal(legacy_on, on_resp)
    np.testing.assert_array_equal(legacy_off, off_resp)


def test_row_masks_follow_the_observation_of_each_row():
    on_mask, off_mask = row_on_off_masks(24)
    assert on_mask.tolist() == [True] * 4 + [False] * 4 + [True] * 4 + [False] * 4 + \
        [True] * 4 + [False] * 4
    assert (on_mask ^ off_mask).all()


def test_leak_criterion_passes_only_when_the_clean_rows_stay_at_zero():
    clean = expected_clean_off_rows("stem")
    on_mask, off_mask = row_on_off_masks(24)
    profile = np.where(on_mask, 10.0, 4.0)
    profile[clean] = 0.0
    ok = leak_criterion(profile, clean, zero_tol=1e-3)
    assert ok["passed"] and ok["max_clean_frac"] == 0.0
    assert ok["n_clean"] == 4 and ok["n_dirty"] == 8
    assert ok["off_share_dirty"] == pytest.approx(1.0)

    profile[clean[0]] = 0.5  # 5% of the ON mean in a row that sees no ON pixel
    bad = leak_criterion(profile, clean, zero_tol=1e-3)
    assert not bad["passed"] and bad["max_clean_frac"] == pytest.approx(0.05)
    assert bad["off_share_dirty"] < 1.0

    # numerical noise below the tolerance is still zero
    profile[clean[0]] = 10.0 * 5e-4
    assert leak_criterion(profile, clean, zero_tol=1e-3)["passed"]


def test_leak_criterion_has_nothing_to_pass_without_clean_rows():
    profile = np.ones(6)
    assert not leak_criterion(profile, expected_clean_off_rows("layer3"), 1e-3)["passed"]


# ---- G2-ON: negatives come from the ON observations only ----

def test_on_observation_tokens_marks_rows_0_2_4_row_major():
    on_tok = on_observation_tokens(6, 4)
    assert on_tok.reshape(6, 4).all(axis=1).tolist() == [True, False, True, False, True, False]
    assert on_observation_tokens(24, 2).reshape(24, 2)[:, 0].tolist() == \
        row_on_off_masks(24)[0].tolist()


def test_token_groups_keep_only_on_observation_negatives_for_the_gate():
    nh, nw = 6, 4
    on_tok = on_observation_tokens(nh, nw)
    pmax = np.zeros((2, nh * nw))
    pmax[0, 0] = AFFECTED_THR + 0.1        # line through an ON token
    pmax[0, 1] = (AFFECTED_THR + UNAFFECTED_THR) / 2   # in between: excluded
    pmax[1, 9] = AFFECTED_THR + 1.0        # row 2, an ON observation
    affected, unaffected, unaffected_on = token_groups(pmax, on_tok)
    assert affected.sum() == 2 and affected[0, 0] and affected[1, 9]
    assert not unaffected[0, 1] and not affected[0, 1]
    # every OFF-observation token is unaffected, and none of them is a gate negative
    assert unaffected[:, ~on_tok].all()
    assert not unaffected_on[:, ~on_tok].any()
    np.testing.assert_array_equal(unaffected_on[:, on_tok], unaffected[:, on_tok])
    assert unaffected_on.sum() == unaffected.sum() - 2 * (~on_tok).sum()


# ---- the probe's premise: an ON-only injection leaves OFF pixels bit-identical ----

def test_off_pixels_are_bit_identical_after_injection_and_preprocessing():
    pytest.importorskip("setigen")
    import yaml

    from scripts.debug.injection_vs_rfi_test import inject_narrowband_on_only, preprocess_raw
    from scripts.debug.teacher_row_leakage_test import ROOT

    with open(ROOT / "configs/data/gbt_fine.yaml") as f:
        preproc = yaml.safe_load(f)["preprocessing"]
    rng = np.random.default_rng(0)
    # float32, like the cache written by scripts/preprocess_cache.py
    raw = (rng.chisquare(4, size=(3, N_OBS, OBS_PX, 1024)) * 1e5).astype(np.float32)
    for i, snippet in enumerate(raw):
        clean = preprocess_raw(snippet, preproc).reshape(N_OBS, OBS_PX, 1024)
        inj = preprocess_raw(inject_narrowband_on_only(snippet, snr=40, drift_rate=0.3,
                                                       seed=42 + i), preproc)
        inj = inj.reshape(N_OBS, OBS_PX, 1024)
        for obs in range(N_OBS):
            if obs in ON_OBS:
                assert np.abs(inj[obs] - clean[obs]).max() > 1.0
            else:
                np.testing.assert_array_equal(inj[obs], clean[obs])
