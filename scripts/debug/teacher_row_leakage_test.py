"""ON->OFF row-leakage probe for a UDMA teacher's feature grid.

Measures, without any training, whether a candidate teacher LOCALIZES an
ON-only signal or bleeds it into the OFF rows. Inject a narrowband line into
obs {0,2,4} of a quiet frame, take the feature displacement
``||P(x+s) - P(x)||`` per (row, col) cell, and compare ON feature-rows against
OFF feature-rows. A localizing teacher keeps the displacement in ON rows (high
ratio); a leaky one spreads it (ratio -> 1).

The measurement is clean because ``inject_narrowband_on_only`` leaves OFF
frames byte-identical and ``preprocess_raw`` normalizes per observation, so any
OFF-row motion is pure receptive-field leakage.

``--architecture`` selects the target under test, each isolating one variable:

    resnet18 (default)  out-of-domain CNN, swept across stem/layer1/layer2/
                        layer3 -- tests the receptive-field hypothesis
    vit_mae             domain-matched teacher, the localizing reference
    udma_student        a trained student's own trunk -- tests architecture
                        vs. training target
    convnextv2_fcmae    ImageNet domain, reconstruction pretraining -- tests
                        objective vs. domain (needs ``pip install timm``)
    udma_map            a trained UDMA model end to end: its teacher, both
                        students and the four disagreement maps (st1, st2,
                        ss, fused cob) -- tests whether the ANOMALY MAP leaks,
                        which is what the ON/OFF short-list filter reads

For ``udma_map`` the map stages are one scalar per cell, so the displacement
is ``|m(x+s) - m(x)|``. They are also read on the level the short-list rule
sees: each row's map maximum against the background, the 99th percentile of
the OFF-row maxima of the clean frames (same stage, same model). With OFF
pixels unchanged, an OFF row above that background is response the model
generated from ON pixels. When the teacher is confined to one observation
(``cnn_obs``) its OFF rows must not move at all: that is the instrument check.

Cost: n_frames * len(snr_list) * len(stages) frozen forwards. No training.

Outcome (2026-07-16): all three hypotheses refuted; only domain-matched
training localizes. Full record in ``docs/03_teacher-localization.md``.

Besides the ON/OFF means the probe prints and saves the response of every
feature row. For ResNet-18 the rows whose receptive field holds no ON pixel
are known from the geometry (``expected_clean_off_rows``): they must stay at
numerical zero, since the OFF pixels do not change. This is a check of the
instrument, with a PASS/FAIL line — a clean row that moves means the test has
a leakage path other than the receptive field (preprocessing, BatchNorm, the
injector). ``--confine obs`` applies ResNet-18 to each observation on its own,
where every OFF row is clean.

Usage (run on the data host; --cache must point at a preprocessed cache):

    # out-of-domain CNN, receptive-field sweep -- default
    python scripts/debug/teacher_row_leakage_test.py --cache <cache_dir>

    # domain-matched ViT-MAE teacher, read at block 3
    python scripts/debug/teacher_row_leakage_test.py \
        --architecture vit_mae --layer 3 \
        --checkpoint <vit_mae.ckpt> \
        --model_config configs/model/vit_mae.yaml --cache <cache_dir>

    # a trained UDMA student's trunk
    python scripts/debug/teacher_row_leakage_test.py \
        --architecture udma_student --student ae \
        --checkpoint <udma.ckpt> \
        --model_config configs/model/udma.yaml --cache <cache_dir>

    # ConvNeXt V2 / FCMAE control
    python scripts/debug/teacher_row_leakage_test.py \
        --architecture convnextv2_fcmae --cache <cache_dir>

    # ResNet-18 confined to one observation at a time
    python scripts/debug/teacher_row_leakage_test.py --confine obs --cache <cache_dir>

    # a trained UDMA model: teacher, students and anomaly maps
    python scripts/debug/teacher_row_leakage_test.py \
        --architecture udma_map --checkpoint <udma.ckpt> \
        --model_config configs/model/udma.yaml --cache <cache_dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.debug.injection_vs_rfi_test import preprocess_raw, inject_narrowband_on_only
from src.models.observation_fold import fold_observations, unfold_rows

INPUT_SHAPE = (96, 1024, 1)
INPUT_HW = (96, 1024)
N_OBS = 6
ON_OBS = (0, 2, 4)

# Stage -> approximate theoretical vertical receptive field (px), for the
# printout only (helps read the ratio trend against the 16-px observation
# boundary). Values are ResNet-18 theoretical max RF on a (96,1024) input.
STAGE_RF = {"stem": 11, "layer1": 43, "layer2": 99, "layer3": 211}
STAGES = ("stem", "layer1", "layer2", "layer3")
# Stage -> vertical stride (px per feature row). Every strided layer of
# ResNet-18 pads symmetrically, so feature row ``i`` is centred on input row
# ``stride * i`` and sees ``stride * i ± (STAGE_RF - 1) / 2``, cut at the frame.
STAGE_STRIDE = {"stem": 4, "layer1": 4, "layer2": 8, "layer3": 16}
OBS_PX = INPUT_HW[0] // N_OBS


def expected_clean_off_rows(stage: str, confine: str = "none") -> np.ndarray:
    """Feature rows of OFF observations whose receptive field holds no ON
    pixel, for ResNet-18 at ``stage``. Confined to one observation, every OFF
    row is clean."""
    stride = STAGE_STRIDE[stage]
    nh = INPUT_HW[0] // stride
    rows_per_obs = nh // N_OBS
    half = (STAGE_RF[stage] - 1) // 2
    clean = []
    for i in range(nh):
        if (i // rows_per_obs) in ON_OBS:
            continue
        if confine == "none":
            lo, hi = max(0, stride * i - half), min(INPUT_HW[0] - 1, stride * i + half)
            if any((r // OBS_PX) in ON_OBS for r in range(lo, hi + 1)):
                continue
        clean.append(i)
    return np.array(clean, dtype=int)


class ResNetStages(nn.Module):
    """Frozen ImageNet ResNet-18 exposing features at each depth stage, so the
    same probe runs across the receptive-field sweep. Input (B,1,96,1024) is
    replicated to 3 channels with NO ImageNet mean/std renorm — same deliberate
    simplification as ``ResNetTeacher`` (the gate validated it)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        # pretrained=False: random weights, for geometry tests (no download).
        b = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)
        self.layer1 = b.layer1
        self.layer2 = b.layer2
        self.layer3 = b.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "ResNetStages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw) at the requested stage."""
        h = self.stem(x.repeat(1, 3, 1, 1))
        if stage == "stem":
            return h
        h = self.layer1(h)
        if stage == "layer1":
            return h
        h = self.layer2(h)
        if stage == "layer2":
            return h
        h = self.layer3(h)
        if stage == "layer3":
            return h
        raise ValueError(f"unknown stage {stage!r}")


class ObsConfined(nn.Module):
    """Any ``features(x, stage)`` model applied to each observation on its own:
    the frame is folded into ``N_OBS`` strips, and the strips' feature rows are
    stacked back in observation order."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        return unfold_rows(self.model.features(fold_observations(x, N_OBS), stage), N_OBS)


CONVNEXT_STAGES = ("stem", "stage0", "stage1", "stage2")
# Downsampling factor labels only (not pixel RF — ConvNeXt's 7x7 depthwise
# kernels stacked over variable block counts [3,3,9,3] make a precise
# theoretical pixel RF non-trivial to state correctly; the /N factor is what
# actually matters for reading this alongside ResNetStages, since stage2's
# grid (6,64) at /16 lines up with ResNet's layer3, the direct comparison
# point). stage3 (/32) is excluded: its (3,32) grid has fewer rows than the
# 6 cadence observations, incompatible with row_on_off_masks.
CONVNEXT_RF = {"stem": "/4", "stage0": "/4", "stage1": "/8", "stage2": "/16"}


class ConvNeXtV2Stages(nn.Module):
    """Frozen ConvNeXt V2 (``convnextv2_tiny.fcmae``, timm) exposing features at
    each depth stage — the domain-vs-objective isolating control
    (2026-07-16): SAME domain (ImageNet) and SAME family (CNN) as
    :class:`ResNetStages`, but pretrained via masked-patch RECONSTRUCTION
    (FCMAE, self-supervised, no classification head) instead of
    classification. If this localizes where ResNet-18 doesn't, the deciding
    variable is the pretraining OBJECTIVE, not the domain; if it stays
    diffuse like ResNet-18, domain-matching is confirmed necessary regardless
    of objective. Input (B,1,96,1024) replicated to 3 channels, no ImageNet
    mean/std renorm — same simplification as :class:`ResNetStages`, and
    fully convolutional (no positional embeddings to reconcile with a
    non-224x224 input, unlike a ViT-MAE-on-ImageNet alternative)."""

    def __init__(self, model_name: str = "convnextv2_tiny.fcmae"):
        super().__init__()
        import timm
        m = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.stem = m.stem
        self.stages_list = m.stages
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "ConvNeXtV2Stages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw) at the requested stage."""
        h = self.stem(x.repeat(1, 3, 1, 1))
        if stage == "stem":
            return h
        h = self.stages_list[0](h)
        if stage == "stage0":
            return h
        h = self.stages_list[1](h)
        if stage == "stage1":
            return h
        h = self.stages_list[2](h)
        if stage == "stage2":
            return h
        raise ValueError(f"unknown stage {stage!r}")


class ViTMAEStages(nn.Module):
    """Domain-matched ViT-MAE teacher, exposing the SAME ``features(x, stage) ->
    (B,C,nh,nw)`` interface as :class:`ResNetStages` so it plugs into
    ``stage_displacement`` unchanged. A ViT has no receptive-field sweep (global
    attention), so it contributes ONE pseudo-stage read at the deployed
    transformer block (``layer``, 1-indexed; -1 = final). Token grid comes from
    the model config's ``patch_size`` — (6,64) for the 0000.fil teacher, rows
    mapping 1:1 to the 6 observations. Reads tokens exactly as
    ``teacher_sensitivity_test.encode_tokens_layer`` (patch_embed + pos_embed,
    blocks 1..k, final norm only when k == depth)."""

    def __init__(self, checkpoint, model_cfg, device, layer=-1):
        super().__init__()
        from scripts.debug.encode_separation_test import load_model
        self.model = load_model(checkpoint, model_cfg, device, require_encode=False)
        if not (hasattr(self.model, "patch_embed") and hasattr(self.model, "encoder")
                and hasattr(self.model.encoder, "layers")):
            raise SystemExit("--architecture vit_mae requires a ViT-MAE checkpoint "
                             "(architecture: vit_mae in --model_config).")
        self.layer = layer
        ph, pw = model_cfg["patch_size"]
        self.grid_size = (INPUT_HW[0] // ph, INPUT_HW[1] // pw)
        self.stage_name = "final" if layer == -1 else f"block{layer}"
        self.eval()

    def train(self, mode: bool = True) -> "ViTMAEStages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str = None) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw) token-feature grid at ``self.layer``."""
        m = self.model
        tok = m.patch_embed(x) + m.pos_embed
        n_layers = len(m.encoder.layers)
        k = n_layers if self.layer in (-1, n_layers) else self.layer
        if not 0 <= k <= n_layers:
            raise SystemExit(f"--layer must be in 0..{n_layers} or -1, got {self.layer}")
        for j, blk in enumerate(m.encoder.layers, start=1):
            if j > k:
                break
            tok = blk(tok)
        if k == n_layers and getattr(m.encoder, "norm", None) is not None:
            tok = m.encoder.norm(tok)
        b, n, d = tok.shape
        nh, nw = self.grid_size
        return tok.reshape(b, nh, nw, d).permute(0, 3, 1, 2)


def load_udma(checkpoint, model_cfg, device, architecture: str) -> nn.Module:
    """A UDMA model from a Lightning checkpoint, in eval mode on ``device``
    (teacher weights included: they live in the checkpoint)."""
    from src.models import build_autoencoder
    model = build_autoencoder(INPUT_SHAPE, model_cfg, loss="mse")
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    if not hasattr(model, "student_ae"):
        raise SystemExit(f"--architecture {architecture} requires a UDMA checkpoint "
                         "(architecture: udma in --model_config).")
    return model


class UDMAStudentStages(nn.Module):
    """A trained UDMA student's own output feature grid, exposing the SAME
    ``features(x, stage) -> (B,C,nh,nw)`` interface as :class:`ResNetStages` /
    :class:`ViTMAEStages`. One pseudo-stage (the student's final projection
    head output, ``(B, teacher.channels, 6, 64)``) — no depth sweep, since the
    question here isn't receptive field, it's "does the SAME student
    architecture localize when trained against a domain-matched target vs an
    out-of-domain one" (2026-07-16, see udma_teacher_rf_leakage_refuted's
    confound note). Loads the full UDMA model (teacher + both students) via
    ``build_autoencoder`` from ``model_config`` (``architecture: udma``,
    either ``teacher.type: vit_mae`` or ``cnn_distilled``) and a Lightning
    checkpoint, exactly like ``encode_separation_test.load_model``, then reads
    out only the requested student's raw output (``student_ae`` or
    ``student_mem``'s projection head, ignoring the teacher's own features and
    the disagreement maps entirely — this measures the STUDENT's learned
    localization, not the teacher's)."""

    def __init__(self, checkpoint, model_cfg, device, student="ae"):
        super().__init__()
        model = load_udma(checkpoint, model_cfg, device, "udma_student")
        self.model = model
        self.student = {"ae": model.student_ae, "mem": model.student_mem}[student]
        self.grid_size = model.teacher.grid_size
        self.stage_name = f"student_{student}"
        self.eval()

    def train(self, mode: bool = True) -> "UDMAStudentStages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str = None) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw) student output feature grid."""
        out = self.student(x)
        if isinstance(out, tuple):  # memory student returns (out, attention)
            out = out[0]
        return out


MAP_STAGES = ("st1", "st2", "ss", "cob")
UDMA_MAP_STAGES = ("teacher", "student_ae", "student_mem") + MAP_STAGES


class UDMAMapStages(nn.Module):
    """A whole trained UDMA model behind the ``features(x, stage)`` interface,
    one stage per piece of the anomaly map (phase 06a, 2026-11-23):

    - ``teacher``: the normalized teacher tokens (the students' target);
    - ``student_ae`` / ``student_mem``: each student's output grid;
    - ``st1``, ``st2``, ``ss``, ``cob``: the disagreement maps of
      ``UDMA.anomaly_map_components``, as ``(B, 1, nh, nw)``.

    Reading them together says where an OFF-row response of the map comes
    from: a teacher and a student that move together on an OFF row cancel in
    ``(T - S)^2``; a confined teacher that stays still does not."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        self.grid_size = model.teacher.grid_size
        self.teacher_confined = bool(getattr(model.teacher, "per_observation", False))

    @classmethod
    def from_checkpoint(cls, checkpoint, model_cfg, device) -> "UDMAMapStages":
        return cls(load_udma(checkpoint, model_cfg, device, "udma_map"))

    def train(self, mode: bool = True) -> "UDMAMapStages":
        return super().train(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        """(B,1,96,1024) -> (B,C,nh,nw); C = 1 for the map stages."""
        m = self.model
        if stage == "teacher":
            return m.teacher(x)
        if stage in ("student_ae", "student_mem"):
            out = getattr(m, stage)(x)
            return out[0] if isinstance(out, tuple) else out
        if stage in MAP_STAGES:
            return m.anomaly_map_components(x)[stage].unsqueeze(1)
        raise ValueError(f"unknown stage {stage!r}")


@torch.no_grad()
def row_level(model, frames, stage, device, batch=32) -> np.ndarray:
    """(N, nh): each row's maximum over the frequency axis of a map stage's own
    value (not a displacement) -- the quantity ``full_row_hits`` compares with
    the cadence threshold."""
    rows = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i + batch]).float().unsqueeze(1).to(device)
        rows.append(model.features(x, stage)[:, 0].amax(dim=2).cpu().numpy())
    return np.concatenate(rows)


def map_background(clean_rows: np.ndarray, quantile: float = 0.99) -> float:
    """Background level of a map stage: ``quantile`` of the OFF-row maxima of
    the clean (uninjected) frames."""
    _, off_mask = row_on_off_masks(clean_rows.shape[1])
    return float(np.quantile(clean_rows[:, off_mask], quantile))


def off_rows_above_background(inj_rows: np.ndarray, background: float) -> dict:
    """Rows of the injected frames against the background.

    ``row_level``: (nh,) mean over frames of row maximum / background.
    ``share_off_ge2``: share of frames with at least two OFF rows above the
    background (the shape of an ``off_leak`` hit; the background is a proxy
    for the cadence ceiling, not the production threshold)."""
    _, off_mask = row_on_off_masks(inj_rows.shape[1])
    above = inj_rows[:, off_mask] > background
    return {
        "row_level": inj_rows.mean(axis=0) / background,
        "share_off_ge2": float((above.sum(axis=1) >= 2).mean()),
        "share_off_ge1": float(above.any(axis=1).mean()),
    }


def row_on_off_masks(nh: int) -> tuple:
    """Boolean (nh,) masks labelling each feature row ON or OFF by which of the
    6 observations it belongs to (ON = obs 0/2/4)."""
    rows_per_obs = nh // N_OBS
    if rows_per_obs == 0:
        raise ValueError(f"grid rows {nh} < {N_OBS} observations")
    obs_of_row = np.arange(nh) // rows_per_obs
    on_mask = np.isin(obs_of_row, ON_OBS)
    return on_mask, ~on_mask


@torch.no_grad()
def row_displacement(model, f_clean, f_inj, stage, device, batch=32) -> np.ndarray:
    """Response of every feature row to the injection at one stage.

    Returns (N, nh): per frame, each feature row's MAX displacement over the
    frequency axis (mirrors ``full_row_hits``'s per-row max, no column window)."""
    rows = []
    for i in range(0, len(f_clean), batch):
        xc = torch.from_numpy(f_clean[i:i + batch]).float().unsqueeze(1).to(device)
        xi = torch.from_numpy(f_inj[i:i + batch]).float().unsqueeze(1).to(device)
        disp = (model.features(xi, stage) - model.features(xc, stage)).norm(dim=1)  # (b, nh, nw)
        rows.append(disp.max(dim=2).values.cpu().numpy())
    return np.concatenate(rows)


def on_off_response(row_max: np.ndarray) -> tuple:
    """(N, nh) row responses -> (on_resp, off_resp), each (N,): per-frame mean
    over that group's feature rows."""
    on_mask, off_mask = row_on_off_masks(row_max.shape[1])
    return row_max[:, on_mask].mean(axis=1), row_max[:, off_mask].mean(axis=1)


def stage_displacement(model, f_clean, f_inj, stage, device, batch=32):
    """Per-frame ON-row and OFF-row response to the injection at one stage."""
    return on_off_response(row_displacement(model, f_clean, f_inj, stage, device, batch))


def leak_criterion(profile: np.ndarray, clean_off: np.ndarray, zero_tol: float) -> dict:
    """Instrument check on a per-row profile ``(nh,)`` (mean over frames).

    The rows in ``clean_off`` see no ON pixel, so they must not respond:
    passes when the largest of them is at most ``zero_tol`` times the mean ON
    row. ``off_share_dirty`` is the fraction of the summed OFF response held by
    the remaining OFF rows (1.0 when all the leak is where geometry puts it).
    """
    on_mask, off_mask = row_on_off_masks(len(profile))
    on_mean = float(profile[on_mask].mean())
    clean = np.zeros(len(profile), dtype=bool)
    clean[clean_off] = True
    dirty = off_mask & ~clean
    off_sum = float(profile[off_mask].sum())
    max_clean = float(profile[clean].max()) if clean.any() else float("nan")
    return {
        "on_mean": on_mean,
        "max_clean_frac": max_clean / (on_mean + 1e-12),
        "off_share_dirty": float(profile[dirty].sum()) / off_sum if off_sum > 0 else float("nan"),
        "n_clean": int(clean.sum()),
        "n_dirty": int(dirty.sum()),
        "passed": bool(clean.any() and max_clean <= zero_tol * on_mean),
    }


def _fmt_ratio(on_m: float, off_m: float) -> str:
    """ON/OFF for the printout; an OFF response of exactly zero has no ratio."""
    return "inf" if off_m == 0 else f"{on_m / off_m:.2f}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--architecture",
                   choices=["resnet18", "convnextv2_fcmae", "vit_mae", "udma_student",
                            "udma_map"],
                   default="resnet18",
                   help="resnet18: out-of-domain P, RF sweep across stem/layer1/2/3 (default). "
                        "convnextv2_fcmae: SAME domain+family as resnet18, but masked-"
                        "reconstruction pretraining instead of classification — isolates "
                        "objective from domain (needs `pip install timm` on the runner). "
                        "vit_mae: domain-matched teacher, single stage at --layer (localizing "
                        "reference) — requires --checkpoint/--model_config. udma_student: a "
                        "trained UDMA student's own output grid — requires --checkpoint/"
                        "--model_config (architecture: udma) and --student. udma_map: a "
                        "trained UDMA model's teacher, students and anomaly maps — requires "
                        "--checkpoint/--model_config (architecture: udma).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Required for --architecture vit_mae/udma_student/udma_map; ignored "
                        "otherwise.")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml",
                   help="Model config: vit_mae.yaml (patch_size -> token grid) for --architecture "
                        "vit_mae, or udma.yaml/udma_cnn_teacher.yaml for udma_student.")
    p.add_argument("--layer", type=int, default=3,
                   help="ViT-MAE transformer block to read (1-indexed; -1 = final). Default 3 = "
                        "the deployed teacher_layer in configs/model/udma.yaml. vit_mae only.")
    p.add_argument("--student", choices=["ae", "mem"], default="ae",
                   help="Which UDMA student to read (udma_student only).")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--n_frames", type=int, default=200, help="Quiet injection sites.")
    p.add_argument("--snr_list", type=float, nargs="+", default=[10, 20, 40])
    p.add_argument("--drift_rate", type=float, default=0.3)
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES,
                   help="ResNet depth stages to sweep (resnet18 only).")
    p.add_argument("--convnext_stages", nargs="+", default=list(CONVNEXT_STAGES),
                   choices=CONVNEXT_STAGES,
                   help="ConvNeXt V2 depth stages to sweep (convnextv2_fcmae only).")
    p.add_argument("--confine", choices=["none", "obs"], default="none",
                   help="resnet18 only. obs: apply the network to each observation on its own "
                        "(16-px strips), so no feature row can see another observation.")
    p.add_argument("--random_weights", action="store_true",
                   help="resnet18 only: untrained ResNet-18 instead of the ImageNet weights — "
                        "what the geometry alone gives, to read next to the pretrained run.")
    p.add_argument("--zero_tol", type=float, default=1e-3,
                   help="A clean OFF row counts as zero when its response is at most this "
                        "fraction of the mean ON row (GPU convolutions are not bit-exact).")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/teacher_row_leakage")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if (args.confine != "none" or args.random_weights) and args.architecture != "resnet18":
        raise SystemExit("--confine and --random_weights apply to --architecture resnet18 only.")
    rng = np.random.default_rng(args.seed)

    with open(args.data_config) as f:
        preproc = yaml.safe_load(f)["preprocessing"]

    if args.architecture == "vit_mae":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --architecture vit_mae.")
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
        print(f"Loading domain-matched ViT-MAE teacher from {args.checkpoint} "
              f"(block {args.layer})")
        model = ViTMAEStages(args.checkpoint, model_cfg, args.device, layer=args.layer)
        stages = [model.stage_name]
        rf_label = {model.stage_name: "global"}
        print(f"  Token grid: {model.grid_size} — global attention, no RF sweep")
    elif args.architecture == "udma_student":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --architecture udma_student.")
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
        teacher_type = model_cfg.get("teacher", {}).get("type", "vit_mae")
        print(f"Loading UDMA student '{args.student}' from {args.checkpoint} "
              f"(teacher.type={teacher_type})")
        model = UDMAStudentStages(args.checkpoint, model_cfg, args.device, student=args.student)
        stages = [model.stage_name]
        rf_label = {model.stage_name: f"teacher={teacher_type}"}
        print(f"  Student output grid: {model.grid_size}")
    elif args.architecture == "udma_map":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --architecture udma_map.")
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
        teacher_type = model_cfg.get("teacher", {}).get("type", "vit_mae")
        print(f"Loading UDMA model from {args.checkpoint} (teacher.type={teacher_type})")
        model = UDMAMapStages.from_checkpoint(args.checkpoint, model_cfg, args.device)
        stages = list(UDMA_MAP_STAGES)
        rf_label = {s: ("map" if s in MAP_STAGES else "features") for s in stages}
        print(f"  Grid: {model.grid_size}; teacher confined to one observation: "
              f"{model.teacher_confined}")
    elif args.architecture == "convnextv2_fcmae":
        print("Loading ConvNeXt V2 (ImageNet, FCMAE self-supervised, frozen) — "
              "domain-vs-objective isolating control")
        model = ConvNeXtV2Stages().to(args.device)
        stages = args.convnext_stages
        rf_label = {s: CONVNEXT_RF[s] for s in stages}
    else:
        print(f"Loading ResNet-18 ({'RANDOM weights' if args.random_weights else 'ImageNet'}, "
              "frozen) — teacher P, all stages"
              + (" — confined to one observation" if args.confine == "obs" else ""))
        if args.random_weights:
            torch.manual_seed(args.seed)
        model = ResNetStages(pretrained=not args.random_weights).to(args.device)
        if args.confine == "obs":
            model = ObsConfined(model)
        stages = args.stages
        rf_label = {s: (f"{OBS_PX}px" if args.confine == "obs" else f"{STAGE_RF[s]}px")
                    for s in stages}

    # ---- quiet injection sites (lowest hot-fraction), same selection as the gate ----
    npy_path = Path(args.cache) / f"{args.split}.npy"
    print(f"Loading cache: {npy_path}")
    arr = np.load(str(npy_path), mmap_mode="r")
    pool_idx = rng.choice(arr.shape[0], size=min(args.n_frames * 4, arr.shape[0]), replace=False)
    raw_pool = np.array(arr[pool_idx])
    del arr

    pre_pool = np.array([preprocess_raw(raw_pool[i], preproc) for i in range(len(raw_pool))])
    hot = np.array([(f > 5.0).mean() for f in pre_pool])
    quiet_sel = np.argsort(hot)[:args.n_frames]
    raw_quiet = raw_pool[quiet_sel]
    f_quiet = pre_pool[quiet_sel]
    print(f"  Quiet frames: {len(quiet_sel)}")

    results = {}  # (stage, snr) -> (on_mean, off_mean, ratio)
    profiles = {}  # (stage, snr) -> (nh,) mean response of each feature row
    is_map = args.architecture == "udma_map"
    map_stages = [s for s in stages if s in MAP_STAGES] if is_map else []
    backgrounds = {s: map_background(row_level(model, f_quiet, s, args.device))
                   for s in map_stages}
    levels = {}  # (stage, snr) -> off_rows_above_background dict
    for snr in args.snr_list:
        f_inj = np.array([
            preprocess_raw(
                inject_narrowband_on_only(raw_quiet[i], snr=snr,
                                          drift_rate=args.drift_rate, seed=args.seed + i),
                preproc)
            for i in range(len(raw_quiet))
        ])
        for stage in stages:
            row_max = row_displacement(model, f_quiet, f_inj, stage, args.device)
            profiles[(stage, snr)] = row_max.mean(axis=0)
            on_resp, off_resp = on_off_response(row_max)
            on_m, off_m = float(on_resp.mean()), float(off_resp.mean())
            ratio = on_m / (off_m + 1e-12)
            results[(stage, snr)] = (on_m, off_m, ratio)
            if stage in map_stages:
                levels[(stage, snr)] = off_rows_above_background(
                    row_level(model, f_inj, stage, args.device), backgrounds[stage])

    # ---- report ----
    print(f"\n{'='*72}")
    print("ON->OFF ROW LEAKAGE  (ON-only injection; ON/OFF = feature-row response)")
    print("higher ON/OFF ratio = better localization; ratio -> 1 = diffuse/leaky")
    print(f"{'='*72}")
    for snr in args.snr_list:
        print(f"\n  SNR = {snr:g}")
        print(f"    {'stage':>8s}  {'vert.RF':>7s}  {'ON_resp':>9s}  {'OFF_resp':>9s}  {'ON/OFF':>7s}")
        for stage in stages:
            on_m, off_m, ratio = results[(stage, snr)]
            print(f"    {stage:>8s}  {rf_label[stage]:>7s}  {on_m:9.4f}  {off_m:9.4f}  "
                  f"{_fmt_ratio(on_m, off_m):>7s}")

    # ---- per-row profile: where in the OFF observations the response sits ----
    is_resnet = args.architecture == "resnet18"
    print(f"\n{'='*72}")
    print("PER-ROW PROFILE  (row response / mean ON row; OFF pixels are unchanged,")
    print("so an OFF row can move only if its receptive field holds ON pixels)")
    print(f"{'='*72}")
    for stage in stages:
        nh = len(profiles[(stage, args.snr_list[0])])
        rows_per_obs = nh // N_OBS
        on_mask, _ = row_on_off_masks(nh)
        clean = set(expected_clean_off_rows(stage, args.confine).tolist()) if is_resnet else set()
        print(f"\n  {stage}  ({nh} rows, {rows_per_obs} per observation)")
        print(f"    {'row':>3s}  {'obs':>3s}  {'':3s}  "
              + "  ".join(f"{'SNR ' + format(s, 'g'):>8s}" for s in args.snr_list)
              + ("  geometry" if is_resnet else ""))
        for i in range(nh):
            frac = [profiles[(stage, s)][i] / (profiles[(stage, s)][on_mask].mean() + 1e-12)
                    for s in args.snr_list]
            tag = ""
            if is_resnet and not on_mask[i]:
                tag = "  no ON px" if i in clean else "  sees ON px"
            print(f"    {i:3d}  {i // rows_per_obs:3d}  {'ON ' if on_mask[i] else 'OFF'}  "
                  + "  ".join(f"{f:8.4f}" for f in frac) + tag)

    # ---- instrument check (resnet18): rows that see no ON pixel must not move ----
    criteria = {}
    if is_resnet:
        print(f"\n{'='*72}")
        print(f"INSTRUMENT CHECK  (clean OFF rows <= {args.zero_tol:g} x mean ON row)")
        print(f"{'='*72}")
        for stage in stages:
            clean = expected_clean_off_rows(stage, args.confine)
            if len(clean) == 0:
                print(f"  {stage:>8s}: every OFF row sees ON pixels — nothing to check")
                continue
            checks = [leak_criterion(profiles[(stage, s)], clean, args.zero_tol)
                      for s in args.snr_list]
            criteria[stage] = checks
            worst = max(c["max_clean_frac"] for c in checks)
            share = min(c["off_share_dirty"] for c in checks)
            ok = all(c["passed"] for c in checks)
            share_txt = f"{share:.4f}" if checks[0]["n_dirty"] else "n/a (no such rows)"
            print(f"  {stage:>8s}: clean rows {clean.tolist()}  max response {worst:.2e} x ON  "
                  f"|  OFF response in the rows that see ON px: {share_txt}  "
                  f"[{'PASS' if ok else 'FAIL'}]")
        if any(not all(c["passed"] for c in v) for v in criteria.values()):
            print("\n  FAIL: a row with no ON pixel in its receptive field moved. The probe has a\n"
                  "  leakage path other than the receptive field (preprocessing, BatchNorm mode,\n"
                  "  the injector touching OFF frames). Find it before reading any ratio above.")

    if is_map:
        print(f"\n{'='*72}")
        print("MAP ROWS AGAINST THE BACKGROUND  (row max / q99 of clean OFF-row maxima;")
        print("OFF pixels are unchanged, so an OFF row above 1 is model-generated)")
        print(f"{'='*72}")
        for stage in map_stages:
            print(f"\n  {stage}  (background {backgrounds[stage]:.4f})")
            print(f"    {'SNR':>5s}  " + "  ".join(f"{'row ' + str(i):>7s}" for i in range(N_OBS))
                  + f"  {'>=1 OFF':>8s}  {'>=2 OFF':>8s}")
            for snr in args.snr_list:
                lv = levels[(stage, snr)]
                print(f"    {snr:5g}  " + "  ".join(f"{v:7.2f}" for v in lv["row_level"])
                      + f"  {lv['share_off_ge1']:8.1%}  {lv['share_off_ge2']:8.1%}")
        if model.teacher_confined:
            print(f"\n{'='*72}")
            print(f"INSTRUMENT CHECK  (confined teacher: OFF rows <= {args.zero_tol:g} x mean ON row)")
            print(f"{'='*72}")
            off_rows = np.flatnonzero(row_on_off_masks(model.grid_size[0])[1])
            checks = [leak_criterion(profiles[("teacher", s)], off_rows, args.zero_tol)
                      for s in args.snr_list]
            criteria["teacher"] = checks
            ok = all(c["passed"] for c in checks)
            worst = max(c["max_clean_frac"] for c in checks)
            print(f"  teacher: max OFF response {worst:.2e} x ON  [{'PASS' if ok else 'FAIL'}]")
            if not ok:
                print("\n  FAIL: an OFF row of a teacher confined to one observation moved. The\n"
                      "  probe or the model has a path across observations other than the\n"
                      "  students. Find it before reading the map rows above.")

    # Headline: ratio trend across stages at the highest SNR (clearest signal).
    ref_snr = max(args.snr_list)
    print(f"\n{'='*72}\nVERDICT (ratio trend @ SNR {ref_snr:g}, deepest->shallowest)\n{'='*72}")
    ratios = {s: results[(s, ref_snr)][2] for s in stages}
    for stage in stages:
        tag = ("LOCALIZED" if ratios[stage] >= 3.0
               else "partial" if ratios[stage] >= 1.5 else "DIFFUSE")
        on_m, off_m, _ = results[(stage, ref_snr)]
        print(f"  {stage:>8s} (RF~{rf_label[stage]}): ON/OFF = {_fmt_ratio(on_m, off_m)}  [{tag}]")
    if args.architecture == "vit_mae":
        print("\n  Read: this is the domain-matched teacher's localizing ON/OFF ratio at its\n"
              "  deployed block — the apples-to-apples reference for the ResNet stages (which\n"
              "  stayed ~2-3, RF-invariant, 2026-07-16). A markedly higher ratio here confirms\n"
              "  localization is a domain-matching property, not a receptive-field one.")
    elif args.architecture == "udma_student":
        print("\n  Read: this is the SAME student CNN architecture (build_encoder trunk), only\n"
              "  the training TARGET differs by checkpoint (vit_mae domain-matched vs\n"
              "  cnn_distilled out-of-domain). Compare this ratio against the SAME test run on\n"
              "  the other checkpoint: if the vit_mae-target student localizes and the\n"
              "  cnn_distilled-target student doesn't, that's a second, architecture-independent\n"
              "  confirmation the TARGET's domain (not the CNN trunk) is the lever — corroborates\n"
              "  the teacher-level finding without any new training.")
    elif is_map:
        print("\n  Read: run it on two checkpoints that differ only in the teacher and compare\n"
              "  stage by stage. The students' OFF response is the leak each student makes;\n"
              "  the teacher's is the leak its target makes. Where both move, st1/st2 can stay\n"
              "  small on the OFF rows; where only the students move, st1/st2 carry the leak\n"
              "  into the map. The background table is what the short-list filter would see.")
    elif args.architecture == "convnextv2_fcmae":
        print("\n  Read: this is the domain-vs-objective isolating control (2026-07-16) — SAME\n"
              "  domain (ImageNet) and family (CNN) as resnet18, but pretrained via masked-patch\n"
              "  RECONSTRUCTION (FCMAE) instead of classification. Compare stage2 (/16 grid,\n"
              "  matches resnet18's layer3) directly against resnet18 layer3's ratio (~2.17):\n"
              "  if this localizes markedly better, the pretraining OBJECTIVE is the lever, not\n"
              "  domain -> a CNN teacher distilled from THIS backbone could be paper-faithful\n"
              "  (out-of-domain, generic) AND localizing, no audio/domain-closer backbone hunt\n"
              "  needed. If it stays diffuse like resnet18, domain-matching is confirmed\n"
              "  necessary regardless of pretraining objective.")
    else:
        print("\n  Read: the ON/OFF mean mixes OFF rows that see ON pixels with rows that do not,\n"
              "  so read it together with the per-row profile above. A stage tests confinement\n"
              "  only if its OFF rows hold no ON pixel; without --confine obs that is true of\n"
              "  4 stem rows out of 12 and of no row from layer1 on.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_tag = args.architecture
    if args.random_weights:
        out_tag = f"{out_tag}_random"
    if args.confine != "none":
        out_tag = f"{out_tag}_{args.confine}"
    if args.architecture == "udma_student":
        out_tag = f"udma_student_{args.student}_{args.checkpoint.parents[1].name}"
    if is_map:
        out_tag = f"udma_map_{args.checkpoint.parents[1].name}"
    out_npz = args.out_dir / f"teacher_row_leakage_{out_tag}.npz"
    np.savez(
        out_npz,
        architecture=args.architecture,
        stages=np.array(stages),
        snr_list=np.array(args.snr_list),
        on_resp=np.array([[results[(s, snr)][0] for s in stages] for snr in args.snr_list]),
        off_resp=np.array([[results[(s, snr)][1] for s in stages] for snr in args.snr_list]),
        ratio=np.array([[results[(s, snr)][2] for s in stages] for snr in args.snr_list]),
        confine=args.confine,
        seed=args.seed,
        n_frames=len(quiet_sel),
        frame_idx=pool_idx[quiet_sel],
        # per-row profiles, one array (n_snr, nh) per stage (nh differs by stage)
        **{f"row_profile_{s}": np.array([profiles[(s, snr)] for snr in args.snr_list])
           for s in stages},
        **{f"clean_off_rows_{s}": expected_clean_off_rows(s, args.confine)
           for s in stages if is_resnet},
        instrument_check=np.array([all(c["passed"] for c in criteria[s]) if s in criteria
                                   else True for s in stages]) if (is_resnet or is_map) else np.array([]),
        teacher_confined=bool(is_map and model.teacher_confined),
        **{f"map_background_{s}": backgrounds[s] for s in map_stages},
        **{f"map_row_level_{s}": np.array([levels[(s, snr)]["row_level"] for snr in args.snr_list])
           for s in map_stages},
        **{f"map_share_off_ge2_{s}": np.array([levels[(s, snr)]["share_off_ge2"]
                                               for snr in args.snr_list]) for s in map_stages},
        **{f"map_share_off_ge1_{s}": np.array([levels[(s, snr)]["share_off_ge1"]
                                               for snr in args.snr_list]) for s in map_stages},
    )
    print(f"\nSaved -> {out_npz}")


if __name__ == "__main__":
    main()
