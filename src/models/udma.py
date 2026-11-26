"""
UDMA — Unsupervised Distillation and Memory-enhanced Autoencoder.

5th ``build_autoencoder`` backbone (``architecture: udma``). Adapts Qi et al.
2024 ("Unsupervised Spectrum Anomaly Detection With Distillation and Memory
Enhanced Autoencoders", IEEE IoT Journal 11(24):39361) to this repo: two
lightweight CNN "students" are trained to regress the frozen, self-supervised
ViT-MAE encoder's token features (not pixels). Disagreement between what the
students predict and what the teacher actually produced is the anomaly signal
— on in-distribution RFI/noise the students learn to agree with the teacher
(and, via the ``map_ss`` term, with each other); on never-seen morphology they
cannot, and the prediction gap opens up. This is a feature-space generalisation
of the validated pixel-space probe ``‖AE(x)−MemAE(x)‖²``
(``scripts/debug/encode_separation_test.py``, ``DisagreementPair``), attacking
its diagnosed failure mode (recon-MSE measures predictability, not
anomalousness — the RFI tail swamps the operating point below SNR 10) by
moving the score off the pixel manifold where that tail lives.

Full design rationale, gate results, and pre-registered acceptance bars:
``docs/02_udma-architecture.md`` (Q1-Q10). Teacher fitness (G1-G3b) was
verified before this module was written (``scripts/debug/teacher_sensitivity_test.py``);
do not skip that gate for a new teacher checkpoint/layer.

Components:
- ``TeacherViT``: frozen ``ViTMAE.encode_tokens_at`` wrapper. Reshapes the
  ``(B, 384, 128)`` token sequence (patch grid 6x64 over a (96,1024) input) to
  ``(B, 128, 6, 64)`` and applies an offline per-channel Norm (``fit_normalization``).
- ``FeatureStudent``: the same CNN trunk as ``Autoencoder``/``MemAE``
  (``build_encoder``, 4 blocks -> 16x spatial reduction) produces
  ``(B, 64, 6, 64)`` — the SAME (6,64) grid as the teacher tokens, by
  construction (patch_size (16,16) on (96,1024) == encoder's 16x downsampling).
  A small projection head maps this to ``(B, 128, 6, 64)`` to match the
  teacher's channel count. ``memory=True`` inserts the validated
  ``MemoryUnit`` between trunk and head (the "MemAE" student).
- ``UDMA``: teacher + two students (AE-student, MemAE-student); joint loss
  (Q4), three disagreement maps + fused score (Q5).

No pixel decoder anywhere — the "decoder" for each student is the projection
head onto the shared token grid; this is what eliminates the pixel-space
dilution (384 positions vs ~98k pixels) that limited every prior recon scorer.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from .encoder import build_encoder
from .memory import MemoryUnit
from .observation_fold import fold_observations, unfold_rows
from .vit_mae import ViTMAE, build_vit_mae

__all__ = ["TeacherViT", "TeacherCNN", "FeatureStudent", "UDMA", "build_udma"]

_ROOT = Path(__file__).resolve().parents[2]


class _FrozenTokenTeacher(nn.Module):
    """Shared base for a frozen, Norm-ed token-grid teacher: buffers
    (``mu``/``sigma``), eval-mode pinning, ``forward``, and
    ``fit_normalization`` are identical regardless of what produces the raw
    ``(B, channels, nh, nw)`` grid (a ViT-MAE block or a distilled CNN trunk)
    — subclasses set ``self.channels``/``self.grid_size`` and implement
    ``_raw_tokens`` in ``__init__``, and MUST call ``self.eval()`` at the end
    of their own ``__init__`` (registering the ``mu``/``sigma`` buffers here
    doesn't pin eval mode by itself — a subclass with BatchNorm layers left
    in the default ``training=True`` state would normalize with per-batch
    stats instead of the frozen running stats until something up the module
    hierarchy calls ``.train()``/``.eval()``, which does not happen for
    standalone use like the teacher-sensitivity gate scripts).
    """

    def train(self, mode: bool = True) -> "_FrozenTokenTeacher":
        return super().train(False)

    def _raw_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,C,H,W) -> (B, channels, nh, nw)``, before Norm. Override in subclasses."""
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._raw_tokens(x)
        return (raw - self.mu.view(1, -1, 1, 1)) / (self.sigma.view(1, -1, 1, 1) + 1e-6)

    @torch.no_grad()
    def fit_normalization(self, loader, device: torch.device, max_batches: Optional[int] = None) -> None:
        """Offline per-channel mean/std of the raw (pre-Norm) token/feature grid
        over ``loader``.

        One-shot (Q2/Q7): call once on the training set, then the resulting
        ``mu``/``sigma`` buffers are saved with the UDMA checkpoint.
        """
        self.eval()
        total = torch.zeros(self.channels, dtype=torch.float64, device=device)
        total_sq = torch.zeros(self.channels, dtype=torch.float64, device=device)
        count = 0
        for i, x in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            raw = self._raw_tokens(x).double()  # (B, D, nh, nw)
            flat = raw.permute(1, 0, 2, 3).reshape(self.channels, -1)
            total += flat.sum(dim=1)
            total_sq += (flat ** 2).sum(dim=1)
            count += flat.shape[1]
        mean = total / count
        var = (total_sq / count - mean ** 2).clamp_min(0.0)
        self.mu.copy_(mean.to(self.mu.dtype))
        self.sigma.copy_(var.sqrt().to(self.sigma.dtype))


class TeacherViT(_FrozenTokenTeacher):
    """Frozen ViT-MAE encoder, read at an intermediate transformer block.

    Wraps a checkpointed :class:`ViTMAE` (never trained further — parameters
    have ``requires_grad=False`` and the module is pinned in ``eval()`` mode
    regardless of the parent Lightning module's train/eval state, since it has
    no BatchNorm to need train-mode statistics). ``forward`` returns the
    Norm-ed token grid ``(B, 128, 6, 64)`` that is the UDMA students' regression
    target.
    """

    def __init__(self, vit: ViTMAE, teacher_layer: int = 3):
        super().__init__()
        self.vit = vit
        self.teacher_layer = teacher_layer
        for p in self.vit.parameters():
            p.requires_grad_(False)
        nh, nw = self.vit.grid_size
        self.grid_size = (nh, nw)
        self.channels = self.vit.patch_embed.proj.out_channels
        self.register_buffer("mu", torch.zeros(self.channels))
        self.register_buffer("sigma", torch.ones(self.channels))
        self.eval()

    def _raw_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,C,H,W) -> (B, channels, nh, nw)``, before Norm."""
        nh, nw = self.grid_size
        tokens = self.vit.encode_tokens_at(x, layer=self.teacher_layer)  # (B, nh*nw, D)
        b, n, d = tokens.shape
        return tokens.reshape(b, nh, nw, d).permute(0, 3, 1, 2)  # (B, D, nh, nw)


class TeacherCNN(_FrozenTokenTeacher):
    """Frozen CNN teacher distilled from a generic pretrained backbone P
    (paper-faithful route, Qi et al. 2024 Eq. 2 / Bergmann "Uninformed
    Students" — teacher feature space anchored out-of-domain by construction;
    spectrum data enters only as distillation input, never as a learning
    target for P itself). Trunk = :func:`build_encoder` with the same
    parametrisation as the UDMA students (``docs/02_udma-architecture.md §11D8) — its own ``latent_dim``-channel 1x1 projection already lands on the
    target channel count, no separate head needed.

    ``per_observation=True`` builds the trunk for one observation
    (``tchans // n_obs`` rows) and applies it to each observation on its own,
    so no grid row depends on another observation's pixels (the property the
    ON/OFF filters downstream need; see docs/03_teacher-localization.md §4.2).
    The grid is unchanged: ``n_obs`` rows of ``(tchans // n_obs) / 2**n_blocks``
    each, i.e. (6, 64) for a (96, 1024) input with 4 blocks.

    Trained once by ``scripts/distill_teacher.py`` (frozen P, MSE on a
    (6,64) grid via a distillation-only 1x1 projection D, discarded after
    training) then loaded here as a fixed teacher — same
    ``forward``/``grid_size``/``channels``/``mu``/``sigma``/``fit_normalization``
    interface as :class:`TeacherViT` (both via :class:`_FrozenTokenTeacher`),
    so :func:`build_udma` only needs to branch at construction time
    (``teacher.type: cnn_distilled``).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        filters,
        latent_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        use_batchnorm: bool = True,
        convs_per_block: int = 2,
        per_observation: bool = False,
        n_obs: int = 6,
    ):
        super().__init__()
        n_blocks = len(filters)
        factor = 2 ** n_blocks
        th, fw, c = input_shape
        self.per_observation = per_observation
        self.n_obs = n_obs
        if per_observation:
            if th % n_obs or (th // n_obs) % factor:
                raise ValueError(
                    f"per_observation teacher: tchans {th} must split into {n_obs} "
                    f"observations of a multiple of {factor} rows (2 ** {n_blocks} blocks)."
                )
            trunk_shape = (th // n_obs, fw, c)
        else:
            trunk_shape = input_shape
        self.trunk = build_encoder(
            input_shape=trunk_shape,
            filters=filters,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            convs_per_block=convs_per_block,
            variational=False,
        )
        for p in self.trunk.parameters():
            p.requires_grad_(False)

        self.grid_size = (th // factor, fw // factor)
        self.channels = latent_dim
        self.register_buffer("mu", torch.zeros(self.channels))
        self.register_buffer("sigma", torch.ones(self.channels))
        # Required: trunk has BatchNorm (use_batchnorm=True by default) and
        # this class is also used standalone (e.g. teacher_sensitivity_test.py
        # --architecture cnn_distilled), which never calls .eval() itself —
        # without this, BatchNorm normalizes with per-batch stats instead of
        # the running stats learned during distillation. See class docstring.
        self.eval()

    def _raw_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,C,H,W) -> (B, channels, nh, nw)``, before Norm."""
        if self.per_observation:
            return unfold_rows(self.trunk(fold_observations(x, self.n_obs)), self.n_obs)
        return self.trunk(x)


class FeatureStudent(nn.Module):
    """CNN trunk (shared with ``Autoencoder``/``MemAE``) + projection head onto
    the teacher's token grid, with an optional :class:`MemoryUnit`.

    The trunk's spatial output ``(B, latent_dim, H', W')`` already sits on the
    same ``(H', W')`` grid as the teacher tokens (both come from a 16x
    downsampling of a (96,1024) input) — no upsampling/pooling is needed to
    align them.

    ``squeeze_shape``/``squeeze_channels`` optionally re-impose the paper's
    third capacity limiter (Qi et al. 2024 §V: students are encoder-decoders
    with a narrow ``1x8x32`` bottleneck). Our trunk keeps the full ``(H',W')``
    grid at ``latent_dim`` channels — 24576 dims for the default (6,64)x64 —
    which is ~96x the paper's 256, a deliberate anti-dilution deviation that
    left the student expressive enough to potentially track the teacher onto
    anomalies. The squeeze is placed **after** the memory rather than before
    it, so that enabling it moves exactly one variable: the memory keeps its
    384 per-position queries of dim ``latent_dim`` unchanged. (Paper-faithful
    placement would put it before the memory, but that would also change the
    memory's query geometry from 384x64 to 8x32, confounding the two limiters
    the way the mem_slots=30 experiment already showed is unreadable.)

    ``per_observation=True`` runs the whole student (trunk, memory, squeeze
    and head) on each observation on its own: the input is folded into its
    ``n_obs`` strips of ``tchans // n_obs`` rows, the student maps each strip
    to its own grid rows, and only the final output is unfolded back into
    observation order. Grid row k then depends on observation k only, in eval
    mode (in train mode BatchNorm still couples the strips through the batch
    statistics). Folding only the trunk would not be enough: the head's 3x3
    context convs would mix neighbouring observation rows of the grid. The
    memory attention ``att`` keeps the same ``(B * nh * nw, slots)`` row
    order, so the entropy term is unchanged. A ``per_observation_n_obs``
    buffer is registered only in this mode, so a strict ``load_state_dict``
    fails when a checkpoint and a model config disagree on the mode, in
    either direction, instead of silently loading the weights into the other
    architecture (the conv weights have the same shapes in both).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        filters,
        latent_dim: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        use_batchnorm: bool = True,
        convs_per_block: int = 2,
        memory: bool = False,
        mem_slots: int = 500,
        shrink_threshold: Optional[float] = None,
        head_hidden: int = 128,
        head_context_convs: int = 2,
        squeeze_shape: Optional[Tuple[int, int]] = None,
        squeeze_channels: Optional[int] = None,
        per_observation: bool = False,
        n_obs: int = 6,
    ):
        super().__init__()
        self.per_observation = per_observation
        self.n_obs = n_obs
        if per_observation:
            factor = 2 ** len(list(filters))
            th = input_shape[0]
            if th % n_obs or (th // n_obs) % factor:
                raise ValueError(
                    f"per_observation student: tchans {th} must split into {n_obs} "
                    f"observations of a multiple of {factor} rows (2 ** {len(list(filters))} blocks)."
                )
            # every shape below (trunk, squeeze grid check) is the strip's
            input_shape = (th // n_obs,) + tuple(input_shape[1:])
            self.register_buffer("per_observation_n_obs", torch.tensor(n_obs))
        self.trunk = build_encoder(
            input_shape=input_shape,
            filters=filters,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            convs_per_block=convs_per_block,
            variational=False,
        )
        self.memory = MemoryUnit(mem_slots, latent_dim, shrink_threshold=shrink_threshold) if memory else None

        self.squeeze = None
        if squeeze_shape is not None:
            sh, sw = int(squeeze_shape[0]), int(squeeze_shape[1])
            sc = int(squeeze_channels if squeeze_channels is not None else latent_dim)
            factor = 2 ** len(list(filters))
            gh, gw = input_shape[0] // factor, input_shape[1] // factor
            if gh % sh or gw % sw:
                raise ValueError(
                    f"student.bottleneck.squeeze.shape {(sh, sw)} must divide the trunk "
                    f"grid {(gh, gw)} (input {input_shape[:2]} / {factor}); the squeeze "
                    f"uses non-overlapping kernel=stride=({gh // sh},{gw // sw})."
                )
            ks = (gh // sh, gw // sw)
            self.squeeze = nn.Sequential(
                nn.Conv2d(latent_dim, sc, kernel_size=ks, stride=ks),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(sc, latent_dim, kernel_size=ks, stride=ks),
                nn.ReLU(inplace=True),
            )

        head_layers = []
        prev = latent_dim
        for _ in range(head_context_convs):
            head_layers += [nn.Conv2d(prev, head_hidden, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2)), nn.ReLU(inplace=True)]
            prev = head_hidden
        head_layers.append(nn.Conv2d(prev, out_channels, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor):
        if self.per_observation:
            x = fold_observations(x, self.n_obs)
        z = self.trunk(x)
        att = None
        if self.memory is not None:
            z, att = self.memory(z)
        if self.squeeze is not None:
            z = self.squeeze(z)
        out = self.head(z)
        if self.per_observation:
            out = unfold_rows(out, self.n_obs)
        return (out, att) if self.memory is not None else out


class UDMA(nn.Module):
    """Teacher-student distillation + memory ensemble, scored by feature-space
    disagreement (Q4/Q5).

    ``compute_loss`` trains the two students to regress ``Norm(T(x))``
    (``map_st1``/``map_st2``) while also minimising their mutual disagreement
    on normal data (``map_ss``); at inference, the SAME three squared-error
    maps become the anomaly signal — the AE-student can only copy through what
    it was trained on, the MemAE-student can only redraw learned normal
    prototypes, and neither can track token features it has never regressed
    correctly, so unseen morphology reopens a gap that training closed for
    normal data.
    """

    def __init__(
        self,
        teacher: TeacherViT,
        student_ae: FeatureStudent,
        student_mem: FeatureStudent,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        entropy_weight: float = 2.0e-4,
        score_weights: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        topk_frac: float = 0.02,
    ):
        super().__init__()
        self.teacher = teacher
        self.student_ae = student_ae
        self.student_mem = student_mem
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.entropy_weight = entropy_weight
        self.score_weights = score_weights
        self.topk_frac = topk_frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "UDMA has no pixel decoder/reconstruction — use anomaly_score(x, method=...) "
            "or anomaly_map(x), not model(x)."
        )

    def _forward_students(self, x: torch.Tensor):
        s_ae = self.student_ae(x)  # (B, 128, nh, nw), no memory
        s_mem, att = self.student_mem(x)  # memory=True -> (out, att)
        return s_ae, s_mem, att

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target = self.teacher(x)  # (B, 128, nh, nw), frozen, no grad by construction
        s_ae, s_mem, att = self._forward_students(x)

        st1 = F.mse_loss(s_ae, target)
        st2 = F.mse_loss(s_mem, target)
        ss = F.mse_loss(s_ae, s_mem)
        entropy = self.student_mem.memory.entropy(att)

        total = self.lambda1 * st1 + self.lambda2 * st2 + self.lambda3 * ss + self.entropy_weight * entropy
        return total, {
            "st1": st1.detach(),
            "st2": st2.detach(),
            "ss": ss.detach(),
            "entropy": entropy.detach(),
            # Early-stopping target (Q7): S-T regression quality only, excluding
            # ss/entropy so a degenerate student pair (both collapsed, agreeing
            # everywhere) can't look like convergence.
            "st_sum": (st1 + st2).detach(),
        }

    @torch.no_grad()
    def anomaly_map_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """The three disagreement maps (Q5) plus their fusion, each ``(B, nh, nw)``.

        Exposed separately from ``anomaly_map`` for qualitative inspection
        (``scripts/debug/udma_anomaly_maps.py``) — e.g. checking whether the
        teacher's global attention smears ``st1``/``st2`` into diffuse maps
        (the Q1 risk) even where the fused ``cob`` map still looks localised.
        """
        target = self.teacher(x)
        s_ae, s_mem, _ = self._forward_students(x)
        w1, w2, w3 = self.score_weights
        map_st1 = (target - s_ae).pow(2).mean(dim=1)
        map_st2 = (target - s_mem).pow(2).mean(dim=1)
        map_ss = (s_ae - s_mem).pow(2).mean(dim=1)
        map_cob = w1 * map_st1 + w2 * map_st2 + w3 * map_ss
        return {"st1": map_st1, "st2": map_st2, "ss": map_ss, "cob": map_cob}

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Fused disagreement map ``(B, nh, nw)`` (``map_cob``, Q5)."""
        return self.anomaly_map_components(x)["cob"]

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, method: str = "topk", topk_frac: Optional[float] = None, **kwargs) -> torch.Tensor:
        """Per-sample anomaly score ``(B,)`` from ``map_cob`` (Q5).

        ``method``: ``recon`` (mean over the (nh,nw) grid — duck-type name for
        harness compatibility, ``encode_separation_test.py --scoring recon``),
        ``topk`` (top-``topk_frac`` grid positions, default), or ``max``.
        """
        if method not in ("recon", "topk", "max"):
            raise ValueError(f"UDMA supports method='recon'/'topk'/'max', got '{method}'.")
        m = self.anomaly_map(x).flatten(1)  # (B, nh*nw)
        if method == "recon":
            return m.mean(dim=1)
        if method == "max":
            return m.max(dim=1).values
        frac = topk_frac if topk_frac is not None else self.topk_frac
        k = max(1, int(round(frac * m.shape[1])))
        return m.topk(k, dim=1).values.mean(dim=1)


def _load_teacher_vit(vit_config_path: Path, checkpoint_path: Path, input_shape: Tuple[int, int, int]) -> ViTMAE:
    with open(vit_config_path) as f:
        vit_config = yaml.safe_load(f)
    vit = build_vit_mae(input_shape, vit_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    vit.load_state_dict(state)
    return vit


def _load_teacher_cnn(teacher_cfg: Dict, input_shape: Tuple[int, int, int]) -> TeacherCNN:
    """Build a :class:`TeacherCNN` and load a trunk-only checkpoint saved by
    ``scripts/distill_teacher.py`` (``{"trunk_state_dict", "filters",
    "latent_dim", "convs_per_block", ...}`` — no distillation projection D).

    Architecture comes from the checkpoint itself (self-describing — the
    single source of truth for what was actually distilled), never silently
    from ``teacher_cfg``: a stale/mismatched config value could otherwise
    load a checkpoint's weights into the wrong architecture without any
    shape error (e.g. ``activation`` differs but produces identical tensor
    shapes). ``kernel_size``/``activation``/``use_batchnorm`` fall back to
    :class:`TeacherCNN`'s own defaults only for checkpoints saved before
    2026-07-16 (when those keys were added) — their true value was always
    that default; ``per_observation``/``n_obs`` likewise for checkpoints saved
    before 2026-11 (always a whole-frame trunk).
    """
    checkpoint_path = _ROOT / teacher_cfg["checkpoint"]
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    missing = [k for k in ("filters", "latent_dim", "convs_per_block") if k not in ckpt]
    if missing:
        raise KeyError(
            f"Distilled-teacher checkpoint {checkpoint_path} is missing required "
            f"architecture key(s) {missing} — was it saved by scripts/distill_teacher.py? "
            f"Re-run distillation."
        )
    teacher = TeacherCNN(
        input_shape,
        filters=list(ckpt["filters"]),
        latent_dim=int(ckpt["latent_dim"]),
        kernel_size=tuple(ckpt.get("kernel_size", (3, 3))),
        activation=ckpt.get("activation", "relu"),
        use_batchnorm=bool(ckpt.get("use_batchnorm", True)),
        convs_per_block=int(ckpt["convs_per_block"]),
        per_observation=bool(ckpt.get("per_observation", False)),
        n_obs=int(ckpt.get("n_obs", 6)),
    )
    teacher.trunk.load_state_dict(ckpt["trunk_state_dict"])
    return teacher


def build_udma(
    input_shape: Tuple[int, int, int],
    model_config: Dict,
) -> UDMA:
    """Build a :class:`UDMA` from a merged ``configs/model/udma.yaml``.

    Args:
        input_shape: ``(tchans, fchans, 1)``, must match the teacher's
            ``vit_config``'s ``patch_size`` divisibility (Q6: v1 = (96,1024)).
        model_config: parsed ``udma.yaml`` — ``teacher`` (checkpoint + vit
            config path + ``teacher_layer``), ``student`` (encoder section,
            reusing the ``convae.yaml`` schema), memory hyperparameters, loss
            lambdas, and scoring weights/``topk_frac``.
    """
    teacher_cfg = model_config["teacher"]
    teacher_type = teacher_cfg.get("type", "vit_mae")
    if teacher_type == "cnn_distilled":
        # Paper-faithful route (D6-D8): teacher distilled from an out-of-domain
        # generic backbone (scripts/distill_teacher.py), not self-supervised
        # in-domain. See docs/03_teacher-localization.md
        teacher = _load_teacher_cnn(teacher_cfg, input_shape)
    elif teacher_type == "vit_mae":
        vit_config_path = _ROOT / teacher_cfg["vit_config"]
        checkpoint_path = _ROOT / teacher_cfg["checkpoint"]
        teacher_layer = int(teacher_cfg.get("teacher_layer", 3))
        vit = _load_teacher_vit(vit_config_path, checkpoint_path, input_shape)
        teacher = TeacherViT(vit, teacher_layer=teacher_layer)
    else:
        raise ValueError(f"teacher.type must be 'vit_mae' or 'cnn_distilled', got '{teacher_type}'.")

    norm_stats_path = teacher_cfg.get("norm_stats")
    if norm_stats_path is not None:
        # Precomputed offline (Q2/Q7 one-shot): scripts/fit_udma_teacher_norm.py.
        # Without this, mu/sigma stay at their identity default (0/1) and
        # training regresses raw, unnormalized teacher features.
        stats = torch.load(str(_ROOT / norm_stats_path), map_location="cpu")
        teacher.mu.copy_(stats["mu"])
        teacher.sigma.copy_(stats["sigma"])

    student_cfg = model_config["student"]
    enc_cfg = student_cfg["encoder"]
    filters = list(enc_cfg["filters"])
    kernel_size = tuple(enc_cfg.get("kernel_size", (3, 3)))
    activation = enc_cfg.get("activation", "relu")
    use_batchnorm = enc_cfg.get("use_batchnorm", True)
    convs_per_block = int(enc_cfg.get("convs_per_block", 2))
    bottleneck_cfg = student_cfg["bottleneck"]
    latent_dim = int(bottleneck_cfg["latent_dim"])
    squeeze_cfg = bottleneck_cfg.get("squeeze") or {}
    squeeze_shape = squeeze_cfg.get("shape")
    squeeze_channels = squeeze_cfg.get("channels")
    head_hidden = int(student_cfg.get("head_hidden", 128))
    head_context_convs = int(student_cfg.get("head_context_convs", 2))
    mem_slots = int(student_cfg.get("mem_slots", 500))
    shrink_threshold = student_cfg.get("shrink_threshold", None)
    # Students confined to one observation each (phase 06a, 2026-11-26): with
    # a confined teacher too, no map row can depend on another observation.
    students_per_obs = bool(student_cfg.get("per_observation", False))
    n_obs = int(getattr(teacher, "n_obs", 6)) if getattr(teacher, "per_observation", False) \
        else int(student_cfg.get("n_obs", 6))

    n_blocks = len(filters)
    factor = 2 ** n_blocks
    th, fw, _ = input_shape
    if th % factor or fw % factor:
        raise ValueError(
            f"Input spatial dims {(th, fw)} must be divisible by {factor} "
            f"(2 ** {n_blocks} student downsampling blocks)."
        )
    if students_per_obs and (th % n_obs or (th // n_obs) % factor):
        raise ValueError(
            f"student.per_observation: tchans {th} must split into {n_obs} observations "
            f"of a multiple of {factor} rows (2 ** {n_blocks} student downsampling blocks)."
        )
    student_grid = (th // factor, fw // factor)
    if student_grid != teacher.grid_size:
        raise ValueError(
            f"Student trunk grid {student_grid} (input {(th, fw)} / {factor}) must "
            f"match the teacher's patch grid {teacher.grid_size} (Q3) — the two feed "
            f"the same disagreement maps. Adjust student.encoder.filters or the ViT "
            f"patch_size so both downsample to the same (H', W')."
        )
    out_channels = teacher.channels

    def _make_student(memory: bool) -> FeatureStudent:
        return FeatureStudent(
            input_shape=input_shape,
            filters=filters,
            latent_dim=latent_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
            convs_per_block=convs_per_block,
            memory=memory,
            mem_slots=mem_slots,
            shrink_threshold=shrink_threshold,
            head_hidden=head_hidden,
            head_context_convs=head_context_convs,
            squeeze_shape=squeeze_shape,
            squeeze_channels=squeeze_channels,
            per_observation=students_per_obs,
            n_obs=n_obs,
        )

    student_ae = _make_student(memory=False)
    student_mem = _make_student(memory=True)

    loss_cfg = model_config.get("loss", {})
    score_cfg = model_config.get("scoring", {})

    return UDMA(
        teacher=teacher,
        student_ae=student_ae,
        student_mem=student_mem,
        lambda1=float(loss_cfg.get("lambda1", 1.0)),
        lambda2=float(loss_cfg.get("lambda2", 1.0)),
        lambda3=float(loss_cfg.get("lambda3", 1.0)),
        entropy_weight=float(loss_cfg.get("entropy_weight", 2.0e-4)),
        score_weights=tuple(score_cfg.get("weights", (0.5, 0.5, 0.5))),
        topk_frac=float(score_cfg.get("topk_frac", 0.02)),
    )
