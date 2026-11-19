"""ResNet-18 (ImageNet, frozen) — "P" in docs/03_teacher-localization.md: the out-of-domain generic backbone the UDMA paper (Qi et al. 2024, Eq. 2) distills
its teacher CNN from. Read at ``layer3`` (256 ch, stride /16 -> native (6,64) grid on a
(96,1024) input) so its token grid matches the ViT-MAE teacher's without any downstream
architecture change.

Exposes the minimal interface ``scripts/debug/teacher_sensitivity_test.py``'s
``encode_tokens_layer()`` expects from a ViT-MAE (``patch_embed``, ``pos_embed``,
``encoder.layers``, ``encoder.norm``) so the SAME gate (G1-G3 thresholds) runs unmodified on
this backbone. P has no transformer blocks of its own — the entire conv stack through
``layer3`` is the ``patch_embed``, and ``encoder.layers`` is empty (nothing further to
iterate).

``confine`` selects what each grid cell is allowed to see:

    none  the whole frame (the July 2026 distillation target). layer3's
          vertical receptive field spans every observation, and row ``n`` is
          centred on the first pixel row of observation ``n``.
    obs   P is applied to each observation on its own (16 x 1024 strips): the
          cell sees its own observation only, full context along frequency.
    tile  P is applied to a 16 x ``tile_width`` crop centred on the cell and
          pooled to one vector — a patch descriptor in the sense of Bergmann
          et al. 2020; confined along frequency as well.

With ``obs`` and ``tile`` an ON-only signal cannot move an OFF row by
construction.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.models.observation_fold import fold_observations, unfold_rows

__all__ = ["ResNetTeacher", "CONFINE_MODES"]

CONFINE_MODES = ("none", "obs", "tile")
_STRIDE = 16  # layer3 downsampling, and the grid cell size in pixels


class _EmptyEncoder(nn.Module):
    """Stand-in for ViTMAE's transformer encoder: zero blocks, no final norm —
    P's entire feature extraction happens in ``patch_embed``."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = None


class ResNetTeacher(nn.Module):
    """Frozen ResNet-18 (ImageNet-pretrained), read at ``layer3``.

    Input: ``(B, 1, 96, 1024)`` standardized snippets (this repo's median/MAD
    ``core_transform``) — replicated to 3 channels, with NO ImageNet mean/std
    renormalization. That is a deliberate simplification the gate itself is
    meant to validate (docs/02_udma-architecture.md §11),
    not an oversight.

    Args:
        confine: ``none`` | ``obs`` | ``tile`` (module docstring).
        n_obs: observations stacked along time (rows of the output grid when
            confined).
        tile_width: crop width in channels for ``tile``; the 16-px cell plus
            equal context on both sides. Frame edges are zero-padded, which is
            the median level after standardization.
        tile_chunk: crops per forward for ``tile`` (memory bound only).
        pretrained: ``False`` builds random weights — for geometry tests, which
            do not depend on the weights and must not download anything.
    """

    grid_size = (6, 64)
    channels = 256

    def __init__(self, confine: str = "none", n_obs: int = 6, tile_width: int = 64,
                 tile_chunk: int = 4096, pretrained: bool = True):
        super().__init__()
        if confine not in CONFINE_MODES:
            raise ValueError(f"confine must be one of {CONFINE_MODES}, got {confine!r}")
        if (tile_width - _STRIDE) % 2 or tile_width < _STRIDE or tile_width % _STRIDE:
            raise ValueError(f"tile_width must be a multiple of {_STRIDE} with even context, "
                             f"got {tile_width}")
        self.confine = confine
        self.n_obs = n_obs
        self.tile_width = tile_width
        self.tile_chunk = tile_chunk
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        for p in self.stem.parameters():
            p.requires_grad_(False)
        # No positional embedding: conv receptive fields already encode
        # position, unlike a ViT patch sequence. Plain float broadcasts fine
        # against the token tensor in encode_tokens_layer's `+ pos_embed`.
        self.pos_embed = 0.0
        self.encoder = _EmptyEncoder()
        self.eval()

    def train(self, mode: bool = True) -> "ResNetTeacher":
        return super().train(False)

    @torch.no_grad()
    def patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, 96, 1024) -> (B, nh*nw, channels)`` token sequence, row-major
        over ``(nh, nw)`` — matches ``patch_pixels()``'s patch ordering."""
        feat = self.forward(x)
        b, c, nh, nw = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(b, nh * nw, c)

    def _layer3(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x.repeat(1, 3, 1, 1))

    def _tiles(self, strips: torch.Tensor) -> torch.Tensor:
        """``(N, 1, hs, W) -> (N, channels, 1, W // 16)``: one pooled descriptor
        per cell, each computed from its own ``hs x tile_width`` crop."""
        n, _, hs, w = strips.shape
        pad = (self.tile_width - _STRIDE) // 2
        crops = F.pad(strips, (pad, pad)).unfold(3, self.tile_width, _STRIDE)  # (N,1,hs,nw,tw)
        nw = crops.shape[3]
        crops = crops.permute(0, 3, 1, 2, 4).reshape(n * nw, 1, hs, self.tile_width)
        out = [self._layer3(crops[i:i + self.tile_chunk]).mean(dim=(2, 3))
               for i in range(0, len(crops), self.tile_chunk)]
        return torch.cat(out).reshape(n, nw, self.channels).permute(0, 2, 1).unsqueeze(2)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, 96, 1024) -> (B, channels, nh, nw)`` spatial feature grid —
        the distillation target for ``scripts/distill_teacher.py``."""
        if self.confine == "none":
            feat = self._layer3(x)
        else:
            strips = fold_observations(x, self.n_obs)
            per_obs = self._layer3(strips) if self.confine == "obs" else self._tiles(strips)
            feat = unfold_rows(per_obs, self.n_obs)
        expected = (x.shape[2] // _STRIDE, x.shape[3] // _STRIDE)
        assert tuple(feat.shape[-2:]) == expected, (
            f"expected grid {expected} from a {tuple(x.shape[-2:])} input, "
            f"got {tuple(feat.shape[-2:])}"
        )
        return feat
