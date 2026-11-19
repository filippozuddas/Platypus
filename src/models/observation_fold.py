"""Fold a stacked cadence frame into its observations, and back.

The ``0000`` input stacks ``n_obs`` observations along the time axis. A network
applied to the folded batch sees one observation per sample, so no output can
depend on pixels of another observation — the property a teacher (and its
distillation target) needs for the ON/OFF logic downstream, where a feature row
labelled OFF must not move when only ON pixels change.
"""

import torch

__all__ = ["fold_observations", "unfold_rows"]


def fold_observations(x: torch.Tensor, n_obs: int = 6) -> torch.Tensor:
    """``(B, C, H, W) -> (B * n_obs, C, H // n_obs, W)``.

    Observation ``k`` of sample ``b`` lands at batch index ``b * n_obs + k``.
    """
    b, c, h, w = x.shape
    if h % n_obs:
        raise ValueError(f"height {h} is not a multiple of n_obs={n_obs}")
    hs = h // n_obs
    return x.reshape(b, c, n_obs, hs, w).permute(0, 2, 1, 3, 4).reshape(b * n_obs, c, hs, w)


def unfold_rows(f: torch.Tensor, n_obs: int = 6) -> torch.Tensor:
    """``(B * n_obs, C, nh, nw) -> (B, C, n_obs * nh, nw)``, inverse of
    :func:`fold_observations` on the feature grid: the ``nh`` rows of
    observation ``k`` become rows ``k * nh .. (k + 1) * nh - 1``."""
    bn, c, nh, nw = f.shape
    if bn % n_obs:
        raise ValueError(f"batch {bn} is not a multiple of n_obs={n_obs}")
    b = bn // n_obs
    return f.reshape(b, n_obs, c, nh, nw).permute(0, 2, 1, 3, 4).reshape(b, c, n_obs * nh, nw)
