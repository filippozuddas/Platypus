"""Distillation training loop: P (frozen, out-of-domain) -> T (trainable CNN)
+ D (discardable 1x1 channel-alignment head).

Business logic for ``scripts/distill_teacher.py`` (thin CLI wrapper) — Fase 2
of ``docs/02_udma-architecture.md §11, the paper-faithful
teacher route (Qi et al. 2024, Eq. 2 / Bergmann "Uninformed Students"): a
small CNN is trained to reproduce a frozen, out-of-domain generic backbone's
features on in-domain data.

Loss: ``L = ||D(T(x)) - P(x)||^2``, MSE over the full feature grid, no
normalization (raw feature values on both sides).

``per_observation=True`` trains T on one observation at a time (its trunk is
built for ``tchans // n_obs`` rows and applied to each observation of the
frame), so T's grid row ``k`` depends on observation ``k`` only. Pair it with
a P that is confined the same way (``ResNetTeacher(confine="obs")``): a T
confined by architecture cannot reproduce a target that mixes observations.
"""

import time
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import build_encoder
from src.models.observation_fold import PerObservation

__all__ = ["build_trainable_trunk", "distill_teacher"]


def _cuda_memory(device) -> str:
    """Live and peak CUDA memory for the log line (empty on CPU): a live value
    that climbs from one log line to the next means something is retained
    across steps."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return ""
    gib = 2 ** 30
    return (f"  cuda alloc={torch.cuda.memory_allocated(device) / gib:.2f} GiB"
            f" peak={torch.cuda.max_memory_allocated(device) / gib:.2f} GiB")


def build_trainable_trunk(
    input_shape: Tuple[int, int, int],
    filters,
    latent_dim: int,
    kernel_size: Tuple[int, int] = (3, 3),
    activation: str = "relu",
    use_batchnorm: bool = True,
    convs_per_block: int = 2,
):
    """T's trunk — same parametrisation :class:`~src.models.udma.TeacherCNN`
    uses to load it later, kept as a plain trainable :class:`~src.models.encoder.Encoder`
    here (``TeacherCNN`` itself freezes + pins eval mode, wrong for training)."""
    return build_encoder(
        input_shape=input_shape,
        filters=filters,
        latent_dim=latent_dim,
        kernel_size=kernel_size,
        activation=activation,
        use_batchnorm=use_batchnorm,
        convs_per_block=convs_per_block,
        variational=False,
    )


def distill_teacher(
    P: nn.Module,
    loader,
    input_shape: Tuple[int, int, int],
    device: str,
    filters,
    latent_dim: int,
    p_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),
    activation: str = "relu",
    use_batchnorm: bool = True,
    convs_per_block: int = 2,
    epochs: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    log_every_n_steps: int = 50,
    log_fn: Callable[[str], None] = print,
    per_observation: bool = False,
    n_obs: int = 6,
) -> Tuple[nn.Module, List[float]]:
    """Train T (+ a discardable D) to reproduce frozen ``P``'s features.

    Args:
        P: frozen teacher candidate, callable as ``P(x) -> (B, p_channels, nh, nw)``.
        loader: yields ``(B, 1, tchans, fchans)`` batches (no labels).
        input_shape: ``(tchans, fchans, 1)`` — used only to validate T's grid
            matches ``P.grid_size`` before training starts.
        p_channels: ``P``'s output channel count (``D`` projects T's
            ``latent_dim`` up to this for the loss only).
        per_observation: train T's trunk on single observations (module
            docstring). The returned trunk is the per-observation one, to be
            loaded with ``TeacherCNN(per_observation=True)``.

    Returns:
        ``(T, losses)`` — the trained trunk (``D`` is discarded) and the
        per-step loss history (for a curve plot / sanity check).

    Raises:
        ValueError: if T's trunk grid (from ``input_shape``/``filters``)
            doesn't match ``P.grid_size``.
        RuntimeError: if ``loader`` yields zero batches (e.g. ``batch_size``
            larger than the dataset with ``drop_last=True``) — training would
            otherwise silently produce an untrained, unsaved T.
    """
    n_blocks = len(filters)
    factor = 2 ** n_blocks
    th, fw, c = input_shape
    t_grid = (th // factor, fw // factor)
    if t_grid != P.grid_size:
        raise ValueError(
            f"T's trunk grid {t_grid} (input {input_shape[:2]} / {factor}) must match "
            f"P's grid {P.grid_size} -- adjust filters."
        )
    if per_observation and (th % n_obs or (th // n_obs) % factor):
        raise ValueError(
            f"per_observation: tchans {th} must split into {n_obs} observations of a "
            f"multiple of {factor} rows."
        )

    trunk_shape = (th // n_obs, fw, c) if per_observation else input_shape
    T = build_trainable_trunk(
        trunk_shape, filters, latent_dim, kernel_size, activation, use_batchnorm, convs_per_block,
    ).to(device)
    T_frame = PerObservation(T, n_obs) if per_observation else T
    D = nn.Conv2d(latent_dim, p_channels, 1).to(device)

    optimizer = torch.optim.AdamW(
        list(T.parameters()) + list(D.parameters()), lr=lr, weight_decay=weight_decay,
    )

    # Accumulate detached GPU scalars (no host sync) rather than calling
    # .item() every step -- forcing a CUDA device-to-host sync on every
    # iteration of the hot loop stalls the async queue. .item() is only
    # called at log_every_n_steps and once, batched, at the very end.
    loss_tensors: List[torch.Tensor] = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)
            with torch.no_grad():
                target = P(x)

            pred = D(T_frame(x))
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clone: the scalar from F.mse_loss keeps the whole (pred - target)
            # buffer as storage (48 MiB at batch 128), and detach() shares it.
            loss_tensors.append(loss.detach().clone())
            if step % log_every_n_steps == 0:
                elapsed = time.time() - t0
                log_fn(f"  epoch {epoch} step {step}  loss={loss.item():.4f}  "
                       f"({elapsed:.0f}s elapsed){_cuda_memory(device)}")
            step += 1

        if not loss_tensors:
            raise RuntimeError(
                "Distillation loader yielded zero batches -- check --batch_size against "
                "the dataset size (drop_last=True silently drops an only/too-small batch)."
            )
        log_fn(f"Epoch {epoch} done, {step} steps, last loss={loss_tensors[-1].item():.4f}")

    losses = torch.stack(loss_tensors).cpu().tolist()
    return T, losses
