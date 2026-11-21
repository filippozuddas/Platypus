"""One-shot: fit the UDMA teacher's per-channel token normalization (Q2/Q7).

Checklist step 3 (docs/02_udma-architecture.md) — MUST run before any
UDMA training launch. ``TeacherViT.mu``/``sigma`` default to identity (0/1);
without this step the students would regress raw, unnormalized teacher
features, which violates Q2 and invalidates the run.

Usage (server, not dev machine):
    PYTHONPATH=/path/to/Platypus python scripts/fit_udma_teacher_norm.py \\
        --config configs/training/udma_gbt_fine.yaml \\
        --out outputs/udma_teacher_norm/gbt_fine_block3.pt

Then point ``configs/model/udma.yaml``'s ``teacher.norm_stats`` at ``--out``
before starting ``scripts/train.py``.

Any ``teacher.norm_stats`` already in the config is ignored here (the fit uses
the raw, pre-normalization teacher features), so the config can name the file
this run is about to create. An existing ``--out`` is never overwritten; a
provenance record (data identity, teacher checkpoint hash, git state) is
written next to it as ``<out>.json``.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import build_datasets
from src.models.autoencoder import build_autoencoder
from src.utils.config import load_config
from src.utils.provenance import write_provenance

# Stop rule for the cnn_obs cycle (phase 06a, approved 2026-11-21): the
# teacher divides by sigma, so a channel far below the others would dominate
# the students' loss and the anomaly map. The phase-04 teacher had sigma
# 0.98-2.05, a max/min ratio of ~2.
SIGMA_REL_FLOOR = 0.01


def sigma_check(sigma: torch.Tensor, rel_floor: float = SIGMA_REL_FLOOR) -> tuple[bool, list[str]]:
    """Per-channel sigma summary. ``ok`` is False when any channel has
    sigma below ``rel_floor`` times the median; the stats are still saved,
    the decision on the next step is the user's."""
    s = sigma.detach().double().cpu().flatten()
    median = s.median().item()
    floor = rel_floor * median
    low = (s < floor).nonzero().flatten().tolist()
    lines = [
        f"sigma per channel: min {s.min().item():.4g}  median {median:.4g}  max {s.max().item():.4g}"
        f"  (max/min {s.max().item() / max(s.min().item(), 1e-30):.3g})",
        f"channels below {rel_floor:g} x median ({floor:.4g}): {len(low)}/{s.numel()}"
        + (f"  -> {low}" if low else ""),
    ]
    ok = not low
    lines.append("SIGMA CHECK: PASS" if ok else
                 "SIGMA CHECK: STOP - do not train the students; decide first")
    return ok, lines


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/training/udma_gbt_fine.yaml"),
                    help="Training config referencing configs/model/udma.yaml")
    p.add_argument("--out", type=Path, required=True, help="Output path for {mu, sigma} tensors")
    p.add_argument("--max_batches", type=int, default=None,
                    help="Cap the number of train batches (debugging only; full train set by default)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; choose another --out")
    cfg = load_config(args.config)
    cfg["model"].get("teacher", {}).pop("norm_stats", None)
    architecture = cfg["model"].get("architecture")
    if architecture != "udma":
        raise SystemExit(f"--config's model must be architecture: udma, got '{architecture}'.")

    data_cfg = cfg["data"]
    cadence_list_path = Path(data_cfg["dataset"]["cadence_list"])
    cadence_list = [
        line.strip().split()
        for line in cadence_list_path.read_text().splitlines()
        if line.strip()
    ]
    train_ds, val_ds = build_datasets(
        cadence_list, data_cfg, val_fraction=0.15, seed=cfg["training"]["seed"],
    )
    if hasattr(val_ds, "close"):
        val_ds.close()
    print(f"Fitting teacher normalization on {len(train_ds)} train snippets.")

    loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["hardware"].get("num_workers", 4),
    )

    frame = data_cfg["frame"]
    input_shape = (frame["tchans"], frame["fchans"], 1)
    model = build_autoencoder(
        input_shape=input_shape,
        model_config=cfg["model"],
        loss=cfg["training"]["loss"],
        learning_rate=cfg["training"]["learning_rate"],
    )
    model.to(args.device)

    teacher_layer = getattr(model.teacher, "teacher_layer", None)
    layer_info = f"layer {teacher_layer}" if teacher_layer is not None else type(model.teacher).__name__
    print(f"Teacher: {layer_info}, channels: {model.teacher.channels}, grid: {model.teacher.grid_size}")
    model.teacher.fit_normalization(loader, args.device, max_batches=args.max_batches)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mu": model.teacher.mu.cpu(), "sigma": model.teacher.sigma.cpu()}, args.out)
    write_provenance(args.out.with_name(args.out.name + ".json"), cfg,
                     n_train_snippets=len(train_ds), max_batches=args.max_batches)
    print(f"Saved -> {args.out}")
    print(f"Add to configs/model/udma.yaml: teacher.norm_stats: {args.out}")
    for line in sigma_check(model.teacher.sigma)[1]:
        print(line)


if __name__ == "__main__":
    main()
