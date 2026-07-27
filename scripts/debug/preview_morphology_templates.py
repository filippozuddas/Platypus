"""Draw what the injectors actually inject — before spending hours on a sweep.

A sensitivity sweep reports numbers, not pictures, so a morphology that is
mis-sized, clipped or invisible produces a perfectly clean-looking low detection
rate. That is not a hypothetical: the first version of `smiley_face` sized its
template to ONE observation (16 time bins), which is exactly one row of the
(6, 64) anomaly map — the sweep then measured frequency extent while reporting
"shape recovery", and the mistake was only caught by eye afterwards.

So: render every morphology on a flat synthetic background, plot it, look at it.
No model, no checkpoint, no real data, no GPU — seconds on a laptop.

Two figures:

* **gallery** — one row per morphology, several seeds across the columns, each
  panel the full (96, 1024) cadence block with observation dividers. This is the
  figure that answers "does it look like the thing I named it".
* **zoom** — the same signals cropped to the injected region, where a shape that
  covers 3% of the band is actually legible.

The background is flat gaussian noise, so anything visible is injected. Amplitude
uses each morphology's real SNR convention, so the relative faintness of the
energy-matched canvas shapes against a narrowband carrier is shown honestly
rather than normalised away per panel.

Usage:
    PYTHONPATH=. python scripts/debug/preview_morphology_templates.py \
        --out_dir outputs/sweeps/morphology_preview
    PYTHONPATH=. python scripts/debug/preview_morphology_templates.py \
        --morphologies smiley_face random_2d --n_seeds 6 --snr 50
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.morphologies import MORPHOLOGIES, build_morphology

INK = "#0b0b0b"
INK_SOFT = "#52514e"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/sweeps/morphology_preview")
    p.add_argument("--morphologies", nargs="+", default=list(MORPHOLOGIES),
                   choices=list(MORPHOLOGIES))
    p.add_argument("--n_seeds", type=int, default=4,
                   help="Sites drawn per morphology — the parameter spread matters "
                        "as much as any single draw.")
    p.add_argument("--snr", type=float, default=50.0,
                   help="High by default: this checks geometry, not detectability.")
    p.add_argument("--noise_mean", type=float, default=10.0)
    p.add_argument("--noise_std", type=float, default=0.35)
    p.add_argument("--seed0", type=int, default=0)
    return p.parse_args()


def flat_background(n_obs, tchans_per_obs, fchans, mean, std, seed):
    """Flat gaussian noise: every visible feature is injected, by construction."""
    rng = np.random.default_rng(10_000 + seed)
    return rng.normal(mean, std, size=(n_obs, tchans_per_obs, fchans))


def injected_block(name, data_cfg, seed, snr, geom, noise):
    """One (total_tchans, fchans) block, plus the site metadata that made it."""
    n_obs, tchans_per_obs, fchans = geom
    bg = flat_background(n_obs, tchans_per_obs, fchans, *noise, seed)
    inj = build_morphology(name, data_cfg, seed=seed)
    site = inj.sample_site(fchans, n_obs * tchans_per_obs, n_obs)
    out, info = inj.inject(bg, site, snr)
    excess = out - bg.astype(np.float32)
    return (np.concatenate(list(out), axis=0),
            np.concatenate(list(excess), axis=0), info)


def draw_block(ax, block, n_obs, tchans_per_obs, title=None, extent=None):
    ax.imshow(block, aspect="auto", origin="upper", cmap="viridis",
              interpolation="nearest", extent=extent)
    for i in range(1, n_obs):
        y = i * tchans_per_obs - 0.5
        if extent is None:
            ax.axhline(y, color="white", linewidth=0.9, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, color=INK_SOFT, fontsize=8, pad=3)


def bounding_box(excess, pad_cols=12, pad_rows=2, floor=0.02):
    """Rows/cols where the injection actually put power, padded for context."""
    mask = excess > floor * excess.max() if excess.max() > 0 else excess > np.inf
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = max(0, rows[0] - pad_rows), min(excess.shape[0], rows[-1] + pad_rows + 1)
    c0, c1 = max(0, cols[0] - pad_cols), min(excess.shape[1], cols[-1] + pad_cols + 1)
    return r0, r1, c0, c1


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    frame = data_cfg["frame"]
    fchans = frame["fchans"]
    tchans_per_obs = frame["tchans"]
    n_obs = 6
    geom = (n_obs, tchans_per_obs, fchans)
    noise = (args.noise_mean, args.noise_std)
    print(f"canvas: {n_obs} x {tchans_per_obs} x {fchans}  "
          f"(total_tchans={n_obs * tchans_per_obs})  snr={args.snr:g}")

    morphs = args.morphologies
    seeds = [args.seed0 + i for i in range(args.n_seeds)]

    blocks = {}
    for name in morphs:
        for seed in seeds:
            blocks[(name, seed)] = injected_block(name, data_cfg, seed, args.snr,
                                                  geom, noise)

    for figname, zoom in (("gallery", False), ("zoom", True)):
        fig, axes = plt.subplots(len(morphs), len(seeds),
                                 figsize=(3.0 * len(seeds), 1.7 * len(morphs)),
                                 squeeze=False)
        for r, name in enumerate(morphs):
            for c, seed in enumerate(seeds):
                block, excess, info = blocks[(name, seed)]
                ax = axes[r][c]
                if zoom:
                    box = bounding_box(excess)
                    if box is None:
                        ax.set_visible(False)
                        continue
                    r0, r1, c0, c1 = box
                    draw_block(ax, block[r0:r1, c0:c1], n_obs, tchans_per_obs,
                               title=f"rows {r0}-{r1}, chans {c0}-{c1}")
                    # Dividers in cropped coordinates: a canvas shape must be seen
                    # crossing them, which is the whole point of the change.
                    for i in range(1, n_obs):
                        y = i * tchans_per_obs - 0.5 - r0
                        if 0 <= y <= (r1 - r0):
                            ax.axhline(y, color="white", linewidth=0.9, alpha=0.85)
                else:
                    draw_block(ax, block, n_obs, tchans_per_obs,
                               title=f"seed {seed}")
                if c == 0:
                    ax.set_ylabel(name, color=INK, fontsize=9, rotation=0,
                                  ha="right", va="center", labelpad=8)
        title = ("What each injector puts in the data — cropped to the signal"
                 if zoom else
                 "What each injector puts in the data — full cadence block, "
                 "white lines are observation boundaries")
        fig.suptitle(f"{title}   (SNR {args.snr:g}, flat noise)",
                     color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "pdf"):
            fig.savefig(args.out_dir / f"morphology_preview_{figname}.{ext}",
                        dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {figname}: {args.out_dir / f'morphology_preview_{figname}.png'}")

    # Geometry table: the numbers to sanity-check before committing to a sweep.
    print(f"\n{'morphology':<22} {'seed':>4} {'rows':>10} {'chans':>12} "
          f"{'map cells':>10} {'peak':>8} {'obs hit':>8}")
    for name in morphs:
        for seed in seeds:
            block, excess, info = blocks[(name, seed)]
            box = bounding_box(excess, pad_cols=0, pad_rows=0)
            if box is None:
                print(f"{name:<22} {seed:>4}   NOTHING INJECTED")
                continue
            r0, r1, c0, c1 = box
            n_hit = sum(bool(excess[i * tchans_per_obs:(i + 1) * tchans_per_obs].max() > 0)
                        for i in range(n_obs))
            # One anomaly-map cell is 16 rows x 16 channels for the (6,64) map.
            cells = f"{int(np.ceil((r1 - r0) / 16))}x{int(np.ceil((c1 - c0) / 16))}"
            print(f"{name:<22} {seed:>4} {f'{r0}-{r1}':>10} {f'{c0}-{c1}':>12} "
                  f"{cells:>10} {excess.max():>8.3f} {n_hit:>8}")


if __name__ == "__main__":
    main()
