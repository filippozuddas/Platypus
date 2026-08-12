"""Slide-5 figure: one row per chosen morphology, example | recovery curve.

Built for the final presentation per the 2026-08-11 mentor review: no
before/after comparison, no legend clutter — one signal example (cropped to
where the injector actually put power) next to its detection-only curve
(blue "detected by the model" line, `shipped` arm only — see
CLAUDE.md "only shipped-arm numbers, not fixed").

Reuses the exact rendering/scoring code from the two scripts this project
already trusts, rather than reimplementing either:
  * `preview_morphology_templates.py` for the injected-signal example panel
    (`injected_block`, `draw_block`, `bounding_box`)
  * `plot_sensitivity_curves.py` for the recovery curve (`series`,
    `rate_with_ci`, `style_axis`, the `s0_sigma3` "detected by the model"
    column, the `shipped` arm's threshold column)

Usage:
    PYTHONPATH=. python scripts/debug/plot_slide5_panels.py \
        --csv outputs/sensitivity/morphologies_v7/pipeline_sensitivity.csv \
        --out outputs/sensitivity/morphologies_v7/figures/slide5_panels.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "debug"))

from preview_morphology_templates import injected_block, draw_block, bounding_box  # noqa: E402
from plot_sensitivity_curves import series, style_axis, INK, INK_SOFT  # noqa: E402

DEFAULT_MORPHS = [
    "narrowband_drift", "narrowband_pulsed", "wideband_continuous",
    "wideband_pulsed", "smiley_face", "random_2d",
]

MODEL_COLOR = "#2a78d6"  # same blue as "detected by the model" in plot_sensitivity_curves


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True,
                   help="pipeline_sensitivity.py output CSV (shipped arm columns).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--morphologies", nargs="+", default=DEFAULT_MORPHS,
                   help="Max 6, in display order top-to-bottom.")
    p.add_argument("--example_seed", type=int, default=0,
                   help="Which injection-preview seed to render as the example.")
    p.add_argument("--example_snr", type=float, default=50.0,
                   help="High SNR for the example panel — this is a geometry "
                        "illustration, not a detectability check (that's the "
                        "curve's job).")
    p.add_argument("--noise_mean", type=float, default=10.0)
    p.add_argument("--noise_std", type=float, default=0.35)
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.morphologies) > 6:
        raise SystemExit(f"slide 5 takes at most 6 classes, got {len(args.morphologies)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    missing = set(args.morphologies) - set(df["morphology"].unique())
    if missing:
        raise SystemExit(f"not in CSV: {sorted(missing)}")

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    frame = data_cfg["frame"]
    n_obs, tchans_per_obs, fchans = 6, frame["tchans"], frame["fchans"]
    geom = (n_obs, tchans_per_obs, fchans)
    noise = (args.noise_mean, args.noise_std)

    n = len(args.morphologies)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 2.35 * n),
                             gridspec_kw={"width_ratios": [1, 1.55]})
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, morph in enumerate(args.morphologies):
        # --- left: example spectrogram, cropped to the injected region ---
        block, excess, _ = injected_block(morph, data_cfg, args.example_seed,
                                          args.example_snr, geom, noise)
        ax_ex = axes[row][0]
        box = bounding_box(excess)
        if box is not None:
            r0, r1, c0, c1 = box
            draw_block(ax_ex, block[r0:r1, c0:c1], n_obs, tchans_per_obs)
            for i in range(1, n_obs):
                y = i * tchans_per_obs - 0.5 - r0
                if 0 <= y <= (r1 - r0):
                    ax_ex.axhline(y, color="white", linewidth=0.8, alpha=0.8)
        else:
            draw_block(ax_ex, block, n_obs, tchans_per_obs)
        ax_ex.set_ylabel(morph.replace("_", " "), color=INK, fontsize=11,
                         rotation=0, ha="right", va="center", labelpad=10)

        # --- right: detection-only curve, shipped arm, no legend ---
        sub = df[df.morphology == morph]
        snrs = sorted(sub["snr"].unique())
        model, lo, hi = series(sub, snrs, lambda d: d["s0_sigma3"])
        ax_c = axes[row][1]
        ax_c.fill_between(snrs, lo, hi, color=MODEL_COLOR, alpha=0.15, linewidth=0)
        ax_c.plot(snrs, model, color=MODEL_COLOR, linewidth=2.4, marker="o",
                  markersize=5.5, markeredgecolor="white", markeredgewidth=1.0)
        ax_c.annotate(f"{model[-1]:.0f}%", xy=(snrs[-1], model[-1]),
                     xytext=(6, 0), textcoords="offset points",
                     color=MODEL_COLOR, fontsize=10.5, va="center", fontweight="bold")
        style_axis(ax_c, "injected SNR" if row == n - 1 else "",
                  "% detected" if row == 0 else "")
        ax_c.set_xlim(min(snrs) - 1, max(snrs) + 8)
        ax_c.set_ylim(-4, 104)

    fig.suptitle("The model detects far more than classic narrowband drift",
                color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
