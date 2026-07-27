"""Detection curves from a pipeline_sensitivity run: vs SNR, and vs each
morphology's own governing parameter.

`pipeline_sensitivity.py` prints one pooled survival table per SNR. That answers
"how complete is the search at SNR X" but not "complete to what *kind* of
signal" — and the sweep already carries the answer, because every site's
morphology parameters are written to the CSV with a `sig_` prefix. Nothing here
needs a rerun: it reads a finished CSV, including `morphologies_v2`'s.

Three figures:

* **cascade** — stage-by-stage survival vs SNR, one panel per morphology. The
  gap between two adjacent lines is what that pipeline stage costs, so a
  morphology that dies at `full_row_hits` looks different from one the scorer
  never saw. This is the figure the sensitivity statement comes from.
* **params** — survival vs each morphology's own variable (pulse period, duty
  cycle, band excursion, shape size, ...), one line per SNR.
* **factorial** — the (frequency extent) x (time structure) 2x2, colour for
  extent and dash for time structure, so a pulsed-signal deficit can be
  attributed to the pulsing rather than to the bandwidth.

**The parameter curves are observational, not a controlled sweep.** Morphology
parameters are sampled per site, not gridded, so a bin's occupants differ in
every other parameter too, and the bins are unbalanced. Read them as "where does
sensitivity fall off", not as an isolated causal effect of one knob. A
controlled version means gridding the parameter at fixed SNR, which is a
different (and more expensive) run.

Usage:
    python scripts/debug/plot_sensitivity_curves.py \
        --csv outputs/sensitivity/morphologies_v3/pipeline_sensitivity.csv \
        --out_dir outputs/sensitivity/morphologies_v3/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Ordinal ramp for the cascade: one hue, light -> dark as the funnel narrows.
# Steps 250..700 of the reference blue ramp; the lightest clears 2:1 on white,
# which a paler step would not.
CASCADE_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
# Same ramp reused for SNR, which is also an ordered magnitude, not an identity.
SNR_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Two categorical slots: frequency extent. Time structure rides on the dash
# pattern instead of a third and fourth hue.
EXTENT_COLORS = {"narrowband": "#2a78d6", "wideband": "#eb6834"}

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"

STAGES = [
    ("s0_sigma3", "scored > 3σ"),
    ("s1_far1pct", "FAR 1% cut"),
    ("s2_on_rows", "≥2 ON rows"),
    ("s3_short_list", "short list"),
    ("survives_pipeline", "reaches a human"),
]

# Per morphology, the variables that morphology actually samples. Columns absent
# from the CSV (an older run, a morphology not swept) are skipped silently — the
# script must stay usable on morphologies_v2, which predates half of these.
PARAM_SPECS = {
    "narrowband_drift": [
        ("drift_rate", "drift rate (Hz/s)"),
        ("sig_width", "linewidth (Hz)"),
    ],
    "narrowband_accel": [
        ("sig_excursion_frac", "band excursion (fraction)"),
        ("sig_accel_hz_s2", "acceleration (Hz/s²)"),
    ],
    "narrowband_sine": [
        ("sig_amplitude_frac", "amplitude (fraction of band)"),
        ("sig_n_cycles", "cycles per cadence"),
        ("sig_peak_drift", "peak drift (Hz/s)"),
    ],
    "narrowband_pulsed": [
        ("sig_period_s", "pulse period (s)"),
        ("sig_duty", "duty cycle"),
        ("sig_pulses_per_obs", "pulses per ON observation"),
    ],
    "wideband_pulsed": [
        ("sig_period_s", "pulse period (s)"),
        ("sig_duty", "duty cycle"),
        ("sig_width_frac", "bandwidth (fraction of band)"),
    ],
    "wideband_continuous": [
        ("sig_width_frac", "bandwidth (fraction of band)"),
    ],
    "smiley_face": [
        ("sig_r_row", "size (time radius, bins)"),
        ("sig_aspect", "aspect ratio (freq/time)"),
        ("sig_mouth_curvature", "mouth curvature"),
    ],
    "random_2d": [
        ("sig_r_row", "size (time radius, bins)"),
        ("sig_aspect", "aspect ratio (freq/time)"),
        ("sig_n_modes", "Fourier modes"),
        ("sig_stroke_sigma", "edge softness (rows)"),
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, nargs="+", required=True,
                   help="One or more pipeline_sensitivity.csv files, concatenated.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--metric", default="survives_pipeline",
                   help="Column plotted against the morphology parameters.")
    p.add_argument("--n_bins", type=int, default=5,
                   help="Quantile bins per numeric parameter.")
    p.add_argument("--min_n", type=int, default=30,
                   help="Bins with fewer injections are dropped, not plotted: a "
                        "detection rate over <30 trials has a CI wider than the "
                        "effect any of these curves is trying to show.")
    return p.parse_args()


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval — correct at the 0% and 100% ends, unlike normal.

    Detection rates here routinely sit at 0 or 1 (SNR=0 control, SNR=50 tail),
    where a normal-approximation interval has zero width and would draw a
    spurious certainty.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def rate_with_ci(series: pd.Series):
    vals = series.astype(bool)
    return wilson(int(vals.sum()), int(len(vals)))


def style_axis(ax, xlabel, ylabel=None, title=None):
    ax.set_ylim(-3, 103)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def grid_of_panels(n, ncols=3, panel=(3.6, 2.8)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def plot_cascade(df, out_dir):
    """Stage-by-stage survival vs SNR, one panel per morphology."""
    morphs = sorted(df["morphology"].unique())
    fig, axes = grid_of_panels(len(morphs))
    records = []

    for ax, morph in zip(axes, morphs):
        sub = df[df.morphology == morph]
        snrs = sorted(sub["snr"].unique())
        for (col, label), colour in zip(STAGES, CASCADE_COLORS):
            if col not in sub.columns:
                continue
            pts = [rate_with_ci(sub[sub.snr == s][col]) for s in snrs]
            rate = [100 * p for p, _, _ in pts]
            lo = [100 * l for _, l, _ in pts]
            hi = [100 * h for _, _, h in pts]
            ax.fill_between(snrs, lo, hi, color=colour, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(snrs, rate, color=colour, linewidth=2.0, marker="o",
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.8,
                    label=label, zorder=3)
            records += [{"figure": "cascade", "morphology": morph, "snr": s,
                         "stage": col, "rate_pct": r, "ci_lo_pct": a, "ci_hi_pct": b,
                         "n": int((sub.snr == s).sum())}
                        for s, r, a, b in zip(snrs, rate, lo, hi)]
        style_axis(ax, "injected SNR", "detected (%)", morph)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK_SOFT, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Pipeline survival by stage — each gap is what that stage costs",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig, records


def plot_params(df, out_dir, metric, n_bins, min_n):
    """Survival vs each morphology's own variable, one line per SNR."""
    panels, records = [], []
    for morph in sorted(df["morphology"].unique()):
        sub = df[df.morphology == morph]
        for col, label in PARAM_SPECS.get(morph, []):
            if col not in sub.columns or sub[col].nunique(dropna=True) < 2:
                continue
            panels.append((morph, col, label))

    if not panels:
        return None, records

    fig, axes = grid_of_panels(len(panels))
    # SNR=0 is the null control: it is 0% everywhere by construction and would
    # only flatten the y-range the real curves need.
    snrs = [s for s in sorted(df["snr"].unique()) if s > 0]
    colours = {s: SNR_RAMP[min(i, len(SNR_RAMP) - 1)] for i, s in enumerate(snrs)}

    for ax, (morph, col, label) in zip(axes, panels):
        sub = df[df.morphology == morph]
        categorical = (sub[col].dtype == object or sub[col].nunique() <= 4)

        for snr in snrs:
            at_snr = sub[sub.snr == snr]
            if at_snr.empty:
                continue
            if categorical:
                levels = sorted(at_snr[col].dropna().unique(), key=str)
                xs = np.arange(len(levels))
                pts = [rate_with_ci(at_snr[at_snr[col] == lv][metric]) for lv in levels]
                counts = [int((at_snr[col] == lv).sum()) for lv in levels]
                ax.set_xticks(xs)
                ax.set_xticklabels([str(lv) for lv in levels], fontsize=8)
                x_labels = [str(lv) for lv in levels]
            else:
                # Quantile bins: parameters are sampled log-ish and unevenly, so
                # equal-width bins would put most injections in one bucket.
                binned = pd.qcut(at_snr[col], q=n_bins, duplicates="drop")
                groups = at_snr.groupby(binned, observed=True)
                xs, pts, counts, x_labels = [], [], [], []
                for interval, g in groups:
                    if len(g) < min_n:
                        continue
                    xs.append(float(g[col].median()))
                    pts.append(rate_with_ci(g[metric]))
                    counts.append(len(g))
                    x_labels.append(f"{interval}")
                xs = np.asarray(xs)

            if len(xs) == 0:
                continue
            rate = [100 * p for p, _, _ in pts]
            lo = [100 * l for _, l, _ in pts]
            hi = [100 * h for _, _, h in pts]
            colour = colours[snr]
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.12, linewidth=0, zorder=2)
            ax.plot(xs, rate, color=colour, linewidth=2.0, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=f"SNR {snr:g}", zorder=3)
            records += [{"figure": "params", "morphology": morph, "parameter": col,
                         "snr": snr, "bin": xl, "x": float(x), "rate_pct": r,
                         "ci_lo_pct": a, "ci_hi_pct": b, "n": c}
                        for xl, x, r, a, b, c in zip(x_labels, xs, rate, lo, hi, counts)]

        style_axis(ax, label, f"{metric} (%)", morph)

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK_SOFT, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{metric} vs each morphology's own variable "
                 f"— observational: parameters are sampled, not gridded",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig, records


def plot_factorial(df, out_dir, metric):
    """The 2x2: colour = frequency extent, dash = time structure."""
    if "sig_freq_extent" not in df.columns:
        return None, []
    fac = df[df.morphology.isin(
        ["narrowband_drift", "narrowband_pulsed",
         "wideband_continuous", "wideband_pulsed"])]
    if fac.empty:
        return None, []

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    records = []
    for extent in ("narrowband", "wideband"):
        for structure, dash in (("continuous", (None, None)), ("pulsed", (5, 2))):
            sub = fac[(fac.sig_freq_extent == extent)
                      & (fac.sig_time_structure == structure)]
            if sub.empty:
                continue
            snrs = sorted(sub["snr"].unique())
            pts = [rate_with_ci(sub[sub.snr == s][metric]) for s in snrs]
            rate = [100 * p for p, _, _ in pts]
            line, = ax.plot(snrs, rate, color=EXTENT_COLORS[extent], linewidth=2.2,
                            marker="o", markersize=5, markeredgecolor="white",
                            markeredgewidth=0.9, label=f"{extent}, {structure}",
                            zorder=3)
            if dash[0]:
                line.set_dashes(list(dash))
            records += [{"figure": "factorial", "extent": extent,
                         "time_structure": structure, "snr": s, "rate_pct": r,
                         "n": int((sub.snr == s).sum())}
                        for s, r in zip(snrs, rate)]

    style_axis(ax, "injected SNR", f"{metric} (%)")
    ax.set_title("Bandwidth or pulsing? The 2×2 separates them",
                 color=INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper left")
    fig.tight_layout()
    return fig, records


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    print(f"{len(df)} injections, {df.morphology.nunique()} morphologies, "
          f"SNR {sorted(df.snr.unique())}")
    if args.metric not in df.columns:
        raise SystemExit(f"--metric {args.metric} not in CSV. "
                         f"Available: {[c for c in df.columns if c.startswith('s')]}")

    all_records = []
    for name, (fig, records) in [
        ("cascade", plot_cascade(df, args.out_dir)),
        ("params", plot_params(df, args.out_dir, args.metric, args.n_bins, args.min_n)),
        ("factorial", plot_factorial(df, args.out_dir, args.metric)),
    ]:
        all_records += records
        if fig is None:
            print(f"  {name}: skipped (columns absent from this CSV)")
            continue
        for ext in ("png", "pdf"):
            path = args.out_dir / f"sensitivity_{name}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {name}: {args.out_dir / f'sensitivity_{name}.png'}")

    # The numbers behind every mark: a figure nobody can read the values off is
    # not a result, and these go straight into the write-up's tables.
    table = args.out_dir / "sensitivity_curves.csv"
    pd.DataFrame(all_records).to_csv(table, index=False)
    print(f"  table: {table}")


if __name__ == "__main__":
    main()
