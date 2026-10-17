"""
Run the production search on real cadences from a cadence list file.

A thin driver over ``src/search/pipeline.py`` (the same chain the
injection-recovery harness runs): loads each cadence's .h5 files into RAM,
slides a (tchans, fchans) window across the frequency axis at stride_infer,
scores every window, forms candidates (FAR quantile, clustering, ON/OFF
diagnostics, short list) and exports the ranked candidates for review.

Each cadence gets its own output folder named:
    cad{idx}_{target}_{fch1_MHz}MHz_{date}

Candidate snippets are saved with per-candidate plots showing
original | reconstruction | error map, either as one multi-page PDF per
cadence per method (default, fast vetting) or individual PNGs
(--plot_format png/both).

Usage (run on the server):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/Platypus \
    python scripts/inference.py \
        --checkpoint outputs/training/<run>/checkpoints/best.ckpt \
        --cadence_list data/processed/inference_cadences.txt \
        --out_dir outputs/inference/run_name \
        --num_workers 32
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.torch_dataset import _load_full_obs
from src.search import pipeline
from src.search.export import export_candidates
from src.search.pipeline import (
    CadenceWindows, SelectionRule, load_model, rank_for_export, read_cadence_meta,
    robust_stats, score_windows, select_candidates,
)

METHODS = ["recon", "topk", "max", "cadence"]

# Names other scripts import from here (pipeline_sensitivity.py,
# recompute_anomaly_maps.py, debug/plot_shortlist_candidates.py), kept until
# those harnesses are retired in favour of the shared chain.
FAR_QUANTILE = SelectionRule.far_quantile
MIN_OFF_POOL = SelectionRule.min_off_pool
MAD_SCALE = pipeline.MAD_SCALE
_worker_init = pipeline._worker_init
_preprocess_at = pipeline._worker_window


def reconstruct_batch(model, snippets: list, device: str) -> np.ndarray:
    x = torch.from_numpy(np.array(snippets)).float().unsqueeze(1).to(device)
    with torch.no_grad():
        recon = model(x)
    return recon.squeeze(1).cpu().numpy()


def make_cadence_dirname(cad_idx: int, meta: dict) -> str:
    fch1_str = f"{meta['fch1_mhz']:.1f}MHz"
    return f"cad{cad_idx:02d}_{meta['source']}_{fch1_str}_{meta['date']}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cadence_list", type=Path, required=True,
                   help="Text file with one cadence per line (6 space-separated .h5 paths)")
    p.add_argument("--data_config", type=Path, default=ROOT / "configs/data/gbt_fine.yaml")
    p.add_argument("--model_config", type=Path, default=ROOT / "configs/model/vit_mae.yaml")
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/inference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_cadences", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--top_k", type=int, default=30,
                   help="Number of top candidates to plot per cadence per method")
    p.add_argument("--plot_format", default="pdf", choices=["pdf", "png", "both"],
                   help="Candidate plot output: one multi-page PDF per cadence per "
                        "method (fast vetting, default), individual PNGs (old "
                        "behaviour), or both.")
    p.add_argument("--methods", nargs="+", default=["recon"], choices=METHODS,
                   help="Anomaly scoring methods to run. The plain Autoencoder/MAE/VAE "
                        "only support 'recon'/'topk'; 'cadence' requires the ViT-MAE "
                        "backbone; UDMA supports 'recon'/'topk'/'max' (its own topk_frac "
                        "default, no 'cadence').")
    p.add_argument("--ignore_short_list", action="store_true",
                   help="Diagnostic only: plot the top_k candidates by on_off_contrast "
                        "regardless of in_short_list (off_leak). The full CSV always "
                        "contains every candidate with the short-list columns either way; "
                        "this only affects which ones get a plot.")
    p.add_argument("--off_ceiling_probe", type=int, default=300,
                   help="Extra background snippets sampled across the whole cadence "
                        "(beyond the candidate clusters) to build the off_noise_ceiling "
                        "pool, so the ceiling reflects the cadence-wide OFF noise floor "
                        "rather than only the OFF-row cells co-located with already-"
                        "flagged candidates (a small, detection-biased sample).")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    preproc = data_cfg["preprocessing"]
    frame = data_cfg["frame"]
    fchans = frame["fchans"]
    tchans = frame["tchans"]
    stride = frame.get("stride_infer", fchans // 2)
    downsample_factor = frame.get("downsample_factor", 1)
    df = data_cfg["raw"]["df"]
    nchans_total = data_cfg["raw"]["nchans"]
    input_shape = (tchans, fchans, 1)
    rule = SelectionRule(off_ceiling_probe=args.off_ceiling_probe)

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    methods = args.methods

    cadence_lines = [
        line.strip().split()
        for line in args.cadence_list.read_text().splitlines()
        if line.strip()
    ]
    if args.max_cadences:
        cadence_lines = cadence_lines[:args.max_cadences]

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, model_cfg, input_shape, args.device)

    print(f"Cadences: {len(cadence_lines)}")
    print(f"Input shape: {input_shape}  methods: {methods}")
    print(f"Frame: {tchans}x{fchans}, stride={stride}, downsample={downsample_factor}")
    print(f"Batch size: {args.batch_size}, workers: {args.num_workers}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {m: [] for m in methods}
    all_rows = []

    for cad_idx, obs_paths in enumerate(cadence_lines):
        obs_paths = [Path(p) for p in obs_paths]

        # Read metadata — try each obs file until one opens
        meta = None
        for obs_path in obs_paths:
            try:
                meta = read_cadence_meta(obs_path)
                break
            except OSError:
                continue
        if meta is None:
            print(f"\nCadence {cad_idx}: SKIPPING — all files corrupt")
            continue
        target_name = meta["source"]
        fch1_mhz = meta["fch1_mhz"]
        # The header's signed foff (Hz/channel) gets the direction of the
        # frequency axis right; the config df is an unsigned magnitude used only
        # if the header carries no foff.
        foff_hz = meta["foff_mhz"] * 1e6
        freq_step_hz = foff_hz if foff_hz else df
        cad_dir = args.out_dir / make_cadence_dirname(cad_idx, meta)
        cad_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Cadence {cad_idx}: {target_name}  fch1={meta['fch1_mhz']:.1f} MHz  "
              f"date={meta['date']}")
        print(f"  -> {cad_dir}")
        print(f"{'='*70}")

        # Load all observations into RAM
        t_load = time.time()
        obs_arrays = []
        corrupt = False
        for i, obs_path in enumerate(obs_paths):
            try:
                arr = _load_full_obs(obs_path, downsample_factor)
            except OSError as e:
                print(f"  SKIPPING cadence — corrupt file obs {i}: {obs_path.name} — {e}")
                corrupt = True
                break
            obs_arrays.append(arr)
            print(f"  Loaded obs {i}: {obs_path.name} -> {arr.shape}")
        if corrupt:
            continue
        print(f"  {sum(a.nbytes for a in obs_arrays) / 1e9:.1f} GB in RAM, "
              f"loaded in {time.time() - t_load:.1f}s")

        with CadenceWindows(obs_arrays, args.out_dir / "_scratch", f"cad{cad_idx:02d}",
                            fchans=fchans, tchans=tchans, stride=stride,
                            preproc=preproc) as windows:
            del obs_arrays
            print(f"  nchans={windows.nchans} -> {len(windows.f_starts)} snippets (stride={stride})")

            cad_arrs = score_windows(model, windows, methods, batch_size=args.batch_size,
                                     num_workers=args.num_workers, device=args.device)
            for j, fs in enumerate(windows.f_starts):
                row = {
                    "cadence_idx": cad_idx,
                    "target": target_name,
                    "f_start": int(fs),
                    "f_center_mhz": fch1_mhz + int(fs) * freq_step_hz / 1e6,
                }
                for m in methods:
                    score = float(cad_arrs[m][j])
                    all_scores[m].append(score)
                    row[f"{m}_score"] = score
                all_rows.append(row)

            n_plots = 0
            for method in methods:
                result = select_candidates(
                    cad_arrs[method], windows.f_starts, model=model, windows=windows,
                    method=method, rule=rule, freq_step_hz=freq_step_hz, fch1_mhz=fch1_mhz,
                    probe_seed=cad_idx, batch_size=args.batch_size, device=args.device,
                )
                result.clusters.to_csv(cad_dir / f"{method}_candidates.csv", index=False)
                ranked, ranked_amaps = rank_for_export(result, ignore_short_list=args.ignore_short_list)
                n_plots += export_candidates(
                    ranked, ranked_amaps, model=model, windows=windows, method=method,
                    top_k=args.top_k, plot_format=args.plot_format, cad_dir=cad_dir,
                    median=result.thresholds["median"], mad_sigma=result.thresholds["mad_sigma"],
                    cad_idx=cad_idx, target=target_name, freq_step_hz=freq_step_hz,
                    fch1_mhz=fch1_mhz, nchans_total=nchans_total, obs_date=meta["date"],
                    device=args.device,
                )

            # Per-cadence frequency profile, one per method
            for method in methods:
                scores = cad_arrs[method]
                fig, ax = plt.subplots(figsize=(16, 4))
                med, mad_s = robust_stats(scores)
                ax.plot(windows.f_starts, scores, linewidth=0.3, alpha=0.7)
                ax.axhline(med + 3 * mad_s, color="orange", ls="--", lw=1,
                           label=f"3s = {med + 3*mad_s:.3f}")
                ax.axhline(med + 5 * mad_s, color="red", ls="--", lw=1,
                           label=f"5s = {med + 5*mad_s:.3f}")
                ax.set_xlabel("Frequency channel (f_start)")
                ax.set_ylabel(f"{method} score")
                ax.set_title(f"{target_name} — {method} score vs frequency")
                ax.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(cad_dir / f"{method}_score_vs_freq.png", dpi=150)
                plt.close()

            print(f"  Saved {n_plots} candidate plots -> {cad_dir}")

    # ---- Global summary ----
    all_arrs = {m: np.array(all_scores[m]) for m in methods}
    n_total = len(next(iter(all_arrs.values()))) if all_arrs else 0

    csv_path = args.out_dir / "inference_scores.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["cadence_idx", "target", "f_start", "f_center_mhz"]
        fieldnames += [f"{m}_score" for m in methods]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved scores -> {csv_path}")

    print(f"\n{'='*70}")
    print(f"GLOBAL SUMMARY -- {n_total} snippets across {len(cadence_lines)} cadences")
    print(f"{'='*70}")

    for name in methods:
        scores = all_arrs[name]
        median, mad_sigma = robust_stats(scores)
        mean, std = scores.mean(), scores.std()
        thresh_3s = median + 3 * mad_sigma
        thresh_5s = median + 5 * mad_sigma
        n_3s = (scores > thresh_3s).sum()
        n_5s = (scores > thresh_5s).sum()

        print(f"\n  {name}:")
        print(f"    mean={mean:.4f}  std={std:.4f}")
        print(f"    median={median:.4f}  MAD_sigma={mad_sigma:.4f}")
        print(f"    min={scores.min():.4f}  max={scores.max():.4f}")
        print(f"    3s (robust)={thresh_3s:.4f}  -> {n_3s} candidates ({n_3s/n_total*100:.3f}%)")
        print(f"    5s (robust)={thresh_5s:.4f}  -> {n_5s} candidates ({n_5s/n_total*100:.3f}%)")

    # Global histograms, one panel per method
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5), squeeze=False)
    for ax, name in zip(axes[0], methods):
        scores = all_arrs[name]
        median, mad_sigma = robust_stats(scores)
        thresh_3 = median + 3 * mad_sigma
        thresh_5 = median + 5 * mad_sigma
        n_3s = (scores > thresh_3).sum()

        clipped = scores[scores < np.percentile(scores, 99.5)]
        ax.hist(clipped, bins=200, alpha=0.7, edgecolor="black", linewidth=0.2)
        ax.axvline(thresh_3, color="orange", ls="--", lw=1.5, label=f"3s = {thresh_3:.3f}")
        ax.axvline(thresh_5, color="red", ls="--", lw=1.5, label=f"5s = {thresh_5:.3f}")
        ax.set_xlabel("Anomaly score (MSE)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} -- {n_3s} candidates > 3s (robust)")
        ax.legend()

    plt.suptitle(f"Score distribution -- {n_total} snippets, {len(cadence_lines)} cadences\n"
                 f"Thresholds: median + k * MAD * 1.4826", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "inference_score_distributions.png", dpi=150)
    plt.close()
    print(f"\nSaved -> {args.out_dir / 'inference_score_distributions.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
