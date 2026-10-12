"""
Offline background extractor — BL Exotica Autoencoder.

Extracts RAW snippets from cadence manifests and saves them as per-split .npy
files for memory-mapped training.

Normalization (bandpass_correct + core_transform) happens in
CachedDataset.__getitem__ so preprocessing hyperparameters can be changed
without re-extracting the cache.  The .npy format enables np.load(mmap_mode='r')
so DataLoader workers share physical memory pages instead of duplicating.

Pre-split mode (format 2, current) — train and val manifests are given:
    PYTHONPATH=. python scripts/preprocess_cache.py \\
        --data-config         configs/data/gbt_fine_nightsplit.yaml \\
        --output              /path/to/cache_dir \\
        --train-cadence-list  data/raw/gbt_0000_nightsplit_train.txt \\
        --val-cadence-list    data/raw/gbt_0000_nightsplit_devcal.txt \\
        --snippets-per-cadence 2200 \\
        --val-snippets-per-cadence 300

  Output directory (see src/data/cache.py):
    train.npy / val.npy               (N, n_obs, tchans_per_obs, fchans) float32 RAW
    train_index.npz / val_index.npz   manifest row and channel offset of every snippet
    meta.json                         manifest sha256, geometry, seed, git state; written last

  Every manifest row is extracted (an unreadable cadence aborts the run).
  Snippet channels are drawn from an RNG seeded per (seed, split, manifest row),
  so the sample does not depend on extraction order. Work happens in
  "<output>.partial" and is renamed to <output> only when complete; re-running
  the same command after an interruption resumes from the last finished cadence.

Legacy mode (internal shuffle-and-fraction split — one manifest in, script
decides train/val/inject-recovery; produces a format-1 cache without index):
    PYTHONPATH=. python scripts/preprocess_cache.py \\
        --config    configs/training/srt_real.yaml \\
        --output    data/processed/cache_gbt_fine \\
        --snippets-per-cadence 18000 \\
        --max-snippets 1100000 \\
        --train-fraction 0.7 \\
        --seed 42
"""

import argparse
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
try:
    import hdf5plugin  # noqa: F401 — registers bitshuffle/LZ4 filters for BL HDF5 files
except ImportError:
    pass
import numpy as np
import yaml
from tqdm import tqdm

from src.data.cache import CACHE_FORMAT_VERSION, load_manifest, sha256_file
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _read_header(path: Path):
    """Return (ntime, nchans) from HDF5 header without loading data."""
    with h5py.File(str(path), "r") as f:
        sh = f["data"].shape   # (ntime, nif, nchans) or (ntime, nchans)
        ntime = sh[0]
        nchans = sh[-1]
    return ntime, nchans


def _validate_cadence(cadence_paths: list[Path]) -> int | None:
    """Check all files in a cadence are readable. Return nchans or None."""
    nchans = None
    for path in cadence_paths:
        try:
            _, nc = _read_header(path)
        except Exception as e:
            print(f"  skip cadence (corrupt {path.name}): {e}")
            return None
        if nchans is None:
            nchans = nc
        elif nc != nchans:
            print(f"  skip cadence (nchans mismatch: {path.name} has {nc}, expected {nchans})")
            return None
    return nchans


def _extract_cadence_snippets(
    cadence_paths: list[Path],
    indices: np.ndarray,
    fchans: int,
    tchans_per_obs: int,
) -> np.ndarray:
    """
    Extract snippets at the given frequency indices from all obs in a cadence.

    Returns (n_snippets, n_obs, tchans_per_obs, fchans) float32 RAW array.
    Each file is opened once and all snippets are read in a single pass.
    """
    n_obs = len(cadence_paths)
    n_snip = len(indices)
    out = np.empty((n_snip, n_obs, tchans_per_obs, fchans), dtype=np.float32)

    for oi, path in enumerate(cadence_paths):
        with h5py.File(str(path), "r", rdcc_nbytes=256 * 1024 * 1024) as hf:
            dset = hf["data"]
            three_d = dset.ndim == 3
            for si, idx in enumerate(indices):
                start = int(idx) * fchans
                if three_d:
                    out[si, oi] = dset[:tchans_per_obs, 0, start : start + fchans]
                else:
                    out[si, oi] = dset[:tchans_per_obs, start : start + fchans]

    return out


# ---------------------------------------------------------------------------
# Pre-split extraction (cache format 2)
# ---------------------------------------------------------------------------

def _git_state() -> dict:
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=repo, capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(dirty.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _plan_presplit(cadences, split_id: int, label: str, spc: int, fchans: int, seed: int):
    """Validate every row and draw its channel blocks. Any unusable row aborts."""
    print(f"\n--- Pre-scan {label}: {len(cadences)} cadences, {spc} snippets each ---")
    n_obs = len(cadences[0])
    plan = []
    for row, cad in enumerate(tqdm(cadences, desc=f"headers {label}")):
        if len(cad) != n_obs:
            raise ValueError(f"{label} row {row}: {len(cad)} observations, expected {n_obs}")
        nchans = _validate_cadence(cad)
        if nchans is None:
            raise ValueError(f"{label} row {row} ({cad[0].name}) is unreadable; "
                             f"fix the file or the manifest before extracting")
        n_avail = nchans // fchans
        if n_avail < spc:
            raise ValueError(f"{label} row {row}: only {n_avail} blocks of {fchans} "
                             f"channels, {spc} requested")
        rng = np.random.default_rng([seed, split_id, row])
        blocks = np.sort(rng.choice(n_avail, size=spc, replace=False))
        plan.append((cad, blocks))
    return plan


def _extract_presplit(plan, partial: Path, label: str, snippet_shape, fchans: int,
                      budget: list) -> bool:
    """Fill {label}.npy cadence by cadence, recording progress after each one.

    Returns False if the cadence budget ran out (test hook for interruption).
    """
    n_obs, tchans_per_obs, _ = snippet_shape
    n_total = sum(len(blocks) for _, blocks in plan)
    npy_path = partial / f"{label}.npy"
    progress_path = partial / f"{label}_progress.json"

    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        mm = np.lib.format.open_memmap(str(npy_path), mode="r+")
        if mm.shape != (n_total, *snippet_shape):
            raise RuntimeError(f"{npy_path} has shape {mm.shape}, expected {(n_total, *snippet_shape)}")
        print(f"\n=== {label}: resuming after {progress['cadences_done']}/{len(plan)} cadences ===")
    else:
        print(f"\n=== {label}: allocating {npy_path.name} {(n_total, *snippet_shape)} ===")
        mm = np.lib.format.open_memmap(str(npy_path), mode="w+", dtype=np.float32,
                                       shape=(n_total, *snippet_shape))
        progress = {"cadences_done": 0, "cursor": 0}
        _write_json_atomic(progress_path, progress)

    done, cursor = progress["cadences_done"], progress["cursor"]
    for cad, blocks in tqdm(plan[done:], desc=label, initial=done, total=len(plan)):
        if budget[0] is not None:
            if budget[0] == 0:
                del mm
                return False
            budget[0] -= 1
        mm[cursor : cursor + len(blocks)] = _extract_cadence_snippets(
            cad, blocks, fchans, tchans_per_obs)
        mm.flush()
        cursor += len(blocks)
        done += 1
        _write_json_atomic(progress_path, {"cadences_done": done, "cursor": cursor})

    assert cursor == n_total, f"BUG: wrote {cursor} but allocated {n_total}"
    del mm

    manifest_row = np.concatenate([np.full(len(b), r, dtype=np.int32) for r, (_, b) in enumerate(plan)])
    chan_start = np.concatenate([b.astype(np.int64) * fchans for _, b in plan])
    n_take = np.array([len(b) for _, b in plan], dtype=np.int64)
    np.savez(partial / f"{label}_index.npz",
             manifest_row=manifest_row, chan_start=chan_start,
             cadence_row_start=np.concatenate([[0], np.cumsum(n_take)[:-1]]),
             cadence_n_take=n_take)
    print(f"  {label}: {cursor:,} snippets written")
    return True


def _same_identity(recorded: dict, identity: dict) -> bool:
    """Compare build settings, identifying manifests by content rather than path."""
    keys = ("format_version", "geometry", "seed")
    if any(recorded.get(k) != identity[k] for k in keys):
        return False
    fields = ("manifest_sha256", "n_cadences", "snippets_per_cadence")
    return all(recorded["splits"][s].get(f) == info[f]
               for s, info in identity["splits"].items() for f in fields)


def build_cache(
    train_manifest: Path,
    val_manifest: Path,
    output: Path,
    *,
    tchans: int,
    fchans: int,
    train_snippets_per_cadence: int,
    val_snippets_per_cadence: int,
    seed: int = 42,
    preprocessing: dict | None = None,
    stop_after_cadences: int | None = None,
) -> bool:
    """Build (or resume) a format-2 cache. Returns True once <output> is complete.

    stop_after_cadences: extract at most this many cadences in this call, then
    return False leaving "<output>.partial" for a later resume (test hook).
    """
    output = Path(output)
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(f"{output} already exists; caches are never overwritten")

    splits = {"train": (Path(train_manifest), train_snippets_per_cadence),
              "val": (Path(val_manifest), val_snippets_per_cadence)}
    cadences = {s: [[Path(p) for p in c] for c in load_manifest(m)] for s, (m, _) in splits.items()}
    n_obs = len(cadences["train"][0])
    if tchans % n_obs:
        raise ValueError(f"tchans={tchans} is not a multiple of n_obs={n_obs}")
    snippet_shape = (n_obs, tchans // n_obs, fchans)

    identity = {
        "format_version": CACHE_FORMAT_VERSION,
        "geometry": {"n_obs": n_obs, "tchans_per_obs": tchans // n_obs, "fchans": fchans},
        "seed": seed,
        "splits": {
            s: {"manifest": str(m), "manifest_sha256": sha256_file(m),
                "n_cadences": len(cadences[s]), "snippets_per_cadence": spc}
            for s, (m, spc) in splits.items()
        },
    }
    partial.mkdir(parents=True, exist_ok=True)
    meta_path = partial / "meta.json"
    if meta_path.exists():
        # Previous run finished writing but died before the rename.
        if not _same_identity(json.loads(meta_path.read_text()), identity):
            raise RuntimeError(f"{partial} holds a finished cache with different settings")
        os.rename(partial, output)
        print(f"\nDone: {output}")
        return True

    identity_path = partial / "identity.json"
    if identity_path.exists():
        if not _same_identity(json.loads(identity_path.read_text()), identity):
            raise RuntimeError(f"{partial} was started with different manifests or settings; "
                               f"remove it or use another --output")
    else:
        _write_json_atomic(identity_path, identity)

    budget = [stop_after_cadences]
    for split_id, (label, (_, spc)) in enumerate(splits.items()):
        plan = _plan_presplit(cadences[label], split_id, label, spc, fchans, seed)
        if not _extract_presplit(plan, partial, label, snippet_shape, fchans, budget):
            print(f"\nStopped early; resume by re-running the same command ({partial}).")
            return False

    meta = dict(identity)
    meta["splits"] = {s: dict(info, n_snippets=info["n_cadences"] * info["snippets_per_cadence"])
                      for s, info in identity["splits"].items()}
    meta.update({
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_state(),
        "preprocessing_at_build": preprocessing,  # informational: the cache stores RAW data
    })
    _write_json_atomic(meta_path, meta)
    for s in splits:
        (partial / f"{s}_progress.json").unlink()
    identity_path.unlink()
    os.rename(partial, output)
    print(f"\nDone: {output}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Extract and cache background snippets")
    p.add_argument("--config", type=Path, default=Path("configs/training/srt_real.yaml"),
                   help="Training config whose data: entry gives frame/preprocessing")
    p.add_argument("--data-config", type=Path, default=None,
                   help="Data config to use directly instead of --config's data: entry")
    p.add_argument("--output", type=Path, default=Path("data/processed/cache_gbt_fine"),
                   help="Output directory (will contain train.npy, val.npy, meta.json)")
    p.add_argument("--snippets-per-cadence", type=int, default=18000,
                   help="Snippets sampled per cadence (controls diversity)")
    p.add_argument("--val-snippets-per-cadence", type=int, default=None,
                   help="Pre-split mode: snippets per val cadence (default: --snippets-per-cadence)")
    p.add_argument("--max-snippets", type=int, default=1_100_000,
                   help="Legacy mode: global cap on total train snippets")
    p.add_argument("--train-fraction", type=float, default=0.7,
                   help="Legacy mode: fraction of cadences used for training; rest → inject-recovery pool")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Legacy mode: fraction of train cadences held out for validation")
    p.add_argument("--exclude-targets", nargs="*", default=None,
                   help="Legacy mode: target names forced into the inject-recovery pool.")
    p.add_argument("--train-cadence-list", type=Path, default=None,
                   help="Pre-split train manifest. Requires --val-cadence-list.")
    p.add_argument("--val-cadence-list", type=Path, default=None,
                   help="Pre-split val manifest, paired with --train-cadence-list.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pre_split = args.train_cadence_list is not None
    if pre_split != (args.val_cadence_list is not None):
        raise ValueError("--train-cadence-list and --val-cadence-list must be given together")

    # ------------------------------------------------------------------ config
    if args.data_config is not None:
        data_cfg = yaml.safe_load(args.data_config.read_text())
    else:
        data_cfg = load_config(args.config)["data"]
    frame_cfg = data_cfg["frame"]
    tchans = frame_cfg["tchans"]           # 96 — total cadence height
    fchans = frame_cfg["fchans"]           # 1024
    cfg_preproc = data_cfg["preprocessing"]
    if frame_cfg.get("downsample_factor", 1) != 1:
        raise ValueError("the extractor does not downsample; downsample_factor must be 1")

    if pre_split:
        build_cache(
            args.train_cadence_list, args.val_cadence_list, args.output,
            tchans=tchans, fchans=fchans,
            train_snippets_per_cadence=args.snippets_per_cadence,
            val_snippets_per_cadence=args.val_snippets_per_cadence or args.snippets_per_cadence,
            seed=args.seed, preprocessing=cfg_preproc,
        )
        return

    _legacy_main(args, data_cfg, tchans, fchans, cfg_preproc)


def _legacy_main(args, data_cfg, tchans, fchans, cfg_preproc):
    """Internal shuffle-and-fraction split (format-1 cache, no index)."""
    np_rng = np.random.default_rng(args.seed)

    def _load_manifest(path: Path):
        return [
            [Path(p) for p in line.strip().split()]
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    args.output.mkdir(parents=True, exist_ok=True)

    cadence_list_path = Path(data_cfg["dataset"]["cadence_list"])
    all_cadences = _load_manifest(cadence_list_path)

    print(f"Total cadences in manifest: {len(all_cadences)}")

    # ------------------------------------------------------ cadence split
    rng = random.Random(args.seed)

    cadences = list(all_cadences)
    rng.shuffle(cadences)

    exclude = set(args.exclude_targets or [])

    def _target_name(paths):
        """Best-effort: filename field before _ON/_OFF."""
        import re
        stem = paths[0].stem
        m = re.search(r"([A-Za-z0-9_]+?)_(ON|OFF)", stem)
        return m.group(1) if m else stem

    excluded = [c for c in cadences if _target_name(c) in exclude]
    cadences  = [c for c in cadences if _target_name(c) not in exclude]

    n_train_total = int(len(cadences) * args.train_fraction)
    train_and_val = cadences[:n_train_total]
    inject_recovery = cadences[n_train_total:] + excluded

    n_val_cad = max(1, int(len(train_and_val) * args.val_fraction))
    val_cadences   = train_and_val[:n_val_cad]
    train_cadences = train_and_val[n_val_cad:]

    print(f"Train cadences : {len(train_cadences)}")
    print(f"Val cadences   : {len(val_cadences)}")
    print(f"Inject-recovery: {len(inject_recovery)}")

    # ------------------------------------- save inject-recovery manifest
    inj_path = args.output / "inject_recovery_cadences.txt"
    with open(inj_path, "w") as f:
        for cad in inject_recovery:
            f.write(" ".join(str(p) for p in cad) + "\n")
    print(f"Inject-recovery cadences saved to: {inj_path}")

    # Determine n_obs and tchans_per_obs from first available cadence
    sample_cad = (train_cadences or val_cadences)[0]
    n_obs = len(sample_cad)
    tchans_per_obs = tchans // n_obs   # e.g. 96 // 6 = 16

    # ====================================================================
    # Pass 1: validate all cadences and compute exact snippet counts
    # ====================================================================
    def _plan_split(split_cadences, n_target, label):
        """Validate headers, compute per-cadence n_take. Returns [(cadence, n_take, n_avail)]."""
        print(f"\n--- Pre-scan {label}: {len(split_cadences)} cadences, target {n_target} ---")
        plan = []
        total = 0
        for cad in split_cadences:
            if total >= n_target:
                break
            nchans = _validate_cadence(cad)
            if nchans is None:
                continue
            n_avail = nchans // fchans
            n_take = min(args.snippets_per_cadence, n_avail, n_target - total)
            if n_take == 0:
                continue
            plan.append((cad, n_take, n_avail))
            total += n_take

        print(f"  {label}: {len(plan)} valid cadences, {total} snippets planned")
        return plan, total

    n_val_target = max(1000, int(args.max_snippets * args.val_fraction / (1 - args.val_fraction)))

    train_plan, n_train = _plan_split(train_cadences, args.max_snippets, "train")
    val_plan, n_val = _plan_split(val_cadences, n_val_target, "val")

    if n_train == 0:
        print("ERROR: no valid train snippets. Aborting.")
        return
    if n_val == 0:
        print("ERROR: no valid val snippets. Aborting.")
        return

    snippet_shape = (n_obs, tchans_per_obs, fchans)
    snippet_bytes = np.dtype(np.float32).itemsize * n_obs * tchans_per_obs * fchans
    train_gb = n_train * snippet_bytes / 1e9
    val_gb = n_val * snippet_bytes / 1e9
    print(f"\nData geometry:")
    print(f"  tchans per obs  : {tchans_per_obs}")
    print(f"  n_obs per cadence: {n_obs}")
    print(f"  snippet shape   : {snippet_shape}")
    print(f"  train: {n_train:,} snippets ({train_gb:.1f} GB)")
    print(f"  val  : {n_val:,} snippets ({val_gb:.1f} GB)")
    print(f"  total: {train_gb + val_gb:.1f} GB")

    # ====================================================================
    # Pass 2: allocate memmap files and extract directly to disk
    # ====================================================================
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    def _extract_to_memmap(plan, n_total, label, out_path):
        """Allocate a .npy memmap and fill it cadence by cadence."""
        shape = (n_total, *snippet_shape)
        print(f"\n=== {label}: allocating {out_path.name} {shape} ===")
        mm = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=np.float32, shape=shape,
        )

        cursor = 0
        for cad, n_take, n_avail in tqdm(plan, desc=label):
            indices = np_rng.choice(n_avail, size=n_take, replace=False)
            indices.sort()

            try:
                snippets = _extract_cadence_snippets(cad, indices, fchans, tchans_per_obs)
            except Exception as e:
                print(f"\nFATAL: extraction failed for {cad[0].name} "
                      f"(passed header check but failed on data read): {e}")
                print("Aborting — partial file may exist. Re-run after fixing the corrupt file.")
                raise

            mm[cursor : cursor + n_take] = snippets
            cursor += n_take
            mm.flush()

        assert cursor == n_total, f"BUG: wrote {cursor} but allocated {n_total}"
        del mm
        print(f"  {label}: {cursor:,} snippets written to {out_path}")

    train_path = out_dir / "train.npy"
    val_path = out_dir / "val.npy"

    _extract_to_memmap(train_plan, n_train, "train", train_path)
    _extract_to_memmap(val_plan, n_val, "val", val_path)

    # ---------------------------------------------------------- metadata
    meta = {
        "n_train": n_train,
        "n_val": n_val,
        "shape_per_snippet": list(snippet_shape),
        "n_train_cadences": len(train_plan),
        "n_val_cadences": len(val_plan),
        "n_heldout_cadences": len(inject_recovery),
        "snippets_per_cadence": args.snippets_per_cadence,
        "seed": args.seed,
        "split_mode": "internal-shuffle",
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "preprocessing": cfg_preproc,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone.")
    print(f"  {train_path}  ({n_train:,} train snippets)")
    print(f"  {val_path}  ({n_val:,} val snippets)")
    print(f"  {meta_path}")
    print(f"  {inj_path}  ({len(inject_recovery)} cadences for inject-recovery)")
    print(f"\nAdd to configs/data/gbt_fine.yaml:")
    print(f"  dataset:")
    print(f"    cache_dir: {out_dir}")


if __name__ == "__main__":
    main()
