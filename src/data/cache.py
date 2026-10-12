"""
Identity and validation of snippet caches built by scripts/preprocess_cache.py.

A format-2 cache directory holds, per split (train, val):
  {split}.npy        — (N, n_obs, tchans_per_obs, fchans) float32 RAW snippets
  {split}_index.npz  — per-sample manifest_row / chan_start, per-cadence row_start / n_take
and a meta.json written last, with "complete": true, the sha256 of the manifest
each split was extracted from, and the snippet geometry.

The cache directory only appears under its final name once complete (the
extractor works in "<name>.partial" and renames at the end), so a missing or
partial cache can never be mistaken for a valid one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import numpy as np

CACHE_FORMAT_VERSION = 2
SPLITS = ("train", "val")


class CacheError(RuntimeError):
    """The cache is missing, incomplete, or does not match what the caller expects."""


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Union[str, Path]) -> List[List[str]]:
    """One cadence per non-empty line, observation paths separated by whitespace."""
    return [line.split() for line in Path(path).read_text().splitlines() if line.strip()]


def verify_cache(
    cache_dir: Union[str, Path],
    *,
    tchans: int,
    fchans: int,
    manifests: Optional[Mapping[str, Union[str, Path, None]]] = None,
) -> Dict:
    """
    Check a format-2 cache against the frame geometry and, optionally, the
    manifests the caller expects each split to come from. Returns meta.json.

    Raises CacheError on any mismatch; never falls back to anything else.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        partial = cache_dir.with_name(cache_dir.name + ".partial")
        hint = f" (an unfinished build exists at {partial})" if partial.exists() else ""
        raise CacheError(f"cache directory not found: {cache_dir}{hint}")

    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        raise CacheError(f"{meta_path} missing: cache is incomplete")
    meta = json.loads(meta_path.read_text())

    if meta.get("format_version") != CACHE_FORMAT_VERSION:
        raise CacheError(
            f"{cache_dir}: format_version {meta.get('format_version')!r}, expected "
            f"{CACHE_FORMAT_VERSION}. Legacy caches carry no manifest identity or index."
        )
    if meta.get("complete") is not True:
        raise CacheError(f"{cache_dir}: meta.json does not mark the cache complete")

    geom = meta["geometry"]
    if geom["fchans"] != fchans or geom["n_obs"] * geom["tchans_per_obs"] != tchans:
        raise CacheError(
            f"{cache_dir}: cache geometry n_obs={geom['n_obs']} x tchans_per_obs="
            f"{geom['tchans_per_obs']}, fchans={geom['fchans']} does not match frame "
            f"tchans={tchans}, fchans={fchans}"
        )
    snippet_shape = (geom["n_obs"], geom["tchans_per_obs"], geom["fchans"])

    for split in SPLITS:
        info = meta["splits"][split]
        arr = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
        if arr.shape != (info["n_snippets"], *snippet_shape) or arr.dtype != np.float32:
            raise CacheError(
                f"{cache_dir}/{split}.npy has shape {arr.shape} dtype {arr.dtype}, "
                f"meta expects ({info['n_snippets']}, {snippet_shape}) float32"
            )
        del arr
        with np.load(cache_dir / f"{split}_index.npz") as idx:
            if len(idx["manifest_row"]) != info["n_snippets"]:
                raise CacheError(f"{cache_dir}/{split}_index.npz length does not match meta")

    for split, expected in (manifests or {}).items():
        if expected is None:
            continue
        recorded = meta["splits"][split]["manifest_sha256"]
        actual = sha256_file(expected)
        if actual != recorded:
            raise CacheError(
                f"{cache_dir}: {split} split was extracted from a manifest with sha256 "
                f"{recorded[:12]}…, but {expected} has sha256 {actual[:12]}…"
            )

    return meta


def check_cache_content(
    cache_dir: Union[str, Path],
    manifests: Mapping[str, Union[str, Path]],
    *,
    n_per_split: int,
    seed: int = 0,
) -> List[Dict]:
    """
    Re-read randomly chosen cached snippets from the raw files named by the
    index and compare them exactly. Returns one record per checked snippet
    (split, sample, manifest_row, chan_start, equal). Structural checks and
    manifest identity go through verify_cache first.
    """
    import h5py
    try:
        import hdf5plugin  # noqa: F401 — registers bitshuffle/LZ4 filters for BL HDF5 files
    except ImportError:
        pass

    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    geom = meta["geometry"]
    verify_cache(cache_dir, tchans=geom["n_obs"] * geom["tchans_per_obs"],
                 fchans=geom["fchans"], manifests=manifests)

    rng = np.random.default_rng(seed)
    records = []
    for split, manifest in manifests.items():
        cadences = load_manifest(manifest)
        arr = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
        with np.load(cache_dir / f"{split}_index.npz") as idx:
            rows, chans = idx["manifest_row"], idx["chan_start"]
        n = min(n_per_split, len(arr))
        for i in np.sort(rng.choice(len(arr), size=n, replace=False)):
            row, chan = int(rows[i]), int(chans[i])
            equal = True
            for obs, path in enumerate(cadences[row]):
                with h5py.File(path, "r") as f:
                    dset = f["data"]
                    sel = (slice(0, geom["tchans_per_obs"]),)
                    sel += (0,) if dset.ndim == 3 else ()
                    raw = dset[sel + (slice(chan, chan + geom["fchans"]),)]
                equal &= bool(np.array_equal(arr[i, obs], raw))
            records.append({"split": split, "sample": int(i), "manifest_row": row,
                            "chan_start": chan, "equal": equal})
    return records
