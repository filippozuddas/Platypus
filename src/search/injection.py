"""
Injection catalog and its application to a whole cadence's raw data.

A catalog is fixed before any model sees it and stored as a directory:

  catalog.json  — parameters (seed, manifest identity, geometry, morphologies,
                  SNR list, ON patterns) and the sha256 of sites.csv
  sites.csv     — one row per (manifest row, site, morphology): absolute
                  window position, per-site seed, frozen morphology metadata

Every injection is one element of sites x SNR x ON pattern. Sites are absolute
window positions drawn uniformly over the band from a seed and the cadence's
identity, never from model scores, and shared by all morphologies so their
comparison is paired. A site's morphology parameters come from a seed derived
by hashing (catalog seed, cadence, site, morphology), so they do not depend on
the order of classes, rows or injections.

Applying an injection rebuilds the morphology from its seed, checks the
regenerated metadata against the catalog (a changed injector invalidates the
catalog instead of silently injecting something else), renders it with the
validated injectors of ``src/data/morphologies.py`` into the raw window at the
site, and returns a :class:`~src.search.pipeline.Patch` for
:class:`~src.search.pipeline.CadenceSearch`.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.data.cache import load_manifest, sha256_file
from src.data.morphologies import build_morphology
from src.search.pipeline import CadenceWindows, Patch

CATALOG_VERSION = 1
N_OBS = 6

#: ON observations carrying the signal. ``3on`` is the target population of the
#: 0000 search; the 2-ON pairs measure signals present in only two ON scans;
#: 1-ON patterns are controls (PLAN, phase 06).
ON_PATTERNS: Dict[str, Tuple[int, ...]] = {
    "3on": (0, 2, 4),
    "2on_02": (0, 2), "2on_04": (0, 4), "2on_24": (2, 4),
    "1on_0": (0,), "1on_2": (2,), "1on_4": (4,),
}

SITE_COLUMNS = ["row", "cadence_key", "site", "window_c0", "morphology", "site_seed", "site_meta"]


def _canonical(obj) -> str:
    """Stable JSON for metadata comparison (numpy scalars and tuples normalised)."""
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o)}")
    return json.dumps(json.loads(json.dumps(obj, default=default)), sort_keys=True)


def cadence_key(row_paths: Sequence[str]) -> str:
    """Identity of a manifest row that survives reordering: its first file's name."""
    return Path(row_paths[0]).name


def _hash_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _obs_start_mjd(path: str) -> float:
    """MJD of an observation's start from its GUPPI name (``blcNN_guppi_MJD_SEC_...``);
    the 2026-10-12 header check found it equal to the header ``tstart`` for every file."""
    parts = Path(path).name.split("_")
    return int(parts[2]) + int(parts[3]) / 86400


def night_groups(cadences: Sequence[Sequence[str]], gap_hours: float = 12.0) -> List[List[int]]:
    """Manifest rows grouped into observing nights, in time order.

    A new night starts where consecutive cadence start times are more than
    ``gap_hours`` apart (the 0000 split found at most 4.7 h within a night and
    at least 28 h between nights).
    """
    order = sorted(range(len(cadences)), key=lambda r: _obs_start_mjd(cadences[r][0]))
    nights: List[List[int]] = []
    last = None
    for r in order:
        t = _obs_start_mjd(cadences[r][0])
        if last is None or (t - last) * 24 > gap_hours:
            nights.append([])
        nights[-1].append(r)
        last = t
    return [sorted(n) for n in nights]


def select_rows_per_night(cadences: Sequence[Sequence[str]], per_night: int, seed: int,
                          gap_hours: float = 12.0) -> List[int]:
    """``per_night`` rows from each observing night (all of a smaller night),
    drawn from a seed and the rows' identities, never from model outputs.

    Within a night, rows are taken round-robin over compute nodes (``blcNN``,
    i.e. frequency sub-bands) in seeded order, so the selection spreads across
    the band instead of piling onto one node.
    """
    selected: List[int] = []
    for night in night_groups(cadences, gap_hours):
        key = cadence_key(cadences[min(night, key=lambda r: _obs_start_mjd(cadences[r][0]))])
        rng = np.random.default_rng(_hash_seed(seed, key, "rows"))
        by_node: Dict[str, List[int]] = {}
        for r in night:
            by_node.setdefault(Path(cadences[r][0]).name.split("_")[0], []).append(r)
        queues = [list(rng.permutation(by_node[n])) for n in sorted(by_node)]
        queues = [queues[i] for i in rng.permutation(len(queues))]
        picked: List[int] = []
        while len(picked) < per_night and any(queues):
            for q in queues:
                if q and len(picked) < per_night:
                    picked.append(int(q.pop(0)))
        selected.extend(picked)
    return sorted(selected)


def _sample_site_meta(morphology: str, site_seed: int, data_cfg: dict,
                      fchans: int, tchans: int):
    injector = build_morphology(morphology, data_cfg, seed=site_seed)
    site = injector.sample_site(fchans, tchans, N_OBS)
    return injector, site


@dataclass(frozen=True)
class Injection:
    """One catalog entry: a site's morphology at one SNR and ON pattern."""

    injection_id: str
    row: int
    cadence_key: str
    site: int
    window_c0: int
    morphology: str
    site_seed: int
    site_meta: str
    snr: float
    on_pattern: str

    @property
    def on_indices(self) -> Tuple[int, ...]:
        return ON_PATTERNS[self.on_pattern]


def build_catalog(out_dir: Path, *, manifest: Path, data_cfg: dict, seed: int,
                  morphologies: Sequence[str], snr_list: Sequence[float],
                  on_patterns: Sequence[str], n_sites: int,
                  rows: Optional[Sequence[int]] = None, edge_margin: int = 0,
                  row_selection: Optional[dict] = None) -> Path:
    """Write a new catalog directory (never overwritten). Returns its catalog.json.

    ``row_selection`` records how ``rows`` were chosen (e.g. the rule of
    :func:`select_rows_per_night`) in catalog.json.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; catalogs are never overwritten")
    unknown = [p for p in on_patterns if p not in ON_PATTERNS]
    if unknown:
        raise ValueError(f"unknown ON patterns {unknown}; available: {list(ON_PATTERNS)}")
    frame = data_cfg["frame"]
    fchans, tchans = int(frame["fchans"]), int(frame["tchans"])
    nchans = int(data_cfg["raw"]["nchans"])
    lo, hi = edge_margin, nchans - fchans - edge_margin
    if hi < lo:
        raise ValueError(f"no room for a {fchans}-channel window in {nchans} channels")

    cadences = load_manifest(manifest)
    rows = list(range(len(cadences))) if rows is None else sorted(set(rows))
    site_rows = []
    for row in rows:
        key = cadence_key(cadences[row])
        rng = np.random.default_rng(_hash_seed(seed, key, "window"))
        windows = rng.integers(lo, hi + 1, size=n_sites)
        for site, c0 in enumerate(windows):
            for morphology in morphologies:
                site_seed = _hash_seed(seed, key, site, morphology)
                _, s = _sample_site_meta(morphology, site_seed, data_cfg, fchans, tchans)
                site_rows.append({"row": row, "cadence_key": key, "site": site,
                                  "window_c0": int(c0), "morphology": morphology,
                                  "site_seed": site_seed, "site_meta": _canonical(s.meta)})

    out_dir.mkdir(parents=True)
    sites_path = out_dir / "sites.csv"
    with open(sites_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SITE_COLUMNS)
        writer.writeheader()
        writer.writerows(site_rows)
    catalog = {
        "catalog_version": CATALOG_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest),
                     "n_rows": len(cadences)},
        "rows": rows,
        "row_selection": row_selection,
        "geometry": {"fchans": fchans, "tchans": tchans, "nchans": nchans, "n_obs": N_OBS},
        "edge_margin": edge_margin,
        "n_sites": n_sites,
        "morphologies": list(morphologies),
        "snr_list": [float(s) for s in snr_list],
        "on_patterns": {p: list(ON_PATTERNS[p]) for p in on_patterns},
        "sites_file": "sites.csv",
        "sites_sha256": sha256_file(sites_path),
    }
    path = out_dir / "catalog.json"
    path.write_text(json.dumps(catalog, indent=2))
    return path


class Catalog:
    """A catalog read back from disk, with its file hash checked."""

    def __init__(self, catalog_dir: Path):
        self.dir = Path(catalog_dir)
        self.meta = json.loads((self.dir / "catalog.json").read_text())
        sites_path = self.dir / self.meta["sites_file"]
        if sha256_file(sites_path) != self.meta["sites_sha256"]:
            raise ValueError(f"{sites_path} does not match the sha256 in catalog.json")
        with open(sites_path, newline="") as f:
            self.sites = list(csv.DictReader(f))

    def check_manifest(self, manifest: Path) -> None:
        if sha256_file(manifest) != self.meta["manifest"]["sha256"]:
            raise ValueError(f"{manifest} is not the manifest this catalog was built from")

    def injections(self, row: Optional[int] = None) -> Iterator[Injection]:
        """Every injection, optionally for one manifest row; stable order."""
        for s in self.sites:
            if row is not None and int(s["row"]) != row:
                continue
            for pattern in self.meta["on_patterns"]:
                for snr in self.meta["snr_list"]:
                    yield Injection(
                        injection_id=f"{s['row']}:{s['site']}:{s['morphology']}:{pattern}:{snr:g}",
                        row=int(s["row"]), cadence_key=s["cadence_key"], site=int(s["site"]),
                        window_c0=int(s["window_c0"]), morphology=s["morphology"],
                        site_seed=int(s["site_seed"]), site_meta=s["site_meta"],
                        snr=float(snr), on_pattern=pattern)


def build_patch(windows: CadenceWindows, inj: Injection, data_cfg: dict, *,
                fch1_mhz: float, foff_mhz: float) -> Tuple[Patch, dict]:
    """Render one injection into the raw data at its site.

    Returns the :class:`Patch` (only the ON observations of the pattern) and a
    record of what was injected: the injector's info, the absolute channel span
    actually changed and its sky frequencies from the header's signed ``foff``.
    """
    fchans, tchans = windows.fchans, windows.tchans
    c0, c1 = inj.window_c0, inj.window_c0 + fchans
    injector, site = _sample_site_meta(inj.morphology, inj.site_seed, data_cfg, fchans, tchans)
    if _canonical(site.meta) != inj.site_meta:
        raise ValueError(f"{inj.injection_id}: regenerated site metadata differs from the "
                         f"catalog; the injector changed since the catalog was built")

    raw = windows.raw_block(c0, c1)                      # (n_obs, ntime, fchans)
    out, info = injector.inject(raw, site, inj.snr, on_indices=inj.on_indices)
    out = np.asarray(out, dtype=raw.dtype)
    untouched = [i for i in range(len(raw)) if i not in inj.on_indices]
    for i in untouched:
        if not np.array_equal(out[i], raw[i]):
            raise AssertionError(f"{inj.injection_id}: observation {i} changed outside the ON pattern")

    changed = np.flatnonzero((out != raw).any(axis=(0, 1)))
    record = {k: v for k, v in info.items() if np.isscalar(v) or isinstance(v, (str, tuple, list))}
    record.update(injection_id=inj.injection_id, window_c0=c0, window_c1=c1,
                  on_pattern=inj.on_pattern)
    if changed.size:
        span_lo, span_hi = c0 + int(changed[0]), c0 + int(changed[-1]) + 1
        record.update(span_lo=span_lo, span_hi=span_hi,
                      freq_mhz_at_span_lo=fch1_mhz + span_lo * foff_mhz,
                      freq_mhz_at_span_hi=fch1_mhz + (span_hi - 1) * foff_mhz)
    else:
        record.update(span_lo=None, span_hi=None)
    return Patch(c0=c0, data={i: out[i] for i in inj.on_indices}), record
