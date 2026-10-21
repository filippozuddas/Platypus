"""
Build an immutable injection catalog for a manifest (src/search/injection.py).

Sites are drawn from the seed and each cadence's identity, never from model
scores, and shared by all morphologies. The catalog directory is never
overwritten. Reads only the manifest and the data config, no data files.

Usage:
    PYTHONPATH=. python scripts/build_injection_catalog.py \\
        --manifest data/raw/gbt_0000_nightsplit_devcal.txt \\
        --data_config configs/data/gbt_fine_nightsplit.yaml \\
        --out outputs/injection_catalogs/devcal_v1 \\
        --on_patterns 3on 2on_02 2on_04 2on_24 1on_2 \\
        --rows_per_night 8
"""

import argparse
from pathlib import Path

import yaml

from src.data.cache import load_manifest
from src.data.morphologies import MORPHOLOGIES
from src.search.injection import ON_PATTERNS, Catalog, build_catalog, select_rows_per_night


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data_config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--morphologies", nargs="+", default=list(MORPHOLOGIES), choices=list(MORPHOLOGIES))
    p.add_argument("--snr_list", type=float, nargs="+", default=[0, 10, 15, 20, 30, 50],
                   help="SNR 0 is the no-injection control at each site.")
    p.add_argument("--on_patterns", nargs="+", default=["3on"], choices=list(ON_PATTERNS))
    p.add_argument("--n_sites", type=int, default=25)
    rows = p.add_mutually_exclusive_group()
    rows.add_argument("--rows", type=int, nargs="+", default=None,
                      help="Manifest rows to include (default: all).")
    rows.add_argument("--rows_per_night", type=int, default=None,
                      help="Seeded rows per observing night, spread over nodes "
                           "(src/search/injection.py: select_rows_per_night).")
    p.add_argument("--edge_margin", type=int, default=0)
    args = p.parse_args()

    data_cfg = yaml.safe_load(args.data_config.read_text())
    selection = None
    if args.rows_per_night is not None:
        args.rows = select_rows_per_night(load_manifest(args.manifest), args.rows_per_night,
                                          seed=args.seed)
        selection = {"rule": "select_rows_per_night", "per_night": args.rows_per_night,
                     "seed": args.seed, "gap_hours": 12.0}
    path = build_catalog(args.out, manifest=args.manifest, data_cfg=data_cfg, seed=args.seed,
                         morphologies=args.morphologies, snr_list=args.snr_list,
                         on_patterns=args.on_patterns, n_sites=args.n_sites, rows=args.rows,
                         edge_margin=args.edge_margin, row_selection=selection)
    catalog = Catalog(args.out)
    n_inj = sum(1 for _ in catalog.injections())
    print(f"Catalog -> {path}")
    print(f"  {len(catalog.meta['rows'])} rows x {args.n_sites} sites x "
          f"{len(args.morphologies)} morphologies x {len(args.snr_list)} SNR x "
          f"{len(args.on_patterns)} ON patterns = {n_inj} injections")
    print(f"  sites.csv sha256 {catalog.meta['sites_sha256']}")


if __name__ == "__main__":
    main()
