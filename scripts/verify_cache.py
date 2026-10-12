"""
Verify a format-2 snippet cache: structure, manifest identity, and content.

Structure and manifest sha256 are checked first (src/data/cache.py:
verify_cache); then --n-per-split random snippets per split are re-read from
the raw files named by the index and compared exactly.

Usage:
    PYTHONPATH=. python scripts/verify_cache.py \\
        --cache /path/to/cache_dir \\
        --train-cadence-list data/raw/gbt_0000_nightsplit_train.txt \\
        --val-cadence-list   data/raw/gbt_0000_nightsplit_devcal.txt \\
        --n-per-split 8
"""

import argparse
import sys
import time
from pathlib import Path

from src.data.cache import check_cache_content


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--train-cadence-list", type=Path, required=True)
    p.add_argument("--val-cadence-list", type=Path, required=True)
    p.add_argument("--n-per-split", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    records = check_cache_content(
        args.cache, {"train": args.train_cadence_list, "val": args.val_cadence_list},
        n_per_split=args.n_per_split, seed=args.seed,
    )
    for r in records:
        status = "OK  " if r["equal"] else "DIFF"
        print(f"{status} {r['split']:5s} sample={r['sample']:>8d} "
              f"row={r['manifest_row']:>4d} chan_start={r['chan_start']}")
    n_bad = sum(not r["equal"] for r in records)
    print(f"\nStructure and manifest identity OK; {len(records) - n_bad}/{len(records)} "
          f"snippets identical to raw ({time.time() - t0:.0f} s).")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
