"""
Annotate a search run with the cross-target veto (src/search/veto.py).

For every row of --run_dir, counts the other targets of the same night and
compute node whose candidates (read from --reference_dir, a background scan of
the same manifest) overlap each background candidate and each injection
region, and writes background_candidates_veto.csv / injections_veto.csv next
to the originals, which are left untouched.

Usage:
    python scripts/cross_target_veto.py \\
        --manifest data/raw/gbt_0000_nightsplit_devcal.txt \\
        --reference_dir outputs/devscan/nightsplit_04b \\
        --run_dir outputs/recovery/devcal_v1 --min_targets 2
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.cache import load_manifest
from src.search.veto import count_other_targets, same_session_rows

FCHANS = 1024


def _row_dirs(run_dir: Path):
    return {int(d.name[3:6]): d for d in sorted(run_dir.glob("row*")) if (d / "DONE").exists()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reference_dir", type=Path, required=True)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--min_targets", type=int, default=2)
    p.add_argument("--tol_channels", type=int, default=0)
    args = p.parse_args()

    sessions = same_session_rows(load_manifest(args.manifest))
    refs = {r: pd.read_csv(d / "background_candidates.csv")
            for r, d in _row_dirs(args.reference_dir).items()}
    n_bg = n_bg_veto = n_inj = n_inj_veto = 0
    for row, d in _row_dirs(args.run_dir).items():
        others = sessions.get(row, [])
        missing = [o for o in others if o not in refs]
        if missing:
            raise SystemExit(f"row {row}: reference rows {missing} missing from {args.reference_dir}")
        references = [refs[o] for o in others]

        bg = pd.read_csv(d / "background_candidates.csv")
        bg["n_other_targets"] = count_other_targets(
            bg.f_start_first.to_numpy(), bg.f_start_last.to_numpy() + FCHANS, references,
            FCHANS, args.tol_channels)
        bg["cross_target_veto"] = bg.n_other_targets >= args.min_targets
        bg.to_csv(d / "background_candidates_veto.csv", index=False)
        n_bg += int(bg.in_short_list.sum())
        n_bg_veto += int((bg.in_short_list & bg.cross_target_veto).sum())

        if (d / "injections.csv").exists():
            inj = pd.read_csv(d / "injections.csv", usecols=["injection_id", "region_lo", "region_hi",
                                                              "in_short_list"])
            inj["n_other_targets"] = count_other_targets(
                inj.region_lo.to_numpy(), inj.region_hi.to_numpy(), references, FCHANS,
                args.tol_channels)
            inj["cross_target_veto"] = inj.n_other_targets >= args.min_targets
            inj[["injection_id", "n_other_targets", "cross_target_veto"]].to_csv(
                d / "injections_veto.csv", index=False)
            sl = inj.in_short_list.astype(str).str.lower().eq("true")
            n_inj += int(sl.sum())
            n_inj_veto += int((sl & inj.cross_target_veto).sum())

    summary = {"reference_dir": str(args.reference_dir), "min_targets": args.min_targets,
               "tol_channels": args.tol_channels, "background_short_list": n_bg,
               "background_short_list_vetoed": n_bg_veto, "injections_short_listed": n_inj,
               "injections_short_listed_vetoed": n_inj_veto}
    (args.run_dir / f"cross_target_veto_min{args.min_targets}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
