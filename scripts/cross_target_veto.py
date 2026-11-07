"""
Final short list of a search run under a named rule: the rule's verdict plus
the cross-target veto (src/search/veto.py), and the reviewed PDF.

The veto needs the other cadences of the same night and compute node, so the
per-cadence search cannot apply it: this stage runs once the night is scanned.
For each row of --run_dir it

* re-decides ``in_short_list`` under --rule from the saved anomaly maps
  (allowed only if the run's recorded rule differs in verdict fields alone, see
  ``VERDICT_FIELDS``; the maps must first reproduce the recorded verdict);
* counts the other targets whose clusters (read from --reference_dir, a scan of
  the same manifest with the same clusters) overlap each candidate and each
  injection region, and drops from the final list those met in at least
  ``veto_min_targets`` of them (the rule's value, or --min_targets);
* writes ``background_candidates_<tag>.csv`` / ``injections_<tag>.csv`` next to
  the originals, which are left untouched, and ``<method>_candidates_<tag>.pdf``:
  the scan's PDF without the pages that left the list (needs ``pypdf``). A
  re-run with the same tag rewrites these three; the scan's files are never
  modified.

The PDF written during a scan is therefore the pre-veto one; the reviewed PDF is
this stage's. Injection verdicts are re-decided only when the run used --rule's
verdict fields (the association of an injection to a cluster depends on them).

Usage:
    python scripts/cross_target_veto.py --rule hybrid_v4 \\
        --manifest data/raw/gbt_0000_nightsplit_devcal.txt \\
        --reference_dir outputs/devscan/hybrid_v3_04b \\
        --run_dir outputs/devscan/hybrid_v3_04b
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cache import load_manifest
from src.search.candidates import short_list_order
from src.search.export import subset_pdf
from src.search.pipeline import (
    RULES, VERDICT_FIELDS, rule_from_record, shares_clusters, short_list_verdict,
)
from src.search.veto import count_other_targets, same_session_rows

FCHANS = 1024


def _row_dirs(run_dir: Path):
    return {int(d.name[3:6]): d for d in sorted(run_dir.glob("row*")) if (d / "DONE").exists()}


def _recorded_rule(run_dir: Path):
    prov = json.loads((run_dir / "provenance.json").read_text())
    return rule_from_record(prov["run_identity"]["selection_rule"])


def _as_bool(col: pd.Series) -> np.ndarray:
    return col.astype(str).str.lower().eq("true").to_numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rule", required=True, choices=list(RULES))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reference_dir", type=Path, required=True)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--min_targets", type=int, default=None,
                   help="Override the rule's veto_min_targets (diagnostics); changes the tag.")
    p.add_argument("--tol_channels", type=int, default=0)
    p.add_argument("--no_pdf", action="store_true", help="Skip the reviewed PDF.")
    args = p.parse_args()

    rule = RULES[args.rule]
    min_targets = rule.veto_min_targets if args.min_targets is None else args.min_targets
    if min_targets < 1:
        raise SystemExit(f"rule {args.rule} has no cross-target veto; pass --min_targets")
    tag = args.rule if args.min_targets is None else f"{args.rule}_veto{min_targets}"
    recorded, reference = _recorded_rule(args.run_dir), _recorded_rule(args.reference_dir)
    for name, r in (("run_dir", recorded), ("reference_dir", reference)):
        if not shares_clusters(r, rule):
            raise SystemExit(f"{name} was scanned with a rule that changes clusters or "
                             f"thresholds relative to {args.rule}")
    same_verdict = all(getattr(recorded, k) == getattr(rule, k) for k in VERDICT_FIELDS
                       if k != "veto_min_targets")

    sessions = same_session_rows(load_manifest(args.manifest))
    refs = {r: pd.read_csv(d / "background_candidates.csv")
            for r, d in _row_dirs(args.reference_dir).items()}
    n = dict(bg_recorded=0, bg_rule=0, bg_vetoed=0, bg_final=0, inj_rule=0, inj_vetoed=0,
             inj_final=0, pdf_rows=0)
    for row, d in _row_dirs(args.run_dir).items():
        others = sessions.get(row, [])
        missing = [o for o in others if o not in refs]
        if missing:
            raise SystemExit(f"row {row}: reference rows {missing} missing from {args.reference_dir}")
        references = [refs[o] for o in others]

        bg = pd.read_csv(d / "background_candidates.csv")
        stored = _as_bool(bg.in_short_list)
        if len(bg):
            amaps = np.load(d / "background_maps.npz")["amap"]
            off_ceiling = json.loads((d / "background_thresholds.json").read_text())["off_ceiling"]
            if len(amaps) != len(bg) or not np.array_equal(
                    short_list_verdict(amaps, off_ceiling, recorded), stored):
                raise SystemExit(f"row {row}: saved maps do not reproduce the recorded verdict")
            verdict = stored if same_verdict else short_list_verdict(amaps, off_ceiling, rule)
        else:
            verdict = stored
        bg["in_short_list_rule"] = verdict
        bg["n_other_targets"] = count_other_targets(
            bg.f_start_first.to_numpy(), bg.f_start_last.to_numpy() + FCHANS, references,
            FCHANS, args.tol_channels)
        bg["cross_target_veto"] = bg.n_other_targets >= min_targets
        final = verdict & ~bg.cross_target_veto.to_numpy()
        bg["in_short_list_final"] = final
        bg.to_csv(d / f"background_candidates_{tag}.csv", index=False)
        n["bg_recorded"] += int(stored.sum())
        n["bg_rule"] += int(verdict.sum())
        n["bg_vetoed"] += int((verdict & bg.cross_target_veto.to_numpy()).sum())
        n["bg_final"] += int(final.sum())

        pdfs = sorted(d.glob("*_candidates.pdf"))
        if pdfs and not args.no_pdf:
            order = short_list_order(stored, bg.on_off_contrast.to_numpy())
            if not set(np.flatnonzero(final)) <= set(order.tolist()):
                raise SystemExit(f"row {row}: the final list holds candidates absent from "
                                 f"{pdfs[0].name}; the reviewed PDF cannot be cut from it")
            keep = [k for k, i in enumerate(order) if final[i]]
            if keep:
                subset_pdf(pdfs[0], d / f"{pdfs[0].stem}_{tag}.pdf", keep, len(order))
                n["pdf_rows"] += 1

        if (d / "injections.csv").exists() and same_verdict:
            inj = pd.read_csv(d / "injections.csv", usecols=["injection_id", "region_lo",
                                                              "region_hi", "in_short_list"])
            inj["n_other_targets"] = count_other_targets(
                inj.region_lo.to_numpy(), inj.region_hi.to_numpy(), references, FCHANS,
                args.tol_channels)
            inj["cross_target_veto"] = inj.n_other_targets >= min_targets
            sl = _as_bool(inj.in_short_list)
            inj["in_short_list_final"] = sl & ~inj.cross_target_veto.to_numpy()
            inj[["injection_id", "n_other_targets", "cross_target_veto",
                 "in_short_list_final"]].to_csv(d / f"injections_{tag}.csv", index=False)
            n["inj_rule"] += int(sl.sum())
            n["inj_vetoed"] += int((sl & inj.cross_target_veto.to_numpy()).sum())
            n["inj_final"] += int(inj.in_short_list_final.sum())

    has_inj = any((d / "injections.csv").exists() for d in _row_dirs(args.run_dir).values())
    summary = {"rule": args.rule, "selection_rule": json.loads(json.dumps(asdict(rule))),
               "recorded_rule_same_verdict": same_verdict, "min_targets": min_targets,
               "tol_channels": args.tol_channels, "reference_dir": str(args.reference_dir),
               "background_short_list_recorded": n["bg_recorded"],
               "background_short_list_rule": n["bg_rule"],
               "background_vetoed": n["bg_vetoed"], "background_short_list_final": n["bg_final"],
               "rows_with_reviewed_pdf": n["pdf_rows"]}
    if has_inj and same_verdict:
        summary.update(injections_short_listed=n["inj_rule"], injections_vetoed=n["inj_vetoed"],
                       injections_short_list_final=n["inj_final"])
    elif has_inj:
        summary["injections"] = ("not re-decided: the run used other verdict fields; "
                                 "re-decide offline from row_max_* / off_ceiling")
    (args.run_dir / f"short_list_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
