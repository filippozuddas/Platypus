"""
Background search and injection recovery, one load per cadence.

For each manifest row: load the six observations once, run the production
search (src/search/pipeline.py) and write its background outputs (window
scores, candidates, cluster maps, thresholds — the development-scan products);
then, if a catalog is given, re-search the cadence once per catalog injection
(src/search/injection.py, src/search/recovery.py) and append one outcome row
per injection to injections.csv.

Resumable: a row with a DONE file is skipped, and injections already in its
injections.csv are not repeated. A run directory is tied to one checkpoint,
manifest, catalog and selection rule (provenance.json); re-running into it
with different ones is refused.

Usage (server):
    PYTHONPATH=. python scripts/search_and_inject.py \\
        --checkpoint outputs/training/<run>/checkpoints/last.ckpt \\
        --model_config configs/model/udma_nightsplit.yaml \\
        --data_config configs/data/gbt_fine_nightsplit.yaml \\
        --manifest data/raw/gbt_0000_nightsplit_devcal.txt \\
        --catalog outputs/injection_catalogs/<name> \\
        --out_dir outputs/recovery/<name>
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from src.data.cache import load_manifest, sha256_file
from src.data.torch_dataset import _load_full_obs
from src.search.export import export_candidates
from src.search.injection import Catalog, cadence_key
from src.search.pipeline import (
    CadenceSearch, CadenceWindows, SelectionRule, enable_determinism, load_model,
    rank_for_export, read_cadence_meta,
)
from src.search.recovery import run_injections, write_background
from src.utils.provenance import write_provenance

METHOD = "topk"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_config", type=Path, required=True)
    p.add_argument("--data_config", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--catalog", type=Path, default=None,
                   help="Injection catalog; without it only the background search runs.")
    p.add_argument("--rows", type=int, nargs="+", default=None,
                   help="Manifest rows (default: the catalog's rows, or every row).")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--off_ceiling_probe", type=int, default=SelectionRule.off_ceiling_probe)
    p.add_argument("--top_k", type=int, default=30,
                   help="Review budget per cadence: sets 'exported' and the PDF length.")
    p.add_argument("--export_pdf", action="store_true",
                   help="Also write the background review PDF of each cadence.")
    p.add_argument("--no_deterministic", action="store_true")
    return p.parse_args()


def _identity(args, rule, catalog):
    return {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest),
        "catalog_sites_sha256": catalog.meta["sites_sha256"] if catalog else None,
        "selection_rule": asdict(rule),
        "method": METHOD,
    }


def main():
    args = parse_args()
    data_cfg = yaml.safe_load(args.data_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())
    frame = data_cfg["frame"]
    fchans, tchans = frame["fchans"], frame["tchans"]
    stride = frame.get("stride_infer", fchans // 2)
    rule = SelectionRule(off_ceiling_probe=args.off_ceiling_probe)

    catalog = Catalog(args.catalog) if args.catalog else None
    if catalog:
        catalog.check_manifest(args.manifest)
    cadences = load_manifest(args.manifest)
    rows = args.rows if args.rows is not None else (catalog.meta["rows"] if catalog
                                                    else list(range(len(cadences))))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(args, rule, catalog)
    prov_path = args.out_dir / "provenance.json"
    if prov_path.exists():
        recorded = json.loads(prov_path.read_text())["run_identity"]
        if recorded != identity:
            raise SystemExit(f"{args.out_dir} belongs to a run with a different checkpoint, "
                             f"manifest, catalog or rule; use another --out_dir")
    numerics = {"deterministic": False} if args.no_deterministic else \
        {"deterministic": True, **enable_determinism()}
    if not prov_path.exists():
        write_provenance(prov_path, {"data": data_cfg, "model": model_cfg},
                         run_identity=identity, numerics=numerics,
                         manifest={"path": str(args.manifest), "n_rows": len(cadences)},
                         catalog=str(args.catalog) if catalog else None,
                         batch_size=args.batch_size, top_k=args.top_k)

    model = load_model(args.checkpoint, model_cfg, (tchans, fchans, 1), args.device)
    print(f"Rows: {len(rows)}  catalog: {args.catalog}  numerics: {numerics}")

    for row in rows:
        paths = [Path(p) for p in cadences[row]]
        key = cadence_key(cadences[row])
        cad_dir = args.out_dir / f"row{row:03d}_{Path(key).stem}"
        if (cad_dir / "DONE").exists():
            print(f"row {row}: done, skipping")
            continue
        cad_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        meta = read_cadence_meta(paths[0])
        foff_hz = meta["foff_mhz"] * 1e6
        freq_step_hz = foff_hz if foff_hz else data_cfg["raw"]["df"]
        print(f"\nrow {row}: {meta['source']} fch1={meta['fch1_mhz']:.1f} MHz -> {cad_dir}")
        obs = [_load_full_obs(p, frame.get("downsample_factor", 1)) for p in paths]
        t_load = time.time() - t0

        with CadenceWindows(obs, args.out_dir / "_scratch", f"row{row:03d}", fchans=fchans,
                            tchans=tchans, stride=stride, preproc=data_cfg["preprocessing"]) as windows:
            del obs
            if catalog and windows.nchans != catalog.meta["geometry"]["nchans"]:
                raise SystemExit(f"row {row}: {windows.nchans} channels, catalog expects "
                                 f"{catalog.meta['geometry']['nchans']}")
            search = CadenceSearch(model, windows, method=METHOD, rule=rule,
                                   freq_step_hz=freq_step_hz, fch1_mhz=meta["fch1_mhz"],
                                   probe_seed=row, batch_size=args.batch_size,
                                   device=args.device, num_workers=args.num_workers)
            write_background(search, cad_dir)
            t_bg = time.time() - t0 - t_load

            if args.export_pdf:
                ranked, amaps = rank_for_export(search.background)
                th = search.background.thresholds
                export_candidates(ranked, amaps, model=model, windows=windows, method=METHOD,
                                  top_k=args.top_k, plot_format="pdf", cad_dir=cad_dir,
                                  median=th["median"], mad_sigma=th["mad_sigma"], cad_idx=row,
                                  target=meta["source"], freq_step_hz=freq_step_hz,
                                  fch1_mhz=meta["fch1_mhz"], nchans_total=windows.nchans,
                                  obs_date=meta["date"], device=args.device)

            n_inj = 0
            if catalog:
                n_inj = run_injections(search, catalog.injections(row=row), data_cfg,
                                       fch1_mhz=meta["fch1_mhz"], foff_mhz=meta["foff_mhz"],
                                       out_csv=cad_dir / "injections.csv", top_k=args.top_k)
        timing = {"load_s": round(t_load, 1), "background_s": round(t_bg, 1),
                  "injections_s": round(time.time() - t0 - t_load - t_bg, 1),
                  "n_injections_this_call": n_inj,
                  "n_candidates": len(search.background.clusters)}
        (cad_dir / "DONE").write_text(json.dumps(timing, indent=2))
        print(f"row {row}: {timing}")


if __name__ == "__main__":
    main()
