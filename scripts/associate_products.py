#!/usr/bin/env python3
"""Night-split association of 0001/0002 header tables; writes draft manifests.

Example:
    python scripts/associate_products.py \
        --headers outputs/split_20261104/exotica_0001_headers.csv \
                  outputs/split_20261104/exotica_0002_headers.csv \
        --headers_0000 outputs/split_20261012/exotica_0000_headers.csv \
        --out_dir outputs/split_20261104/association
"""

import argparse
import sys
from pathlib import Path

# Allow invocation from the repository without importing ML/data dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import product_association as pa  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headers", type=Path, nargs="+", required=True,
                        help="scan_headers.py CSVs, one product each")
    parser.add_argument("--headers_0000", type=Path,
                        help="0000 scan_headers.py CSV, for band and start checks")
    parser.add_argument("--manifest_dir", type=Path, default=Path("data/raw"),
                        help="Directory with gbt_0000_nightsplit_{train,devcal,final}.txt")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="New directory; refused if it exists")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"Output already exists: {args.out_dir}; choose a new name")

    inputs: dict[str, object] = {"headers": {}, "manifests": {}}
    product_rows: dict[str, list[dict]] = {}
    for path in args.headers:
        rows, bad = pa.load_headers(path)
        products = {r["product"] for r in rows}
        if len(products) != 1 or products & product_rows.keys():
            parser.error(f"{path}: expected one new product per CSV, got {sorted(products)}")
        product_rows[products.pop()] = rows
        inputs["headers"][str(path)] = {"sha256": pa.sha256(path), "rows": len(rows),
                                        "rows_with_errors": len(bad)}
    manifests = {}
    for split in pa.SPLITS:
        path = args.manifest_dir / f"gbt_0000_nightsplit_{split}.txt"
        manifests[split] = pa.load_manifest(path)
        inputs["manifests"][str(path)] = {"sha256": pa.sha256(path),
                                          "rows": len(manifests[split])}
    headers_0000 = None
    if args.headers_0000:
        headers_0000, _ = pa.load_headers(args.headers_0000)
        inputs["headers_0000"] = {"path": str(args.headers_0000),
                                  "sha256": pa.sha256(args.headers_0000),
                                  "rows": len(headers_0000)}

    result = pa.associate(product_rows, manifests, headers_0000)
    pa.write_outputs(result, args.out_dir, inputs)
    for product, info in pa.summary(result, inputs)["products"].items():
        print(f"{product}: {info['files']} files, {info['sequences']} sequences, "
              f"manifest rows {info['manifest_rows']}, flags {info['flags']}")
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
