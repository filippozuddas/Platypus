#!/usr/bin/env python3
"""Inventory GBT resolution products without reading spectrogram arrays.

This is a discovery step, not a cadence builder or a final train/test split.
Observation matches are inferred from filenames and remain provisional until
checked against headers and observing-session metadata.

Example:
    python scripts/inventory_products.py /content/wd_mybook2/ \
        --out outputs/product_inventory_20261006/summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = {
    "train": ROOT / "data/raw/gbt_0000_train_cadences.txt",
    "val": ROOT / "data/raw/gbt_0000_val_cadences.txt",
    "heldout": ROOT / "data/raw/gbt_0000_heldout_cadences.txt",
}
PRODUCT_NAMES = {"fine": "0000", "time": "0001", "mid": "0002"}
PRODUCT_RE = re.compile(
    r"\.(000[012]|fine|time|mid)\.(h5|hdf5|fil)$", re.IGNORECASE
)
NODE_RE = re.compile(r"^blc[0-9a-z]+_", re.IGNORECASE)
DATA_EXTENSIONS = {".h5", ".hdf5", ".fil"}


def identify(path: Path) -> tuple[str | None, str | None]:
    """Return product and node-independent filename key when recognizable."""
    match = PRODUCT_RE.search(path.name)
    if match is None:
        return None, None
    product = PRODUCT_NAMES.get(match.group(1).lower(), match.group(1))
    core = path.name[: match.start()].removesuffix(".rawspec")
    key = NODE_RE.sub("", core).lower()
    return product, key or None


def node_scan_key(path: Path) -> str | None:
    """Keep the recorder node when matching a scan across resolution products."""
    match = PRODUCT_RE.search(path.name)
    if match is None:
        return None
    return path.name[: match.start()].removesuffix(".rawspec").lower() or None


def legacy_splits(
    manifests: dict[str, Path] | None = None,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, object]]]:
    key_splits: dict[str, set[str]] = defaultdict(set)
    node_splits: dict[str, set[str]] = defaultdict(set)
    provenance: dict[str, dict[str, object]] = {}
    for split, manifest in (MANIFESTS if manifests is None else manifests).items():
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing 0000 manifest: {manifest}")
        count = 0
        rows = 0
        content = manifest.read_bytes()
        for line in content.decode("utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows += 1
            for name in line.split():
                product, key = identify(Path(name))
                if product == "0000" and key:
                    key_splits[key].add(split)
                    node_key = node_scan_key(Path(name))
                    if node_key:
                        node_splits[node_key].add(split)
                    count += 1
        provenance[split] = {
            "path": str(manifest), "rows": rows, "file_entries": count,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return key_splits, node_splits, provenance


def output_paths(out: Path) -> tuple[Path, Path]:
    """Reject existing artifacts and a parent that is a file before inventorying."""
    csv_path = out.with_name(out.stem + ".files.csv")
    for parent in out.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Output parent is a file: {parent}; choose a new directory")
    for path in (out, csv_path):
        if path.exists():
            raise ValueError(f"Output already exists: {path}; choose a new output name")
    if out == csv_path:
        raise ValueError("Summary and CSV output paths must differ")
    return out, csv_path


def safe_value(value: object) -> object:
    """Convert common HDF5 scalar attributes to JSON-friendly values."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sample_header(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path)}
    try:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    raise ValueError("HDF5 dataset 'data' is absent")
                dataset = handle["data"]
                result["shape"] = list(dataset.shape)
                attrs = dataset.attrs
                for name in ("fch1", "foff", "tsamp", "tstart", "source_name", "nifs"):
                    if name in attrs:
                        result[name] = safe_value(attrs[name])
        else:
            import blimpy

            waterfall = blimpy.Waterfall(str(path), load_data=False)
            for name in ("fch1", "foff", "tsamp", "tstart", "source_name", "nifs", "nchans"):
                if name in waterfall.header:
                    result[name] = safe_value(waterfall.header[name])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def spaced_samples(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = {round(i * (len(paths) - 1) / (count - 1)) for i in range(count)}
    return [paths[i] for i in sorted(indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing the new products")
    parser.add_argument("--out", type=Path, help="Write summary JSON and sibling .files.csv")
    parser.add_argument("--samples-per-product", type=int, default=3)
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"Directory does not exist: {args.root}")
    if args.samples_per_product < 2:
        parser.error("--samples-per-product must be at least 2")
    if args.out:
        try:
            output_paths(args.out)
        except ValueError as exc:
            parser.error(str(exc))

    old_splits, old_node_splits, manifest_provenance = legacy_splits()
    rows: list[dict[str, str]] = []
    product_paths: dict[str, list[Path]] = defaultdict(list)
    product_keys: dict[str, set[str]] = defaultdict(set)

    for path in args.root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        product, key = identify(path)
        node_key = node_scan_key(path)
        rows.append({
            "path": str(path),
            "product": product or "unknown",
            "observation_key": key or "",
            "node_observation_key": node_key or "",
            "legacy_0000_split_by_scan": ",".join(sorted(old_splits.get(key or "", set()))),
            "legacy_0000_split_same_node": ",".join(
                sorted(old_node_splits.get(node_key or "", set()))
            ),
        })
        if product:
            product_paths[product].append(path)
            if key:
                product_keys[product].add(key)

    rows.sort(key=lambda row: (row["product"], row["observation_key"], row["path"]))
    product_summary: dict[str, dict[str, object]] = {}
    for product in ("0000", "0001", "0002"):
        paths = sorted(product_paths.get(product, []))
        keys = product_keys.get(product, set())
        split_counts = Counter(
            split for key in keys for split in old_splits.get(key, set())
        )
        product_summary[product] = {
            "files": len(paths),
            "files_matching_same_node_in_0000": sum(
                node_scan_key(p) in old_node_splits for p in paths
            ),
            "files_with_new_node_in_known_scan": sum(
                identify(p)[1] in old_splits and node_scan_key(p) not in old_node_splits
                for p in paths
            ),
            "files_with_new_scan": sum(identify(p)[1] not in old_splits for p in paths),
            "filename_observation_keys": len(keys),
            "keys_in_existing_0000_manifests": len(keys & old_splits.keys()),
            "keys_not_in_existing_0000_manifests": len(keys - old_splits.keys()),
            "matched_keys_by_existing_split": dict(sorted(split_counts.items())),
            "example_filenames": [p.name for p in spaced_samples(paths, min(3, len(paths)))]
            if len(paths) >= 2 else [p.name for p in paths],
            "sample_headers": [
                sample_header(p) for p in spaced_samples(paths, args.samples_per_product)
            ]
            if paths else [],
        }

    unknown = [row["path"] for row in rows if row["product"] == "unknown"]
    summary = {
        "schema_version": 2,
        "root": str(args.root),
        "data_files": len(rows),
        "existing_0000_manifests": manifest_provenance,
        "existing_0000_filename_observation_keys": len(old_splits),
        "existing_0000_keys_in_multiple_old_splits": sum(
            len(splits) > 1 for splits in old_splits.values()
        ),
        "products": product_summary,
        "keys_shared_by_new_0001_and_0002": len(
            product_keys.get("0001", set()) & product_keys.get("0002", set())
        ),
        "unrecognized_file_count": len(unknown),
        "unrecognized_filename_examples": unknown[:10],
        "interpretation": (
            "Scan and same-node matches are distinct filename-based associations. "
            "A split inherited from another node does not establish exact file membership. "
            "Header samples cover only a few files. "
            "Verify timing, frequency coverage and observing-session groups before splitting."
        ),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        csv_path = args.out.with_name(args.out.stem + ".files.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "path", "product", "observation_key", "node_observation_key",
                    "legacy_0000_split_by_scan", "legacy_0000_split_same_node",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"File list saved: {csv_path}")


if __name__ == "__main__":
    main()
