"""Associate 0001/0002 files with the 0000 night split, from header tables only.

The independence unit is the observing night. A sequence of a new product
inherits the split of the 0000 manifest row with the same node and scans; a
sequence with no such row (e.g. a node whose 0000 copy is missing) takes the
split of the night it falls in. Nights are clusters of sequence starts and are
named S01, S02, ... in time order among the nights that contain 0000 rows,
which reproduces the 17 nights of the 0000 split verification.

Inputs are the CSVs written by ``scripts/scan_headers.py``; no array is read.
Example: ``python scripts/associate_products.py --help``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.product_inventory import identify, node_scan_key

SPLITS = ("train", "devcal", "final")
SEQUENCE_GAP_S = 700  # file starts are ~325 s apart inside a cadence, >=855 s between
NIGHT_GAP_S = 12 * 3600  # 0000: <=4.69 h inside a night, >=28.21 h between nights
BAND_TOL_MHZ = 0.01
TSTART_TOL_S = 1.0
# Flags that keep a sequence out of the manifests until reviewed.
BLOCKING = {
    "pattern", "mixed_band", "partial_0000_match", "no_0000_night",
    "night_split_conflict", "band_vs_0000", "tstart_vs_0000",
}


def _start_s(row: dict) -> float:
    return float(row["tstart_mjd"]) * 86400


def _name_start_s(path: str) -> int:
    """Start time encoded in a BL filename: {node}_guppi_{mjd}_{sec}_..."""
    fields = Path(path).name.split("_")
    return int(fields[2]) * 86400 + int(fields[3])


@dataclass
class Sequence:
    product: str
    files: list[dict]
    split: str = ""
    night: str = ""
    match: str = ""  # node_scan | night | "" when unassigned
    scan_split: str = ""  # split of 0000 rows sharing the scans on any node
    flags: list[str] = field(default_factory=list)

    @property
    def node(self) -> str:
        return self.files[0]["node"]

    @property
    def start_s(self) -> float:
        return _start_s(self.files[0])

    @property
    def node_keys(self) -> frozenset[str]:
        return frozenset(node_scan_key(Path(r["file"])) for r in self.files)

    @property
    def scan_keys(self) -> frozenset[str]:
        return frozenset(identify(Path(r["file"]))[1] for r in self.files)

    @property
    def blocked(self) -> bool:
        return not self.split or any(f.split("_")[0] == "files" or f in BLOCKING
                                     for f in self.flags)


@dataclass
class Night:
    name: str
    split: str
    start_s: float
    end_s: float
    rows_0000: int
    sequences: Counter = field(default_factory=Counter)


@dataclass
class Association:
    sequences: dict[str, list[Sequence]]
    nights: list[Night]
    missing_in_product: dict[str, list[list[str]]]
    night_gaps_s: dict[str, float]

    def manifest_lines(self, product: str, split: str) -> list[str]:
        """Assigned, unblocked sequences, ordered like the 0000 manifests."""
        chosen = [s for s in self.sequences[product] if s.split == split and not s.blocked]
        chosen.sort(key=lambda s: (s.start_s, s.node))
        return [" ".join(r["file"] for r in s.files) for s in chosen]


def group_sequences(product: str, rows: list[dict]) -> list[Sequence]:
    """Split each node's time-ordered files wherever consecutive starts jump."""
    by_node: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_node[row["node"]].append(row)
    sequences = []
    for node_rows in by_node.values():
        node_rows.sort(key=_start_s)
        current = [node_rows[0]]
        for row in node_rows[1:]:
            if _start_s(row) - _start_s(current[-1]) > SEQUENCE_GAP_S:
                sequences.append(Sequence(product, current))
                current = [row]
            else:
                current.append(row)
        sequences.append(Sequence(product, current))
    sequences.sort(key=lambda s: (s.start_s, s.node))
    return sequences


def shape_flags(seq: Sequence) -> list[str]:
    flags = []
    targets = [r["target"] for r in seq.files]
    if len(targets) != 6:
        flags.append(f"files_{len(targets)}")
    elif not (targets[0] == targets[2] == targets[4]
              and len({targets[0], targets[1], targets[3], targets[5]}) == 4):
        flags.append("pattern")
    if len({(r["f_start_MHz"], r["f_stop_MHz"]) for r in seq.files}) > 1:
        flags.append("mixed_band")
    if len({r["ntime"] for r in seq.files}) > 1:
        flags.append("ragged_ntime")
    return flags


def header_flags(seq: Sequence, headers_0000: dict[str, dict]) -> list[str]:
    """Same node and scan must mean the same band and start time as in 0000."""
    flags = set()
    for row in seq.files:
        ref = headers_0000.get(node_scan_key(Path(row["file"])))
        if ref is None:
            continue
        if any(abs(float(row[k]) - float(ref[k])) > BAND_TOL_MHZ
               for k in ("f_start_MHz", "f_stop_MHz")):
            flags.add("band_vs_0000")
        if abs(_start_s(row) - _start_s(ref)) > TSTART_TOL_S:
            flags.add("tstart_vs_0000")
    return sorted(flags)


def _cluster_nights(starts: list[tuple[float, str, object]]) -> list[list[tuple]]:
    starts = sorted(starts, key=lambda item: item[0])
    clusters = [[starts[0]]] if starts else []
    for item in starts[1:]:
        if item[0] - clusters[-1][-1][0] > NIGHT_GAP_S:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def associate(
    product_rows: dict[str, list[dict]],
    manifests: dict[str, list[list[str]]],
    headers_0000: list[dict] | None = None,
) -> Association:
    """Assign every sequence of every product to a split, or flag why not."""
    rows_0000 = [(split, row) for split in SPLITS for row in manifests.get(split, [])]
    row_keys = [frozenset(node_scan_key(Path(n)) for n in row) for _, row in rows_0000]
    by_node_key: dict[str, int] = {}
    scan_splits: dict[str, set[str]] = defaultdict(set)
    for index, (split, row) in enumerate(rows_0000):
        for name in row:
            by_node_key[node_scan_key(Path(name))] = index
            scan_splits[identify(Path(name))[1]].add(split)
    header_index = {node_scan_key(Path(r["file"])): r for r in headers_0000 or []}

    sequences = {p: group_sequences(p, rows) for p, rows in product_rows.items()}
    starts: list[tuple[float, str, object]] = [
        (_name_start_s(row[0]), "0000", split) for split, row in rows_0000
    ]
    starts += [(s.start_s, "new", s) for seqs in sequences.values() for s in seqs]

    clusters = _cluster_nights(starts)
    nights: list[Night] = []
    for cluster in clusters:
        splits = {ref for _, kind, ref in cluster if kind == "0000"}
        name = f"S{len(nights) + 1:02d}" if splits else ""
        if len(splits) > 1:
            raise ValueError(f"Night {name} holds 0000 rows from splits {sorted(splits)}")
        split = next(iter(splits), "")
        if name:
            nights.append(Night(name, split, cluster[0][0], cluster[-1][0],
                                sum(kind == "0000" for _, kind, _ in cluster)))
        for _, kind, ref in cluster:
            if kind == "new":
                ref.night, ref.split = name, split
                if name:
                    nights[-1].sequences[ref.product] += 1

    matched: dict[str, set[int]] = defaultdict(set)
    for product, seqs in sequences.items():
        for seq in seqs:
            seq.flags = shape_flags(seq) + header_flags(seq, header_index)
            seq.scan_split = ",".join(sorted(
                set().union(*(scan_splits.get(k, set()) for k in seq.scan_keys))))
            night_split = seq.split
            hits = {by_node_key[k] for k in seq.node_keys if k in by_node_key}
            if len(hits) == 1 and row_keys[index := next(iter(hits))] == seq.node_keys:
                matched[product].add(index)
                seq.match, seq.split = "node_scan", rows_0000[index][0]
                if night_split != seq.split:
                    seq.flags.append("night_split_conflict")
            elif hits:
                seq.split = ""
                seq.flags.append("partial_0000_match")
            elif not seq.night:
                seq.flags.append("no_0000_night")
            else:
                seq.match = "night"

    missing = {
        product: [row for index, (_, row) in enumerate(rows_0000)
                  if index not in matched[product]]
        for product in sequences
    }
    gaps = {
        "max_inside_night_s": max(
            (b[0] - a[0] for cluster in clusters
             for a, b in zip(cluster, cluster[1:])), default=0.0),
        "min_between_nights_s": min(
            (b.start_s - a.end_s for a, b in zip(nights, nights[1:])), default=0.0),
    }
    return Association(sequences, nights, missing, gaps)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def load_headers(path: Path) -> tuple[list[dict], list[dict]]:
    """Readable rows and rows carrying a parse or header error."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    good, bad = [], []
    for row in rows:
        (bad if row.get("parse_error") or row.get("header_error") else good).append(row)
    return good, bad


def load_manifest(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split() for line in lines if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(result: Association, inputs: dict[str, object]) -> dict[str, object]:
    products = {}
    for product, seqs in result.sequences.items():
        products[product] = {
            "files": sum(len(s.files) for s in seqs),
            "sequences": len(seqs),
            "manifest_rows": {
                split: len(result.manifest_lines(product, split)) for split in SPLITS
            },
            "sequences_by_match": dict(Counter(s.match or "unassigned" for s in seqs)),
            "flags": dict(Counter(f for s in seqs for f in s.flags)),
            "blocked_sequences": [
                {"first_file": s.files[0]["file"], "night": s.night, "flags": s.flags}
                for s in seqs if s.blocked
            ],
            "0000_rows_without_full_counterpart": len(result.missing_in_product[product]),
            "night_matched_scan_split_disagreements": sum(
                s.match == "night" and s.scan_split not in ("", s.split) for s in seqs
            ),
            "ntime": dict(Counter(r["ntime"] for s in seqs for r in s.files)),
        }
    return {
        "schema_version": 1,
        "inputs": inputs,
        "sequence_gap_s": SEQUENCE_GAP_S,
        "night_gap_s": NIGHT_GAP_S,
        "night_gaps_s": result.night_gaps_s,
        "nights": [
            {"night": n.name, "split": n.split, "rows_0000": n.rows_0000,
             "sequences": dict(n.sequences),
             "start_mjd": round(n.start_s / 86400, 5), "end_mjd": round(n.end_s / 86400, 5)}
            for n in result.nights
        ],
        "products": products,
    }


def write_outputs(result: Association, out_dir: Path, inputs: dict[str, object]) -> None:
    """Manifests, per-sequence table, summary and hashes in a new directory."""
    out_dir.mkdir(parents=True, exist_ok=False)
    written = []
    for product in result.sequences:
        for split in SPLITS:
            path = out_dir / f"gbt_{product}_nightsplit_{split}.txt"
            lines = result.manifest_lines(product, split)
            path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            written.append(path)
    table = out_dir / "sequences.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product", "split", "night", "match", "scan_split", "node",
                         "start_mjd", "f_start_MHz", "f_stop_MHz", "on_target",
                         "n_files", "flags", "first_file"])
        for product, seqs in result.sequences.items():
            for s in seqs:
                first = s.files[0]
                writer.writerow([product, s.split, s.night, s.match, s.scan_split, s.node,
                                 first["tstart_mjd"], first["f_start_MHz"],
                                 first["f_stop_MHz"], first["target"], len(s.files),
                                 ";".join(s.flags), first["file"]])
    missing = out_dir / "missing_in_product.txt"
    missing.write_text("".join(
        f"{product} {' '.join(row)}\n"
        for product, rows in result.missing_in_product.items() for row in rows
    ), encoding="utf-8")
    written += [table, missing]
    (out_dir / "summary.json").write_text(
        json.dumps(summary(result, inputs), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in written), encoding="utf-8"
    )
