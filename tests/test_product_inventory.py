"""Metadata identity regressions; no telescope arrays or model required."""

from pathlib import Path

import pytest

from src.product_inventory import identify, legacy_splits, node_scan_key, output_paths


def test_scan_match_does_not_assign_an_unseen_node_to_legacy_split(tmp_path):
    old = "blc00_guppi_60098_12345_TARGET_0001.rawspec.0000.h5"
    manifest = tmp_path / "train.txt"
    manifest.write_text(old + "\n", encoding="utf-8")
    scans, nodes, _ = legacy_splits({"train": manifest})
    new = Path(old.replace("blc00", "blc50").replace(".0000.h5", ".0001.h5"))
    product, scan = identify(new)
    assert product == "0001"
    assert scans[scan] == {"train"}
    assert node_scan_key(new) not in nodes
    same = Path(old.replace(".0000.h5", ".mid.h5"))
    assert nodes[node_scan_key(same)] == {"train"}
    unrelated = Path(str(new).replace("60098", "60099"))
    assert identify(unrelated)[1] not in scans


def test_manifest_hash_retains_row_order(tmp_path):
    first = "blc00_guppi_60098_12345_A_0001.rawspec.0000.h5"
    second = "blc00_guppi_60099_12345_B_0001.rawspec.0000.h5"
    manifest = tmp_path / "train.txt"
    manifest.write_text(f"{first}\n{second}\n", encoding="utf-8")
    scans_before, _, before = legacy_splits({"train": manifest})
    manifest.write_text(f"{second}\n{first}\n", encoding="utf-8")
    scans_after, _, after = legacy_splits({"train": manifest})
    assert scans_before == scans_after
    assert before["train"]["sha256"] != after["train"]["sha256"]
    assert after["train"]["rows"] == 2


def test_output_rejects_historical_extensionless_file_as_parent(tmp_path):
    historical = tmp_path / "product_inventory"
    historical.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="parent is a file"):
        output_paths(historical / "new.json")
    with pytest.raises(ValueError, match="already exists"):
        output_paths(historical)
    assert output_paths(tmp_path / "new" / "summary.json") == (
        tmp_path / "new" / "summary.json", tmp_path / "new" / "summary.files.csv"
    )
