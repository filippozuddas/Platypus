"""Filename identity regressions; no telescope arrays or model required."""

from pathlib import Path

from src.product_inventory import identify, node_scan_key


def test_scan_key_ignores_the_node_and_node_scan_key_keeps_it():
    old = Path("blc00_guppi_60098_12345_TARGET_0001.rawspec.0000.h5")
    other_node = Path(old.name.replace("blc00", "blc50").replace(".0000.h5", ".0001.h5"))
    assert identify(old) == ("0000", "guppi_60098_12345_target_0001")
    assert identify(other_node) == ("0001", identify(old)[1])
    assert node_scan_key(other_node) != node_scan_key(old)
    same_node = Path(old.name.replace(".0000.h5", ".mid.h5"))
    assert identify(same_node)[0] == "0002"
    assert node_scan_key(same_node) == node_scan_key(old)
    assert identify(Path(old.name.replace("60098", "60099")))[1] != identify(old)[1]


def test_unrecognized_names_have_no_key():
    assert identify(Path("notes.txt")) == (None, None)
    assert node_scan_key(Path("blc00_guppi_60098_12345_TARGET_0001.h5")) is None
