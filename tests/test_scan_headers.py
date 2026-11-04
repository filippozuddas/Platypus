"""Header scan regressions on tiny synthetic HDF5 files; no telescope arrays."""

import csv

import h5py
import numpy as np
import pytest

from scripts import scan_headers


def _write_filterbank(path, *, compression=None, chunks=None, mask=True):
    with h5py.File(path, "w") as f:
        data = f.create_dataset(
            "data", data=np.zeros((8, 1, 16), dtype="float32"),
            chunks=chunks, compression=compression,
        )
        data.attrs.update({
            "fch1": 1500.0, "foff": -0.5, "tsamp": 1.0,
            "tstart": 60055.0, "source_name": b"MESSIER67",
        })
        if mask:
            f.create_dataset("mask", data=np.zeros((8, 1, 16), dtype="uint8"))


def test_layout_fields_come_from_metadata(tmp_path):
    path = tmp_path / "blc00_guppi_60055_00000_MESSIER67_0055.rawspec.0001.h5"
    _write_filterbank(path, compression="gzip", chunks=(4, 1, 16))
    row = scan_headers.read_header_h5py(path)
    assert row["dtype"] == "float32"
    assert row["chunks"] == "4x1x16"
    assert row["filter_ids"] == "1"  # H5Z_FILTER_DEFLATE
    assert row["has_mask"] is True
    assert row["file_size_bytes"] == path.stat().st_size
    assert 0 < row["storage_bytes"] <= 8 * 16 * 4
    assert row["nchans"] == 16 and row["ntime"] == 8


def test_contiguous_dataset_has_no_chunks_or_filters(tmp_path):
    path = tmp_path / "blc00_guppi_60055_00000_MESSIER67_0055.rawspec.0002.h5"
    _write_filterbank(path, mask=False)
    row = scan_headers.read_header_h5py(path)
    assert row["chunks"] == ""
    assert row["filter_ids"] == ""
    assert row["has_mask"] is False
    assert row["storage_bytes"] == 8 * 16 * 4


def test_csv_keeps_layout_columns(tmp_path):
    path = tmp_path / "blc00_guppi_60055_00000_MESSIER67_0055.rawspec.0001.h5"
    _write_filterbank(path, chunks=(4, 1, 16))
    out = tmp_path / "headers.csv"
    scan_headers.write_csv(out, [scan_headers._process_one(str(path))])
    with out.open() as handle:
        (row,) = list(csv.DictReader(handle))
    for column in ("dtype", "chunks", "filter_ids", "storage_bytes",
                   "file_size_bytes", "has_mask"):
        assert row[column] != "", column


def test_existing_output_is_refused(tmp_path):
    out = tmp_path / "headers.csv"
    out.write_text("old\n")
    with pytest.raises(FileExistsError):
        scan_headers.write_csv(out, [])
    assert out.read_text() == "old\n"
