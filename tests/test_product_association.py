"""Night-split association of new products, on synthetic header tables only."""

import pytest

from src import product_association as pa

DAY = 60055
SCAN_STEP_S = 325


def _seq(node, start_s, *, product="0001", on="ON", offs=("A", "B", "C"),
         band=(1000.0, 1187.5), scan0=10, day=DAY, ntime=858112, n=6):
    """Header rows of one ABACAD sequence, as scan_headers writes them."""
    targets = [on, offs[0], on, offs[1], on, offs[2]][:n]
    rows = []
    for i, target in enumerate(targets):
        sec = start_s + i * SCAN_STEP_S
        name = f"{node}_guppi_{day}_{sec:05d}_{target}_{scan0 + i:04d}.rawspec.{product}.h5"
        rows.append({
            "file": f"/data/{name}", "node": node, "target": target,
            "tstart_mjd": str(day + sec / 86400), "ntime": str(ntime),
            "f_start_MHz": str(band[0]), "f_stop_MHz": str(band[1]),
        })
    return rows


def _manifest_row(rows):
    return [r["file"].replace(".0001.h5", ".0000.h5") for r in rows]


def _by_first_file(result, product="0001"):
    return {s.files[0]["file"]: s for s in result.sequences[product]}


def test_same_node_and_scan_inherits_the_manifest_split():
    seq = _seq("blc00", 1000)
    result = pa.associate({"0001": seq}, {"train": [_manifest_row(seq)]})
    (s,) = result.sequences["0001"]
    assert (s.split, s.match, s.night, s.flags) == ("train", "node_scan", "S01", [])


def test_sequence_without_0000_counterpart_takes_the_split_of_its_night():
    known = _seq("blc00", 1000)
    extra = _seq("blc50", 1000)  # same scans on a node absent from 0000
    result = pa.associate({"0001": known + extra}, {"devcal": [_manifest_row(known)]})
    s = _by_first_file(result)[extra[0]["file"]]
    assert (s.split, s.match, s.night) == ("devcal", "night", "S01")
    assert s.scan_split == "devcal"


def test_night_without_0000_rows_is_left_unassigned():
    known = _seq("blc00", 1000)
    lone = _seq("blc00", 1000, day=DAY + 10)
    result = pa.associate({"0001": known + lone}, {"final": [_manifest_row(known)]})
    s = _by_first_file(result)[lone[0]["file"]]
    assert s.split == "" and "no_0000_night" in s.flags
    assert lone[0]["file"] not in "\n".join(result.manifest_lines("0001", "final"))


def test_nights_are_numbered_by_time_and_must_not_mix_splits():
    first, second = _seq("blc00", 1000), _seq("blc00", 1000, day=DAY + 3)
    result = pa.associate(
        {"0001": second + first},
        {"final": [_manifest_row(second)], "train": [_manifest_row(first)]},
    )
    assert [(n.name, n.split) for n in result.nights] == [("S01", "train"), ("S02", "final")]
    late = _seq("blc01", 20000)  # same night as `first`, but listed in final
    with pytest.raises(ValueError, match="S01"):
        pa.associate({"0001": first + late},
                     {"train": [_manifest_row(first)], "final": [_manifest_row(late)]})


def test_broken_sequences_are_flagged_and_kept_out_of_manifests():
    short = _seq("blc00", 1000, n=5)
    bad_pattern = _seq("blc01", 1000, offs=("A", "A", "C"))
    mixed = _seq("blc02", 1000)
    mixed[3]["f_start_MHz"] = "1187.5"
    rows = short + bad_pattern + mixed
    result = pa.associate({"0001": rows}, {"train": [_manifest_row(_seq("blc03", 1000))]})
    flags = {s.node: s.flags for s in result.sequences["0001"]}
    assert "files_5" in flags["blc00"]
    assert "pattern" in flags["blc01"]
    assert "mixed_band" in flags["blc02"]
    assert result.manifest_lines("0001", "train") == []


def test_partial_overlap_with_a_0000_row_is_not_assigned():
    seq = _seq("blc00", 1000)
    row = _manifest_row(seq)
    row[5] = row[5].replace("_0015.", "_0099.")
    result = pa.associate({"0001": seq}, {"train": [row]})
    (s,) = result.sequences["0001"]
    assert s.split == "" and "partial_0000_match" in s.flags
    assert result.missing_in_product["0001"] == [row]


def test_header_mismatch_with_0000_blocks_the_sequence():
    seq = _seq("blc00", 1000)
    headers_0000 = [dict(r, file=r["file"].replace(".0001.h5", ".0000.h5")) for r in seq]
    headers_0000[0] = dict(headers_0000[0], f_start_MHz="2000.0")
    result = pa.associate({"0001": seq}, {"train": [_manifest_row(seq)]}, headers_0000)
    (s,) = result.sequences["0001"]
    assert "band_vs_0000" in s.flags and s.blocked
    assert result.manifest_lines("0001", "train") == []


def test_unequal_ntime_is_reported_but_kept():
    seq = _seq("blc00", 1000)
    seq[4]["ntime"] = "847872"
    result = pa.associate({"0001": seq}, {"train": [_manifest_row(seq)]})
    (s,) = result.sequences["0001"]
    assert s.flags == ["ragged_ntime"] and s.split == "train"
    assert len(result.manifest_lines("0001", "train")) == 1


def test_manifest_rows_are_ordered_by_first_start_then_node():
    late, early_b, early_a = _seq("blc00", 5000), _seq("blc02", 1000), _seq("blc01", 1000)
    rows = late + early_b + early_a
    manifest = [_manifest_row(s) for s in (late, early_b, early_a)]
    result = pa.associate({"0001": rows}, {"train": manifest})
    lines = result.manifest_lines("0001", "train")
    assert [line.split()[0].split("/")[-1][:5] for line in lines] == ["blc01", "blc02", "blc00"]
    assert lines[0] == " ".join(r["file"] for r in early_a)


def test_outputs_refuse_an_existing_directory(tmp_path):
    seq = _seq("blc00", 1000)
    result = pa.associate({"0001": seq}, {"train": [_manifest_row(seq)]})
    out = tmp_path / "association"
    pa.write_outputs(result, out, inputs={})
    assert (out / "gbt_0001_nightsplit_train.txt").read_text().count("\n") == 1
    assert (out / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        pa.write_outputs(result, out, inputs={})
