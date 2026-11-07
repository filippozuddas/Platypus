"""The final short-list stage (scripts/cross_target_veto.py) on a toy night:
three targets on one compute node, one cadence with five candidates."""

import json
import sys
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from scripts import cross_target_veto  # noqa: E402
from src.search.pipeline import RULES  # noqa: E402

OFF_CEILING = 1.0
# Row 0: clean 3-ON | 3-ON + one bright OFF | clean, met by both other targets |
# clean, met by one other target | a single lit ON (never short-listed).
F_STARTS = [0, 10_240, 20_480, 30_720, 40_960]
CONTRAST = [3.0, 5.0, 4.0, 2.0, 1.0]


def _maps():
    m = np.full((5, 6, 64), 0.1)
    m[:4, 0, 5] = m[:4, 2, 5] = m[:4, 4, 5] = 4.0
    m[1, 3, 9] = 4.0
    m[4, 0, 5] = 4.0
    return m


def _cadence(sec, target):
    return [f"/d/blc01_guppi_60000_{sec + 320 * o:05d}_{target}_{o:04d}.rawspec.0000.h5"
            for o in range(6)]


def _write_row(d, f_starts, in_short_list, maps, contrast):
    d.mkdir(parents=True)
    pd.DataFrame({"f_start_peak": f_starts, "f_start_first": f_starts, "f_start_last": f_starts,
                  "on_off_contrast": contrast, "in_short_list": in_short_list}) \
        .to_csv(d / "background_candidates.csv", index=False)
    np.savez(d / "background_maps.npz", amap=maps)
    (d / "background_thresholds.json").write_text(json.dumps({"off_ceiling": OFF_CEILING}))
    (d / "DONE").write_text("{}")


def make_run(tmp_path, recorded, stored, pdf=True):
    """Run directory scanned with rule ``recorded``; row 0's stored verdict ``stored``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(" ".join(_cadence(s, t)) for s, t in
                                  ((10_000, "A"), (12_000, "B"), (14_000, "C"))) + "\n")
    run = tmp_path / "run"
    run.mkdir()
    (run / "provenance.json").write_text(json.dumps(
        {"run_identity": {"selection_rule": json.loads(json.dumps(asdict(RULES[recorded])))}}))
    _write_row(run / "row000_A", F_STARTS, stored, _maps(), CONTRAST)
    quiet = np.full((2, 6, 64), 0.1)
    _write_row(run / "row001_B", [20_480, 30_720], [False, False], quiet, [1.0, 1.0])
    _write_row(run / "row002_C", [20_480], [False], quiet[:1], [1.0])
    if pdf:
        # One page per short-listed candidate, in review order; page width
        # (1 + candidate index) inches identifies the candidate.
        order = np.argsort(-np.asarray(CONTRAST))
        with PdfPages(run / "row000_A" / "topk_candidates.pdf") as out:
            for i in order:
                if stored[i]:
                    fig = plt.figure(figsize=(1 + i, 1))
                    out.savefig(fig)
                    plt.close(fig)
    return manifest, run


def run_stage(monkeypatch, manifest, run, *extra):
    monkeypatch.setattr(sys, "argv", ["cross_target_veto.py", "--rule", "hybrid_v4",
                                      "--manifest", str(manifest), "--reference_dir", str(run),
                                      "--run_dir", str(run), *extra])
    cross_target_veto.main()
    return json.loads((run / "short_list_hybrid_v4.json").read_text())


def test_v3_scan_re_decided_under_v4_with_the_veto_and_the_reviewed_pdf(tmp_path, monkeypatch):
    pypdf = pytest.importorskip("pypdf")
    manifest, run = make_run(tmp_path, "hybrid_v3", [True, True, True, True, False])
    s = run_stage(monkeypatch, manifest, run)
    assert (s["background_short_list_recorded"], s["background_short_list_rule"],
            s["background_vetoed"], s["background_short_list_final"]) == (4, 3, 1, 2)
    assert not s["recorded_rule_same_verdict"] and s["min_targets"] == 2
    bg = pd.read_csv(run / "row000_A" / "background_candidates_hybrid_v4.csv")
    assert bg.in_short_list_rule.tolist() == [True, False, True, True, False]
    assert bg.n_other_targets.tolist() == [0, 0, 2, 1, 0]
    assert bg.in_short_list_final.tolist() == [True, False, False, True, False]
    # The scan's PDF is untouched; the reviewed one keeps candidates 0 and 3, in order.
    assert len(pypdf.PdfReader(run / "row000_A" / "topk_candidates.pdf").pages) == 4
    pages = pypdf.PdfReader(run / "row000_A" / "topk_candidates_hybrid_v4.pdf").pages
    assert [round(float(p.mediabox.width) / 72) for p in pages] == [1, 4]


def test_v4_scan_keeps_its_verdict_and_vetoes_injections(tmp_path, monkeypatch):
    manifest, run = make_run(tmp_path, "hybrid_v4", [True, False, True, True, False], pdf=False)
    pd.DataFrame({"injection_id": ["a", "b"], "region_lo": [100, 20_600],
                  "region_hi": [200, 20_700], "in_short_list": [True, True]}) \
        .to_csv(run / "row000_A" / "injections.csv", index=False)
    s = run_stage(monkeypatch, manifest, run, "--no_pdf")
    assert s["recorded_rule_same_verdict"] and s["background_short_list_final"] == 2
    assert (s["injections_short_listed"], s["injections_vetoed"],
            s["injections_short_list_final"]) == (2, 1, 1)
    inj = pd.read_csv(run / "row000_A" / "injections_hybrid_v4.csv")
    assert inj.in_short_list_final.tolist() == [True, False]


def test_refuses_other_clusters_or_maps_that_disagree(tmp_path, monkeypatch):
    manifest, run = make_run(tmp_path, "production", [True, True, True, True, False], pdf=False)
    with pytest.raises(SystemExit, match="changes clusters"):
        run_stage(monkeypatch, manifest, run, "--no_pdf")
    manifest, run = make_run(tmp_path / "b", "hybrid_v3", [True, True, False, True, False],
                             pdf=False)
    with pytest.raises(SystemExit, match="do not reproduce"):
        run_stage(monkeypatch, manifest, run, "--no_pdf")
