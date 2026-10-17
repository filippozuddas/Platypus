"""Injection catalog: immutability, model-independent and order-independent
sites, and application to a whole cadence's raw data."""

import json

import numpy as np
import pytest
import yaml

from src.data.morphologies import MORPHOLOGIES
from src.search.injection import Catalog, build_catalog, build_patch
from src.search.pipeline import CadenceWindows, PatchedWindows

FCHANS, TCHANS, NCHANS, NTIME = 1024, 96, 40960, 16
FCH1_MHZ, FOFF_MHZ = 8000.0, -2.7939677238464355e-6


@pytest.fixture(scope="module")
def data_cfg():
    with open("configs/data/gbt_fine_nightsplit.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["raw"]["nchans"] = NCHANS   # toy cadence
    return cfg


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.txt"
    path.write_text("".join(" ".join(f"/data/cad{r}_obs{o}.h5" for o in range(6)) + "\n"
                            for r in range(3)))
    return path


def _build(tmp_path, name, manifest, data_cfg, **kw):
    args = dict(seed=7, morphologies=list(MORPHOLOGIES), snr_list=[0, 30],
                on_patterns=["3on", "2on_02"], n_sites=4)
    args.update(kw)
    build_catalog(tmp_path / name, manifest=manifest, data_cfg=data_cfg, **args)
    return Catalog(tmp_path / name)


def test_catalog_is_deterministic_and_never_overwritten(tmp_path, manifest, data_cfg):
    a = _build(tmp_path, "a", manifest, data_cfg)
    b = _build(tmp_path, "b", manifest, data_cfg)
    assert a.meta["sites_sha256"] == b.meta["sites_sha256"]
    assert len(list(a.injections())) == 3 * 4 * len(MORPHOLOGIES) * 2 * 2
    with pytest.raises(FileExistsError):
        _build(tmp_path, "a", manifest, data_cfg)


def test_sites_do_not_depend_on_class_order_or_row_selection(tmp_path, manifest, data_cfg):
    full = _build(tmp_path, "full", manifest, data_cfg)
    reordered = _build(tmp_path, "rev", manifest, data_cfg, morphologies=list(MORPHOLOGIES)[::-1],
                       rows=[2, 0])
    key = lambda s: (s["row"], s["site"], s["morphology"])
    full_sites = {key(s): s for s in full.sites if s["row"] in ("0", "2")}
    assert full_sites == {key(s): s for s in reordered.sites}


def test_sites_are_shared_across_morphologies(tmp_path, manifest, data_cfg):
    catalog = _build(tmp_path, "c", manifest, data_cfg)
    windows = {}
    for s in catalog.sites:
        windows.setdefault((s["row"], s["site"]), set()).add(s["window_c0"])
    assert all(len(v) == 1 for v in windows.values())
    seeds = {(s["row"], s["site"], s["morphology"]): s["site_seed"] for s in catalog.sites}
    assert len(set(seeds.values())) == len(seeds)


def test_tampered_catalog_is_rejected(tmp_path, manifest, data_cfg):
    catalog = _build(tmp_path, "t", manifest, data_cfg)
    sites = catalog.dir / "sites.csv"
    sites.write_text(sites.read_text().replace(",0,", ",1,", 1))
    with pytest.raises(ValueError, match="sha256"):
        Catalog(catalog.dir)
    other = manifest.with_name("other.txt")
    other.write_text(manifest.read_text() + "x\n")
    with pytest.raises(ValueError, match="not the manifest"):
        _build(tmp_path, "u", manifest, data_cfg).check_manifest(other)


@pytest.fixture
def cadence_windows(tmp_path):
    rng = np.random.default_rng(1)
    obs = [rng.random((NTIME, NCHANS), dtype=np.float32) + 1.0 for _ in range(6)]
    with CadenceWindows(obs, tmp_path / "w", "w", fchans=FCHANS, tchans=TCHANS, stride=512,
                        preproc={"bandpass_method": "polynomial", "poly_degree": 3}) as w:
        yield w, obs


def test_every_morphology_applies_on_only_and_contained(tmp_path, manifest, data_cfg, cadence_windows):
    windows, obs = cadence_windows
    catalog = _build(tmp_path, "apply", manifest, data_cfg, rows=[0])
    seen = set()
    for inj in catalog.injections(row=0):
        patch, record = build_patch(windows, inj, data_cfg, fch1_mhz=FCH1_MHZ, foff_mhz=FOFF_MHZ)
        assert set(patch.data) == set(inj.on_indices)       # OFFs and unused ONs: not in the patch
        if inj.snr == 0:
            assert record["span_lo"] is None                 # control: nothing injected
            for i, block in patch.data.items():
                np.testing.assert_array_equal(block, obs[i][:, patch.c0:patch.c1])
            continue
        seen.add(inj.morphology)
        assert record["window_c0"] <= record["span_lo"] < record["span_hi"] <= record["window_c1"]
        # Signed foff: channel up, frequency down.
        assert record["freq_mhz_at_span_lo"] > record["freq_mhz_at_span_hi"]
        patched = PatchedWindows(windows, patch)
        assert len(patched.affected) >= 1
    assert seen == set(MORPHOLOGIES)


def test_changed_injector_is_detected(tmp_path, manifest, data_cfg, cadence_windows):
    windows, _ = cadence_windows
    catalog = _build(tmp_path, "m", manifest, data_cfg, rows=[0])
    inj = next(i for i in catalog.injections(row=0) if i.snr > 0)
    meta = json.loads(inj.site_meta)
    meta["tampered"] = True
    tampered = inj.__class__(**{**inj.__dict__, "site_meta": json.dumps(meta, sort_keys=True)})
    with pytest.raises(ValueError, match="injector changed"):
        build_patch(windows, tampered, data_cfg, fch1_mhz=FCH1_MHZ, foff_mhz=FOFF_MHZ)
