"""Format-2 snippet cache: extraction layout, identity, resume, and strict loading."""

import json
import warnings

import h5py
import numpy as np
import pytest

from scripts.preprocess_cache import build_cache
from src.data.cache import CacheError, sha256_file, verify_cache
from src.data.torch_dataset import CachedDataset, build_datasets

N_OBS, NTIME, NCHANS, FCHANS, TCHANS = 6, 16, 4096, 1024, 96   # 4 channel blocks per file


def _write_cadence(root, name, seed, nchans=NCHANS):
    rng = np.random.default_rng(seed)
    paths = []
    for obs in range(N_OBS):
        path = root / f"{name}_obs{obs}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=rng.random((NTIME, 1, nchans), dtype=np.float32) + 1.0)
        paths.append(str(path))
    return paths


def _write_manifest(path, cadences):
    path.write_text("".join(" ".join(c) + "\n" for c in cadences))
    return path


@pytest.fixture
def manifests(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    train = [_write_cadence(raw, f"train{i}", seed=i) for i in range(2)]
    val = [_write_cadence(raw, f"val{i}", seed=10 + i) for i in range(2)]
    return (_write_manifest(tmp_path / "train.txt", train),
            _write_manifest(tmp_path / "val.txt", val), train, val)


def _build(train_m, val_m, out, **kw):
    return build_cache(train_m, val_m, out, tchans=TCHANS, fchans=FCHANS,
                       train_snippets_per_cadence=3, val_snippets_per_cadence=2, seed=42, **kw)


def _cache_cfg(cache_dir, train_m, val_m, source="cache"):
    return {
        "frame": {"tchans": TCHANS, "fchans": FCHANS},
        "preprocessing": {"bandpass_method": "median", "mad_epsilon": 1e-6},
        "dataset": {"source": source, "cache_dir": str(cache_dir),
                    "train_cadence_list": str(train_m), "val_cadence_list": str(val_m)},
    }


def test_layout_index_and_content_match_raw(manifests, tmp_path):
    train_m, val_m, train, val = manifests
    out = tmp_path / "cache"
    assert _build(train_m, val_m, out)

    meta = json.loads((out / "meta.json").read_text())
    assert meta["complete"] is True and meta["format_version"] == 2
    assert meta["splits"]["train"]["manifest_sha256"] == sha256_file(train_m)
    assert meta["splits"]["val"]["n_snippets"] == 4
    assert not (tmp_path / "cache.partial").exists()

    for split, cadences, spc in (("train", train, 3), ("val", val, 2)):
        arr = np.load(out / f"{split}.npy")
        idx = np.load(out / f"{split}_index.npz")
        assert arr.shape == (len(cadences) * spc, N_OBS, NTIME, FCHANS)
        np.testing.assert_array_equal(idx["manifest_row"], np.repeat(np.arange(len(cadences)), spc))
        np.testing.assert_array_equal(idx["cadence_row_start"], np.arange(len(cadences)) * spc)
        for i, (row, chan) in enumerate(zip(idx["manifest_row"], idx["chan_start"])):
            for obs, path in enumerate(cadences[row]):
                with h5py.File(path, "r") as f:
                    np.testing.assert_array_equal(arr[i, obs], f["data"][:, 0, chan:chan + FCHANS])


def test_existing_output_is_never_overwritten(manifests, tmp_path):
    train_m, val_m, *_ = manifests
    out = tmp_path / "cache"
    out.mkdir()
    with pytest.raises(FileExistsError):
        _build(train_m, val_m, out)


@pytest.mark.parametrize("stop_after", [1, 2, 3])
def test_interrupted_build_resumes_to_identical_cache(manifests, tmp_path, stop_after):
    train_m, val_m, *_ = manifests
    reference = tmp_path / "reference"
    _build(train_m, val_m, reference)

    out = tmp_path / "cache"
    assert not _build(train_m, val_m, out, stop_after_cadences=stop_after)
    assert not out.exists() and (tmp_path / "cache.partial").exists()
    with pytest.raises(CacheError, match="unfinished build"):
        verify_cache(out, tchans=TCHANS, fchans=FCHANS)

    assert _build(train_m, val_m, out)
    for name in ("train.npy", "val.npy"):
        assert (out / name).read_bytes() == (reference / name).read_bytes()
    for name in ("train_index.npz", "val_index.npz"):
        a, b = np.load(out / name), np.load(reference / name)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_resume_with_different_settings_is_refused(manifests, tmp_path):
    train_m, val_m, *_ = manifests
    out = tmp_path / "cache"
    _build(train_m, val_m, out, stop_after_cadences=1)
    with pytest.raises(RuntimeError, match="different manifests or settings"):
        build_cache(train_m, val_m, out, tchans=TCHANS, fchans=FCHANS,
                    train_snippets_per_cadence=2, val_snippets_per_cadence=2, seed=42)


def test_unreadable_row_aborts_instead_of_skipping(manifests, tmp_path):
    train_m, val_m, train, _ = manifests
    bad = train[0][:-1] + _write_cadence(tmp_path / "raw", "short", seed=99, nchans=2048)[-1:]
    bad_m = _write_manifest(tmp_path / "bad.txt", [bad, train[1]])
    with pytest.raises(ValueError, match="row 0"):
        _build(bad_m, val_m, tmp_path / "cache")


def test_loader_rejects_mismatched_or_incomplete_cache(manifests, tmp_path):
    train_m, val_m, train, _ = manifests
    out = tmp_path / "cache"
    _build(train_m, val_m, out)

    other_m = _write_manifest(tmp_path / "other.txt", train[:1])
    with pytest.raises(CacheError, match="sha256"):
        build_datasets([], _cache_cfg(out, other_m, val_m))
    with pytest.raises(CacheError, match="geometry"):
        verify_cache(out, tchans=TCHANS, fchans=512)

    meta = json.loads((out / "meta.json").read_text())
    meta["splits"]["train"]["n_snippets"] += 1
    (out / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(CacheError, match="train.npy"):
        verify_cache(out, tchans=TCHANS, fchans=FCHANS)

    (out / "meta.json").unlink()
    with pytest.raises(CacheError, match="incomplete"):
        verify_cache(out, tchans=TCHANS, fchans=FCHANS)
    with pytest.raises(CacheError, match="not found"):
        build_datasets([], _cache_cfg(tmp_path / "missing", train_m, val_m))


def test_cache_mode_loads_and_legacy_fallback_warns(manifests, tmp_path):
    train_m, val_m, train, _ = manifests
    out = tmp_path / "cache"
    _build(train_m, val_m, out)

    train_ds, val_ds = build_datasets([], _cache_cfg(out, train_m, val_m))
    assert isinstance(train_ds, CachedDataset)
    assert (len(train_ds), len(val_ds)) == (6, 4)
    assert train_ds[0].shape == (1, TCHANS, FCHANS)

    legacy = _cache_cfg(tmp_path / "missing", train_m, val_m)
    del legacy["dataset"]["source"]
    legacy["frame"].update(stride_train=FCHANS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            build_datasets(train, legacy)
        except Exception:
            pass  # only the warning matters here; raw reading is out of scope
    assert any("falling back to reading raw files" in str(w.message) for w in caught)
