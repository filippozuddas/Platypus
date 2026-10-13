"""Provenance records and the observing-night reference configuration."""

import json
from pathlib import Path

from src.data.cache import sha256_file
from src.utils.config import load_config
from src.utils.provenance import write_provenance

ROOT = Path(__file__).resolve().parent.parent


def test_provenance_records_cache_meta_and_teacher_hashes(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "meta.json").write_text(json.dumps({"format_version": 2, "seed": 42}))
    ckpt = tmp_path / "teacher.ckpt"
    ckpt.write_bytes(b"weights")
    cfg = {
        "data": {"dataset": {"source": "cache", "cache_dir": str(cache)}},
        "model": {"teacher": {"checkpoint": str(ckpt), "norm_stats": str(tmp_path / "absent.pt")}},
    }

    record = write_provenance(tmp_path / "provenance.json", cfg, n_train_snippets=6)

    assert json.loads((tmp_path / "provenance.json").read_text()) == record
    assert record["data"]["cache_meta"] == {"format_version": 2, "seed": 42}
    assert record["teacher"]["checkpoint"]["sha256"] == sha256_file(ckpt)
    assert record["teacher"]["norm_stats"]["sha256"] is None
    assert record["n_train_snippets"] == 6
    assert set(record["git"]) == {"commit", "dirty", "modified"}


def test_nightsplit_reference_config_is_consistent():
    cfg = load_config(ROOT / "configs/training/udma_gbt_fine_nightsplit.yaml")
    dataset = cfg["data"]["dataset"]
    assert dataset["source"] == "cache"
    assert dataset["cadence_list"] == dataset["train_cadence_list"]
    assert dataset["train_cadence_list"].endswith("nightsplit_train.txt")
    assert dataset["val_cadence_list"].endswith("nightsplit_devcal.txt")

    teacher = cfg["model"]["teacher"]
    assert teacher["checkpoint"].startswith("outputs/training/20260624_084754_057f87c/")
    assert teacher["norm_stats"].endswith("gbt_fine_nightsplit_block3.pt")

    # Hyperparameters of the 20260707_093113_6d0d1ba run.
    training = cfg["training"]
    assert (training["batch_size"], training["learning_rate"], training["epochs"]) == (256, 1e-3, 60)
    assert (training["optimizer"], training["weight_decay"]) == ("adamw", 0.01)
    assert (training["monitor"], training["patience"]) == ("val_st_sum", 8)
