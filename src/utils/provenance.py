"""Provenance records written next to caches, teacher statistics and training runs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.data.cache import sha256_file

_ROOT = Path(__file__).resolve().parent.parent.parent


def git_state() -> Dict[str, Any]:
    """HEAD commit and whether tracked files differ from it (None if git is unavailable)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=_ROOT, capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(dirty.strip()),
                "modified": [line[3:] for line in dirty.splitlines()]}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "modified": None}


def environment() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in ("numpy", "torch", "pytorch_lightning", "h5py"):
        module = sys.modules.get(name)
        versions[name] = getattr(module, "__version__", None) if module else None
    return versions


def data_identity(cfg_data: Dict[str, Any]) -> Dict[str, Any]:
    """The cache meta.json for dataset.source: cache, else the configured paths."""
    dataset = cfg_data.get("dataset", {})
    if dataset.get("source") == "cache":
        meta = json.loads((Path(dataset["cache_dir"]) / "meta.json").read_text())
        return {"source": "cache", "cache_dir": dataset["cache_dir"], "cache_meta": meta}
    return {"source": dataset.get("source"), "cache_dir": dataset.get("cache_dir"),
            "cadence_list": dataset.get("cadence_list")}


def teacher_identity(model_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """sha256 of the UDMA teacher checkpoint and normalization statistics, if any."""
    teacher = model_cfg.get("teacher")
    if not teacher:
        return None
    out: Dict[str, Any] = {}
    for key in ("checkpoint", "norm_stats"):
        path = teacher.get(key)
        if path is not None:
            full = _ROOT / path
            out[key] = {"path": path, "sha256": sha256_file(full) if full.is_file() else None}
    return out


def write_provenance(path: Path, cfg: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """Write a JSON provenance record and return it."""
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git": git_state(),
        "environment": environment(),
        "data": data_identity(cfg["data"]),
        "teacher": teacher_identity(cfg.get("model", {})),
        **extra,
    }
    Path(path).write_text(json.dumps(record, indent=2))
    return record
