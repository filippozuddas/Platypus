"""Filename keys that match one GBT scan across resolution products.

`identify` gives a node-independent scan key, `node_scan_key` keeps the
recorder node. They are distinct associations: the same scan recorded by
another node is different data. The N00a sampling discovery that used to live
here was replaced by `scripts/scan_headers.py` + `src/product_association.py`
(removed after 6cb228a).
"""

from __future__ import annotations

import re
from pathlib import Path

PRODUCT_NAMES = {"fine": "0000", "time": "0001", "mid": "0002"}
PRODUCT_RE = re.compile(
    r"\.(000[012]|fine|time|mid)\.(h5|hdf5|fil)$", re.IGNORECASE
)
NODE_RE = re.compile(r"^blc[0-9a-z]+_", re.IGNORECASE)


def identify(path: Path) -> tuple[str | None, str | None]:
    """Return product and node-independent filename key when recognizable."""
    match = PRODUCT_RE.search(path.name)
    if match is None:
        return None, None
    product = PRODUCT_NAMES.get(match.group(1).lower(), match.group(1))
    core = path.name[: match.start()].removesuffix(".rawspec")
    key = NODE_RE.sub("", core).lower()
    return product, key or None


def node_scan_key(path: Path) -> str | None:
    """Keep the recorder node when matching a scan across resolution products."""
    match = PRODUCT_RE.search(path.name)
    if match is None:
        return None
    return path.name[: match.start()].removesuffix(".rawspec").lower() or None
