#!/usr/bin/env python3
"""CLI for metadata-only GBT product discovery; not a full header audit."""

import sys
from pathlib import Path

# Allow invocation from the repository without importing ML/data dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product_inventory import main  # noqa: E402


if __name__ == "__main__":
    main()
