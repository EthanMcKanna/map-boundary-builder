from __future__ import annotations

import os

NETWORK_BLOCK_ENV = "MAP_BOUNDARY_BLOCK_NETWORK"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def network_blocked() -> bool:
    return os.environ.get(NETWORK_BLOCK_ENV, "").strip().lower() in _TRUE_VALUES
