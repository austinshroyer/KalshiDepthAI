from __future__ import annotations

import shutil
from pathlib import Path


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9
