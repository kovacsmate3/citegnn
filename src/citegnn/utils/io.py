"""Small I/O helpers for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write one JSON artifact to disk, creating parents as needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return output_path

