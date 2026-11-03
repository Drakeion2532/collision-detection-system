# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Transfer-learning traffic collision detection and automated reporting."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Accident", "NonAccident"]   # label 0 = accident, 1 = non-accident (as in the paper)


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

__author__ = "Shubhan Punde"
__provenance__ = "sp2532-2be7050d798f"
