"""Transfer-learning traffic collision detection and automated reporting."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Accident", "NonAccident"]   # label 0 = accident, 1 = non-accident (as in the paper)


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)
