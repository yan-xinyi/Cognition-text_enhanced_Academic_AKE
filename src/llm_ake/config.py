from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .util import load_json


def project_root_from(path: Path) -> Path:
    return path.resolve()


def load_experiment(root: Path) -> Dict[str, Any]:
    return load_json(root / "configs" / "experiment.json")


def parse_condition(config: Dict[str, Any], condition: str) -> Tuple[str, str, Dict[str, Any]]:
    if condition not in config["conditions"]:
        raise ValueError(f"Unknown condition {condition}; expected one of {config['conditions']}")
    model_key, view = condition.split("-", 1)
    return model_key, view, config["models"][model_key]

