from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognition_ake.models import METHODS
from cognition_ake.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a controlled cognition-text AKE model")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ake.yaml")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--domain", choices=["PMC", "LIS", "IEEE"], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 52, 62], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("train_index", "validation_index", "test_index"):
        path = Path(config["data"][key])
        config["data"][key] = str(path if path.is_absolute() else ROOT / path)
    output = Path(config["output_root"])
    output = output if output.is_absolute() else ROOT / output
    result = train(config, args.method, args.domain, args.seed, output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

