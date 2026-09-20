from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognition_ake.evaluation import evaluate_cutoffs, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset-level micro P/R/F1@k")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_cutoffs(read_jsonl(args.predictions), args.k)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

