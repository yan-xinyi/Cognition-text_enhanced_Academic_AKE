from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognition_ake.bootstrap import paired_document_bootstrap
from cognition_ake.evaluation import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired document bootstrap for multi-seed micro F1@5")
    parser.add_argument("--candidate", nargs="+", required=True, type=Path)
    parser.add_argument("--baseline", nargs="+", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = paired_document_bootstrap(
        [read_jsonl(path) for path in args.candidate],
        [read_jsonl(path) for path in args.baseline],
        k=args.k,
        replicates=args.replicates,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

