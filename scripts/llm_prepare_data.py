from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT
from llm_ake.data_prepare import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gold-free TA/FT inputs from academic_gold_free.jsonl")
    parser.add_argument("--source-gold-free", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare_dataset(ROOT, args.source_gold_free)
    print(f"Prepared {manifest['test_unique_doc_ids']} unique test documents")
    print(ROOT / "data" / "manifests" / "dataset_manifest.json")


if __name__ == "__main__":
    main()

