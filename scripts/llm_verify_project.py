from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _bootstrap import ROOT
from llm_ake.config import load_experiment, parse_condition
from llm_ake.governance import verify_prediction_freeze, verify_prompt_freeze
from llm_ake.prompt import load_prompt
from llm_ake.util import iter_jsonl, load_json, sha256_file


FORBIDDEN_FIELDS = {"kw", "present_kws", "gold", "gold_all", "gold_present", "gold_keyphrases", "rf"}


def verify_prepared() -> None:
    config = load_experiment(ROOT)
    load_prompt(ROOT / "prompts" / "ZEROSHOT_PROMPT_v1.txt")
    manifest_path = ROOT / "data" / "manifests" / "llm_dataset_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise RuntimeError("Dataset manifest is not PASS")
    expected_counts = manifest["test_domain_counts"]
    all_ids = set()
    for domain in config["domains"]:
        view_ids = {}
        for view in config["views"]:
            path = ROOT / "data" / "inference" / domain / f"{view}.jsonl"
            records = list(iter_jsonl(path))
            if len(records) != int(expected_counts[domain]):
                raise RuntimeError(f"Unexpected count in {path}: {len(records)}")
            if any(FORBIDDEN_FIELDS & set(row) for row in records):
                raise RuntimeError(f"Gold/leakage field found in {path}")
            view_ids[view] = {str(row["doc_id"]) for row in records}
            if len(view_ids[view]) != len(records):
                raise RuntimeError(f"Duplicate IDs in {path}")
            if sha256_file(path) != next(item["sha256"] for item in manifest["files"] if item["path"] == path.relative_to(ROOT).as_posix()):
                raise RuntimeError(f"Manifest hash mismatch: {path}")
        if view_ids["TA"] != view_ids["FT"]:
            raise RuntimeError(f"TA/FT ID mismatch for {domain}")
        all_ids.update(view_ids["TA"])
        for view in config["views"]:
            smoke = list(iter_jsonl(ROOT / "data" / "smoke" / domain / f"{view}.jsonl"))
            if len(smoke) != 1 or smoke[0]["split"] != "dev" or FORBIDDEN_FIELDS & set(smoke[0]):
                raise RuntimeError(f"Invalid smoke input for {domain}/{view}")
    if len(all_ids) != int(manifest["test_unique_doc_ids"]):
        raise RuntimeError("Global test ID count mismatch")
    sealed = load_json(ROOT / "data" / "_sealed_test_gold" / "SEALED_GOLD_MANIFEST.json")
    if sealed.get("status") != "SEALED" or len(sealed.get("files", [])) != 3:
        raise RuntimeError("Sealed gold manifest is invalid")
    for item in sealed["files"]:
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Sealed gold hash mismatch: {path}")
    access = load_json(ROOT / "outputs" / "governance" / "test_gold_access.json")
    if access.get("status") != "none" or int(access.get("access_count", 0)) != 0:
        raise RuntimeError("Prepared project must have test_gold_access=none")
    print({"status": "PASS", "stage": "prepared", "test_documents": len(all_ids), "domains": expected_counts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepared", "prompt-frozen", "predictions-frozen"], default="prepared")
    args = parser.parse_args()
    verify_prepared()
    if args.stage in {"prompt-frozen", "predictions-frozen"}:
        verify_prompt_freeze(ROOT)
        print({"status": "PASS", "stage": "prompt-frozen"})
    if args.stage == "predictions-frozen":
        verify_prediction_freeze(ROOT)
        print({"status": "PASS", "stage": "predictions-frozen"})


if __name__ == "__main__":
    main()
