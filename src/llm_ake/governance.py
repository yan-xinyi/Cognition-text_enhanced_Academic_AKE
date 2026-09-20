from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from .config import load_experiment, parse_condition
from .util import atomic_write_json, iter_jsonl, load_json, sha256_file, utc_now


def _frozen_files(root: Path) -> List[Path]:
    manifest = load_json(root / "data" / "manifests" / "llm_dataset_manifest.json")
    paths = [root / item["path"] for item in manifest["files"] if item["dataset"] == "inference"]
    return [
        root / "configs" / "experiment.json",
        root / "prompts" / "ZEROSHOT_PROMPT_v1.txt",
        root / "data" / "manifests" / "llm_dataset_manifest.json",
        *paths,
    ]


def _hash_map(root: Path, paths: List[Path]) -> Dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def freeze_prompt(root: Path) -> Dict[str, Any]:
    config = load_experiment(root)
    smoke_log = root / "outputs" / "ENGINEERING_SMOKE_TEST_LOG.csv"
    if not smoke_log.exists():
        raise RuntimeError("ENGINEERING_SMOKE_TEST_LOG.csv is missing; run all four Smoke Test conditions first")
    latest: Dict[tuple, Dict[str, str]] = {}
    with smoke_log.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            latest[(row["condition"], row["domain"])] = row
    expected = {(condition, domain) for condition in config["conditions"] for domain in config["domains"]}
    missing = sorted(expected - set(latest))
    if missing:
        raise RuntimeError(f"Smoke Test combinations missing: {missing}")
    failed = [key for key in expected if latest[key].get("api_success") != "True" or latest[key].get("parse_success") != "True"]
    if failed:
        raise RuntimeError(f"Smoke Test API/parser failures must be fixed before freeze: {failed}")
    files = _frozen_files(root)
    freeze = {
        "schema_version": 1,
        "status": "FROZEN",
        "frozen_at_utc": utc_now(),
        "experiment": config["experiment"],
        "model_ids": {key: value["model_id"] for key, value in config["models"].items()},
        "provider_route": config["api"]["provider"],
        "generation": config["generation"],
        "parser": config["parser"],
        "cleaning": config["cleaning"],
        "test_gold_access": "none",
        "smoke_test_combinations": len(expected),
        "frozen_files": _hash_map(root, files),
    }
    atomic_write_json(root / "outputs" / "PROMPT_FREEZE.json", freeze)
    return freeze


def verify_prompt_freeze(root: Path) -> Dict[str, Any]:
    path = root / "outputs" / "PROMPT_FREEZE.json"
    if not path.exists():
        raise RuntimeError("PROMPT_FREEZE.json is missing; formal test inference is locked")
    freeze = load_json(path)
    if freeze.get("status") != "FROZEN" or freeze.get("test_gold_access") != "none":
        raise RuntimeError("PROMPT_FREEZE.json is not a valid pre-test freeze")
    for relative, expected_hash in freeze["frozen_files"].items():
        file_path = root / relative
        if not file_path.exists() or sha256_file(file_path) != expected_hash:
            raise RuntimeError(f"Frozen file changed or is missing: {relative}")
    return freeze


def freeze_predictions(root: Path) -> Dict[str, Any]:
    config = load_experiment(root)
    prompt_freeze = verify_prompt_freeze(root)
    sets: List[Dict[str, Any]] = []
    all_doc_ids = set()
    for condition in config["conditions"]:
        _, view, _ = parse_condition(config, condition)
        for domain in config["domains"]:
            input_path = root / "data" / "inference" / domain / f"{view}.jsonl"
            parsed_path = root / "outputs" / "parsed" / condition / f"{domain}.jsonl"
            if not parsed_path.exists():
                raise RuntimeError(f"Prediction set missing: {parsed_path.relative_to(root)}")
            inputs = list(iter_jsonl(input_path))
            predictions = list(iter_jsonl(parsed_path))
            input_ids = [str(row["doc_id"]) for row in inputs]
            prediction_ids = [str(row["doc_id"]) for row in predictions]
            if len(prediction_ids) != len(set(prediction_ids)):
                raise RuntimeError(f"Duplicate prediction doc_id in {condition}/{domain}")
            if set(input_ids) != set(prediction_ids):
                missing = sorted(set(input_ids) - set(prediction_ids))[:10]
                extra = sorted(set(prediction_ids) - set(input_ids))[:10]
                raise RuntimeError(f"Incomplete prediction set {condition}/{domain}; missing={missing}, extra={extra}")
            raw_hashes = {}
            for row in predictions:
                raw_path = root / row["raw_response_path"]
                if not raw_path.exists():
                    raise RuntimeError(f"Raw response missing: {raw_path.relative_to(root)}")
                raw_hashes[row["doc_id"]] = sha256_file(raw_path)
            sets.append({
                "condition": condition,
                "domain": domain,
                "view": view,
                "documents": len(predictions),
                "input_path": input_path.relative_to(root).as_posix(),
                "input_sha256": sha256_file(input_path),
                "parsed_path": parsed_path.relative_to(root).as_posix(),
                "parsed_sha256": sha256_file(parsed_path),
                "raw_response_hashes": raw_hashes,
                "parse_failures": sum(not bool(row.get("parse_success")) for row in predictions),
                "output_count_violations": sum(not bool(row.get("output_count_compliant")) for row in predictions),
            })
            all_doc_ids.update(input_ids)
    required = int(config["governance"]["required_formal_inference_sets"])
    if len(sets) != required:
        raise RuntimeError(f"Expected {required} inference sets, found {len(sets)}")
    freeze = {
        "schema_version": 1,
        "status": "FROZEN",
        "frozen_at_utc": utc_now(),
        "prompt_freeze_sha256": sha256_file(root / "outputs" / "PROMPT_FREEZE.json"),
        "test_gold_access_at_freeze": "none",
        "inference_sets": len(sets),
        "unique_test_documents": len(all_doc_ids),
        "sets": sets,
    }
    atomic_write_json(root / "outputs" / "TEST_PREDICTIONS_FREEZE.json", freeze)
    return freeze


def verify_prediction_freeze(root: Path) -> Dict[str, Any]:
    path = root / "outputs" / "TEST_PREDICTIONS_FREEZE.json"
    if not path.exists():
        raise RuntimeError("TEST_PREDICTIONS_FREEZE.json is missing; test gold remains locked")
    freeze = load_json(path)
    if freeze.get("status") != "FROZEN" or freeze.get("test_gold_access_at_freeze") != "none":
        raise RuntimeError("Invalid prediction freeze")
    if sha256_file(root / "outputs" / "PROMPT_FREEZE.json") != freeze["prompt_freeze_sha256"]:
        raise RuntimeError("PROMPT_FREEZE.json changed after prediction freeze")
    for item in freeze["sets"]:
        parsed = root / item["parsed_path"]
        if sha256_file(parsed) != item["parsed_sha256"]:
            raise RuntimeError(f"Parsed predictions changed: {item['parsed_path']}")
        for doc_id, expected_hash in item["raw_response_hashes"].items():
            match = next(row for row in iter_jsonl(parsed) if str(row["doc_id"]) == str(doc_id))
            raw = root / match["raw_response_path"]
            if sha256_file(raw) != expected_hash:
                raise RuntimeError(f"Raw response changed for {doc_id}")
    return freeze
