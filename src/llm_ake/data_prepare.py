from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .cleaning import compose_document
from .config import load_experiment
from .util import sha256_file, sha256_text, utc_now, write_jsonl, atomic_write_json


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [len(row["document"]) for row in rows]
    return {
        "documents": len(rows),
        "characters_min": min(lengths) if lengths else 0,
        "characters_median": int(statistics.median(lengths)) if lengths else 0,
        "characters_max": max(lengths) if lengths else 0,
        "truncated_documents": sum(bool(row["cleaning_audit"]["truncated"]) for row in rows),
        "leakage_heading_hits_after": sum(
            int((row["cleaning_audit"].get("fulltext") or {}).get("leakage_heading_hits_after", 0)) for row in rows
        ),
    }


def _prepare_rows(records: Iterable[Dict[str, Any]], view: str, max_characters: int) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for record in records:
        document, audit = compose_document(record, view=view, max_characters=max_characters)
        output.append({
            "doc_id": str(record["doc_id"]),
            "domain": str(record["domain"]),
            "split": str(record["split"]),
            "view": view,
            "document": document,
            "document_sha256": sha256_text(document),
            "cleaning_audit": audit,
        })
    return sorted(output, key=lambda row: row["doc_id"])


def prepare_dataset(root: Path, source_gold_free: Path) -> Dict[str, Any]:
    config = load_experiment(root)
    domains = list(config["domains"])
    max_characters = int(config["cleaning"]["max_document_characters"])
    test_by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    dev_by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_test = set()
    split_counts = Counter()
    with source_gold_free.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            split = str(record.get("split"))
            domain = str(record.get("domain"))
            split_counts[split] += 1
            if domain not in domains:
                continue
            if split == "test":
                doc_id = str(record["doc_id"])
                if doc_id in seen_test:
                    raise ValueError(f"Duplicate test doc_id {doc_id} at line {line_number}")
                seen_test.add(doc_id)
                test_by_domain[domain].append(record)
            elif split == "dev":
                dev_by_domain[domain].append(record)

    if set(test_by_domain) != set(domains):
        raise ValueError(f"Test domains mismatch: {sorted(test_by_domain)}")
    smoke_records = {
        domain: sorted(dev_by_domain[domain], key=lambda row: str(row["doc_id"]))[:1]
        for domain in domains
    }
    generated: List[Dict[str, Any]] = []
    for dataset_name, source_map in (("inference", test_by_domain), ("smoke", smoke_records)):
        for domain in domains:
            for view in config["views"]:
                rows = _prepare_rows(source_map[domain], view=view, max_characters=max_characters)
                output_path = root / "data" / dataset_name / domain / f"{view}.jsonl"
                write_jsonl(output_path, rows)
                generated.append({
                    "dataset": dataset_name,
                    "domain": domain,
                    "view": view,
                    "path": output_path.relative_to(root).as_posix(),
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                    **_summarise(rows),
                })

    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "source": {
            "path_at_packaging": str(source_gold_free.resolve()),
            "bytes": source_gold_free.stat().st_size,
            "sha256": sha256_file(source_gold_free),
            "gold_free_fields_only": ["doc_id", "domain", "split", "title", "abstract", "fulltext"],
        },
        "source_split_counts": dict(split_counts),
        "test_domain_counts": {domain: len(test_by_domain[domain]) for domain in domains},
        "test_unique_doc_ids": len(seen_test),
        "smoke_selection": {
            domain: [str(row["doc_id"]) for row in smoke_records[domain]] for domain in domains
        },
        "files": generated,
    }
    atomic_write_json(root / "data" / "manifests" / "llm_dataset_manifest.json", manifest)
    return manifest
