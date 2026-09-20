from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .config import load_experiment, parse_condition
from .governance import verify_prompt_freeze
from .openrouter import call_openrouter
from .parser import normalise, parse_numbered_list, source_presence
from .prompt import render_messages
from .util import (
    append_csv,
    atomic_write_json,
    iter_jsonl,
    load_dotenv,
    load_json,
    safe_doc_id,
    sha256_file,
    utc_now,
    write_jsonl,
)


API_AUDIT_FIELDS = [
    "timestamp_utc", "dataset", "condition", "domain", "view", "doc_id", "model_id",
    "configured_provider", "actual_provider", "request_id", "request_hash", "http_status",
    "retry_count", "latency_seconds", "input_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens", "reported_cost", "status", "error",
]


def _usage(response: Dict[str, Any]) -> Dict[str, Any]:
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "reasoning_tokens": details.get("reasoning_tokens", usage.get("reasoning_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "reported_cost": usage.get("cost"),
    }


def _parse_record(raw: Dict[str, Any], document: str) -> Dict[str, Any]:
    predictions = parse_numbered_list(raw.get("assistant_text", ""), maximum=10)
    presence = [source_presence(value, document) for value in predictions]
    normalized = [normalise(value) for value in predictions]
    duplicate_count = len(normalized) - len(set(normalized))
    return {
        "doc_id": raw["doc_id"],
        "domain": raw["domain"],
        "split": raw["split"],
        "condition": raw["condition"],
        "model_key": raw["model_key"],
        "model_id": raw["model_id"],
        "view": raw["view"],
        "provider_route": raw["provider_route"],
        "request_id": raw.get("request_id"),
        "document_sha256": raw["document_sha256"],
        "prompt_sha256": raw["prompt_sha256"],
        "predictions": predictions,
        "normalized_predictions": normalized,
        "source_presence_exact": [item["exact"] for item in presence],
        "source_presence_nfkc_casefold": [item["nfkc_casefold"] for item in presence],
        "parsed_count": len(predictions),
        "output_count_compliant": 5 <= len(predictions) <= 10,
        "parse_success": bool(predictions),
        "duplicate_count": duplicate_count,
        "raw_response_path": raw["raw_response_path"],
    }


def run_inference_set(root: Path, condition: str, domain: str, dataset: str) -> Dict[str, Any]:
    config = load_experiment(root)
    if domain not in config["domains"]:
        raise ValueError(f"Unknown domain {domain}")
    model_key, view, model = parse_condition(config, condition)
    if dataset not in {"smoke", "inference"}:
        raise ValueError("dataset must be smoke or inference")
    load_dotenv(root / ".env")
    prompt_path = root / "prompts" / "ZEROSHOT_PROMPT_v1.txt"
    prompt_sha = sha256_file(prompt_path)
    if dataset == "inference":
        verify_prompt_freeze(root)
    input_path = root / "data" / dataset / domain / f"{view}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_condition = condition if dataset == "inference" else f"SMOKE-{condition}"
    raw_dir = root / "outputs" / "raw_responses" / output_condition / domain
    error_dir = root / "outputs" / "errors" / output_condition / domain
    parsed_path = root / "outputs" / "parsed" / output_condition / f"{domain}.jsonl"
    audit_path = root / "outputs" / "llm_api_audit.csv"
    rows = list(iter_jsonl(input_path))
    parsed_rows: List[Dict[str, Any]] = []
    failures = 0
    calls = 0
    for row in rows:
        doc_id = str(row["doc_id"])
        raw_path = raw_dir / f"{safe_doc_id(doc_id)}.json"
        if raw_path.exists():
            raw = load_json(raw_path)
            if raw.get("status") == "success":
                if raw.get("document_sha256") != row["document_sha256"] or raw.get("prompt_sha256") != prompt_sha:
                    raise RuntimeError(f"Existing response hash mismatch for {doc_id}; do not overwrite a formal response")
                parsed_rows.append(_parse_record(raw, row["document"]))
                continue
        messages = render_messages(prompt_path, row["document"])
        result = call_openrouter(config, model["model_id"], messages)
        calls += 1
        response = result.response_json
        usage = _usage(response)
        raw = {
            "schema_version": 1,
            "status": "success" if result.ok else "error",
            "timestamp_utc": utc_now(),
            "dataset": dataset,
            "condition": condition,
            "domain": domain,
            "split": row["split"],
            "doc_id": doc_id,
            "model_key": model_key,
            "model_id": model["model_id"],
            "view": view,
            "provider_route": config["api"]["provider"],
            "actual_provider": response.get("provider"),
            "request_id": response.get("id"),
            "request_hash": result.request_hash,
            "document_sha256": row["document_sha256"],
            "prompt_sha256": prompt_sha,
            "http_status": result.http_status,
            "latency_seconds": result.latency_seconds,
            "attempts": result.attempts,
            "assistant_text": result.assistant_text,
            "usage": usage,
            "response_json": response,
            "error": result.error,
            "raw_response_path": raw_path.relative_to(root).as_posix(),
        }
        audit_row = {
            "timestamp_utc": raw["timestamp_utc"],
            "dataset": dataset,
            "condition": condition,
            "domain": domain,
            "view": view,
            "doc_id": doc_id,
            "model_id": model["model_id"],
            "configured_provider": ",".join(config["api"]["provider"]["order"]),
            "actual_provider": raw["actual_provider"],
            "request_id": raw["request_id"],
            "request_hash": result.request_hash,
            "http_status": result.http_status,
            "retry_count": max(0, len(result.attempts) - 1),
            "latency_seconds": round(result.latency_seconds, 6),
            **usage,
            "status": raw["status"],
            "error": result.error,
        }
        append_csv(audit_path, API_AUDIT_FIELDS, audit_row)
        if result.ok:
            atomic_write_json(raw_path, raw)
            parsed_rows.append(_parse_record(raw, row["document"]))
            if (error_dir / f"{safe_doc_id(doc_id)}.json").exists():
                os.unlink(error_dir / f"{safe_doc_id(doc_id)}.json")
        else:
            failures += 1
            atomic_write_json(error_dir / f"{safe_doc_id(doc_id)}.json", raw)
    write_jsonl(parsed_path, sorted(parsed_rows, key=lambda item: item["doc_id"]))

    if dataset == "smoke":
        for item in parsed_rows:
            append_csv(
                root / "outputs" / "ENGINEERING_SMOKE_TEST_LOG.csv",
                ["timestamp_utc", "condition", "domain", "doc_id", "api_success", "parse_success", "parsed_count", "output_count_compliant", "source_presence_rate", "raw_response_path"],
                {
                    "timestamp_utc": utc_now(),
                    "condition": condition,
                    "domain": domain,
                    "doc_id": item["doc_id"],
                    "api_success": True,
                    "parse_success": item["parse_success"],
                    "parsed_count": item["parsed_count"],
                    "output_count_compliant": item["output_count_compliant"],
                    "source_presence_rate": (sum(item["source_presence_nfkc_casefold"]) / item["parsed_count"] if item["parsed_count"] else 0.0),
                    "raw_response_path": item["raw_response_path"],
                },
            )
    summary = {
        "dataset": dataset,
        "condition": condition,
        "domain": domain,
        "documents_expected": len(rows),
        "documents_parsed": len(parsed_rows),
        "api_calls_this_run": calls,
        "failures": failures,
        "parsed_path": parsed_path.relative_to(root).as_posix(),
    }
    if failures:
        raise RuntimeError(f"{failures} API/network failures remain in {condition}/{domain}; rerun the same command to resume")
    return summary

