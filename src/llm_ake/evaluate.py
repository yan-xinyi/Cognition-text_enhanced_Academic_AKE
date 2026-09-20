from __future__ import annotations

import csv
import json
import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from nltk.tokenize import TreebankWordTokenizer

from .config import load_experiment, parse_condition
from .governance import verify_prediction_freeze
from .util import atomic_write_json, iter_jsonl, load_json, sha256_file, utc_now


TOKENIZER = TreebankWordTokenizer()


def normalise_token(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold().strip()


def phrase_tokens(value: object) -> List[str]:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return [normalise_token(token) for token in TOKENIZER.tokenize(text) if any(character.isalnum() for character in token)]


def normalise_phrase(value: object) -> str:
    return " ".join(phrase_tokens(value))


def occurrences(sequence: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase or len(phrase) > len(sequence):
        return False
    width = len(phrase)
    return any(list(sequence[index : index + width]) == list(phrase) for index in range(len(sequence) - width + 1))


def _load_sealed_gold(root: Path) -> Dict[str, Dict[str, Any]]:
    raw_dir = root / "data" / "_sealed_test_gold" / "raw"
    records: Dict[str, Dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        for row in data:
            doc_id = str(row["id"])
            if doc_id in records:
                raise ValueError(f"Duplicate sealed test doc_id: {doc_id}")
            document = "\n\n".join([str(row.get("te", "")), str(row.get("ab", "")), str(row.get("ft", ""))])
            sequence = phrase_tokens(document)
            normalized_all = []
            normalized_present = []
            for raw_phrase in row.get("kw", []):
                tokens = phrase_tokens(raw_phrase)
                if not tokens:
                    continue
                key = " ".join(tokens)
                if key not in normalized_all:
                    normalized_all.append(key)
                if occurrences(sequence, tokens) and key not in normalized_present:
                    normalized_present.append(key)
            records[doc_id] = {
                "doc_id": doc_id,
                "domain": str(row["dn"]),
                "gold_all": normalized_all,
                "gold_present": normalized_present,
            }
    return records


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _counts(predictions: Sequence[str], gold: Sequence[str], k: int) -> Tuple[int, int, int]:
    # Protocol: Top-k first, then normalize and set-deduplicate.
    predicted = {normalise_phrase(value) for value in list(predictions)[:k] if normalise_phrase(value)}
    gold_set = set(gold)
    return len(predicted & gold_set), len(predicted), len(gold_set)


def _prf(tp: int, predicted: int, gold: int) -> Tuple[float, float, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2.0 * tp / (predicted + gold) if predicted + gold else 0.0
    return precision, recall, f1


def _quantile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _holm(rows: List[Dict[str, Any]]) -> None:
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family_rows in by_family.values():
        ordered = sorted(family_rows, key=lambda row: row["raw_p"])
        running = 0.0
        total = len(ordered)
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (total - index) * row["raw_p"])
            running = max(running, adjusted)
            row["holm_p"] = running


def _bootstrap(
    document_counts: Dict[str, Dict[str, Dict[str, Tuple[int, int, int]]]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    comparisons = [
        ("B_fulltext_contribution", "L70B-FT", "L70B-TA"),
        ("B_fulltext_contribution", "Q9B-FT", "Q9B-TA"),
        ("D_llm_family", "L70B-FT", "Q9B-FT"),
    ]
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    rng = random.Random(int(config["evaluation"]["bootstrap_seed"]))
    domains = list(config["domains"])
    output: List[Dict[str, Any]] = []
    for family, left, right in comparisons:
        observed_domain_deltas = []
        for domain in domains:
            left_values = list(document_counts[left][domain].values())
            right_values = list(document_counts[right][domain].values())
            left_totals = tuple(sum(value[index] for value in left_values) for index in range(3))
            right_totals = tuple(sum(value[index] for value in right_values) for index in range(3))
            observed_domain_deltas.append(_prf(*left_totals)[2] - _prf(*right_totals)[2])
        observed = sum(observed_domain_deltas) / len(observed_domain_deltas)
        deltas: List[float] = []
        for _ in range(replicates):
            domain_deltas = []
            for domain in domains:
                doc_ids = sorted(set(document_counts[left][domain]) & set(document_counts[right][domain]))
                left_tp = left_pred = left_gold = 0
                right_tp = right_pred = right_gold = 0
                for _index in range(len(doc_ids)):
                    doc_id = doc_ids[rng.randrange(len(doc_ids))]
                    ltp, lp, lg = document_counts[left][domain][doc_id]
                    rtp, rp, rg = document_counts[right][domain][doc_id]
                    left_tp += ltp; left_pred += lp; left_gold += lg
                    right_tp += rtp; right_pred += rp; right_gold += rg
                domain_deltas.append(_prf(left_tp, left_pred, left_gold)[2] - _prf(right_tp, right_pred, right_gold)[2])
            deltas.append(sum(domain_deltas) / len(domain_deltas))
        less_equal = sum(value <= 0 for value in deltas)
        greater_equal = sum(value >= 0 for value in deltas)
        raw_p = min(1.0, 2.0 * min((less_equal + 1) / (replicates + 1), (greater_equal + 1) / (replicates + 1)))
        output.append({
            "family": family,
            "left": left,
            "right": right,
            "delta_equal_domain_micro_f1_at_5": observed,
            "ci_low": _quantile(deltas, 0.025),
            "ci_high": _quantile(deltas, 0.975),
            "raw_p": raw_p,
            "holm_p": None,
            "replicates": replicates,
            "unit": "paired_document_stratified_by_domain",
        })
    _holm(output)
    return output


def unlock_and_score(root: Path, explicit_unlock: bool) -> Dict[str, Any]:
    if not explicit_unlock:
        raise RuntimeError("Explicit --unlock-test-gold is required")
    config = load_experiment(root)
    prediction_freeze = verify_prediction_freeze(root)
    access_path = root / "outputs" / "governance" / "test_gold_access.json"
    access = load_json(access_path)
    if int(access.get("access_count", 0)) != 0 or access.get("status") != "none":
        raise RuntimeError("Test gold has already been unlocked; repeated scoring is blocked")
    access.update({
        "status": "in_progress",
        "access_count": 1,
        "first_access_at_utc": utc_now(),
        "purpose": "one_time_final_micro_evaluation_after_prediction_freeze",
        "prediction_freeze_sha256": sha256_file(root / "outputs" / "TEST_PREDICTIONS_FREEZE.json"),
    })
    atomic_write_json(access_path, access)

    gold = _load_sealed_gold(root)
    expected_docs = int(prediction_freeze["unique_test_documents"])
    if len(gold) != expected_docs:
        raise RuntimeError(f"Sealed gold document count {len(gold)} != frozen predictions {expected_docs}")

    prediction_rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    document_counts: Dict[str, Dict[str, Dict[str, Tuple[int, int, int]]]] = defaultdict(lambda: defaultdict(dict))
    ks = [3, 5, 10]
    for condition in config["conditions"]:
        model_key, view, model = parse_condition(config, condition)
        for domain in config["domains"]:
            parsed_path = root / "outputs" / "parsed" / condition / f"{domain}.jsonl"
            records = list(iter_jsonl(parsed_path))
            totals = {k: [0, 0, 0] for k in ks}
            for record in records:
                doc_id = str(record["doc_id"])
                gold_record = gold.get(doc_id)
                if not gold_record or gold_record["domain"] != domain:
                    raise RuntimeError(f"Gold/domain mismatch for {doc_id}")
                predictions = list(record.get("predictions", []))
                for k in ks:
                    counts = _counts(predictions, gold_record["gold_present"], k)
                    totals[k] = [totals[k][index] + counts[index] for index in range(3)]
                    if k == 5:
                        document_counts[condition][domain][doc_id] = counts
                prediction_rows.append({
                    "doc_id": doc_id,
                    "domain": domain,
                    "condition": condition,
                    "model_id": model["model_id"],
                    "view": view,
                    "predictions_json": json.dumps(predictions, ensure_ascii=False),
                    "normalized_predictions_json": json.dumps([normalise_phrase(value) for value in predictions], ensure_ascii=False),
                    "gold_present_count": len(gold_record["gold_present"]),
                })
                exact = record.get("source_presence_exact", [])
                relaxed = record.get("source_presence_nfkc_casefold", [])
                diagnostics.append({
                    "doc_id": doc_id,
                    "domain": domain,
                    "condition": condition,
                    "view": view,
                    "parsed_count": record.get("parsed_count", 0),
                    "parse_success": record.get("parse_success", False),
                    "output_count_compliant": record.get("output_count_compliant", False),
                    "duplicate_count": record.get("duplicate_count", 0),
                    "source_presence_exact_rate": sum(bool(value) for value in exact) / len(exact) if exact else 0.0,
                    "source_presence_nfkc_casefold_rate": sum(bool(value) for value in relaxed) / len(relaxed) if relaxed else 0.0,
                })
            for k in ks:
                tp, predicted, gold_count = totals[k]
                precision, recall, f1 = _prf(tp, predicted, gold_count)
                metrics_rows.append({
                    "domain": domain,
                    "condition": condition,
                    "model_id": model["model_id"],
                    "view": view,
                    "k": k,
                    "documents": len(records),
                    "tp": tp,
                    "predicted": predicted,
                    "gold_present": gold_count,
                    "micro_precision": precision,
                    "micro_recall": recall,
                    "micro_f1": f1,
                    "micro_precision_percent": 100 * precision,
                    "micro_recall_percent": 100 * recall,
                    "micro_f1_percent": 100 * f1,
                })

    bootstrap_rows = _bootstrap(document_counts, config)
    _write_csv(root / "outputs" / "parsed_predictions.csv", list(prediction_rows[0].keys()), prediction_rows)
    _write_csv(root / "outputs" / "test_compliance_diagnostics.csv", list(diagnostics[0].keys()), diagnostics)
    _write_csv(root / "outputs" / "llm_test_micro_results.csv", list(metrics_rows[0].keys()), metrics_rows)
    _write_csv(root / "outputs" / "llm_paired_bootstrap.csv", list(bootstrap_rows[0].keys()), bootstrap_rows)

    access.update({
        "status": "completed",
        "completed_at_utc": utc_now(),
        "gold_documents": len(gold),
        "no_prompt_or_prediction_changes_allowed": True,
    })
    atomic_write_json(access_path, access)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "prediction_freeze_verified": True,
        "test_gold_access": access,
        "test_documents": len(gold),
        "conditions": len(config["conditions"]),
        "inference_sets": len(prediction_freeze["sets"]),
        "parsed_prediction_rows": len(prediction_rows),
        "diagnostic_rows": len(diagnostics),
        "metric_rows": len(metrics_rows),
        "bootstrap_rows": len(bootstrap_rows),
        "outputs": {
            name: sha256_file(root / "outputs" / name)
            for name in [
                "parsed_predictions.csv",
                "test_compliance_diagnostics.csv",
                "llm_test_micro_results.csv",
                "llm_paired_bootstrap.csv",
            ]
        },
    }
    atomic_write_json(root / "outputs" / "verification_report.json", report)
    return report

