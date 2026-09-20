from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .normalization import unique_ranked


def read_jsonl(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prediction_phrases(record: Mapping[str, Any]) -> List[str]:
    raw = record.get("predictions", [])
    return [item[0] if isinstance(item, (list, tuple)) else item for item in raw]


def document_counts(record: Mapping[str, Any], k: int = 5) -> Dict[str, int]:
    predicted = set(unique_ranked(prediction_phrases(record), k=k))
    gold = set(unique_ranked(record.get("gold_present", [])))
    return {"tp": len(predicted & gold), "predicted": len(predicted), "gold": len(gold)}


def prf(counts: Mapping[str, int]) -> Dict[str, float | int]:
    p = counts["tp"] / counts["predicted"] if counts["predicted"] else 0.0
    r = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
    f1 = 2.0 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, **{k: int(counts[k]) for k in ("tp", "predicted", "gold")}}


def evaluate_micro(records: Sequence[Mapping[str, Any]], k: int = 5) -> Dict[str, float | int]:
    counts = {"tp": 0, "predicted": 0, "gold": 0}
    for record in records:
        row = document_counts(record, k=k)
        for key in counts:
            counts[key] += row[key]
    return {"documents": len(records), "k": int(k), **prf(counts)}


def evaluate_cutoffs(records: Sequence[Mapping[str, Any]], cutoffs: Iterable[int] = (3, 5, 10)) -> Dict[str, dict]:
    return {str(k): evaluate_micro(records, int(k)) for k in cutoffs}


def index_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in records:
        key = f"{row.get('domain', '')}::{row['doc_id']}"
        if key in indexed:
            raise ValueError(f"Duplicate document key: {key}")
        indexed[key] = row
    return indexed

