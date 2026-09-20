from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List


NUMBERED_LINE = re.compile(r"^\s*(10|[1-9])\.\s+(.+?)\s*$")
OUTER_QUOTES = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
)


def normalise(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold().strip()


def strip_outer_quotes(value: str) -> str:
    result = value.strip()
    for left, right in OUTER_QUOTES:
        if len(result) >= 2 and result.startswith(left) and result.endswith(right):
            return result[len(left) : len(result) - len(right)].strip()
    return result


def parse_numbered_list(response_text: str, maximum: int = 10) -> List[str]:
    parsed: List[str] = []
    for line in str(response_text or "").splitlines():
        match = NUMBERED_LINE.match(line)
        if not match:
            continue
        rank = int(match.group(1))
        if rank > maximum:
            continue
        phrase = strip_outer_quotes(match.group(2))
        if phrase:
            parsed.append(phrase)
        if len(parsed) >= maximum:
            break
    return parsed


def ordered_unique_after_top_k(values: Iterable[str], top_k: int) -> List[str]:
    seen = set()
    result: List[str] = []
    for raw in list(values)[:top_k]:
        key = normalise(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def source_presence(phrase: str, document: str) -> Dict[str, bool]:
    phrase_text = str(phrase)
    document_text = str(document)
    return {
        "exact": phrase_text in document_text,
        "nfkc_casefold": normalise(phrase_text) in normalise(document_text),
    }

