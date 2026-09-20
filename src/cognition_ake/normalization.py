from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


_SPACE = re.compile(r"\s+")
_BOUNDARY = " \t\r\n.,;:!?()[]{}\"'`“”‘’"


def normalize_phrase(value: object) -> str:
    """Normalize phrases for exact present-keyphrase matching."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = _SPACE.sub(" ", text).strip(_BOUNDARY)
    return text


def unique_ranked(values: Iterable[object], k: int | None = None) -> List[str]:
    """Preserve first occurrence while deduplicating normalized phrases."""
    output: List[str] = []
    seen = set()
    for value in values:
        phrase = normalize_phrase(value)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        output.append(phrase)
        if k is not None and len(output) >= int(k):
            break
    return output

