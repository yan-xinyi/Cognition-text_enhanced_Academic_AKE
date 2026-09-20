from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple


INLINE_SPACE = re.compile(r"[\t \f\v]+")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
KEYWORD_HEADING = re.compile(r"^\s*(KEYWORDS?|KEY\s+WORDS|INDEX\s+TERMS|AUTHOR\s+KEYWORDS)\s*[:.\-—–]?\s*", re.I)
REFERENCE_HEADING = re.compile(r"^\s*(REFERENCES|BIBLIOGRAPHY)\s*[:.\-—–]?\s*$", re.I)
MARKER_ANYWHERE = re.compile(r"\b(KEYWORDS?|KEY\s+WORDS|INDEX\s+TERMS|AUTHOR\s+KEYWORDS|REFERENCES|BIBLIOGRAPHY)\b", re.I)


def _normalise_lines(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _leakage_heading_hits(text: str) -> int:
    hits = 0
    for paragraph in [part.strip() for part in PARAGRAPH_BREAK.split(text) if part.strip()]:
        first_line = paragraph.split("\n", 1)[0].strip()
        if KEYWORD_HEADING.match(first_line) or REFERENCE_HEADING.fullmatch(first_line):
            hits += 1
    return hits


def clean_fulltext(value: object) -> Tuple[str, Dict[str, Any]]:
    original = _normalise_lines(value)
    paragraphs = [paragraph.strip() for paragraph in PARAGRAPH_BREAK.split(original) if paragraph.strip()]
    kept: List[str] = []
    removed_keyword_blocks = 0
    references_truncated = False
    for paragraph in paragraphs:
        first_line = paragraph.split("\n", 1)[0].strip()
        if REFERENCE_HEADING.fullmatch(first_line):
            references_truncated = True
            break
        if KEYWORD_HEADING.match(first_line):
            removed_keyword_blocks += 1
            continue
        kept.append(paragraph)
    cleaned = "\n\n".join(kept).strip()
    return cleaned, {
        "original_characters": len(original),
        "cleaned_characters": len(cleaned),
        "marker_mentions_before": len(MARKER_ANYWHERE.findall(original)),
        "marker_mentions_after": len(MARKER_ANYWHERE.findall(cleaned)),
        "leakage_heading_hits_before": _leakage_heading_hits(original),
        "leakage_heading_hits_after": _leakage_heading_hits(cleaned),
        "removed_keyword_blocks": removed_keyword_blocks,
        "references_truncated": references_truncated,
    }


def _truncate_by_paragraphs(text: str, budget: int) -> Tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    paragraphs = [part for part in PARAGRAPH_BREAK.split(text) if part.strip()]
    head_budget = int(budget * 0.75)
    tail_budget = budget - head_budget
    head: List[str] = []
    head_chars = 0
    for paragraph in paragraphs:
        cost = len(paragraph) + (2 if head else 0)
        if head_chars + cost > head_budget:
            break
        head.append(paragraph)
        head_chars += cost
    tail: List[str] = []
    tail_chars = 0
    for paragraph in reversed(paragraphs[len(head) :]):
        cost = len(paragraph) + (2 if tail else 0)
        if tail_chars + cost > tail_budget:
            break
        tail.append(paragraph)
        tail_chars += cost
    tail.reverse()
    marker = "\n\n[DETERMINISTIC MIDDLE TRUNCATION]\n\n"
    result = "\n\n".join(head) + marker + "\n\n".join(tail)
    return result[:budget], True


def compose_document(record: Dict[str, Any], view: str, max_characters: int) -> Tuple[str, Dict[str, Any]]:
    title = _normalise_lines(record.get("title"))
    abstract = _normalise_lines(record.get("abstract"))
    # The frozen prompt states that the first line is the title. Do not inject
    # synthetic labels such as "Title" or "Full Text" into the source text.
    ta = f"{title}\n\n{abstract}".strip()
    if view == "TA":
        document, truncated = _truncate_by_paragraphs(ta, max_characters)
        return document, {
            "view": view,
            "truncated": truncated,
            "final_characters": len(document),
            "fulltext": None,
        }
    if view != "FT":
        raise ValueError(f"Unsupported view: {view}")
    fulltext, audit = clean_fulltext(record.get("fulltext"))
    combined = f"{ta}\n\n{fulltext}".strip()
    document, truncated = _truncate_by_paragraphs(combined, max_characters)
    return document, {
        "view": view,
        "truncated": truncated,
        "final_characters": len(document),
        "fulltext": audit,
    }
