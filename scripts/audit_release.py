from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cff", ".csv", ".example", ""}
SKIP = {".git", ".venv", ".pytest_cache", "__pycache__"}
PRIVATE_PARTS = {"raw", "processed", "_sealed_test_gold", "checkpoints", "outputs", "raw_responses"}
SECRET_PATTERNS = {
    "OpenRouter key": re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    "generic API assignment": re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"][^'\"\s]{12,}"),
}
ABSOLUTE_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:[\\/](?:Users|严欣怡文件|博五|Mytools)[\\/]"),
    "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
}


def main() -> None:
    findings = []
    large_files = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP for part in path.parts) or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 20 * 1024 * 1024:
            large_files.append({"path": str(relative), "bytes": path.stat().st_size})
        if any(part in PRIVATE_PARTS for part in relative.parts):
            findings.append({"path": str(relative), "type": "private-directory-content"})
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in {**SECRET_PATTERNS, **ABSOLUTE_PATTERNS}.items():
            if pattern.search(text):
                findings.append({"path": str(relative), "type": label})
    status = "PASS" if not findings and not large_files else "FAIL"
    report = {"status": status, "root": ".", "findings": findings, "large_files": large_files}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
