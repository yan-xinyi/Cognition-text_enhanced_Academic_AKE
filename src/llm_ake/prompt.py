from __future__ import annotations

from pathlib import Path
from typing import Tuple


SYSTEM_MARKER = "# System Prompt"
TASK_MARKER = "# Task Instruction"


def load_prompt(path: Path) -> Tuple[str, str, str]:
    full = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip() + "\n"
    if SYSTEM_MARKER not in full or TASK_MARKER not in full or "{document}" not in full:
        raise ValueError(f"Prompt markers or {{document}} placeholder missing: {path}")
    before_task, after_task = full.split(TASK_MARKER, 1)
    system = before_task.split(SYSTEM_MARKER, 1)[1].strip()
    user_template = (TASK_MARKER + after_task).strip()
    return system, user_template, full


def render_messages(path: Path, document: str):
    system, user_template, _ = load_prompt(path)
    user = user_template.replace("{document}", document)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

