from __future__ import annotations

from _bootstrap import ROOT
from llm_ake.governance import freeze_prompt


if __name__ == "__main__":
    result = freeze_prompt(ROOT)
    print(f"Prompt protocol frozen at {result['frozen_at_utc']}")
    print(ROOT / "outputs" / "PROMPT_FREEZE.json")

