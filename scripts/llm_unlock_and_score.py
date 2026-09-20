from __future__ import annotations

import argparse

from _bootstrap import ROOT
from llm_ake.evaluate import unlock_and_score


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time test-gold unlock and final scoring")
    parser.add_argument("--unlock-test-gold", action="store_true")
    args = parser.parse_args()
    result = unlock_and_score(ROOT, explicit_unlock=args.unlock_test_gold)
    print(f"Final scoring complete for {result['test_documents']} documents")
    print(ROOT / "outputs" / "verification_report.json")


if __name__ == "__main__":
    main()

