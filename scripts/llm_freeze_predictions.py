from __future__ import annotations

from _bootstrap import ROOT
from llm_ake.governance import freeze_predictions


if __name__ == "__main__":
    result = freeze_predictions(ROOT)
    print(f"Frozen {result['inference_sets']} inference sets for {result['unique_test_documents']} documents")
    print(ROOT / "outputs" / "TEST_PREDICTIONS_FREEZE.json")

