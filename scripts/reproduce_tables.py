from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def near(actual: float, expected: float, tolerance: float = 0.02) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected:.2f}, found {actual:.4f}")


def main() -> None:
    validation = pd.read_csv(ROOT / "results" / "validation" / "structure_selection_micro_f1_at_5.csv")
    test = pd.read_csv(ROOT / "results" / "test" / "main_test_micro_f1_at_5.csv")
    coverage = pd.read_csv(ROOT / "results" / "domain_analysis" / "proxy_et_coverage_expansion.csv")
    expected_validation = {"TEXT": 32.20, "CogAlign": 34.64, "FS-CA": 35.01, "ET-ATT-5D": 34.48}
    expected_test = {"TEXT": 30.98, "CogAlign": 31.41, "ET-ATT-5D": 32.09, "L70B-TA": 21.08, "Q9B-TA": 14.99}
    for method, expected in expected_validation.items():
        near(float(validation.loc[validation.method == method, "overall_micro_f1_at_5_pct"].iloc[0]), expected)
    for method, expected in expected_test.items():
        near(float(test.loc[test.system == method, "f1_percent"].iloc[0]), expected)
    test_coverage = coverage[coverage.split == "test"]
    if float(test_coverage.proxy_complete_keyphrase_coverage_percent.min()) < 99.0:
        raise AssertionError("Frozen test Proxy ET keyphrase coverage unexpectedly fell below 99%")

    report = {
        "status": "PASS",
        "checks": {
            "validation_rows": len(expected_validation),
            "test_rows": len(expected_test),
            "minimum_test_proxy_keyphrase_coverage_percent": round(
                float(test_coverage.proxy_complete_keyphrase_coverage_percent.min()), 4
            ),
        },
    }
    destination = ROOT / "results" / "reproduction_check.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
