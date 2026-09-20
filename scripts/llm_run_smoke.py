from __future__ import annotations

import argparse

from _bootstrap import ROOT
from llm_ake.config import load_experiment
from llm_ake.inference import run_inference_set


def main() -> None:
    config = load_experiment(ROOT)
    parser = argparse.ArgumentParser(description="Run non-test engineering smoke calls")
    parser.add_argument("--model", required=True, choices=sorted(config["models"]))
    parser.add_argument("--view", required=True, choices=config["views"])
    parser.add_argument("--domain", choices=config["domains"], action="append")
    args = parser.parse_args()
    condition = f"{args.model}-{args.view}"
    for domain in args.domain or config["domains"]:
        print(run_inference_set(ROOT, condition, domain, dataset="smoke"))


if __name__ == "__main__":
    main()

