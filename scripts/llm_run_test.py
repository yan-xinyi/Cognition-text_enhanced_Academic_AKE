from __future__ import annotations

import argparse

from _bootstrap import ROOT
from llm_ake.config import load_experiment
from llm_ake.inference import run_inference_set


def main() -> None:
    config = load_experiment(ROOT)
    parser = argparse.ArgumentParser(description="Run one frozen formal test condition across all domains")
    parser.add_argument("--condition", required=True, choices=config["conditions"])
    parser.add_argument("--domain", choices=config["domains"], action="append")
    args = parser.parse_args()
    for domain in args.domain or config["domains"]:
        print(run_inference_set(ROOT, args.condition, domain, dataset="inference"))


if __name__ == "__main__":
    main()

