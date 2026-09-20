from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from .evaluation import document_counts, index_records, prf


def _micro_for_keys(records: Mapping[str, dict], keys: Sequence[str], k: int) -> float:
    counts = {"tp": 0, "predicted": 0, "gold": 0}
    for key in keys:
        row = document_counts(records[key], k=k)
        for name in counts:
            counts[name] += row[name]
    return float(prf(counts)["f1"])


def paired_document_bootstrap(
    candidate_seeds: Sequence[Sequence[dict]],
    baseline_seeds: Sequence[Sequence[dict]],
    *,
    k: int = 5,
    replicates: int = 10000,
    seed: int = 20260901,
    stratify_domain: bool = True,
) -> Dict[str, float | int | str]:
    """Paired document bootstrap with seeds averaged inside each replicate."""
    if len(candidate_seeds) != len(baseline_seeds) or not candidate_seeds:
        raise ValueError("Candidate and baseline must contain the same non-zero number of seeds")
    candidate = [index_records(rows) for rows in candidate_seeds]
    baseline = [index_records(rows) for rows in baseline_seeds]
    keys = sorted(candidate[0])
    for mapping in candidate[1:] + baseline:
        if sorted(mapping) != keys:
            raise ValueError("All paired prediction files must contain identical document keys")

    domains: Dict[str, list[str]] = {}
    for key in keys:
        domain = str(candidate[0][key].get("domain", "ALL")) if stratify_domain else "ALL"
        domains.setdefault(domain, []).append(key)

    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled: list[str] = []
        for domain_keys in domains.values():
            positions = rng.integers(0, len(domain_keys), size=len(domain_keys))
            sampled.extend(domain_keys[position] for position in positions)
        candidate_f1 = np.mean([_micro_for_keys(rows, sampled, k) for rows in candidate])
        baseline_f1 = np.mean([_micro_for_keys(rows, sampled, k) for rows in baseline])
        deltas[index] = candidate_f1 - baseline_f1

    observed = np.mean([_micro_for_keys(rows, keys, k) for rows in candidate]) - np.mean(
        [_micro_for_keys(rows, keys, k) for rows in baseline]
    )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    left = (np.count_nonzero(deltas <= 0) + 1) / (replicates + 1)
    right = (np.count_nonzero(deltas >= 0) + 1) / (replicates + 1)
    return {
        "delta_f1": float(observed),
        "delta_f1_pp": float(observed * 100),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "ci_low_pp": float(lower * 100),
        "ci_high_pp": float(upper * 100),
        "p_two_sided": float(min(1.0, 2.0 * min(left, right))),
        "replicates": int(replicates),
        "unit": "paired_document_stratified_by_domain_seed_mean_within_replicate" if stratify_domain else "paired_document_seed_mean_within_replicate",
    }

