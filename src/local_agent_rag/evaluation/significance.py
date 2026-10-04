"""Paired McNemar tests without a dataframe dependency."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from local_agent_rag.evaluation.metrics import PIPELINES

csv.field_size_limit(2**31 - 1)


def _exact_two_sided_binomial(discordant_a: int, discordant_b: int) -> float:
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    probability = 2.0 * sum(math.comb(n, k) for k in range(lower + 1)) / (2**n)
    return min(1.0, probability)


def mcnemar_counts(baseline: list[int], target: list[int]) -> dict[str, Any]:
    if len(baseline) != len(target):
        raise ValueError("McNemar inputs must have equal length.")
    b1_t1 = sum(b == 1 and t == 1 for b, t in zip(baseline, target))
    b1_t0 = sum(b == 1 and t == 0 for b, t in zip(baseline, target))
    b0_t1 = sum(b == 0 and t == 1 for b, t in zip(baseline, target))
    b0_t0 = sum(b == 0 and t == 0 for b, t in zip(baseline, target))
    discordant = b1_t0 + b0_t1
    exact = discordant < 25
    if exact:
        statistic = float(min(b1_t0, b0_t1))
        pvalue = _exact_two_sided_binomial(b1_t0, b0_t1)
    elif discordant == 0:
        statistic, pvalue = 0.0, 1.0
    else:
        statistic = (abs(b1_t0 - b0_t1) - 1) ** 2 / discordant
        pvalue = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "both_correct": b1_t1,
        "baseline_correct_target_incorrect": b1_t0,
        "baseline_incorrect_target_correct": b0_t1,
        "both_incorrect": b0_t0,
        "discordant_pairs": discordant,
        "exact": exact,
        "statistic": statistic,
        "pvalue": pvalue,
    }


def evaluate_mcnemar(csv_path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    baseline = [int(row["rag_llm_score"]) for row in rows]
    results: dict[str, dict[str, Any]] = {}
    for name, pipeline in PIPELINES.items():
        if name == "baseline":
            continue
        target = [int(row[f"{pipeline['prefix']}_llm_score"]) for row in rows]
        results[name] = mcnemar_counts(baseline, target)
    return results
