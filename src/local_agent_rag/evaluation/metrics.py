"""Deterministic metric computation for the five reported pipelines."""

from __future__ import annotations

import ast
import csv
import re
import string
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

csv.field_size_limit(2**31 - 1)

PIPELINES: dict[str, dict[str, Any]] = {
    "baseline": {"prefix": "rag", "display": "Baseline", "has_loops": False},
    "linear_v1": {"prefix": "linear_v1", "display": "Linear V1", "has_loops": False},
    "corrective_v1": {"prefix": "correction_v1", "display": "Corrective V1", "has_loops": True},
    "linear_v2": {"prefix": "linear_v2", "display": "Linear V2", "has_loops": False},
    "corrective_v2": {"prefix": "correction_v2", "display": "Corrective V2", "has_loops": True},
}


def normalize_answer(value: object) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: object, ground_truth: object) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: object, ground_truth: object) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _percentile(values: list[float], fraction: float) -> float:
    """Match the index-based percentile rule used for the reported runs."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def evaluate_csv(csv_path: str | Path) -> dict[str, Any]:
    """Compute all reported metrics from one model's judged CSV."""
    with Path(csv_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    seen: set[str] = set()
    total_gold_documents = 0
    accumulators: dict[str, dict[str, Any]] = {
        name: {
            "hits": 0,
            "distribution": {0: 0, 1: 0, 2: 0},
            "em": 0.0,
            "f1": 0.0,
            "llm": 0.0,
            "latencies": [],
            "ttfts": [],
            "loops": [],
            "errors": 0,
        }
        for name in PIPELINES
    }

    for row in rows:
        question_id = row["question_id"]
        if question_id in seen:
            raise ValueError(f"Duplicate question_id in {csv_path}: {question_id}")
        seen.add(question_id)
        gold_titles = set(ast.literal_eval(row["gold_titles"]))
        total_gold_documents += len(gold_titles)
        gold_answer = row["gold_answer"]

        for name, pipeline in PIPELINES.items():
            prefix = pipeline["prefix"]
            current = accumulators[name]
            answer = str(row.get(f"{prefix}_answer", ""))
            if answer == "ERROR":
                current["errors"] += 1
                continue
            titles = set(ast.literal_eval(row[f"{prefix}_titles"]))
            matches = len(gold_titles.intersection(titles))
            current["hits"] += matches
            current["distribution"][matches] += 1
            current["em"] += exact_match_score(answer, gold_answer)
            current["f1"] += f1_score(answer, gold_answer)
            current["llm"] += int(row.get(f"{prefix}_llm_score", 0) or 0)
            current["latencies"].append(float(row[f"{prefix}_latency"]))
            current["ttfts"].append(float(row[f"{prefix}_ttft"]))
            if pipeline["has_loops"]:
                current["loops"].append(int(row.get(f"{prefix}_loops", 1)))

    output: dict[str, Any] = {
        "questions": len(rows),
        "unique_question_ids": len(seen),
        "total_gold_documents": total_gold_documents,
        "pipelines": {},
    }
    for name, pipeline in PIPELINES.items():
        current = accumulators[name]
        valid = len(rows) - current["errors"]
        if valid <= 0:
            raise ValueError(f"Pipeline {name} has no valid rows in {csv_path}")
        record = {
            "display": pipeline["display"],
            "valid": valid,
            "errors": current["errors"],
            "context_recall": current["hits"] / total_gold_documents * 100,
            "retrieval_distribution": current["distribution"],
            "llm_accuracy": current["llm"] / valid * 100,
            "exact_match": current["em"] / valid * 100,
            "f1": current["f1"] / valid * 100,
            "ttft_mean": sum(current["ttfts"]) / valid,
            "ttft_median": median(current["ttfts"]),
            "ttft_p90": _percentile(current["ttfts"], 0.90),
            "latency_mean": sum(current["latencies"]) / valid,
            "latency_median": median(current["latencies"]),
            "latency_p90": _percentile(current["latencies"], 0.90),
        }
        if pipeline["has_loops"]:
            record["loops_mean"] = sum(current["loops"]) / valid
        output["pipelines"][name] = record

    baseline = output["pipelines"]["baseline"]
    for name in ("linear_v1", "corrective_v1", "linear_v2", "corrective_v2"):
        record = output["pipelines"][name]
        latency_delta = record["latency_mean"] - baseline["latency_mean"]
        accuracy_delta = record["llm_accuracy"] - baseline["llm_accuracy"]
        record["accuracy_gain"] = accuracy_delta
        record["latency_delta"] = latency_delta
        record["efficiency"] = accuracy_delta / latency_delta if latency_delta else None
    return output
