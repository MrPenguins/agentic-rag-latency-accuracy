#!/usr/bin/env python
"""Run all five reported pipelines for one model configuration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_agent_rag.runtime import configure, vectorstore
from local_agent_rag.retrieval import hybrid_search_with_score
from local_agent_rag.pipelines.baseline import run_standard_rag
from local_agent_rag.pipelines.linear_v1 import run_linear_agent as run_linear_v1
from local_agent_rag.pipelines.corrective_v1 import run_correction_agent as run_corrective_v1
from local_agent_rag.pipelines.linear_v2 import run_linear_agent as run_linear_v2
from local_agent_rag.pipelines.corrective_v2 import run_correction_agent as run_corrective_v2

HEADERS = [
    "question_id", "question", "gold_answer", "gold_titles",
    "rag_answer", "rag_latency", "rag_ttft", "rag_titles",
    "linear_v1_answer", "linear_v1_latency", "linear_v1_ttft", "linear_v1_titles",
    "correction_v1_answer", "correction_v1_latency", "correction_v1_ttft",
    "correction_v1_titles", "correction_v1_loops",
    "linear_v2_answer", "linear_v2_latency", "linear_v2_ttft", "linear_v2_titles",
    "correction_v2_answer", "correction_v2_latency", "correction_v2_ttft",
    "correction_v2_titles", "correction_v2_loops",
]


def _gold_titles(supporting_facts: list[list[object]]) -> list[str]:
    return list(set(str(fact[0]) for fact in supporting_facts))


def _safe_call(label: str, function, *args):
    try:
        return function(*args)
    except Exception as error:
        print(f"Error in {label}: {error}")
        if label.startswith("Corrective"):
            return "ERROR", 0.0, [], 0, 0.0
        return "ERROR", 0.0, [], 0.0


def run_benchmark(input_json: Path, output_csv: Path) -> None:
    with input_json.open("r", encoding="utf-8") as stream:
        dataset = json.load(stream)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    processed_ids: set[str] = set()
    if output_csv.exists():
        with output_csv.open("r", encoding="utf-8", newline="") as stream:
            processed_ids = {row["question_id"] for row in csv.DictReader(stream)}
        print(f"Found {len(processed_ids)} completed questions; resuming.")

    new_file = not output_csv.exists()
    with output_csv.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        if new_file:
            writer.writeheader()

        print("Priming the LLM, embedding model, BM25, and ChromaDB...")
        try:
            hybrid_search_with_score("warm up query", None, vectorstore, k=1)
        except Exception as error:
            print(f"Retrieval warm-up warning: {error}")

        for index, item in enumerate(dataset, 1):
            question_id = item["_id"]
            if question_id in processed_ids:
                continue
            question = item["question"]
            print(f"[{index}/{len(dataset)}] {question_id}: {question}")

            rag_answer, rag_latency, rag_titles, rag_ttft = _safe_call(
                "Baseline", run_standard_rag, question, question_id
            )
            l1_answer, l1_latency, l1_titles, l1_ttft = _safe_call(
                "Linear V1", run_linear_v1, question, question_id
            )
            c1_answer, c1_latency, c1_titles, c1_loops, c1_ttft = _safe_call(
                "Corrective V1", run_corrective_v1, question, question_id
            )
            l2_answer, l2_latency, l2_titles, l2_ttft = _safe_call(
                "Linear V2", run_linear_v2, question, question_id
            )
            c2_answer, c2_latency, c2_titles, c2_loops, c2_ttft = _safe_call(
                "Corrective V2", run_corrective_v2, question, question_id
            )

            writer.writerow({
                "question_id": question_id,
                "question": question,
                "gold_answer": item["answer"],
                "gold_titles": str(_gold_titles(item["supporting_facts"])),
                "rag_answer": rag_answer,
                "rag_latency": round(rag_latency, 2),
                "rag_ttft": round(rag_ttft, 2),
                "rag_titles": str(rag_titles),
                "linear_v1_answer": l1_answer,
                "linear_v1_latency": round(l1_latency, 2),
                "linear_v1_ttft": round(l1_ttft, 2),
                "linear_v1_titles": str(l1_titles),
                "correction_v1_answer": c1_answer,
                "correction_v1_latency": round(c1_latency, 2),
                "correction_v1_ttft": round(c1_ttft, 2),
                "correction_v1_titles": str(c1_titles),
                "correction_v1_loops": c1_loops,
                "linear_v2_answer": l2_answer,
                "linear_v2_latency": round(l2_latency, 2),
                "linear_v2_ttft": round(l2_ttft, 2),
                "linear_v2_titles": str(l2_titles),
                "correction_v2_answer": c2_answer,
                "correction_v2_latency": round(c2_latency, 2),
                "correction_v2_ttft": round(c2_ttft, 2),
                "correction_v2_titles": str(c2_titles),
                "correction_v2_loops": c2_loops,
            })
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "benchmark_subset_500.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure(args.config)
    run_benchmark(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
