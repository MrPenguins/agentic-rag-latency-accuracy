"""Small deterministic helpers shared by pipeline implementations."""

from __future__ import annotations

from typing import Any


def parse_search_queries(raw_output: str, original_question: str) -> list[str]:
    """Parse pipe-delimited queries, preserving the original fallback behavior."""
    queries = [query.strip() for query in raw_output.split("|") if query.strip()]
    return queries or [original_question]


def merge_stateful_context(
    retained: list[Any], acquired: list[Any], fallback: list[Any], k: int
) -> list[Any]:
    """Retain at most k-1 prior documents so one slot remains for new evidence."""
    selected = list(retained[: max(0, k - 1)])
    selected.extend(acquired[: k - len(selected)])
    if len(selected) < k:
        selected.extend(fallback[: k - len(selected)])
    return selected
