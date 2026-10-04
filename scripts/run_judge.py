#!/usr/bin/env python
"""Run the optional current DeepSeek judge over one raw result CSV."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_agent_rag.evaluation.judge import run_llm_judge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY before running the optional API judge.")
    run_llm_judge(
        str(args.input.resolve()),
        str(args.output.resolve()),
        api_key,
        model=args.judge_model,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
