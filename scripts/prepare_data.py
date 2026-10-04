#!/usr/bin/env python
"""Download HotpotQA and build the filtered ChromaDB used by the paper."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_agent_rag.runtime import configure, get_config

DATASET_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "llama3.1_8b.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild generated local artifacts.")
    args = parser.parse_args()
    configure(args.config)
    config = get_config()
    raw_path = Path(config["paths"]["raw_dataset"])
    vector_path = Path(config["paths"]["vectorstore"])
    bm25_path = Path(config["paths"]["bm25_cache"])

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        print(f"Downloading HotpotQA from {DATASET_URL}")
        response = requests.get(DATASET_URL, timeout=120)
        response.raise_for_status()
        raw_path.write_bytes(response.content)

    if args.force:
        if vector_path.exists():
            shutil.rmtree(vector_path)
        if bm25_path.exists():
            bm25_path.unlink()
    elif vector_path.exists():
        raise SystemExit(f"Vectorstore already exists at {vector_path}; pass --force to rebuild it.")

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    subset = json.loads((ROOT / "data" / "benchmark_subset_500.json").read_text(encoding="utf-8"))
    filtered = [item for item in data if len(item["context"]) == 10]
    filtered_ids = {item["_id"] for item in filtered}
    missing = sorted({item["_id"] for item in subset} - filtered_ids)
    if missing:
        raise ValueError(f"The fixed subset contains IDs absent from the filtered source: {missing[:5]}")

    documents = []
    for item in filtered:
        for title, sentences in item["context"]:
            documents.append(Document(
                page_content=f"Title: {title}\nText: {''.join(sentences)}",
                metadata={"question_id": item["_id"], "title": title},
            ))
    print(f"Embedding {len(documents)} paragraphs from {len(filtered)} questions...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=config["models"]["embedding_name"],
        model_kwargs={"device": config["models"]["device"]},
        show_progress=True,
    )
    Chroma.from_documents(documents, embedding_model, persist_directory=str(vector_path))
    print(f"Vectorstore written to {vector_path}")


if __name__ == "__main__":
    main()
