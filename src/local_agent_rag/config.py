"""Configuration loading with project-root-relative path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_DATABASE_KEYS = {"k_retrieval", "hybrid_alpha"}
REQUIRED_MODEL_KEYS = {
    "llm_name",
    "llm_temperature",
    "embedding_name",
    "device",
    "max_retries",
}


def _resolve_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, validate, and normalize one experiment configuration."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}

    config = deepcopy(loaded)
    missing_db = REQUIRED_DATABASE_KEYS - set(config.get("database", {}))
    missing_model = REQUIRED_MODEL_KEYS - set(config.get("models", {}))
    if missing_db or missing_model:
        raise ValueError(
            f"Invalid configuration {config_path}: "
            f"missing database keys={sorted(missing_db)}, "
            f"model keys={sorted(missing_model)}"
        )

    if int(config["database"]["k_retrieval"]) != 2:
        raise ValueError("The reported experiments require k_retrieval=2.")
    if float(config["database"]["hybrid_alpha"]) != 0.5:
        raise ValueError("The reported experiments require hybrid_alpha=0.5.")
    if int(config["models"]["max_retries"]) != 3:
        raise ValueError("The reported experiments require max_retries=3.")
    if float(config["models"]["llm_temperature"]) != 0:
        raise ValueError("The reported experiments require llm_temperature=0.")

    paths = config.setdefault("paths", {})
    paths.setdefault("vectorstore", "artifacts/vectorstore")
    paths.setdefault("bm25_cache", "artifacts/BM25_CACHE/bm25_index.pkl")
    paths.setdefault("raw_dataset", "artifacts/dataset/hotpot_dev_distractor_v1.json")
    for key, value in list(paths.items()):
        paths[key] = _resolve_path(str(value))

    config["_config_path"] = str(config_path.resolve())
    return config
