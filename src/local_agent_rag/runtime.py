"""Lazy model and vector-store initialization.

Importing this module never starts Ollama, loads an embedding model, or opens
ChromaDB. Resources are constructed only when a pipeline is actually called.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_agent_rag.config import PROJECT_ROOT, load_config


K = 2
MAX_RETRIES = 3
_config: dict[str, Any] | None = None
_resources: "RuntimeResources | None" = None


@dataclass
class RuntimeResources:
    llm: Any
    vectorstore: Any
    embedding_model: Any


def configure(config_path: str | Path) -> dict[str, Any]:
    """Select a model configuration and reset any existing lazy resources."""
    global _config, _resources
    _config = load_config(config_path)
    _resources = None
    return _config


def get_config() -> dict[str, Any]:
    """Return the active configuration, defaulting to Qwen 3.5 9B."""
    global _config
    if _config is None:
        _config = load_config(PROJECT_ROOT / "configs" / "qwen3.5_9b.yaml")
    return _config


def get_resources(*, warm_up: bool = True) -> RuntimeResources:
    """Construct heavyweight resources on first actual use."""
    global _resources
    if _resources is not None:
        return _resources

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama

    config = get_config()
    embedding_model = HuggingFaceEmbeddings(
        model_name=config["models"]["embedding_name"],
        model_kwargs={"device": config["models"]["device"]},
    )
    vectorstore = Chroma(
        persist_directory=config["paths"]["vectorstore"],
        embedding_function=embedding_model,
    )
    llm = ChatOllama(
        model=config["models"]["llm_name"],
        temperature=config["models"]["llm_temperature"],
        reasoning=bool(config["models"].get("reasoning", False)),
    )
    _resources = RuntimeResources(llm, vectorstore, embedding_model)

    if warm_up:
        print("Initializing models and warming up GPU...")
        print(f"   LLM: {config['models']['llm_name']}")
        llm.invoke("Hi")
        embedding_model.embed_query("Warm up the GPU memory pool.")
        print("GPU is fully warm. Ready to benchmark.")
    return _resources


class _LazyProxy:
    """Forward attribute access to a lazily created runtime resource."""

    def __init__(self, getter: Callable[[], Any]):
        object.__setattr__(self, "_getter", getter)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_getter")(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_getter")(), name, value)


llm = _LazyProxy(lambda: get_resources().llm)
vectorstore = _LazyProxy(lambda: get_resources().vectorstore)
