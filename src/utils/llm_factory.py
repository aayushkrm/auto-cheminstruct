"""LLM client factory — creates LangChain chat models from AutoChemConfig.

Uses Fireworks AI (OpenAI-compatible) as the provider.
Wired via ChatOpenAI with provider-specific base_url and api_key.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import AutoChemConfig


def create_llm(config: AutoChemConfig, **kwargs) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance from AutoChemConfig.

    Uses Fireworks AI as the provider (OpenAI-compatible).

    Args:
        config: Validated AutoChemConfig.
        **kwargs: Override for temperature, max_tokens, model, etc.

    Returns:
        Configured ChatOpenAI instance.
    """
    llm_cfg = config.llm

    params = {
        "model": llm_cfg.model,
        "base_url": llm_cfg.base_url,
        "api_key": llm_cfg.api_key or "not-needed",
        "temperature": llm_cfg.temperature,
        "max_tokens": llm_cfg.max_tokens,
        "timeout": llm_cfg.request_timeout,
        "max_retries": 3,
    }
    params.update(kwargs)

    return ChatOpenAI(**params)


def create_embedding_model(config: AutoChemConfig):
    """Create an embedding model for RAG. Uses same provider as LLM."""
    from langchain_openai import OpenAIEmbeddings

    llm_cfg = config.llm

    return OpenAIEmbeddings(
        model=config.rag.embedding_model,
        base_url=llm_cfg.base_url,
        api_key=llm_cfg.api_key or "not-needed",
    )
