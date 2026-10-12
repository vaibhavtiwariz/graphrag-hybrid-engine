from __future__ import annotations

import logging
from functools import lru_cache

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _ollama_reachable(base_url: str) -> bool:
    import httpx

    try:
        return httpx.get(f"{base_url}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def llm_endpoint() -> tuple[str, str]:
    settings = get_settings()
    if settings.llm_provider == "groq":
        if settings.groq_api_key:
            return GROQ_BASE_URL, settings.groq_api_key
        logger.warning(
            "GROQ_API_KEY is empty - falling back to local Ollama. "
            "Get a free key at https://console.groq.com/keys and add it to .env "
            "for ~10x faster responses."
        )
        if _ollama_reachable(settings.ollama_base_url):
            if not settings.llm_model.startswith("llama3"):
                settings.llm_model = "llama3.2:3b"
            return f"{settings.ollama_base_url.rstrip('/')}/v1", "ollama"
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is empty and local Ollama is unreachable. "
            "Either paste a free Groq key from https://console.groq.com/keys into .env, "
            "or start the Docker stack with 'docker compose up -d'."
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Paste it into the .env file, then restart the API."
            )
        return settings.openai_base_url, settings.openai_api_key
    return f"{settings.ollama_base_url.rstrip('/')}/v1", "ollama"


def completion_extra_kwargs() -> dict:
    settings = get_settings()
    if settings.llm_provider == "groq" and settings.llm_model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}


@lru_cache(maxsize=1)
def raw_client() -> OpenAI:
    base_url, api_key = llm_endpoint()
    return OpenAI(base_url=base_url, api_key=api_key, timeout=180)


@lru_cache(maxsize=1)
def structured_client():
    import instructor

    return instructor.from_openai(raw_client(), mode=instructor.Mode.JSON)


@lru_cache(maxsize=1)
def chat_llm():
    settings = get_settings()
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    from langchain_openai import ChatOpenAI

    base_url, api_key = llm_endpoint()
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
    )
