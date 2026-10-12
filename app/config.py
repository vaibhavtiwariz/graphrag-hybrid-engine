from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "GraphRAG Hybrid Engine"

    llm_provider: Literal["ollama", "openai", "groq"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    groq_api_key: str | None = None
    llm_model: str = "llama3.1:8b"

    embedding_model_name: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_device: str | None = None
    use_fp16: bool = False

    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    enable_rerank: bool = True

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nimbus_hybrid_chunks"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag123"
    neo4j_database: str = "neo4j"

    redis_url: str = "redis://localhost:6379/0"

    chunk_size: int = 800
    chunk_overlap: int = 120

    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_graph: int = 8
    final_top_k: int = 6
    rrf_k: int = 40
    rerank_input_cap: int = 16
    rerank_score_floor: float = 0.05
    rerank_max_length: int = 512
    grade_bypass_threshold: float = 0.78
    text2cypher_min_entities: int = 1
    parallel_retrieval: bool = True

    max_crag_iterations: int = 1
    min_relevant_docs: int = 1
    relevance_ratio: float = 0.2
    enable_web_fallback: bool = True
    web_results_k: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
