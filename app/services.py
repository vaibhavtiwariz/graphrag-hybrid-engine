from __future__ import annotations

from functools import lru_cache

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedder():
    from app.core.embeddings import BGEM3Embedder

    return BGEM3Embedder()


@lru_cache(maxsize=1)
def get_qdrant_store():
    from app.stores.qdrant_store import QdrantStore

    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embed_dim)
    store.ensure_collection()
    return store


@lru_cache(maxsize=1)
def get_graph_store():
    from app.stores.neo4j_store import GraphStore

    settings = get_settings()
    store = GraphStore(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    store.ensure_constraints()
    return store


@lru_cache(maxsize=1)
def get_reranker():
    from app.retrieval.reranker import BGEReranker

    return BGEReranker()


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    from app.retrieval.graph_retriever import GraphRetriever
    from app.retrieval.hybrid import HybridRetriever

    return HybridRetriever(
        embedder=get_embedder(),
        qdrant=get_qdrant_store(),
        graph_retriever=GraphRetriever(get_graph_store()),
        reranker=get_reranker(),
    )


def get_web_search_fn():
    from app.crag.web_search import duckduckgo_search_fn

    return duckduckgo_search_fn


def ensure_stores() -> None:
    get_graph_store()
    get_qdrant_store()
