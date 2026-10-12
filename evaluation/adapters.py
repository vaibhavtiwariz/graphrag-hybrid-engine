from __future__ import annotations

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings


class BGEEmbeddingsAdapter(Embeddings):
    """Wraps the local BGEM3 embedder in the LangChain Embeddings interface for Ragas."""

    def __init__(self) -> None:
        from app.services import get_embedder

        self._embedder = get_embedder()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        dense, _ = self._embedder.encode(list(texts))
        return dense

    def embed_query(self, text: str) -> list[float]:
        dense, _ = self._embedder.encode([text])
        return dense[0]
