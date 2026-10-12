from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter

from app.config import Settings, get_settings
from app.core.embeddings import BGEM3Embedder
from app.retrieval.fusion import rrf_fuse
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.reranker import BGEReranker, prune_by_score_floor
from app.stores.qdrant_store import QdrantStore, ScoredChunk


def _norm_hit_id(hit_id: str) -> str:
    """Graph hits prefix chunk ids with 'g:'; strip it so the same chunk
    found by vector search and graph traversal merges into one candidate."""
    return hit_id[2:] if hit_id.startswith("g:") else hit_id


@dataclass
class HybridResult:
    question: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    graph_meta: dict = field(default_factory=dict)
    fused_scores: list[tuple[str, float]] = field(default_factory=list)


class HybridRetriever:
    def __init__(
        self,
        embedder: BGEM3Embedder,
        qdrant: QdrantStore,
        graph_retriever: GraphRetriever,
        reranker: BGEReranker,
        settings: Settings | None = None,
    ) -> None:
        self.embedder = embedder
        self.qdrant = qdrant
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.settings = settings or get_settings()

    def retrieve(self, question: str) -> HybridResult:
        cfg = self.settings
        trace: list[dict] = []

        if os.environ.get("RETRIEVAL_BASELINE") == "1":
            # Baseline mode (ablation): dense-only vector retrieval, no graph,
            # no fusion, no rerank. Used to measure what the hybrid stack adds.
            start = perf_counter()
            dense_vector, _ = self.embedder.encode_one(question)
            hits = self.qdrant.search_dense(dense_vector, cfg.final_top_k)
            trace.append(
                {
                    "stage": "baseline",
                    "detail": {"mode": "vector-only", "hits": len(hits)},
                    "ms": round((perf_counter() - start) * 1000, 1),
                }
            )
            return HybridResult(
                question=question, chunks=hits, trace=trace, graph_meta={}, fused_scores=[]
            )

        def _vector_stage():
            stage_start = perf_counter()
            dense_vector, sparse_vector = self.embedder.encode_one(question)
            dense_hits = self.qdrant.search_dense(dense_vector, cfg.top_k_dense)
            sparse_hits = self.qdrant.search_sparse(sparse_vector, cfg.top_k_sparse)
            return (
                dense_hits,
                sparse_hits,
                round((perf_counter() - stage_start) * 1000, 1),
            )

        def _graph_stage():
            stage_start = perf_counter()
            graph_chunks, graph_meta = self.graph_retriever.retrieve(question, k=cfg.top_k_graph)
            return graph_chunks, graph_meta, round((perf_counter() - stage_start) * 1000, 1)

        if cfg.parallel_retrieval:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as executor:
                vector_future = executor.submit(_vector_stage)
                graph_future = executor.submit(_graph_stage)
                dense_hits, sparse_hits, vector_ms = vector_future.result()
                graph_chunks, graph_meta, graph_ms = graph_future.result()
        else:
            dense_hits, sparse_hits, vector_ms = _vector_stage()
            graph_chunks, graph_meta, graph_ms = _graph_stage()

        graph_meta["chunks"] = len(graph_chunks)
        trace.append(
            {
                "stage": "vector_search",
                "detail": {"dense": len(dense_hits), "sparse": len(sparse_hits)},
                "ms": vector_ms,
            }
        )
        trace.append(
            {
                "stage": "graph_search",
                "detail": graph_meta,
                "ms": graph_ms,
            }
        )

        start = perf_counter()
        fused = rrf_fuse(
            [
                [_norm_hit_id(hit.id) for hit in dense_hits],
                [_norm_hit_id(hit.id) for hit in sparse_hits],
                [_norm_hit_id(hit.id) for hit in graph_chunks],
            ],
            k=cfg.rrf_k,
        )
        # Merge duplicate contexts: the same chunk can surface from both vector
        # search (bare id) and graph traversal ("g:"-prefixed id). Keep the
        # vector hit's metadata since it carries the real retrieval score.
        pool: dict[str, ScoredChunk] = {}
        for hit in [*dense_hits, *sparse_hits, *graph_chunks]:
            key = _norm_hit_id(hit.id)
            existing = pool.get(key)
            if existing is None or (existing.kind == "graph" and hit.kind != "graph"):
                pool[key] = hit
        candidates = [pool[cid] for cid, _ in fused]
        capped = candidates[: cfg.rerank_input_cap]
        trace.append(
            {
                "stage": "rrf_fusion",
                "detail": {"candidates": len(candidates), "to_rerank": len(capped)},
                "ms": round((perf_counter() - start) * 1000, 1),
            }
        )

        start = perf_counter()
        reranked = self.reranker.rerank(question, list(capped), top_k=cfg.final_top_k)
        final = prune_by_score_floor(reranked, cfg.rerank_score_floor)
        trace.append(
            {
                "stage": "rerank",
                "detail": {"returned": len(final), "pruned": len(reranked) - len(final)},
                "ms": round((perf_counter() - start) * 1000, 1),
            }
        )

        return HybridResult(
            question=question,
            chunks=final,
            trace=trace,
            graph_meta=graph_meta,
            fused_scores=fused[:20],
        )
