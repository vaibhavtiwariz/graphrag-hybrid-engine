from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


def point_id_for(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class ScoredChunk:
    __slots__ = ("id", "text", "doc_id", "title", "score", "kind", "rerank_score")

    def __init__(
        self,
        id: str,
        text: str,
        doc_id: str = "",
        title: str = "",
        score: float = 0.0,
        kind: str = "chunk",
        rerank_score: float | None = None,
    ) -> None:
        self.id = id
        self.text = text
        self.doc_id = doc_id
        self.title = title
        self.score = score
        self.kind = kind
        self.rerank_score = rerank_score


class VectorRecord:
    __slots__ = ("key", "text", "metadata", "dense", "sparse")

    def __init__(self, key: str, text: str, metadata: dict, dense: list[float], sparse: SparseVector) -> None:
        self.key = key
        self.text = text
        self.metadata = metadata
        self.dense = dense
        self.sparse = sparse


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int) -> None:
        self.client = QdrantClient(url=url, timeout=60)
        self.collection = collection
        self.dim = dim

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(size=self.dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert_records(self, records: list[VectorRecord]) -> int:
        points = [
            PointStruct(
                id=point_id_for(r.key),
                vector={"dense": r.dense, "sparse": r.sparse},
                payload={**r.metadata, "text": r.text},
            )
            for r in records
        ]
        for start in range(0, len(points), 64):
            self.client.upsert(collection_name=self.collection, points=points[start : start + 64])
        return len(points)

    def search_dense(self, vector: list[float], k: int) -> list[ScoredChunk]:
        result = self.client.query_points(
            collection_name=self.collection,
            using="dense",
            query=list(vector),
            limit=k,
            with_payload=True,
        )
        return [self._to_hit(p) for p in result.points]

    def search_sparse(self, sparse_vector: SparseVector, k: int) -> list[ScoredChunk]:
        result = self.client.query_points(
            collection_name=self.collection,
            using="sparse",
            query=sparse_vector,
            limit=k,
            with_payload=True,
        )
        return [self._to_hit(p) for p in result.points]

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def delete_by_doc(self, doc_id: str) -> None:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )

    @staticmethod
    def _to_hit(point) -> ScoredChunk:
        payload = point.payload or {}
        return ScoredChunk(
            id=str(payload.get("key") or point.id),
            text=str(payload.get("text") or ""),
            doc_id=str(payload.get("doc_id") or ""),
            title=str(payload.get("title") or ""),
            kind=str(payload.get("kind") or "chunk"),
            score=float(point.score),
        )
