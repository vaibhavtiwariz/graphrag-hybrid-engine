from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import get_settings
from app.core.embeddings import BGEM3Embedder
from app.core.vector_utils import slugify
from app.ingestion.chunking import split_text
from app.ingestion.extractor import extract_chunk, summarize_document
from app.ingestion.loaders import LoadedDocument, load_document
from app.ingestion.schema import ChunkExtraction
from app.stores.neo4j_store import GraphStore
from app.stores.qdrant_store import QdrantStore, VectorRecord

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".markdown"}


def _retrying(fn, *args, **kwargs):
    from tenacity import Retrying, stop_after_attempt, wait_fixed

    for attempt in Retrying(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True):
        with attempt:
            return fn(*args, **kwargs)


@dataclass
class IngestReport:
    doc_id: str
    title: str
    chunks: int
    entities: int
    relations: int
    summary: str
    extraction_failures: int = 0


def _merge_extraction_registry(
    registry: dict[str, dict], extraction: ChunkExtraction
) -> None:
    for entity in extraction.entities:
        key = slugify(entity.name)
        if key not in registry:
            registry[key] = {
                "key": key,
                "name": entity.name[:80],
                "type": entity.type,
                "description": entity.description[:240],
            }


def _encode_in_batches(embedder: BGEM3Embedder, texts: list[str], batch_size: int = 8):
    """Encode texts in small batches so one oversized batch cannot fail the whole ingest."""
    dense_all: list = []
    sparse_all: list = []
    for i in range(0, len(texts), batch_size):
        dense, sparse = embedder.encode(texts[i : i + batch_size])
        dense_all.extend(dense)
        sparse_all.extend(sparse)
    return dense_all, sparse_all


def ingest_file(
    path: str | Path,
    *,
    embedder: BGEM3Embedder | None = None,
    qstore: QdrantStore | None = None,
    gstore: GraphStore | None = None,
) -> IngestReport:
    settings = get_settings()
    embedder = embedder or BGEM3Embedder()
    qstore = qstore or QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embed_dim)
    gstore = gstore or GraphStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database
    )
    _retrying(qstore.ensure_collection)
    gstore.ensure_constraints()
    doc: LoadedDocument = load_document(path)
    pieces = split_text(doc.text, settings.chunk_size, settings.chunk_overlap)
    if not pieces:
        raise ValueError(f"No usable text extracted from {doc.path}")

    extractions: list[ChunkExtraction] = []
    failures = 0
    for piece in pieces:
        try:
            extractions.append(extract_chunk(piece, doc.title))
        except Exception as exc:
            failures += 1
            logger.warning("extraction failed for %s chunk: %s", doc.doc_id, exc)
            extractions.append(ChunkExtraction())

    registry: dict[str, dict] = {}
    mentions: list[dict] = []
    relations: list[dict] = []
    seen_mentions: set[tuple[str, str]] = set()

    chunk_ids = [f"{doc.doc_id}::c{i}" for i in range(len(pieces))]
    for idx, extraction in enumerate(extractions):
        chunk_id = chunk_ids[idx]
        _merge_extraction_registry(registry, extraction)
        for entity in extraction.entities:
            mention = {"chunk_id": chunk_id, "entity_key": slugify(entity.name)}
            pair_key = (chunk_id, mention["entity_key"])
            if pair_key not in seen_mentions:
                seen_mentions.add(pair_key)
                mentions.append(mention)
        for relation in extraction.relations:
            source_key = slugify(relation.source)
            target_key = slugify(relation.target)
            if source_key == target_key:
                continue
            for key, fallback_name in ((source_key, relation.source), (target_key, relation.target)):
                if key not in registry:
                    registry[key] = {
                        "key": key,
                        "name": fallback_name,
                        "type": "CONCEPT",
                        "description": "",
                    }
            relations.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "relation_type": (relation.relation_type or "RELATES_TO").upper().replace(" ", "_"),
                    "evidence": relation.evidence[:300],
                    "chunk_id": chunk_id,
                }
            )

    try:
        summary_obj = summarize_document(doc.title, "\n\n".join(pieces))
        summary = summary_obj.summary
    except Exception as exc:
        logger.warning("summary failed for %s: %s", doc.doc_id, exc)
        summary = "\n\n".join(pieces)[:500]

    chunks_payload = [
        {"id": cid, "text": piece, "index": idx}
        for idx, (cid, piece) in enumerate(zip(chunk_ids, pieces, strict=True))
    ]
    chunk_pairs = [[a, b] for a, b in zip(chunk_ids, chunk_ids[1:], strict=False)]

    _retrying(gstore.upsert_document,
        doc_id=doc.doc_id,
        title=doc.title,
        path=doc.path,
        summary=summary,
        chunks=chunks_payload,
        chunk_pairs=chunk_pairs,
        entities=list(registry.values()),
        mentions=mentions,
        relations=[
            {k: r[k] for k in ("source_key", "target_key", "relation_type", "evidence")}
            for r in relations
        ],
    )

    texts = [piece for piece in pieces] + [f"{doc.title}\n\n{summary}"]
    try:
        dense_vectors, sparse_vectors = embedder.encode(texts)
    except Exception as exc:
        logger.warning(
            "batch embedding failed for %s (%s); retrying in batches of 8", doc.doc_id, exc
        )
        dense_vectors, sparse_vectors = _encode_in_batches(embedder, texts, batch_size=8)

    records: list[VectorRecord] = []
    for idx, piece in enumerate(pieces):
        records.append(
            VectorRecord(
                key=chunk_ids[idx],
                text=piece,
                metadata={
                    "key": chunk_ids[idx],
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "kind": "chunk",
                },
                dense=dense_vectors[idx],
                sparse=sparse_vectors[idx],
            )
        )
    summary_key = f"{doc.doc_id}::summary"
    records.append(
        VectorRecord(
            key=summary_key,
            text=f"{doc.title}\n\n{summary}",
            metadata={
                "key": summary_key,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "kind": "summary",
            },
            dense=dense_vectors[-1],
            sparse=sparse_vectors[-1],
        )
    )
    _retrying(qstore.upsert_records, records)

    return IngestReport(
        doc_id=doc.doc_id,
        title=doc.title,
        chunks=len(pieces),
        entities=len(registry),
        relations=len(relations),
        summary=summary,
        extraction_failures=failures,
    )


def ingest_directory(path: str | Path) -> list[IngestReport]:
    root = Path(path)
    files = sorted(
        f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )
    reports: list[IngestReport] = []
    errors: list[str] = []
    for file_path in files:
        try:
            report = ingest_file(file_path)
            reports.append(report)
            logger.info("ingested %s (%s chunks)", report.title, report.chunks)
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")
            logger.error("failed to ingest %s: %s", file_path, exc)
    if errors and not reports:
        raise RuntimeError(f"All documents failed: {errors[:3]}")
    return reports


def report_to_dict(report: IngestReport) -> dict:
    return asdict(report)
