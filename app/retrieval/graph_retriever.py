from __future__ import annotations

import hashlib
import json

from app.config import get_settings
from app.core.llm import structured_client
from app.core.vector_utils import keyword_terms
from app.ingestion.schema import CypherQuery
from app.retrieval.cypher_guard import is_safe_readonly
from app.stores.neo4j_store import GraphStore
from app.stores.qdrant_store import ScoredChunk

CYPHER_SYSTEM = """You convert a natural language question into ONE read-only Neo4j Cypher query.
Graph schema:
(:Document {id, title, summary})
(:Chunk {id, text, idx, doc_id})
(:Entity {key, name, type, description})  types: PRODUCT, COMPONENT, SUPPLIER, PERSON, TEAM,
INCIDENT, POLICY, LOCATION, PROCESS, METRIC
(:Document)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:NEXT]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED {type, evidence}]->(:Entity) with relation types like SUPPLIED_BY, PART_OF,
CAUSED_BY, FIXED_BY, CONTACT_FOR, AFFECTS, REQUIRES, GOVERNED_BY, LOCATED_IN.
Rules: start with MATCH or OPTIONAL MATCH; single statement ending with RETURN; always add a LIMIT
of at most 50; never write data and never call procedures; prefer matching Entity by name with
toLower(e.name) CONTAINS for fuzzy names."""

GRAPH_SCHEMA_TEXT = (
    "Nodes: Document(id,title,summary), Chunk(id,text,idx), "
    "Entity(key,name,type,description). "
    "Relationships: (Document)-HAS_CHUNK->(Chunk), (Chunk)-NEXT->(Chunk), "
    "(Chunk)-MENTIONS->(Entity), (Entity)-RELATED{type,evidence}->(Entity)."
)


class GraphRetriever:
    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def retrieve(self, question: str, k: int = 8) -> tuple[list[ScoredChunk], dict]:
        settings = get_settings()
        meta: dict = {"entities": [], "triples": [], "cypher": None}
        terms = keyword_terms(question)

        matched = self.store.find_entity_keys_by_terms(terms, limit=15)
        keys = [row["key"] for row in matched]
        meta["entities"] = [row["name"] for row in matched]

        expanded = self.store.expand_neighborhood(keys, hops=2, limit=40) if keys else []
        expanded_keys = [row["key"] for row in expanded]
        all_keys = list(dict.fromkeys(keys + expanded_keys))

        chunks: list[ScoredChunk] = []
        if all_keys:
            direct_rows = (
                self.store.chunks_for_entity_keys(keys, limit=max(k * 2, 8)) if keys else []
            )
            only_expanded = [key for key in expanded_keys if key not in set(keys)]
            expanded_rows = (
                self.store.chunks_for_entity_keys(only_expanded, limit=k) if only_expanded else []
            )
            seen: set[str] = set()
            for row in [*direct_rows, *expanded_rows]:
                chunk_id = row["chunk_id"]
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                chunks.append(
                    ScoredChunk(
                        id=f"g:{chunk_id}",
                        text=row.get("text") or "",
                        doc_id=row.get("doc_id") or "",
                        title=row.get("title") or "",
                        score=0.0,
                        kind="graph",
                    )
                )
            meta["triples"] = self.store.relation_triples(all_keys, limit=24)
            meta["direct_entities"] = len(keys)
            meta["expanded_entities"] = len(only_expanded)

        if len(matched) >= get_settings().text2cypher_min_entities:
            meta["cypher"] = "skipped: keyword entity match already sufficient"
        else:
            blob = self._try_text2cypher(question, settings.llm_model)
            if blob:
                digest = hashlib.sha1(question.encode()).hexdigest()[:10]
                chunks.append(
                    ScoredChunk(
                        id=f"g:cypher:{digest}",
                        text=blob,
                        doc_id="knowledge-graph",
                        title="Knowledge graph traversal result",
                        score=0.0,
                        kind="graph",
                    )
                )

        return chunks[: max(k * 2, 10)], meta

    def _try_text2cypher(self, question: str, model: str) -> str | None:
        try:
            completion = structured_client().chat.completions.create(
                model=model,
                response_model=CypherQuery,
                messages=[
                    {"role": "system", "content": CYPHER_SYSTEM},
                    {
                        "role": "user",
                        "content": f"{GRAPH_SCHEMA_TEXT}\n\nQuestion: {question}",
                    },
                ],
                temperature=0,
                max_retries=2,
            )
            candidate = completion.cypher.strip().rstrip(";")
            if not is_safe_readonly(candidate):
                return None
            rows = self.store.run_readonly(candidate)
            if not rows:
                return None
            return json.dumps(rows, default=str)[:1500]
        except Exception:
            return None
