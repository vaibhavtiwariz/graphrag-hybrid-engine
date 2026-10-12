from __future__ import annotations

from neo4j import GraphDatabase

from app.retrieval.cypher_guard import assert_safe_readonly

DOC_META_UPSERT = """
MERGE (d:Document {id: $doc_id})
ON CREATE SET d.created_at = datetime()
SET d.title = $title, d.source_path = $path, d.summary = $summary
"""

CHUNKS_UPSERT = """
UNWIND $chunks AS ch
MATCH (d:Document {id: $doc_id})
MERGE (c:Chunk {id: ch.id})
SET c.text = ch.text, c.idx = ch.index, c.doc_id = $doc_id
MERGE (d)-[:HAS_CHUNK]->(c)
"""

CHUNK_ORDER_UPSERT = """
UNWIND $pairs AS pair
MATCH (a:Chunk {id: pair[0]}), (b:Chunk {id: pair[1]})
MERGE (a)-[:NEXT]->(b)
"""

ENTITY_UPSERT = """
UNWIND $entities AS e
MERGE (en:Entity {key: e.key})
ON CREATE SET en.name = e.name, en.type = e.type,
              en.description = e.description, en.created_at = datetime()
ON MATCH SET en.name = e.name, en.type = e.type, en.description = e.description
SET en.last_seen_at = datetime()
"""

MENTION_UPSERT = """
UNWIND $mentions AS m
MATCH (c:Chunk {id: m.chunk_id})
MATCH (en:Entity {key: m.entity_key})
MERGE (c)-[:MENTIONS]->(en)
"""

RELATION_UPSERT = """
UNWIND $relations AS r
MATCH (a:Entity {key: r.source_key}), (b:Entity {key: r.target_key})
MERGE (a)-[rel:RELATED {type: r.relation_type}]->(b)
SET rel.evidence = r.evidence
"""

COUNTS_QUERY = """
CALL { MATCH (d:Document) RETURN count(d) AS documents }
CALL { MATCH (c:Chunk) RETURN count(c) AS chunks }
CALL { MATCH (e:Entity) RETURN count(e) AS entities }
CALL { MATCH ()-[r:RELATED]->() RETURN count(r) AS relations }
RETURN documents, chunks, entities, relations
"""


class GraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self._driver.close()

    def healthy(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
        ]
        with self._driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

    def upsert_document(
        self,
        *,
        doc_id: str,
        title: str,
        path: str,
        summary: str,
        chunks: list[dict],
        chunk_pairs: list[list[str]],
        entities: list[dict],
        mentions: list[dict],
        relations: list[dict],
    ) -> None:
        def _tx(tx):
            tx.run(DOC_META_UPSERT, doc_id=doc_id, title=title, path=path, summary=summary)
            if chunks:
                tx.run(CHUNKS_UPSERT, doc_id=doc_id, chunks=chunks)
            if chunk_pairs:
                tx.run(CHUNK_ORDER_UPSERT, pairs=chunk_pairs)
            if entities:
                tx.run(ENTITY_UPSERT, entities=entities)
            if mentions:
                tx.run(MENTION_UPSERT, mentions=mentions)
            if relations:
                tx.run(RELATION_UPSERT, relations=relations)

        with self._driver.session(database=self.database) as session:
            session.execute_write(_tx)

    def find_entity_keys_by_terms(self, terms: list[str], limit: int = 15) -> list[dict]:
        cleaned = [t.lower() for t in terms if t and len(t) >= 3]
        if not cleaned:
            return []
        query = (
            "UNWIND $terms AS term "
            "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS term "
            "RETURN DISTINCT e.key AS key, e.name AS name, e.type AS type "
            "ORDER BY size(e.name) ASC LIMIT $limit"
        )
        return self._read(query, terms=cleaned, limit=limit)

    def expand_neighborhood(self, keys: list[str], hops: int = 2, limit: int = 40) -> list[dict]:
        if not keys:
            return []
        hops = max(1, min(int(hops), 3))
        query = (
            f"UNWIND $keys AS key "
            f"MATCH (e:Entity {{key: key}}) "
            f"MATCH p = (e)-[:RELATED*1..{hops}]-(n:Entity) "
            f"RETURN DISTINCT n.key AS key, n.name AS name, n.type AS type, min(length(p)) AS distance "
            f"ORDER BY distance ASC LIMIT $limit"
        )
        return self._read(query, keys=keys, limit=limit)

    def chunks_for_entity_keys(self, keys: list[str], limit: int = 20) -> list[dict]:
        if not keys:
            return []
        query = (
            "UNWIND $keys AS key "
            "MATCH (c:Chunk)-[:MENTIONS]->(:Entity {key: key}) "
            "OPTIONAL MATCH (d:Document {id: c.doc_id}) "
            "RETURN DISTINCT c.id AS chunk_id, c.text AS text, c.doc_id AS doc_id, "
            "coalesce(d.title, '') AS title LIMIT $limit"
        )
        return self._read(query, keys=keys, limit=limit)

    def relation_triples(self, keys: list[str], limit: int = 24) -> list[dict]:
        if not keys:
            return []
        query = (
            "UNWIND $keys AS key "
            "MATCH (a:Entity {key: key})-[r:RELATED]-(b:Entity) "
            "WHERE b.key IN $keys AND a.key <> b.key "
            "RETURN DISTINCT a.name AS source, r.type AS relation_type, b.name AS target "
            "LIMIT $limit"
        )
        return self._read(query, keys=keys, limit=limit)

    def entity_subgraph(self, name: str, depth: int = 1, limit: int = 200) -> dict:
        depth = max(1, min(int(depth), 3))
        query = (
            f"MATCH (a:Entity) WHERE toLower(a.name) = toLower($name) "
            f"MATCH p = (a)-[:RELATED*1..{depth}]-(b:Entity) "
            f"RETURN a.name AS source, b.name AS target, "
            f"[rel IN relationships(p) | rel.type] AS relation_types LIMIT $limit"
        )
        rows = self._read(query, name=name, limit=limit)
        edges: list[dict] = []
        nodes: set[str] = set()
        for row in rows:
            nodes.add(row["source"])
            nodes.add(row["target"])
            for rel in row.get("relation_types") or ["RELATED"]:
                edges.append({"source": row["source"], "relation": rel, "target": row["target"]})
        return {"center": name, "nodes": sorted(nodes), "edges": edges}

    def counts(self) -> dict:
        rows = self._read(COUNTS_QUERY)
        return rows[0] if rows else {}

    def list_documents(self) -> list[dict]:
        query = (
            "MATCH (d:Document) "
            "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk) "
            "WITH d, count(c) AS chunks "
            "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity) "
            "RETURN d.id AS id, d.title AS title, d.summary AS summary, "
            "chunks, count(DISTINCT e) AS entities "
            "ORDER BY d.title"
        )
        return self._read(query)

    def delete_document(self, doc_id: str) -> int:
        query = (
            "MATCH (d:Document {id: $doc_id}) "
            "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk) "
            "WITH d, collect(c) AS chunks "
            "DETACH DELETE d "
            "FOREACH (c IN chunks | DETACH DELETE c)"
        )
        with self._driver.session(database=self.database) as session:
            summary = session.run(query, doc_id=doc_id).consume()
            return summary.counters.nodes_deleted

    def run_readonly(self, cypher: str, params: dict | None = None) -> list[dict]:
        safe = assert_safe_readonly(cypher)

        def _work(tx):
            return tx.run(safe, **(params or {})).data()

        with self._driver.session(database=self.database) as session:
            return session.execute_read(_work)

    def _read(self, query: str, **params):
        with self._driver.session(database=self.database) as session:
            return session.execute_read(lambda tx: tx.run(query, **params).data())
