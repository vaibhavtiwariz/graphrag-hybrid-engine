from __future__ import annotations

import pytest

from app.retrieval.cypher_guard import CypherSafetyError, assert_safe_readonly, is_safe_readonly


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'kestrel' RETURN e LIMIT 10",
        "OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(x) RETURN c LIMIT 5",
        "UNWIND $keys AS k MATCH (e:Entity {key: k}) RETURN e.name",
        "WITH 1 AS x MATCH (n) RETURN count(n)",
    ],
)
def test_accepts_readonly_queries(cypher):
    assert is_safe_readonly(cypher)


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (n:Entity {name: 'evil'})",
        "MERGE (n:Entity {key: 'evil'})",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) DELETE n",
        "MATCH (n) SET n.hacked = true",
        "DROP DATABASE neo4j",
        "CALL db.labels() YIELD label RETURN label",
        "LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row",
        "MATCH (a) CREATE (a)-[:X]->(a)",
    ],
)
def test_rejects_mutating_or_procedural_queries(cypher):
    assert not is_safe_readonly(cypher)


def test_rejects_multi_statement_and_empty():
    assert not is_safe_readonly("MATCH (n) RETURN n; DROP DATABASE neo4j")
    assert not is_safe_readonly("")
    assert not is_safe_readonly(None)
    assert not is_safe_readonly("RETURN 1")


def test_assert_raises():
    with pytest.raises(CypherSafetyError):
        assert_safe_readonly("DELETE FROM nothing")
