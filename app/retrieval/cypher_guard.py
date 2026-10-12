from __future__ import annotations

import re

FORBIDDEN_RE = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|call|load\s+csv|foreach)\b",
    re.IGNORECASE,
)
ALLOWED_START_RE = re.compile(r"^\s*(match|optional\s+match|unwind|with)\b", re.IGNORECASE)


class CypherSafetyError(ValueError):
    pass


def is_safe_readonly(cypher: str | None) -> bool:
    if not cypher:
        return False
    query = cypher.strip().rstrip(";")
    if ";" in query:
        return False
    if FORBIDDEN_RE.search(query):
        return False
    return bool(ALLOWED_START_RE.match(query))


def assert_safe_readonly(cypher: str | None) -> str:
    if not is_safe_readonly(cypher):
        raise CypherSafetyError(f"Refusing to execute non read-only Cypher: {cypher!r}")
    return cypher.strip().rstrip(";")
