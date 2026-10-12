from __future__ import annotations

import re
from collections.abc import Mapping

from qdrant_client.models import SparseVector


def lexical_weights_to_sparse(lexical_weights: Mapping[str, float]) -> SparseVector:
    items = sorted(
        (int(token), float(weight))
        for token, weight in dict(lexical_weights or {}).items()
        if float(weight) > 0.0
    )
    return SparseVector(
        indices=[index for index, _ in items],
        values=[value for _, value in items],
    )


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")

STOPWORDS = frozenset(
    """a about after all also an and any are as at be been before between both
    but by can could did do does down during each every few for from had has
    have how if in into is it its many may more most must no not of on only or
    other our out over per same should so some such than that the their them
    then there these they this those to too under up use used using very via
    was we were what when where which who whom why will with would you your
    issue question answer cause caused done being make made take given need
    wants know let just like well""".split()
)


def keyword_terms(text: str, limit: int = 14) -> list[str]:
    seen: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS or token in seen:
            continue
        seen.append(token)
        if len(seen) >= limit:
            break
    return seen


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"
