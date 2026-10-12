from __future__ import annotations

import pytest
from qdrant_client.models import SparseVector

from app.core.vector_utils import keyword_terms, lexical_weights_to_sparse, slugify


def test_lexical_weights_filters_zero_and_negative():
    sparse = lexical_weights_to_sparse({"10": 1.5, "20": 0.0, "30": -2.0, "5": 0.25})
    assert isinstance(sparse, SparseVector)
    assert list(sparse.indices) == [5, 10]
    assert list(sparse.values) == [0.25, 1.5]


def test_lexical_weights_sorts_indices():
    sparse = lexical_weights_to_sparse({"900": 1.0, "3": 0.5, "77": 0.9})
    assert list(sparse.indices) == [3, 77, 900]


def test_keyword_terms_drop_stopwords_and_short_tokens():
    terms = keyword_terms("Which supplier caused the INC-1042 AtlasArm overheating issue?")
    assert "supplier" in terms
    assert "inc-1042" in terms or "atlasarm" in terms
    assert "the" not in terms
    assert "which" not in terms


def test_keyword_terms_respects_limit():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    assert len(keyword_terms(text, limit=4)) == 4


def test_slugify_normalizes_names():
    assert slugify("Kestrel Metals!") == "kestrel-metals"
    assert slugify("  INC--1042 ") == "inc-1042"
    assert slugify("***") == "item"


def test_empty_input_produces_empty_sparse_vector():
    sparse = lexical_weights_to_sparse({})
    assert list(sparse.indices) == []
    assert list(sparse.values) == []


@pytest.mark.parametrize("bad", [None, {}])
def test_none_like_inputs_are_safe(bad):
    sparse = lexical_weights_to_sparse(bad or {})
    assert list(sparse.indices) == []
