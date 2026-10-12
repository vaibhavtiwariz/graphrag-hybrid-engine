from __future__ import annotations

from types import SimpleNamespace

from app.retrieval.reranker import prune_by_score_floor


def hit(rid: str, rerank_score=None, score=0.0):
    return SimpleNamespace(id=rid, rerank_score=rerank_score, score=score)


def test_prune_drops_below_floor():
    hits = [hit("a", 0.9), hit("b", 0.4), hit("c", 0.01)]
    kept = prune_by_score_floor(hits, 0.05)
    assert [h.id for h in kept] == ["a", "b"]


def test_prune_always_keeps_top_hit_when_all_below_floor():
    hits = [hit("a", 0.01), hit("b", 0.001)]
    kept = prune_by_score_floor(hits, 0.5)
    assert [h.id for h in kept] == ["a"]


def test_prune_falls_back_to_base_score_when_rerank_missing():
    hits = [hit("a", None, score=0.8), hit("b", None, score=0.01)]
    kept = prune_by_score_floor(hits, 0.05)
    assert [h.id for h in kept] == ["a"]


def test_prune_empty_input():
    assert prune_by_score_floor([], 0.05) == []
