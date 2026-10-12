from __future__ import annotations

from app.retrieval.fusion import rrf_fuse


def test_rrf_prefers_items_ranked_high_across_lists():
    fused = dict(
        rrf_fuse(
            [["a", "b", "c"], ["b", "c", "d"], ["c", "e"]],
            k=60,
        )
    )
    assert fused["c"] > fused["b"] > fused["a"]
    assert abs(fused["a"] - 1 / 61) < 1e-9
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(fused["c"] - (1 / 63 + 1 / 62 + 1 / 61)) < 1e-9


def test_rrf_single_list_preserves_order():
    ranking = ["x", "y", "z"]
    fused = [cid for cid, _ in rrf_fuse([ranking], k=60)]
    assert fused == ranking


def test_rrf_k_changes_spread():
    top_only = dict(rrf_fuse([["a", "b"], ["a", "b"]], k=1))
    spread = dict(rrf_fuse([["a", "b"], ["a", "b"]], k=1000))
    assert top_only["a"] - top_only["b"] > spread["a"] - spread["b"]


def test_rrf_deduplicates_within_one_ranking_gracefully():
    fused = dict(rrf_fuse([["a", "a", "b"]], k=60))
    assert abs(fused["a"] - (1 / 61 + 1 / 62)) < 1e-9
    assert abs(fused["b"] - 1 / 63) < 1e-9
