from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.crag.components import CRAGComponents
from app.crag.graph import build_crag_graph
from app.ingestion.schema import GradeItem


def fake_docs(n_chunks: int = 6, score: float = 0.2) -> SimpleNamespace:
    chunks = [
        SimpleNamespace(
            id=f"chunk-{i}",
            text=f"passage {i}",
            title=f"title {i}",
            doc_id="doc",
            kind="chunk",
            score=score,
            rerank_score=None,
        )
        for i in range(n_chunks)
    ]
    return SimpleNamespace(chunks=chunks, trace=[], graph_meta={"entities": ["kestrel-metals"]})


def all_irrelevant(question: str, docs) -> list[GradeItem]:
    return [GradeItem(index=i, is_relevant=False, reason="off topic") for i in range(len(docs.chunks))]


def all_relevant(question: str, docs) -> list[GradeItem]:
    return [GradeItem(index=i, is_relevant=True, reason="on point") for i in range(len(docs.chunks))]


def make_components(grader, docs=None) -> CRAGComponents:
    return CRAGComponents(
        retrieve=lambda question: docs or fake_docs(),
        grade=grader,
        rewrite=lambda question, feedback: f"{question} refined",
        web_search=lambda question: [{"title": "web", "url": "http://x", "snippet": "s"}],
        generate=lambda question, docs, web: (
            "final answer",
            [{"id": "1", "title": "t", "source": "kb", "score": 0.5}],
            [],
        ),
    )


def invoke(components, settings_overrides: dict | None = None) -> dict:
    original_settings = __import__("app.crag.nodes", fromlist=["get_settings"])
    real_factory = original_settings.get_settings
    if settings_overrides:
        original_settings.get_settings = lambda: Settings(**settings_overrides)
    try:
        app = build_crag_graph(components)
        final_state = app.invoke(
            {
                "question": "which supplier caused INC-1042?",
                "original": "which supplier caused INC-1042?",
                "iteration": 0,
                "used_web": False,
                "trace": [],
            },
            config={"recursion_limit": 24},
        )
        return dict(final_state)
    finally:
        original_settings.get_settings = real_factory


def test_happy_path_generates_without_correction(monkeypatch):
    import app.crag.nodes as nodes

    monkeypatch.setattr(nodes, "get_settings", lambda: Settings())
    state = invoke(make_components(all_relevant))
    assert state["stop_reason"] == "generated"
    assert state["answer"] == "final answer"
    assert state["used_web"] is False
    assert state["iteration"] == 0


def test_insufficient_context_triggers_rewrite_then_web_fallback(monkeypatch):
    import app.crag.nodes as nodes

    monkeypatch.setattr(
        nodes,
        "get_settings",
        lambda: Settings(max_crag_iterations=1, enable_web_fallback=True),
    )
    state = invoke(make_components(all_irrelevant))
    assert state["iteration"] == 1
    assert state["used_web"] is True
    assert state["stop_reason"] == "generated_with_web"
    assert any(step["stage"] == "rewrite_query" for step in state["trace"])
    assert any(step["stage"] == "web_fallback" for step in state["trace"])


def test_min_relevant_docs_short_circuits_ratio_rule(monkeypatch):
    import app.crag.nodes as nodes

    def two_of_six_relevant(question, docs):
        return [
            GradeItem(index=i, is_relevant=i < 2, reason="")
            for i in range(len(docs.chunks))
        ]

    monkeypatch.setattr(nodes, "get_settings", lambda: Settings(relevance_ratio=0.9))
    state = invoke(make_components(two_of_six_relevant))
    assert state["stop_reason"] == "generated"
    assert state["iteration"] == 0


def test_no_fallback_possible_still_answers(monkeypatch):
    import app.crag.nodes as nodes

    monkeypatch.setattr(
        nodes,
        "get_settings",
        lambda: Settings(max_crag_iterations=0, enable_web_fallback=False),
    )
    state = invoke(make_components(all_irrelevant))
    assert state["stop_reason"] == "generated"
    assert state["answer"]


def test_high_confidence_bypasses_grader(monkeypatch):
    import app.crag.nodes as nodes

    monkeypatch.setattr(nodes, "get_settings", lambda: Settings())
    state = invoke(make_components(all_relevant, docs=fake_docs(score=0.97)))
    assert state["stop_reason"] == "generated_fastpath"
    assert state["answer"] == "final answer"
    stages = [step["stage"] for step in state["trace"]]
    assert "grade_documents" not in stages


def test_low_confidence_still_grades(monkeypatch):
    import app.crag.nodes as nodes

    monkeypatch.setattr(nodes, "get_settings", lambda: Settings())
    state = invoke(make_components(all_relevant, docs=fake_docs(score=0.3)))
    assert state["stop_reason"] == "generated"
    stages = [step["stage"] for step in state["trace"]]
    assert "grade_documents" in stages


def test_empty_retrieval_goes_straight_to_rewrite(monkeypatch):
    import app.crag.nodes as nodes

    empty = SimpleNamespace(chunks=[], trace=[], graph_meta={})
    monkeypatch.setattr(
        nodes, "get_settings", lambda: Settings(max_crag_iterations=1, enable_web_fallback=True)
    )
    state = invoke(make_components(all_irrelevant, docs=empty))
    assert state["iteration"] == 1
    stages = [step["stage"] for step in state["trace"]]
    assert "grade_documents" not in stages
    assert "rewrite_query" in stages
