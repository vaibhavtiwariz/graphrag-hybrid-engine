from __future__ import annotations

import weakref
from dataclasses import dataclass, field

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from app.crag.components import CRAGComponents, CRAGConfig


class CRAGState(TypedDict, total=False):
    question: str
    original: str
    iteration: int
    docs: object
    grades: list
    web_snippets: list
    used_web: bool
    answer: str
    citations: list
    contexts: list
    stop_reason: str
    trace: list


@dataclass
class CRAGResult:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    contexts: list = field(default_factory=list)
    iterations: int = 0
    used_web: bool = False
    stop_reason: str = ""
    trace: list[dict] = field(default_factory=list)


def _config_signature(config: CRAGConfig) -> tuple:
    return (
        config.min_relevant_docs,
        config.relevance_ratio,
        config.max_iterations,
        config.enable_web_fallback,
        config.grade_bypass_threshold,
    )


# Module-level cache: compiling the LangGraph app is expensive, so compile once
# per (components, config) pair. The weakref guards against id() reuse after GC.
_compiled_cache: dict[tuple[int, tuple], tuple] = {}


def _get_compiled_app(components: CRAGComponents, config: CRAGConfig):
    key = (id(components), _config_signature(config))
    entry = _compiled_cache.get(key)
    if entry is not None:
        ref, app = entry
        if ref() is components:
            return app
    app = build_crag_graph(components, config)
    _compiled_cache[key] = (weakref.ref(components), app)
    return app


def build_crag_graph(components: CRAGComponents, config: CRAGConfig | None = None):
    from app.crag import nodes
    from app.crag.nodes import (
        make_generator,
        make_grader,
        make_retrieve,
        make_rewriter,
        make_route_after_grade,
        make_route_after_retrieve,
        make_web_searcher,
    )

    if config is None:
        # Resolve via nodes.get_settings() so test monkeypatching of that
        # symbol keeps working.
        config = CRAGConfig.from_settings(nodes.get_settings())

    graph = StateGraph(CRAGState)
    graph.add_node("retrieve", make_retrieve(components))
    graph.add_node("grade", make_grader(components))
    graph.add_node("rewrite", make_rewriter(components))
    graph.add_node("web_search", make_web_searcher(components))
    graph.add_node("generate", make_generator(components))

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        make_route_after_retrieve(config),
        {"grade": "grade", "rewrite": "rewrite", "web_search": "web_search", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "grade",
        make_route_after_grade(config),
        {"rewrite": "rewrite", "web_search": "web_search", "generate": "generate"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def run_crag(
    question: str,
    components: CRAGComponents | None = None,
    config: CRAGConfig | None = None,
) -> CRAGResult:
    from app.crag import nodes

    components = components or build_default_components()
    if config is None:
        config = CRAGConfig.from_settings(nodes.get_settings())
    app = _get_compiled_app(components, config)
    initial: CRAGState = {
        "question": question.strip(),
        "original": question.strip(),
        "iteration": 0,
        "used_web": False,
        "trace": [],
    }
    final_state = app.invoke(initial, config={"recursion_limit": 24})
    return CRAGResult(
        question=question,
        answer=final_state.get("answer", ""),
        citations=final_state.get("citations", []),
        contexts=final_state.get("contexts", []),
        iterations=final_state.get("iteration", 0),
        used_web=final_state.get("used_web", False),
        stop_reason=final_state.get("stop_reason", ""),
        trace=final_state.get("trace", []),
    )


def build_default_components() -> CRAGComponents:
    from app.crag.nodes import default_components

    return default_components()
