from __future__ import annotations

import os
from time import perf_counter

from app.config import get_settings
from app.crag.components import CRAGComponents, CRAGConfig
from app.ingestion.extractor import generate_answer, grade_documents, rewrite_query
from app.ingestion.schema import AnswerOut, BatchGrades, GradeItem


def _ordered_context(docs) -> list:
    return sorted(getattr(docs, "chunks", []) or [], key=lambda c: 0 if c.kind != "summary" else 1)


def make_retrieve(components: CRAGComponents):
    def retrieve_node(state: dict) -> dict:
        result = components.retrieve(state["question"])
        trace = list(state.get("trace", []))
        trace.extend(getattr(result, "trace", []))
        return {"docs": result, "trace": trace}

    return retrieve_node


def make_grader(components: CRAGComponents):
    def grade_node(state: dict) -> dict:
        start = perf_counter()
        grades: list[GradeItem] = components.grade(state["question"], state["docs"])
        relevant = sum(1 for grade in grades if grade.is_relevant)
        trace = list(state.get("trace", []))
        trace.append(
            {
                "stage": "grade_documents",
                "detail": {
                    "graded": len(grades),
                    "relevant": relevant,
                    "misses": [g.reason[:80] for g in grades if not g.is_relevant][:3],
                },
                "ms": round((perf_counter() - start) * 1000, 1),
            }
        )
        return {"grades": grades, "trace": trace}

    return grade_node


def make_route_after_retrieve(config: CRAGConfig):
    """Closure over CRAGConfig: routes after the retrieve node."""

    def _route(state: dict) -> str:
        if os.environ.get("RETRIEVAL_BASELINE") == "1":
            return "generate"
        chunks = getattr(state.get("docs"), "chunks", []) or []
        if chunks:
            top = max(
                (chunk.rerank_score if chunk.rerank_score is not None else chunk.score)
                for chunk in chunks
            )
            if top >= config.grade_bypass_threshold:
                return "generate"
            return "grade"
        if state.get("iteration", 0) < config.max_iterations:
            return "rewrite"
        if config.enable_web_fallback and not state.get("used_web"):
            return "web_search"
        return "generate"

    return _route


def make_route_after_grade(config: CRAGConfig):
    """Closure over CRAGConfig: routes after the grade node."""

    def _route(state: dict) -> str:
        if os.environ.get("RETRIEVAL_BASELINE") == "1":
            return "generate"
        grades = state.get("grades") or []
        total = len(getattr(state.get("docs"), "chunks", []) or [])
        relevant = sum(1 for grade in grades if grade.is_relevant)

        if total and (
            relevant >= config.min_relevant_docs
            or (relevant > 0 and relevant / max(total, 1) >= config.relevance_ratio)
        ):
            return "generate"
        if state.get("iteration", 0) < config.max_iterations:
            return "rewrite"
        if config.enable_web_fallback and not state.get("used_web"):
            return "web_search"
        return "generate"

    return _route


def route_after_retrieve(state: dict) -> str:
    """Backward-compatible entry point; resolves config from global settings."""
    return make_route_after_retrieve(CRAGConfig.from_settings(get_settings()))(state)


def route_after_grade(state: dict) -> str:
    """Backward-compatible entry point; resolves config from global settings."""
    return make_route_after_grade(CRAGConfig.from_settings(get_settings()))(state)


def make_rewriter(components: CRAGComponents):
    def rewrite_node(state: dict) -> dict:
        start = perf_counter()
        feedback_lines = [
            f"irrelevant passage: {grade.reason}"
            for grade in (state.get("grades") or [])
            if not grade.is_relevant
        ]
        graph_meta = getattr(state.get("docs"), "graph_meta", {}) or {}
        if not graph_meta.get("entities"):
            feedback_lines.append(
                "no knowledge-graph entities matched; target concrete products, SKUs, "
                "incident IDs or policy codes"
            )
        new_query = components.rewrite(
            state["original"], "\n".join(feedback_lines) or "broaden entity coverage"
        )
        trace = list(state.get("trace", []))
        trace.append(
            {
                "stage": "rewrite_query",
                "detail": {"new_query": new_query},
                "ms": round((perf_counter() - start) * 1000, 1),
            }
        )
        return {
            "question": new_query,
            "iteration": state.get("iteration", 0) + 1,
            "trace": trace,
        }

    return rewrite_node


def make_web_searcher(components: CRAGComponents):
    def web_node(state: dict) -> dict:
        start = perf_counter()
        snippets = components.web_search(state["question"])
        trace = list(state.get("trace", []))
        trace.append(
            {
                "stage": "web_fallback",
                "detail": {"results": len(snippets)},
                "ms": round((perf_counter() - start) * 1000, 1),
            }
        )
        return {"web_snippets": snippets, "used_web": True, "trace": trace}

    return web_node


def make_generator(components: CRAGComponents):
    def generate_node(state: dict) -> dict:
        start = perf_counter()
        web_snippets = state.get("web_snippets") or []
        answer_text, citations, contexts = components.generate(
            state["question"], state["docs"], web_snippets
        )
        trace = list(state.get("trace", []))
        grades = state.get("grades") or []
        stop_reason = (
            "generated_with_web"
            if web_snippets
            else ("generated" if grades else "generated_fastpath")
        )
        trace.append(
            {
                "stage": "generate",
                "detail": {
                    "kb_contexts": len(contexts),
                    "web_snippets": len(web_snippets),
                    "graded": len(grades),
                },
                "ms": round((perf_counter() - start) * 1000, 1),
            }
        )
        return {
            "answer": answer_text,
            "citations": citations,
            "contexts": contexts,
            "stop_reason": stop_reason,
            "trace": trace,
        }

    return generate_node


def default_components() -> CRAGComponents:
    from app.services import get_hybrid_retriever, get_web_search_fn

    def _grade(question: str, docs) -> list[GradeItem]:
        excerpts = "\n\n".join(
            f"[{i}] {chunk.title}\n{chunk.text[:800]}" for i, chunk in enumerate(docs.chunks)
        )
        result: BatchGrades = grade_documents(question, excerpts)
        by_index = {item.index: item for item in result.items}
        return [
            by_index.get(i, GradeItem(index=i, is_relevant=False, reason="not graded"))
            for i in range(len(docs.chunks))
        ]

    def _rewrite(question: str, feedback: str) -> str:
        return rewrite_query(question, feedback).query.strip()

    def _generate(question: str, docs, web_snippets: list[dict]):
        ordered = _ordered_context(docs)
        blocks: list[str] = []
        citations: list[dict] = []
        contexts: list = []
        counter = 1
        for chunk in ordered[:6]:
            blocks.append(f"[{counter}] (knowledge-base) {chunk.title}\n{chunk.text}")
            citations.append(
                {
                    "id": chunk.id,
                    "title": chunk.title or chunk.doc_id,
                    "source": chunk.kind,
                    "score": round(chunk.rerank_score if chunk.rerank_score is not None else chunk.score, 4),
                }
            )
            contexts.append(chunk)
            counter += 1
        for snippet in web_snippets:
            blocks.append(
                f"[{counter}] (web) {snippet.get('title', '')} "
                f"{snippet.get('url', '')}\n{snippet.get('snippet', '')}"
            )
            citations.append(
                {
                    "id": snippet.get("url", f"web:{counter}"),
                    "title": snippet.get("title", "Web result"),
                    "source": "web",
                    "score": 0.0,
                }
            )
            contexts.append(snippet)
            counter += 1
        if not blocks:
            answer = AnswerOut(answer="Insufficient information: no context was retrieved.")
        else:
            answer = generate_answer(question, "Context blocks:\n\n" + "\n\n".join(blocks))
        return answer.answer, citations, contexts

    return CRAGComponents(
        retrieve=lambda question: get_hybrid_retriever().retrieve(question),
        grade=_grade,
        rewrite=_rewrite,
        web_search=get_web_search_fn(),
        generate=_generate,
    )
