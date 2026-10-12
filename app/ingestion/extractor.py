from __future__ import annotations

from app.config import get_settings
from app.core.llm import completion_extra_kwargs, structured_client
from app.ingestion.schema import (
    AnswerOut,
    BatchGrades,
    ChunkExtraction,
    DocSummary,
    MultiHopQA,
    RewriteOutput,
)

EXTRACTION_SYSTEM = """You are an information extraction engine building an enterprise knowledge graph.
Extract only facts explicitly present in the passage. Entity names must be short verbatim noun phrases.
Allowed entity types: PRODUCT, COMPONENT, SUPPLIER, PERSON, TEAM, INCIDENT, POLICY, LOCATION, PROCESS, METRIC.
relation_type must be UPPER_SNAKE such as SUPPLIED_BY, PART_OF, CAUSED_BY, FIXED_BY, CONTACT_FOR,
AFFECTS, REQUIRES, GOVERNED_BY, LOCATED_IN, OWNS, SUPERSEDES.
Skip marketing language and boilerplate. Maximum 10 entities and 8 relations per passage. Be terse."""

SUMMARY_SYSTEM = """You summarize enterprise documents for a retrieval index.
Write one dense paragraph (max 120 words) capturing products, components, suppliers, people,
incidents, policies and any concrete identifiers."""

GRADER_SYSTEM = """You grade retrieved passages for answering an enterprise question.
For each numbered passage decide is_relevant: true only if it contains facts needed to answer.
Judge relevance to the question intent, not surface keyword overlap."""

REWRITE_SYSTEM = """You rewrite retrieval queries for a hybrid knowledge-graph RAG system.
Return a single improved query line that names concrete entities, product SKUs, incident IDs or
policy codes implied by the feedback, and drops noisy words."""

ANSWER_SYSTEM = """You answer enterprise questions strictly from the provided context blocks.
Cite blocks inline using their [n] numbers after each claim. If the context is insufficient say
"Insufficient information in the knowledge base." Do not invent entities, numbers or policies.
Keep answers concise: 2-5 sentences unless the question explicitly requires more detail."""


def extract_chunk(text: str, title: str) -> ChunkExtraction:
    settings = get_settings()
    return structured_client().chat.completions.create(
        model=settings.llm_model,
        response_model=ChunkExtraction,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": f"Document: {title}\n\nPassage:\n{text}"},
        ],
        temperature=0,
        max_tokens=2600,
        max_retries=2,
        **completion_extra_kwargs(),
    )


def summarize_document(title: str, combined_text: str) -> DocSummary:
    settings = get_settings()
    trimmed = combined_text[:6000]
    return structured_client().chat.completions.create(
        model=settings.llm_model,
        response_model=DocSummary,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Document: {title}\n\nContent:\n{trimmed}"},
        ],
        temperature=0,
        max_tokens=1500,
        max_retries=2,
        **completion_extra_kwargs(),
    )


def grade_documents(question: str, numbered_excerpts: str) -> BatchGrades:
    settings = get_settings()
    return structured_client().chat.completions.create(
        model=settings.llm_model,
        response_model=BatchGrades,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\n{numbered_excerpts}"},
        ],
        temperature=0,
        max_retries=2,
        **completion_extra_kwargs(),
    )


def rewrite_query(question: str, feedback: str) -> RewriteOutput:
    settings = get_settings()
    return structured_client().chat.completions.create(
        model=settings.llm_model,
        response_model=RewriteOutput,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {
                "role": "user",
                "content": f"Original question: {question}\n\nRetrieval feedback:\n{feedback}",
            },
        ],
        temperature=0,
        max_retries=2,
        **completion_extra_kwargs(),
    )


def generate_answer(question: str, context_blocks: str) -> AnswerOut:
    settings = get_settings()
    return structured_client().chat.completions.create(
        model=settings.llm_model,
        response_model=AnswerOut,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"{context_blocks}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_retries=2,
        **completion_extra_kwargs(),
    )


def generate_multihop_qa(passage_a: str, passage_b: str) -> MultiHopQA:
    settings = get_settings()
    prompt = (
        "Passage A:\n"
        f"{passage_a[:1200]}\n\n"
        "Passage B:\n"
        f"{passage_b[:1200]}\n\n"
        "Write ONE enterprise question that can be answered ONLY by combining facts from both "
        "passages. Name concrete entities or identifiers. Then give a 1-3 sentence reference "
        "answer grounded in those facts."
    )
    return structured_client().chat.completions.create(
        model=settings.llm_model,
        response_model=MultiHopQA,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_retries=2,
        **completion_extra_kwargs(),
    )
