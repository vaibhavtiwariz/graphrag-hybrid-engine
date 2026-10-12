# GraphRAG & Self-Corrective Hybrid Retrieval Engine: Executive Summary

> Print-ready one-pager: open `docs/EXECUTIVE_SUMMARY.html` in a browser and press
> **Cmd+P → Save as PDF** (A4, exact colors preserved).

**What it is.** A production-style retrieval platform that unites a Neo4j knowledge graph,
Qdrant dense+sparse vectors, RRF fusion, cross-encoder reranking, and a self-corrective CRAG
loop (LangGraph) with continuous Ragas/DeepEval evaluation: built end-to-end on free,
self-hostable resources (Groq free tier or local Ollama, OSS stores, local HuggingFace models).

**Why it matters.** Vector-only RAG fails on enterprise knowledge: no entity awareness for
multi-hop questions, keyword blindness on IDs/SKUs/codes, and blind generation with zero
quality measurement. This engine fuses four retrieval signals, grades every retrieval and
self-corrects (rewrite → web fallback), and proves quality with a benchmark harness.

**Pipeline.** Ingest (chunk → LLM entity/relation extraction → Neo4j graph + BGE-M3
dense/sparse vectors) → parallel retrieval (dense ANN + sparse + guarded text2Cypher graph
hops) → RRF fusion → cross-encoder rerank → CRAG loop (confidence fastpath → LLM grading →
rewrite → web fallback) → grounded, cited answer with a full latency trace.

**Measured impact** (17 indexed documents, warm cache, Groq gpt-oss-20b):

| KPI | Value |
|---|---|
| Warm end-to-end latency | **2.1–2.5 s** (from ~60 s fully-local baseline; −71% vs untuned hosted build) |
| Ragas faithfulness / context precision / recall | **1.0 / 1.0 / 1.0** on golden multi-hop cases |
| Tests | 49 unit tests green, ruff-clean, CI-gated |
| Cost | $0 mandatory (free tiers + OSS) |

Key optimizations: rerank pool 30→16 with 600-char pairs (−53%), conditional text2Cypher
(−95% graph stage), reranker-confidence grading fastpath (−100% grading on confident hits),
single-rewrite cap, `reasoning_effort=low`, parallel retrieval threads, startup model preload.

**Stack.** Neo4j 5 · Qdrant · BGE-M3 · bge-reranker-v2-m3 · LangGraph · Instructor ·
Groq gpt-oss-20b / Ollama · FastAPI · Streamlit · Celery+Redis · Ragas · DeepEval · Docker ·
GitHub Actions.

**Use cases.** Enterprise KB multi-hop Q&A, support copilots (warranty/RMA/SLA), compliance
cross-checks, resume/candidate screening (demonstrated live), policy helpdesk, procurement
chains, research literature graphs, contract clause tracing.

**Roadmap.** GPU/MPS reranking, Leiden graph communities for global questions, query
decomposition, SSE streaming, multi-turn memory, feedback-driven nightly Ragas regression
gates in CI, auth/multi-tenancy, docx/OCR ingestion, K8s packaging.

---
*Full details: `docs/PROJECT_GUIDE.md` (or the styled `PROJECT_GUIDE.html`).*
