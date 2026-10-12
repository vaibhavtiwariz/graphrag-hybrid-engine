# GraphRAG & Self-Corrective Hybrid Retrieval Engine

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-graph-008CC1?logo=neo4j&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vectors-DC244C)
![LangGraph](https://img.shields.io/badge/LangGraph-CRAG-1C3C3C?logo=langchain&logoColor=white)
![Ragas](https://img.shields.io/badge/Ragas-evaluation-FF6F00)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

## What this is

A RAG system that answers questions over a document collection by combining a knowledge
graph with vector search. The idea came from a real problem: support teams drown in docs.
Confluence pages, incident reports, policy PDFs, supplier records, all scattered. Plain
vector search handles "what is X" fine, but it falls apart on multi-hop questions like
"which supplier's defect caused this incident and which batch was affected". That question
needs product, incident, component, and supplier records joined together. This engine
extracts those links into a graph, searches both the graph and the vectors, fuses the
results, and then checks its own work before answering.

Everything runs on free, self-hostable pieces: Neo4j, Qdrant, local embedding and rerank
models, and either a free Groq key or local Ollama.

Full detail: `docs/PROJECT_GUIDE.md`.

## Architecture

```
docs (md/txt/pdf)
   │  chunker → Instructor/LLM entity+relation extraction → doc summary
   ▼
Neo4j Property Graph                    Qdrant collection
  (:Document)-[:HAS_CHUNK]->(:Chunk)    "dense"  = BGE-M3 1024-d vectors
  (:Chunk)-[:MENTIONS]->(:Entity)       "sparse" = BGE-M3 lexical weights
  (:Entity)-[:RELATED]->(:Entity)
                 │
question ───────►├─ dense ANN search
                 ├─ sparse lexical search
                 ├─ graph traversal: entity match + LLM text2Cypher (read-only guard)
                 ▼
      Reciprocal Rank Fusion → BGE cross-encoder rerank
                 ▼
      CRAG loop (LangGraph): grade docs → sufficient? generate w/ citations
                             ↳ insufficient → rewrite query → retrieve again
                             ↳ exhausted   → DuckDuckGo web fallback
                 ▼
FastAPI /query (answer + citations + full trace) · Streamlit UI · Celery async ingest
```

## Free stack

| Component | Choice | Cost |
|---|---|---|
| Knowledge graph | Neo4j 5 Community (Docker) / AuraDB Free | free |
| Vector store | Qdrant OSS (Docker) / Cloud free tier | free |
| Dense embeddings | BAAI/bge-m3 local (1024-d) | free |
| Sparse vectors | BGE-M3 lexical weights in Qdrant native sparse index | free |
| Reranker | BAAI/bge-reranker-v2-m3 local cross-encoder | free |
| LLM | Ollama (llama3.2:3b) locally; Groq free tier via `LLM_PROVIDER=groq` | free |
| Structured output | Instructor (OpenAI-compatible endpoint) | free |
| Orchestration | LangGraph state machine | free |
| Web fallback | DuckDuckGo (`ddgs`), keyless | free |
| Async jobs | Celery + Redis | free |
| Evaluation | Ragas + DeepEval | free |
| Serving | FastAPI + Streamlit | free |

## Quickstart

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[heavy,eval,ui,websearch,dev]"

docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b

cp .env.example .env
.venv/bin/python scripts/ingest_cli.py data/sample_corpus

.venv/bin/python -m uvicorn app.api.main:app --port 8000
.venv/bin/python -m streamlit run app/ui/streamlit_app.py
```

Then ask it something multi-hop:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"Which supplier machining defect caused the AtlasArm overheating incident and which batch was affected?"}'
```

The response has the answer, numbered citations, how many CRAG iterations it ran, whether
it fell back to the web, and a per-stage latency trace.

Speed note: Ollama-in-Docker runs CPU-only, so answers take 30-60 seconds. A free Groq
key (`LLM_PROVIDER=groq` plus `GROQ_API_KEY` in `.env`) brings that down to a few seconds
at zero cost. No re-ingestion needed when you switch.

`make` wraps the common commands: `make setup infra-up pull-model ingest api ui eval-gen eval test lint smoke`.

## Running the evaluation

1. The golden set ships with the repo: `evaluation/golden_multi_hop.jsonl` (16 hand-written
   multi-hop questions with reference answers).
2. You can grow it from the graph itself:
   `.venv/bin/python evaluation/dataset_generator.py --num 110`
3. Run the harness (it runs the full pipeline per question, then scores with Ragas):

```bash
.venv/bin/python evaluation/run_eval.py --dataset evaluation/golden_multi_hop.jsonl
```

Metrics: faithfulness, answer relevancy, context precision, context recall. Add `--deepeval`
for G-Eval correctness scoring. Reports land in `evaluation/results/`. CI runs lint and unit
tests on every push.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness of Neo4j / Qdrant / Redis / LLM |
| `POST /query` | hybrid retrieval + CRAG, returns answer, citations, trace |
| `POST /ingest` | enqueue ingestion through Celery (`sync=true` for inline) |
| `POST /ingest/upload` | upload and ingest a single file |
| `GET /ingest/status/{id}` | status of a queued ingestion job |
| `GET /stats` | graph and vector counts |
| `GET /subgraph/{entity}` | neighborhood edges, used by the UI graph view |

## Config knobs

`.env.example` documents every setting. The ones you will touch most:

| Knob | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `groq` / `ollama` / `openai`, one line swaps the brain |
| `GROQ_API_KEY` | - | free key from console.groq.com/keys |
| `TOP_K_DENSE` / `TOP_K_SPARSE` | 10 / 10 | per-modality recall |
| `FINAL_TOP_K` | 6 | chunks kept after reranking |
| `RRF_K` | 60 | fusion constant (lower means top ranks dominate) |
| `MAX_CRAG_ITERATIONS` | 2 | rewrite-loop hard cap |
| `MIN_RELEVANT_DOCS` / `RELEVANCE_RATIO` | 2 / 0.34 | when grading counts as "good enough" |
| `ENABLE_WEB_FALLBACK` | true | DuckDuckGo rescue path when retrieval fails |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 120 | characters per chunk |

## Why this architecture

One honest line per choice:

- **Neo4j:** multi-hop questions need real graph traversals. A vector store cannot do
  product-to-component-to-supplier joins.
- **Qdrant:** dense and sparse vectors live in one collection. One index, one query path,
  no second BM25 pipeline to operate.
- **RRF:** it fuses ranks, not scores. Dense cosine scores, sparse lexical weights, and
  graph hop counts sit on completely different scales, so rank fusion needs no calibration.
- **Local BGE cross-encoder:** it scores the query and passage together, which beats
  bi-encoder similarity, and it is free and close to what hosted rerankers do.
- **CRAG loop:** retrieval fails. It just does. Instead of answering from bad context, the
  system grades what it found and either rewrites the query or falls back to the web.
- **Groq or Ollama:** both are free. Groq is fast, Ollama runs fully offline. One env var
  switches between them.

## Measured results

16 golden multi-hop questions, same corpus, same pipeline code. Ran October 2026.

| mode | faithfulness | answer relevancy | context precision | context recall | web fallback rate |
|---|---|---|---|---|---|
| hybrid (graph + vectors + rerank + CRAG) | 0.83 | 0.65 | 0.88 | 1.00 | 0% |
| vector-only baseline (dense retrieval only) | 1.00 | 0.56 | 0.62 | 1.00 | 0% |

What this says: the graph leg earns its keep on retrieval quality. Context precision
jumps from 0.62 to 0.88, which means far fewer irrelevant chunks reach the generator.
Answer relevancy moves from 0.56 to 0.65. Faithfulness and recall sit at ceiling for
both, the corpus is small and clean, so there is no headroom left to measure there.

How it was measured: Ragas metrics, 16 cases, BGE-M3 embeddings, local cross-encoder
rerank. Hybrid judged with gpt-oss-20b. Baseline judged with gpt-oss-20b for
faithfulness and recall; answer relevancy and precision used qwen3-27b because the
20b daily quota ran out mid-run. Same questions, same corpus, same harness. The full
per-case reports are in `evaluation/results/`.

Latency (median per stage, CPU-only box): vector search 1.3s, graph search 0.2s,
rerank 67s, generation 0.9s. The reranker is the bottleneck on CPU. On a GPU box it
drops to under a second. This is the honest cost of the cross-encoder leg.

## Limitations

- A local LLM on CPU is slow. Expect 30-60 second answers with Ollama-in-Docker. Use Groq
  for anything interactive.
- No auth on the API. Run it on localhost only. Do not expose port 8000 to the internet.
- The golden set is 16 cases. Small. Percentages computed from 16 questions are not
  statistics.
- The Ragas judge grades answers written by the same model family. That inflates
  faithfulness-type scores. Read them as directional, not absolute.
- The web fallback pulls untrusted content from the internet into the answer context. Fine
  for a demo; a real deployment needs source filtering.
- text2Cypher is best-effort. The guard only allows read-only queries, and hard questions
  may return nothing from the graph leg. The pipeline falls back gracefully.

## Future scope

- Incremental ingestion with document hashing, so re-ingesting only touches changed files.
- Typed Neo4j relationships (SUPPLIED_BY, CAUSED_BY) instead of the generic RELATED edge
  with a type property.
- Per-tenant collections and API keys.
- Eval drift monitoring that gates CI: fail the build if faithfulness drops.
- An MCP server so agents can query the engine directly.
- Streaming answers over SSE.
- Multilingual embeddings for non-English corpora.
- Growing the golden set with human-labeled questions instead of synthetic ones.
