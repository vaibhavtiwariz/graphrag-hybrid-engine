#!/usr/bin/env bash
# Full hybrid-vs-baseline comparison for the GraphRAG engine.
#
# Prerequisites (run once):
#   docker compose up -d          # Neo4j, Qdrant, Redis, Ollama
#   cp .env.example .env          # then set LLM_PROVIDER (ollama or groq)
#   docker compose exec ollama ollama pull llama3.1:8b   # if using ollama
#
# What this does:
#   1. Ingests the sample corpus (Neo4j graph + Qdrant vectors).
#   2. Runs the hybrid eval: dense + sparse + graph retrieval, RRF fusion,
#      cross-encoder rerank, CRAG loop, Ragas metrics.
#   3. Runs the vector-only baseline: dense ANN only, single
#      retrieve+generate pass (no graph, no rerank, no rewrite, no web
#      fallback), same Ragas metrics.
#   4. Prints both JSON summaries side by side for the honest comparison.
#
# Reports land in evaluation/results/ as:
#   full_eval_hybrid_<yyyymmdd_hhmm>.json
#   full_eval_vector-only_<yyyymmdd_hhmm>.json
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
PY=".venv/bin/python"

echo "== 1/4 Ingesting sample corpus =="
$PY scripts/ingest_cli.py data/sample_corpus

echo "== 2/4 Hybrid eval (16 golden multi-hop cases) =="
$PY evaluation/run_eval.py --dataset evaluation/golden_multi_hop.jsonl

echo "== 3/4 Vector-only baseline eval =="
$PY evaluation/run_eval.py --dataset evaluation/golden_multi_hop.jsonl --baseline

echo "== 4/4 Summaries =="
latest_hybrid=$(ls -t evaluation/results/full_eval_hybrid_*.json | head -1)
latest_baseline=$(ls -t evaluation/results/full_eval_vector-only_*.json | head -1)

print_summary() {
  "$PY" - "$1" <<'EOF'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)
keys = ("mode", "timestamp", "cases", "baseline_validation",
        "ragas", "web_fallback_rate", "median_stage_ms")
print(json.dumps({k: d.get(k) for k in keys}, indent=2, ensure_ascii=False))
EOF
}

echo "--- hybrid: $latest_hybrid"
print_summary "$latest_hybrid"
echo "--- vector-only baseline: $latest_baseline"
print_summary "$latest_baseline"
