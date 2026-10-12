MODEL ?= llama3.2:3b
PY := .venv/bin/python
UV := $(shell command -v $(HOME)/.local/bin/uv 2>/dev/null || command -v uv 2>/dev/null)

.PHONY: setup infra-up infra-down pull-model ingest api worker ui eval-gen eval smoke reset test lint fmt clean

setup:
	$(UV) python install 3.11
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python .venv/bin/python -e ".[heavy,eval,ui,websearch,dev]"

infra-up:
	docker compose up -d

infra-down:
	docker compose down

pull-model:
	docker compose exec ollama ollama pull $(MODEL)

ingest:
	$(PY) scripts/ingest_cli.py data/sample_corpus

api:
	$(PY) -m uvicorn app.api.main:app --reload --port 8000

worker:
	$(PY) -m celery -A app.tasks.celery_app:celery_app worker -l info -Q celery

ui:
	$(PY) -m streamlit run app/ui/streamlit_app.py

eval-gen:
	$(PY) evaluation/dataset_generator.py --num 110 --out evaluation/generated_qa.jsonl

eval:
	$(PY) evaluation/run_eval.py --dataset evaluation/golden_multi_hop.jsonl --dataset evaluation/generated_qa.jsonl

smoke:
	$(PY) scripts/smoke_test.py

reset:
	$(PY) scripts/reset_kb.py --yes

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check app evaluation scripts tests

fmt:
	.venv/bin/ruff check --fix app evaluation scripts tests

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
