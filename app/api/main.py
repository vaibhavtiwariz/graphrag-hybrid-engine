from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)


class Citation(BaseModel):
    id: str
    title: str
    source: str
    score: float = 0.0


class TraceStep(BaseModel):
    stage: str
    detail: dict = {}
    ms: float = 0.0


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = []
    used_web: bool = False
    iterations: int = 0
    stop_reason: str = ""
    trace: list[TraceStep] = []
    graph_meta: dict = {}
    latency_ms: float = 0.0


class IngestRequest(BaseModel):
    path: str
    sync: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.services import ensure_stores

        ensure_stores()
        logger.info("stores ready")
    except Exception as exc:
        logger.warning("stores unavailable at startup: %s", exc)

    def _preload_models():
        try:
            from app.services import get_embedder, get_hybrid_retriever, get_reranker

            get_embedder()
            get_reranker()
            get_hybrid_retriever()
            logger.info("models preloaded")
        except Exception as exc:
            logger.warning("model preload failed: %s", exc)

    import threading

    threading.Thread(target=_preload_models, daemon=True).start()
    yield


app = FastAPI(
    title=get_settings().app_name,
    version="0.1.0",
    description="GraphRAG + self-corrective hybrid retrieval with continuous evaluation",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    statuses: dict[str, bool | str] = {"api": "ok"}

    try:
        from app.services import get_graph_store

        statuses["neo4j"] = get_graph_store().healthy()
    except Exception:
        statuses["neo4j"] = False

    try:
        from app.services import get_qdrant_store

        statuses["qdrant"] = get_qdrant_store().healthy()
    except Exception:
        statuses["qdrant"] = False

    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        statuses["redis"] = bool(client.ping())
    except Exception:
        statuses["redis"] = False

    provider = settings.llm_provider
    if provider == "groq" and settings.groq_api_key:
        try:
            response = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=4,
            )
            statuses["llm"] = response.status_code == 200
        except Exception:
            statuses["llm"] = False
    elif provider == "openai" and settings.openai_api_key:
        try:
            response = httpx.get(
                f"{settings.openai_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                timeout=4,
            )
            statuses["llm"] = response.status_code == 200
        except Exception:
            statuses["llm"] = False
    else:
        try:
            response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
            statuses["llm"] = response.status_code == 200
        except Exception:
            statuses["llm"] = False

    return statuses


@app.post("/query", response_model=AnswerResponse)
def query(request: QueryRequest) -> AnswerResponse:
    from time import perf_counter

    from app.crag.graph import run_crag

    start = perf_counter()
    try:
        result = run_crag(request.question)
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(status_code=503, detail=f"retrieval pipeline error: {exc}") from exc

    latency = round((perf_counter() - start) * 1000, 1)
    return AnswerResponse(
        question=result.question,
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        used_web=result.used_web,
        iterations=result.iterations,
        stop_reason=result.stop_reason,
        trace=[TraceStep(**step) for step in result.trace],
        graph_meta=_graph_meta(result),
        latency_ms=latency,
    )


def _graph_meta(result) -> dict:
    for step in result.trace:
        if step.get("stage") == "graph_search":
            return step.get("detail", {})
    return {}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    path = Path(request.path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {path}")

    if request.sync:
        from app.ingestion.pipeline import ingest_directory, ingest_file, report_to_dict

        if path.is_dir():
            reports = [report_to_dict(r) for r in ingest_directory(path)]
            return {"mode": "sync", "documents": reports}
        report = ingest_file(path)
        return {"mode": "sync", "documents": [report_to_dict(report)]}

    try:
        from app.tasks.ingest_tasks import (
            ingest_directory_task,
            ingest_file_task,
        )

        task = (
            ingest_directory_task.delay(str(path))
            if path.is_dir()
            else ingest_file_task.delay(str(path))
        )
        return {"mode": "queued", "task_id": task.id}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"queue unavailable ({exc}); retry with sync=true",
        ) from exc


@app.post("/ingest/upload")
async def ingest_upload(file: Annotated[UploadFile, File()]) -> dict:
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / _safe_upload_name(file.filename)
    size = 0
    try:
        with target.open("wb") as fh:
            while True:
                piece = await file.read(1024 * 1024)
                if not piece:
                    break
                size += len(piece)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="upload exceeds 25 MB limit")
                fh.write(piece)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise

    from app.ingestion.pipeline import ingest_file, report_to_dict

    report = ingest_file(target)
    return {"mode": "sync", "document": report_to_dict(report)}


def _safe_upload_name(filename: str | None) -> str:
    """Strip directory components, keep only safe characters, and prefix with a
    uuid so a malicious filename can never escape the upload directory."""
    raw = Path(filename or "").name
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", raw).strip("._")
    if not safe:
        raise HTTPException(status_code=400, detail="invalid upload filename")
    ext = Path(safe).suffix[:12]
    return f"{uuid.uuid4().hex}{ext}"


@app.get("/ingest/status/{task_id}")
def ingest_status(task_id: str) -> dict:
    from app.tasks.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)
    payload = {"task_id": task_id, "state": result.state}
    if result.ready():
        payload["result"] = result.result if result.successful() else str(result.result)
    return payload


@app.get("/documents")
def list_documents() -> dict:
    try:
        from app.services import get_graph_store

        return {"documents": get_graph_store().list_documents()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    try:
        from app.services import get_graph_store, get_qdrant_store

        nodes_deleted = get_graph_store().delete_document(doc_id)
        get_qdrant_store().delete_by_doc(doc_id)
        return {"deleted": doc_id, "nodes_deleted": nodes_deleted}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/stats")
def stats() -> dict:
    output: dict = {}
    try:
        from app.services import get_graph_store

        output["graph"] = get_graph_store().counts()
    except Exception as exc:
        output["graph"] = {"error": str(exc)}
    try:
        from app.services import get_qdrant_store

        output["qdrant_points"] = get_qdrant_store().count()
    except Exception as exc:
        output["qdrant_points"] = None
        output["qdrant_error"] = str(exc)
    return output


@app.get("/subgraph/{entity_name}")
def subgraph(entity_name: str, depth: int = 1) -> dict:
    try:
        from app.services import get_graph_store

        return get_graph_store().entity_subgraph(entity_name, depth=depth)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
