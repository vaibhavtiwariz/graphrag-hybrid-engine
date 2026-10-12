from __future__ import annotations

import logging

from app.ingestion.pipeline import ingest_directory, ingest_file, report_to_dict
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="ingest.file")
def ingest_file_task(path: str) -> dict:
    report = ingest_file(path)
    return report_to_dict(report)


@celery_app.task(name="ingest.directory")
def ingest_directory_task(path: str) -> list[dict]:
    return [report_to_dict(r) for r in ingest_directory(path)]
