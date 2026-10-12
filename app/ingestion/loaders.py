from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.vector_utils import slugify


@dataclass
class LoadedDocument:
    doc_id: str
    title: str
    text: str
    path: str


def _title_from_text(text: str, stem: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return stem


def load_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    doc_id = f"{slugify(path.stem)}-{digest}"
    title = _title_from_text(text, path.stem.replace("_", " ").replace("-", " "))
    return LoadedDocument(doc_id=doc_id, title=title, text=text, path=str(path))
