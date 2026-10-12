from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


def prune_by_score_floor(hits: list, floor: float) -> list:
    if not hits:
        return []
    kept = [
        hit
        for hit in hits
        if (hit.rerank_score if hit.rerank_score is not None else hit.score) >= floor
    ]
    return kept if kept else hits[:1]


class BGEReranker:
    def __init__(self, enabled: bool | None = None) -> None:
        settings = get_settings()
        self.enabled = settings.enable_rerank if enabled is None else enabled
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            settings = get_settings()
            kwargs: dict = {}
            if settings.use_fp16:
                kwargs["use_fp16"] = True
            self._model = FlagReranker(settings.rerank_model_name, **kwargs)
        return self._model

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list:
        if not hits:
            return []
        if not self.enabled:
            return hits[:top_k] if top_k else hits
        try:
            model = self._load()
            pairs = [[query, hit.text[:600]] for hit in hits]
            max_length = get_settings().rerank_max_length
            try:
                scores = model.compute_score(pairs, normalize=True, max_length=max_length)
            except TypeError:
                scores = model.compute_score(pairs, normalize=True)
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            for hit, score in zip(hits, scores, strict=False):
                hit.rerank_score = float(score)
            hits.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        except Exception as exc:
            logger.warning("cross-encoder rerank failed, falling back to fused order: %s", exc)
        return hits[:top_k] if top_k else hits
