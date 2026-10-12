from __future__ import annotations

from app.config import get_settings


def duckduckgo_search_fn(question: str) -> list[dict]:
    settings = get_settings()
    if not settings.enable_web_fallback:
        return []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        raw = DDGS().text(question, max_results=settings.web_results_k)
        results = []
        for item in raw or []:
            results.append(
                {
                    "title": item.get("title", "Web result"),
                    "url": item.get("href") or item.get("url", ""),
                    "snippet": (item.get("body") or item.get("snippet") or "")[:800],
                }
            )
        return results
    except Exception:
        return []
