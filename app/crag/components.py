from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings, get_settings


@dataclass
class CRAGComponents:
    retrieve: Callable[[str], object]
    grade: Callable[[str, object], list]
    rewrite: Callable[[str, str], str]
    web_search: Callable[[str], list]
    generate: Callable[[str, object, list], tuple]


@dataclass
class CRAGConfig:
    min_relevant_docs: int = 2
    relevance_ratio: float = 0.34
    max_iterations: int = 2
    enable_web_fallback: bool = True
    grade_bypass_threshold: float = 0.78

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> CRAGConfig:
        settings = settings or get_settings()
        return cls(
            min_relevant_docs=settings.min_relevant_docs,
            relevance_ratio=settings.relevance_ratio,
            max_iterations=settings.max_crag_iterations,
            enable_web_fallback=settings.enable_web_fallback,
            grade_bypass_threshold=settings.grade_bypass_threshold,
        )
