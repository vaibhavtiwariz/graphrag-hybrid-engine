from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ALLOWED_ENTITY_TYPES = (
    "PRODUCT",
    "COMPONENT",
    "SUPPLIER",
    "PERSON",
    "TEAM",
    "INCIDENT",
    "POLICY",
    "LOCATION",
    "PROCESS",
    "METRIC",
)


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=2)
    type: str = Field(default="CONCEPT")
    description: str = ""

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        upper = (value or "").upper().replace(" ", "_")
        return upper if upper in ALLOWED_ENTITY_TYPES else "CONCEPT"


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation_type: str = "RELATES_TO"
    evidence: str = ""


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=12)
    relations: list[ExtractedRelation] = Field(default_factory=list, max_length=8)


class DocSummary(BaseModel):
    summary: str


class GradeItem(BaseModel):
    index: int
    is_relevant: bool
    reason: str = ""


class BatchGrades(BaseModel):
    items: list[GradeItem] = Field(default_factory=list)


class RewriteOutput(BaseModel):
    query: str = Field(min_length=3)


class AnswerOut(BaseModel):
    answer: str = Field(min_length=1)


class CypherQuery(BaseModel):
    cypher: str = Field(min_length=10)


class MultiHopQA(BaseModel):
    question: str = Field(min_length=10)
    reference_answer: str
