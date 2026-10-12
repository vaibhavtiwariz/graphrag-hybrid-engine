from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.schema import (
    ALLOWED_ENTITY_TYPES,
    AnswerOut,
    BatchGrades,
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelation,
    GradeItem,
    MultiHopQA,
)


def test_entity_type_normalized_to_allowlist():
    entity = ExtractedEntity(name="Kestrel Metals", type="supplier")
    assert entity.type == "SUPPLIER"
    fallback = ExtractedEntity(name="mystery", type="WIDGET")
    assert fallback.type == "CONCEPT"


def test_extraction_defaults_are_empty_lists():
    extraction = ChunkExtraction()
    assert extraction.entities == []
    assert extraction.relations == []


def test_extraction_caps_lengths():
    entities = [ExtractedEntity(name=f"entity {i}") for i in range(15)]
    with pytest.raises(ValidationError):
        ChunkExtraction(entities=entities)
    ok = ChunkExtraction(entities=entities[:12])
    assert len(ok.entities) == 12


def test_relation_defaults():
    relation = ExtractedRelation(source="AtlasArm", target="AeroGrip")
    assert relation.relation_type == "RELATES_TO"
    assert relation.evidence == ""


def test_grade_item_requires_index_and_flag():
    item = GradeItem(index=3, is_relevant=True, reason="mentions supplier")
    assert item.is_relevant is True
    batch = BatchGrades(items=[item])
    assert batch.items[0].index == 3


def test_answer_and_qa_models_require_content():
    with pytest.raises(ValidationError):
        AnswerOut(answer="")
    qa = MultiHopQA(question="Which supplier caused INC-1042?", reference_answer="Kestrel Metals")
    assert qa.question.startswith("Which")
    assert len(ALLOWED_ENTITY_TYPES) >= 10
