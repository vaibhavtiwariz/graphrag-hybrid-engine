from __future__ import annotations

import json
import random
from pathlib import Path

import click

from app.services import get_graph_store


@click.command()
@click.option("--num", default=110, show_default=True, help="Target number of QA pairs")
@click.option(
    "--out",
    default="evaluation/generated_qa.jsonl",
    show_default=True,
)
@click.option("--seed", default=42)
def main(num: int, out: str, seed: int) -> None:
    from app.ingestion.extractor import generate_multihop_qa

    store = get_graph_store()
    counts = store.counts()
    if not counts or not counts.get("chunks"):
        raise SystemExit(
            "Knowledge graph is empty - run ingestion first (make ingest)."
        )

    pairs_query = (
        "MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk) "
        "WHERE c1.id < c2.id AND c1.doc_id <> c2.doc_id "
        "RETURN DISTINCT c1.id AS id_a, c2.id AS id_b, "
        "c1.text AS text_a, c2.text AS text_b, collect(DISTINCT e.name)[..3] AS shared"
    )
    rows = store.run_readonly(pairs_query)
    random.Random(seed).shuffle(rows)

    seen: set[str] = set()
    written = 0
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if written >= num:
                break
            try:
                qa = generate_multihop_qa(row["text_a"], row["text_b"])
            except Exception:
                continue
            question_key = qa.question.strip().lower()
            if question_key in seen:
                continue
            seen.add(question_key)
            record = {
                "id": f"gen-{written:04d}",
                "shared_entities": row.get("shared", []),
                "question": qa.question.strip(),
                "reference": qa.reference_answer.strip(),
                "source_chunks": [row["id_a"], row["id_b"]],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 10 == 0:
                print(f"generated {written}/{num} questions")

    print(f"wrote {written} multi-hop QA pairs to {out_path}")
    print(f"graph stats: {counts}")


if __name__ == "__main__":
    main()
