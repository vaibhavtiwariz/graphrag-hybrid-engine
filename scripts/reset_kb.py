from __future__ import annotations

import argparse

from app.services import get_graph_store, get_qdrant_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe the knowledge graph and vector index")
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    args = parser.parse_args()

    if not args.yes:
        confirm = input("Delete ALL documents/entities/vectors? type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("aborted")
            return

    graph = get_graph_store()
    with graph._driver.session(database=graph.database) as session:
        session.run("MATCH (n) DETACH DELETE n")

    qdrant = get_qdrant_store()
    qdrant.client.delete_collection(qdrant.collection)
    qdrant.ensure_collection()

    print("graph:", graph.counts())
    print("qdrant points:", qdrant.count())
    print("knowledge base is empty - ingest again with scripts/ingest_cli.py or the UI uploader")


if __name__ == "__main__":
    main()
