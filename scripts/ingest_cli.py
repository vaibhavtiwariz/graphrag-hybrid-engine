from __future__ import annotations

import sys
from pathlib import Path

import click

from app.config import get_settings
from app.ingestion.pipeline import ingest_directory, ingest_file, report_to_dict


@click.command()
@click.argument("path", type=click.Path(exists=True))
def main(path: str) -> None:
    target = Path(path).expanduser().resolve()
    reports = (
        ingest_directory(target) if target.is_dir() else [ingest_file(target)]
    )
    total_chunks = 0
    for report in reports:
        data = report_to_dict(report)
        total_chunks += data["chunks"]
        print(
            f"ingested {data['title']}: chunks={data['chunks']} entities={data['entities']} "
            f"relations={data['relations']} failures={data['extraction_failures']}"
        )
    print(f"done: {len(reports)} documents, {total_chunks} chunks (provider={get_settings().llm_provider})")
    sys.exit(0)


if __name__ == "__main__":
    main()
