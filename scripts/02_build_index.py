#!/usr/bin/env python3
"""
Step 2 — Chunk, embed, and index all fetched articles into ChromaDB.

Prerequisite: run 01_fetch_wikipedia.py first.

Usage:
    python scripts/02_build_index.py
    python scripts/02_build_index.py --entity albert_einstein
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.progress import track

from src.config import ENTITY_CATALOG, settings
from src.chunker import chunk_entity
from src.embedder import get_embedder
from src.ingest import load_processed_text
from src.vector_store import (
    init_sqlite,
    upsert_chunks,
    upsert_chunks_sqlite,
    upsert_entity_metadata,
    check_embedding_model_consistency,
    collection_count,
)
from src.utils import configure_logging

console = Console()


def index_entity(entity: dict, embedder) -> int:
    """Chunk + embed + upsert one entity. Returns number of chunks created."""
    text = load_processed_text(entity["entity_id"])
    chunks = chunk_entity(text, entity)

    if not chunks:
        console.print(f"[yellow]⚠[/yellow]  No chunks for {entity['entity_id']}")
        return 0

    embeddings = embedder.embed_batch([c.text for c in chunks])
    upsert_chunks(chunks, embeddings)
    upsert_chunks_sqlite(chunks)
    upsert_entity_metadata(entity, word_count=len(text.split()), embedding_model=embedder.model_name)

    return len(chunks)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Build ChromaDB index")
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    init_sqlite()

    # Guard against embedding model mismatch on re-index
    try:
        check_embedding_model_consistency()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    embedder = get_embedder()
    console.print(f"[bold]Embedding model:[/bold] {embedder.model_name}")

    catalog = ENTITY_CATALOG if not args.entity else [
        e for e in ENTITY_CATALOG if e["entity_id"] == args.entity
    ]

    total_chunks = 0
    for entity in track(catalog, description="Indexing..."):
        n = index_entity(entity, embedder)
        total_chunks += n
        console.print(f"  [green]✓[/green] {entity['entity_name']:30s} ({n} chunks)")

    console.print(f"\n[bold green]Done.[/bold green] Total chunks in store: {collection_count()}")


if __name__ == "__main__":
    main()
