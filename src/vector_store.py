"""ChromaDB abstraction — Option B: single collection + metadata filtering."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import PROJECT_ROOT, ENTITY_CATALOG, settings
from src.chunker import Chunk

logger = logging.getLogger(__name__)


# ── ChromaDB client ───────────────────────────────────────────────────────────

def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(settings.db_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection() -> chromadb.Collection:
    """Return (creating if needed) the single wikipedia_rag collection."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Chunk upsert ──────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """
    Insert or update chunks in ChromaDB.
    `embeddings` must be in the same order as `chunks`.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have equal length")

    collection = get_collection()

    ids = [f"{c.entity_id}_chunk_{c.chunk_index:04d}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "entity_id":       c.entity_id,
            "entity_name":     c.entity_name,
            "entity_type":     c.entity_type,   # "person" | "place" — routing key
            "chunk_index":     c.chunk_index,
            "source_url":      c.source_url,
            "token_count":     c.token_count,
        }
        for c in chunks
    ]

    # TODO: ChromaDB upsert call
    #   collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    raise NotImplementedError


def collection_count() -> int:
    return get_collection().count()


# ── SQLite metadata store (chunk audit log) ───────────────────────────────────

def _sqlite_conn() -> sqlite3.Connection:
    db_path = settings.sqlite_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite() -> None:
    """Create tables if they don't exist."""
    with _sqlite_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id        TEXT PRIMARY KEY,
                entity_name      TEXT NOT NULL,
                entity_type      TEXT NOT NULL,
                wikipedia_title  TEXT NOT NULL,
                wikipedia_url    TEXT NOT NULL,
                ingested_at      TEXT NOT NULL,
                article_word_count INTEGER,
                embedding_model  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id       TEXT PRIMARY KEY,
                entity_id      TEXT NOT NULL REFERENCES entities(entity_id),
                entity_type    TEXT NOT NULL,
                chunk_index    INTEGER NOT NULL,
                raw_text       TEXT NOT NULL,
                token_count    INTEGER,
                source_url     TEXT,
                UNIQUE(entity_id, chunk_index)
            );
        """)


def upsert_entity_metadata(entity: dict, word_count: int, embedding_model: str) -> None:
    """Record a successfully ingested entity in SQLite."""
    url = f"https://en.wikipedia.org/wiki/{entity['wikipedia_title'].replace(' ', '_')}"
    with _sqlite_conn() as conn:
        conn.execute(
            """INSERT INTO entities
               (entity_id, entity_name, entity_type, wikipedia_title,
                wikipedia_url, ingested_at, article_word_count, embedding_model)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(entity_id) DO UPDATE SET
                 ingested_at=excluded.ingested_at,
                 article_word_count=excluded.article_word_count,
                 embedding_model=excluded.embedding_model
            """,
            (
                entity["entity_id"], entity["entity_name"], entity["entity_type"],
                entity["wikipedia_title"], url,
                datetime.now(timezone.utc).isoformat(),
                word_count, embedding_model,
            ),
        )


def upsert_chunks_sqlite(chunks: list[Chunk]) -> None:
    """Persist chunk audit records to SQLite (independent of ChromaDB)."""
    # TODO: bulk INSERT OR REPLACE into chunks table
    raise NotImplementedError


def get_embedding_model_for_entity(entity_id: str) -> str | None:
    """Return the embedding model used when entity was indexed, or None."""
    with _sqlite_conn() as conn:
        row = conn.execute(
            "SELECT embedding_model FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
    return row["embedding_model"] if row else None


def check_embedding_model_consistency() -> None:
    """
    Raise ValueError if any indexed entity used a different embedding model
    than the one currently configured. Prevents silent cross-model query corruption.
    """
    # TODO:
    #   stored_models = set of distinct embedding_model values in entities table
    #   if stored_models and settings.embedding_model not in stored_models:
    #       raise ValueError(f"Embedding model mismatch: ...")
    raise NotImplementedError
