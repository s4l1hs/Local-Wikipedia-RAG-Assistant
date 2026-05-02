"""
Vector store abstraction and ChromaDB implementation.

Design: Single collection (Option B)
─────────────────────────────────────
All chunks (persons + places) live in one ChromaDB collection with
`entity_type` stored in metadata.  Callers filter by type or run across
the whole index:

  query(..., filter={"entity_type": "person"})  →  person chunks only
  query(..., filter={"entity_type": "place"})   →  place chunks only
  query(..., filter=None)                        →  entity-agnostic

Why single-collection over two separate collections (Option A)?

  1. "Both" / ambiguous queries work by natural similarity ranking.
     L2-normalised cosine scores are directly comparable across all chunks
     regardless of entity type — no artificial merge step needed.
  2. Corpus is balanced: 22 people × ~14 chunks + 22 places × ~14 chunks
     ≈ 616 total.  No type-skew that isolation would correct.
  3. One HNSW index is faster to build and search than two.
  4. Simpler persistence: one DB path, one rebuild command.

"Both" query strategy:
  The intent router returns entity_type=None when the query is type-
  ambiguous.  filter=None surfaces the k most relevant chunks regardless
  of type — preferable to forcing equal representation from each type.

Top-k and reranking hook:
  Default top_k=5 (5 × ~400 tok ≈ 2 000 tok, fits in a 4 096-tok LLM).
  Pass top_k_fetch > top_k to retrieve a wider candidate pool; the caller's
  reranker trims back to top_k.  No reranker is implemented now; the
  parameter reserves the architectural slot.

Persistence and rebuild:
  DB lives at PROJECT_ROOT/db/.  It is a *derived* artifact — source of
  truth is data/processed/_chunks.jsonl + .embed_cache/*.pkl.
  Full rebuild (600 chunks, already cached) takes < 10 seconds.
  On corruption: store.reset() then re-run scripts/02_build_index.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import chromadb
import numpy as np

from src.chunker import Chunk
from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DB_PATH      = PROJECT_ROOT / "db"
_UPSERT_BATCH = 500   # documents per ChromaDB upsert call


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """A single retrieved passage with its relevance score."""

    chunk_id:  str
    content:   str
    metadata:  dict[str, Any]
    score:     float   # cosine similarity (0–1); higher = more relevant
    distance:  float   # raw ChromaDB cosine distance; hook for reranker

    @property
    def entity_id(self) -> str:
        return self.metadata.get("entity_id", "")

    @property
    def entity_name(self) -> str:
        return self.metadata.get("entity_name", "")

    @property
    def entity_type(self) -> str:
        return self.metadata.get("entity_type", "")

    @property
    def section_heading(self) -> str:
        return self.metadata.get("section_heading", "")


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class VectorStore(Protocol):
    def add(
        self,
        chunks:     list[Chunk],
        embeddings: np.ndarray,
    ) -> None: ...

    def query(
        self,
        embedding:   np.ndarray,
        top_k:       int,
        filter:      dict | None = None,   # noqa: A002
        top_k_fetch: int | None  = None,
    ) -> list[QueryResult]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


# ── ChromaDB implementation ───────────────────────────────────────────────────

class ChromaVectorStore:
    """
    Single-collection ChromaDB vector store backed by a PersistentClient.

    Implements the VectorStore Protocol (structural, no inheritance needed).
    """

    def __init__(
        self,
        collection_name: str  = settings.chroma_collection_name,
        db_path:         Path = _DB_PATH,
    ) -> None:
        self._collection_name = collection_name
        self._db_path         = db_path

        # ChromaDB 0.5.23 logs ERROR-level telemetry failures even when
        # anonymized_telemetry=False — suppress to avoid confusing output.
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(
            logging.CRITICAL
        )

        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(db_path),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "ChromaVectorStore ready  collection=%s  count=%d",
            collection_name,
            self._col.count(),
        )

    # ── VectorStore interface ─────────────────────────────────────────────────

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Upsert chunks into the collection.

        Idempotent — re-indexing the same chunk_id overwrites the existing
        entry rather than duplicating it.  Processed in batches of
        _UPSERT_BATCH to stay within ChromaDB's per-call limits.
        """
        if not chunks:
            return
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks/embeddings length mismatch: "
                f"{len(chunks)} vs {embeddings.shape[0]}"
            )
        for start in range(0, len(chunks), _UPSERT_BATCH):
            bc = chunks[start : start + _UPSERT_BATCH]
            be = embeddings[start : start + _UPSERT_BATCH]
            self._col.upsert(
                ids=[c.chunk_id for c in bc],
                embeddings=[e.tolist() for e in be],
                documents=[c.content for c in bc],
                metadatas=[_sanitize_meta(c.metadata) for c in bc],
            )
            logger.debug("Upserted batch %d–%d", start, start + len(bc) - 1)

    def query(
        self,
        embedding:   np.ndarray,
        top_k:       int,
        filter:      dict | None = None,   # noqa: A002
        top_k_fetch: int | None  = None,
    ) -> list[QueryResult]:
        """
        Return the top-k most similar chunks.

        Args:
            embedding:   1-D float32 query vector (must be L2-normalised).
            top_k:       Number of results to return.
            filter:      Optional ChromaDB `where` clause, e.g.
                         {"entity_type": "person"} or None for no filter.
            top_k_fetch: Retrieve this many candidates, then trim to top_k.
                         Architectural hook for a future cross-encoder
                         reranker; defaults to top_k (no-op).
        """
        total = self._col.count()
        if total == 0:
            return []

        n_fetch = min(
            top_k_fetch if top_k_fetch is not None else top_k,
            total,
        )

        raw = self._col.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_fetch,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )

        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        if not ids:
            return []

        results = [
            QueryResult(
                chunk_id=cid,
                content=doc,
                metadata=meta,
                score=float(1.0 - dist),
                distance=float(dist),
            )
            for cid, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]
        return results[:top_k]

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        """Drop and recreate the collection, clearing all data."""
        self._client.delete_collection(self._collection_name)
        self._col = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' reset.", self._collection_name)

    # ── Extras ────────────────────────────────────────────────────────────────

    def clear_collection(self, name: str | None = None) -> None:
        """Alias for reset() — convenient for test teardown."""
        self.reset()

    def stats(self) -> dict:
        """Return total count, per-entity-type breakdown, and storage info."""
        total    = self.count()
        by_type: dict[str, int] = {}
        if total > 0:
            for etype in ("person", "place"):
                result         = self._col.get(where={"entity_type": etype}, include=[])
                by_type[etype] = len(result["ids"])
        return {
            "total":      total,
            "by_type":    by_type,
            "collection": self._collection_name,
            "db_path":    str(self._db_path),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_store: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    """Return the shared ChromaVectorStore instance (opened once per process)."""
    global _store
    if _store is None:
        _store = ChromaVectorStore()
    return _store


# ── Private helpers ───────────────────────────────────────────────────────────

def _sanitize_meta(meta: dict) -> dict:
    """
    Drop None values; stringify any non-scalar that slips through.
    ChromaDB metadata must be str | int | float | bool.
    """
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out
