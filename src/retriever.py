"""Top-k semantic retrieval with similarity threshold and misclassification fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import settings
from src.embedder import get_embedder
from src.router import Intent, classify, intent_to_chroma_filter, extract_mentioned_entity_ids
from src.vector_store import get_collection

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    entity_id: str
    entity_name: str
    entity_type: str
    text: str
    source_url: str
    similarity: float   # cosine similarity in [0, 1]; higher = more relevant


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    sim_max: float
    intent: Intent
    used_fallback: bool   # True if type filter was dropped due to low sim_max


def retrieve(query: str, k: int | None = None) -> RetrievalResult:
    """
    Full retrieval pipeline for a single query.

    Steps:
      1. Classify intent.
      2. Embed query.
      3. Query ChromaDB with metadata filter.
      4. If sim_max < LOW_THRESHOLD and a filter was applied → retry without filter.
      5. Enforce entity coverage for comparison queries.
    """
    k = k or settings.top_k
    embedder = get_embedder()
    collection = get_collection()

    intent = classify(query)
    where_clause = intent_to_chroma_filter(intent)

    query_embedding = embedder.embed(query)

    # ── Primary query ─────────────────────────────────────────────────────────
    raw = _chroma_query(collection, query_embedding, k, where_clause)
    sim_max = _compute_sim_max(raw)
    used_fallback = False

    # ── Misclassification fallback (FR-8) ─────────────────────────────────────
    if sim_max < settings.similarity_threshold_low and where_clause is not None:
        logger.debug("sim_max=%.3f below threshold, retrying without filter", sim_max)
        raw = _chroma_query(collection, query_embedding, k, where_clause=None)
        sim_max = _compute_sim_max(raw)
        used_fallback = True

    chunks = _parse_chroma_results(raw)

    # ── Entity coverage enforcement for comparison queries ────────────────────
    chunks = _enforce_entity_coverage(
        query, chunks, query_embedding, collection, sim_max
    )

    return RetrievalResult(
        chunks=chunks,
        sim_max=sim_max,
        intent=intent,
        used_fallback=used_fallback,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _chroma_query(
    collection,
    query_embedding: list[float],
    k: int,
    where_clause: dict | None,
) -> dict:
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        kwargs["where"] = where_clause
    return collection.query(**kwargs)


def _compute_sim_max(raw: dict) -> float:
    """ChromaDB cosine distance → similarity = 1 - distance."""
    distances = raw.get("distances", [[]])[0]
    if not distances:
        return 0.0
    return 1.0 - min(distances)


def _parse_chroma_results(raw: dict) -> list[RetrievedChunk]:
    # TODO: zip ids, documents, metadatas, distances → list[RetrievedChunk]
    # Remember: similarity = 1 - distance
    raise NotImplementedError


def _enforce_entity_coverage(
    query: str,
    chunks: list[RetrievedChunk],
    query_embedding: list[float],
    collection,
    current_sim_max: float,
) -> list[RetrievedChunk]:
    """
    For comparison queries, ensure at least one chunk per named entity.

    If a named entity is absent from `chunks`, fetch its top-1 chunk
    via entity_id filter and append it.
    """
    # TODO:
    #   named_ids = extract_mentioned_entity_ids(query)
    #   covered = {c.entity_id for c in chunks}
    #   for eid in named_ids:
    #       if eid not in covered:
    #           gap = _chroma_query(collection, query_embedding, k=1,
    #                               where_clause={"entity_id": {"$eq": eid}})
    #           chunks.extend(_parse_chroma_results(gap))
    return chunks
