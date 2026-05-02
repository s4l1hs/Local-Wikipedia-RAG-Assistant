"""
Retriever — combines router, embedder, and vector store into one pipeline step.

Retrieve pipeline:
  1. Classify query intent (QueryRouter)
  2. Embed query (Embedder.embed_query — applies BGE instruction prefix)
  3. Query ChromaDB with intent-derived filter
       person       → filter={"entity_type": "person"}
       place        → filter={"entity_type": "place"}
       both/unknown → filter=None, top_k_fetch = top_k × 2 (wider candidate pool)
  4. Entity coverage: for queries naming specific entities, ensure each is
     represented in results (secondary per-entity fetch if missing).
  5. Similarity threshold — drop chunks below threshold; empty result → IDK flag.
  6. Return RetrievalResult.

───────────────────────────────────────────────────────────────────────────────
ULTRATHINK: "both" merge strategy
───────────────────────────────────────────────────────────────────────────────
Score-based merge was chosen over round-robin and hybrid:

  Round-robin artificially enforces 50/50 balance regardless of relevance.
  For "Who visited the Colosseum" (place-leaning), round-robin would pull
  low-scoring person chunks instead of better-matching place chunks. Rejected.

  Score-based merge (selected): single unfiltered ChromaDB call, globally
  ranked by cosine similarity. For explicit comparison queries ("Compare
  Einstein and the Eiffel Tower"), both entity types naturally score high —
  the embedding captures both topics simultaneously. For ambiguous queries
  ("Tell me about Lincoln"), the higher-scoring type dominates — correct.
  top_k_fetch = top_k × 2 ensures a wider initial candidate pool without
  requiring two separate queries.

  Hybrid min-1 guarantee is handled separately by _ensure_entity_coverage(),
  which fires only when a named entity is completely absent from results.
  This fixes the edge case without degrading quality in the common case.

───────────────────────────────────────────────────────────────────────────────
ULTRATHINK: similarity threshold calibration
───────────────────────────────────────────────────────────────────────────────
Default: settings.similarity_threshold_low = 0.30.

bge-small-en-v1.5 background similarity floor for unrelated English sentences
is ~0.25–0.35. Below 0.30 is noise. The default 0.30 is intentionally inclusive
to avoid silent IDK on edge cases; calibrate upward in Phase 7:

  Eval set: 40 in-corpus queries (1/entity, verifiable) + 20 out-of-domain
  Sweep T ∈ {0.25, 0.30, 0.35, 0.40, 0.45, 0.50}
  Metric: Youden's J = TPR + TNR − 1
    TPR = in-corpus queries with ≥1 result above T   (want: 1.0)
    TNR = out-of-domain with 0 results above T       (want: 1.0)
  Expected calibrated threshold: 0.35–0.42.

───────────────────────────────────────────────────────────────────────────────
ULTRATHINK: "I don't know" — three-layer defense
───────────────────────────────────────────────────────────────────────────────
Threshold is a COARSE filter only. High cosine similarity ≠ passage answers
the question. A chunk about Einstein's papers scores high for ANY Einstein
query, including ones it doesn't address. Mitigation across three layers:

  Layer 1 (here): threshold drops clearly out-of-corpus queries.
  Layer 2 (Phase 5 prompt): strict grounding instructions —
    "Answer ONLY from the provided passages."
    "Each factual statement must trace to a specific passage."
    "If passages don't directly answer, say so; do not speculate."
    "Correct false premises using passage evidence."
    similarity_threshold_mid (0.45) triggers a low-confidence hedge prefix.
  Layer 3 (config): temperature=0.1 keeps the LLM in summarization mode,
    suppressing creative confabulation.

  Residual risk: a high-scoring superficially-relevant chunk that doesn't
  contain the specific answer. The fundamental fix is a cross-encoder reranker
  (Phase 7 enhancement) that scores passage–question relevance directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.chunker import Chunk
from src.config import settings
from src.embedder import Embedder, get_embedder
from src.router import QueryRouter, RoutingDecision, get_router
from src.vector_store import ChromaVectorStore, QueryResult, get_vector_store

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Output of a single retrieval call."""

    chunks:          list[Chunk]
    scores:          list[float]
    routing:         RoutingDecision
    query:           str
    below_threshold: int  = 0     # chunks dropped by threshold (observability)
    used_coverage:   bool = False  # True if entity coverage augmentation fired

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def max_score(self) -> float:
        return max(self.scores, default=0.0)

    @property
    def low_confidence(self) -> bool:
        """True when best score is below mid-tier threshold."""
        return self.max_score < settings.similarity_threshold_mid

    def summary(self) -> str:
        lines = [
            f"Query   : {self.query!r}",
            f"Routing : {self.routing.category}  [{self.routing.reasoning}]",
            f"Results : {len(self.chunks)} chunk(s)  "
            f"(dropped={self.below_threshold}, coverage_augmented={self.used_coverage})",
        ]
        for i, (c, s) in enumerate(zip(self.chunks, self.scores), 1):
            meta = c.metadata
            entity = meta.get("entity_name", "?")
            section = meta.get("section_heading") or "(intro)"
            preview = c.content[:80].replace("\n", " ")
            lines.append(f"  #{i}  score={s:.3f}  [{entity} / {section}]  {preview}…")
        if self.is_empty:
            lines.append("  (empty — below similarity threshold)")
        return "\n".join(lines)


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Full retrieval pipeline: route → embed → vector search → threshold filter.

    Thread-safe after construction (all state is read-only).
    """

    def __init__(
        self,
        router:               QueryRouter      | None = None,
        embedder:             Embedder         | None = None,
        store:                ChromaVectorStore | None = None,
        top_k:                int   = settings.top_k,
        similarity_threshold: float = settings.similarity_threshold_low,
        enable_cache:         bool  = False,
    ) -> None:
        self._router    = router   or get_router()
        self._embedder  = embedder or get_embedder()
        self._store     = store    or get_vector_store()
        self._top_k     = top_k
        self._threshold = similarity_threshold
        self._cache: dict[str, RetrievalResult] | None = {} if enable_cache else None

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """
        Run the full retrieval pipeline for *query*.

        Args:
            query:  User question string.
            top_k:  Override the instance-level top_k for this call.

        Returns:
            RetrievalResult with .is_empty = True when no chunk clears the
            similarity threshold (caller should skip LLM generation).
        """
        k = top_k if top_k is not None else self._top_k

        # ── Cache ─────────────────────────────────────────────────────────────
        cache_key = f"{query}::{k}"
        if self._cache is not None and cache_key in self._cache:
            logger.debug("Cache hit: %r", query)
            return self._cache[cache_key]

        # ── Step 1: Route ─────────────────────────────────────────────────────
        routing = self._router.route(query)
        logger.debug("Routing: %s  (%s)", routing.category, routing.reasoning)

        # ── Step 2: Embed ─────────────────────────────────────────────────────
        q_vec = self._embedder.embed_query(query)

        # ── Step 3: Query vector store ────────────────────────────────────────
        chroma_filter = self._router.intent_to_filter(routing)
        # For both/unknown: no filter, fetch 2× for better candidate diversity
        top_k_fetch   = k * 2 if chroma_filter is None else None

        raw: list[QueryResult] = self._store.query(
            q_vec,
            top_k=k,
            filter=chroma_filter,
            top_k_fetch=top_k_fetch,
        )

        # ── Step 4: Entity coverage enforcement ───────────────────────────────
        raw, used_coverage = self._ensure_entity_coverage(raw, q_vec, routing, k)

        # ── Step 5: Similarity threshold ──────────────────────────────────────
        passing     = [r for r in raw if r.score >= self._threshold]
        below_count = len(raw) - len(passing)

        if below_count:
            logger.debug(
                "Dropped %d chunk(s) below threshold %.2f; best=%.3f",
                below_count,
                self._threshold,
                max((r.score for r in raw), default=0.0),
            )

        # ── Step 6: Build result ──────────────────────────────────────────────
        chunks = [Chunk(content=r.content, metadata=r.metadata) for r in passing]
        scores = [r.score for r in passing]

        result = RetrievalResult(
            chunks=chunks,
            scores=scores,
            routing=routing,
            query=query,
            below_threshold=below_count,
            used_coverage=used_coverage,
        )

        if self._cache is not None:
            self._cache[cache_key] = result

        return result

    def clear_cache(self) -> None:
        """Invalidate the in-memory result cache."""
        if self._cache is not None:
            self._cache.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_entity_coverage(
        self,
        results:  list[QueryResult],
        q_vec:    np.ndarray,
        routing:  RoutingDecision,
        top_k:    int,
    ) -> tuple[list[QueryResult], bool]:
        """
        Guarantee that each explicitly named entity appears at least once.

        For a query like "Compare Einstein and the Colosseum", if Einstein
        chunks happen to dominate the top-k, Colosseum might be absent.
        This fetch ensures representation without degrading quality:
          • Only fires when a named entity is truly absent from results.
          • Augmented result re-sorted by score and capped at top_k.
          • Returns (results, used_coverage_flag).
        """
        if not routing.matched_entities:
            return results, False

        covered = {r.entity_id for r in results}
        augmented = list(results)
        fired = False

        for eid in routing.matched_entities:
            if eid not in covered:
                gap = self._store.query(
                    q_vec,
                    top_k=1,
                    filter={"entity_id": eid},
                )
                if gap:
                    augmented.extend(gap)
                    covered.add(eid)
                    fired = True
                    logger.debug("Coverage augment: added chunk for entity %r", eid)

        if fired:
            # Re-rank by score, trim to top_k
            augmented.sort(key=lambda r: r.score, reverse=True)
            augmented = augmented[:top_k]

        return augmented, fired


# ── Module-level singleton ────────────────────────────────────────────────────

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Return the shared Retriever instance (built once per process)."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
