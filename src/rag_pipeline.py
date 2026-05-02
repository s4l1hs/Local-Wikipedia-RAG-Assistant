"""End-to-end RAG orchestration — the single entry point for the UI layer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import settings
from src.llm import IDK_RESPONSE, LLMResponse, generate
from src.retriever import RetrievalResult, RetrievedChunk, retrieve
from src.utils import truncate_to_tokens

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_PREFIX = "Based on limited available information: "


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]
    sim_max: float
    intent: str
    used_fallback: bool
    is_idk: bool
    is_low_confidence: bool


def ask(query: str, show_sources: bool = True) -> RAGResponse:
    """
    Full RAG pipeline for a single user question.

    Decision logic:
      sim_max < LOW_THRESHOLD  → IDK (no LLM call)
      LOW ≤ sim_max < MID      → LLM call with low-confidence prefix
      sim_max ≥ MID            → full answer
    """
    # ── 1. Retrieval ──────────────────────────────────────────────────────────
    result: RetrievalResult = retrieve(query)

    # ── 2. Threshold check — Layer 1 IDK (FR-11) ──────────────────────────────
    if result.sim_max < settings.similarity_threshold_low:
        logger.info("IDK (retrieval): sim_max=%.3f below %.2f",
                    result.sim_max, settings.similarity_threshold_low)
        return RAGResponse(
            answer=IDK_RESPONSE,
            sources=[],
            sim_max=result.sim_max,
            intent=result.intent.value,
            used_fallback=result.used_fallback,
            is_idk=True,
            is_low_confidence=False,
        )

    # ── 3. Context assembly (FR-9) ────────────────────────────────────────────
    context = _assemble_context(result.chunks)

    # ── 4. LLM generation (FR-10) ─────────────────────────────────────────────
    llm_response: LLMResponse = generate(context, query)
    answer = llm_response.answer

    # ── 5. Detect Layer 2 IDK (LLM declined) ─────────────────────────────────
    is_idk = IDK_RESPONSE.lower() in answer.lower()

    # ── 6. Low-confidence prefix ──────────────────────────────────────────────
    is_low_confidence = (
        not is_idk
        and result.sim_max < settings.similarity_threshold_mid
    )
    if is_low_confidence:
        answer = _LOW_CONFIDENCE_PREFIX + answer

    return RAGResponse(
        answer=answer,
        sources=result.chunks if show_sources else [],
        sim_max=result.sim_max,
        intent=result.intent.value,
        used_fallback=result.used_fallback,
        is_idk=is_idk,
        is_low_confidence=is_low_confidence,
    )


def _assemble_context(chunks: list[RetrievedChunk]) -> str:
    """
    Concatenate retrieved chunks into a context string.
    Deduplicates by (entity_id, chunk_index).
    Truncates to MAX_CONTEXT_TOKENS.
    """
    seen: set[tuple[str, int]] = set()
    parts: list[str] = []

    for chunk in sorted(chunks, key=lambda c: c.similarity, reverse=True):
        # TODO: dedup key = (chunk.entity_id, chunk.chunk_index from chunk_id)
        # parts.append(f"[Source: {chunk.entity_name} — {chunk.source_url}]\n{chunk.text}")

        raise NotImplementedError

    full_context = "\n\n".join(parts)
    return truncate_to_tokens(full_context, settings.max_context_tokens)
