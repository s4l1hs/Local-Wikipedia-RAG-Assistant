"""
Retriever unit tests — run with: pytest tests/test_retriever.py -v

All tests use mocked Embedder + ChromaVectorStore so no populated DB is
required.  Embedding is mocked with random unit vectors; vector-store queries
return hand-crafted QueryResult fixtures.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.chunker import Chunk
from src.retriever import Retriever, RetrievalResult
from src.router import QueryRouter, RoutingDecision
from src.vector_store import QueryResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _unit_vec(dim: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _qr(
    chunk_id:   str,
    entity_id:  str,
    entity_name: str,
    entity_type: str,
    score:      float,
    content:    str = "Sample content.",
) -> QueryResult:
    meta = {
        "chunk_id":        chunk_id,
        "entity_id":       entity_id,
        "entity_name":     entity_name,
        "entity_type":     entity_type,
        "category":        "test",
        "section_heading": "Introduction",
        "is_intro":        True,
        "chunk_index":     0,
        "token_count":     10,
        "word_count":      10,
        "source_url":      f"https://en.wikipedia.org/wiki/{entity_id}",
        "wiki_title":      entity_id,
        "created_at":      "2026-01-01T00:00:00+00:00",
    }
    return QueryResult(
        chunk_id=chunk_id,
        content=content,
        metadata=meta,
        score=score,
        distance=1.0 - score,
    )


# Pre-built QueryResult pool
_EINSTEIN_R  = _qr("person::albert_einstein::000", "albert_einstein",  "Albert Einstein",  "person", 0.88)
_CURIE_R     = _qr("person::marie_curie::000",     "marie_curie",      "Marie Curie",      "person", 0.75)
_TESLA_R     = _qr("person::nikola_tesla::000",    "nikola_tesla",     "Nikola Tesla",     "person", 0.82)
_EIFFEL_R    = _qr("place::eiffel_tower::000",     "eiffel_tower",     "Eiffel Tower",     "place",  0.87)
_COLOSSEUM_R = _qr("place::colosseum::000",        "colosseum",        "Colosseum",        "place",  0.79)
_LOW_R       = _qr("person::albert_einstein::001", "albert_einstein",  "Albert Einstein",  "person", 0.22)


def _make_retriever(
    store_query_return: list[QueryResult],
    threshold: float = 0.30,
) -> Retriever:
    """Build a Retriever with mocked embedder and store."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = _unit_vec(seed=42)

    mock_store = MagicMock()
    mock_store.query.return_value = store_query_return
    mock_store.count.return_value = len(store_query_return)

    return Retriever(
        router=QueryRouter(),
        embedder=mock_embedder,
        store=mock_store,
        top_k=5,
        similarity_threshold=threshold,
    )


# ── Tests: routing plumbing ───────────────────────────────────────────────────

def test_einstein_query_routes_person() -> None:
    retriever = _make_retriever([_EINSTEIN_R])
    result = retriever.retrieve("Who was Albert Einstein?")
    assert result.routing.category == "person"


def test_eiffel_query_routes_place() -> None:
    retriever = _make_retriever([_EIFFEL_R])
    result = retriever.retrieve("Where is the Eiffel Tower located?")
    assert result.routing.category == "place"


def test_comparison_routes_both() -> None:
    retriever = _make_retriever([_EINSTEIN_R, _EIFFEL_R])
    result = retriever.retrieve("Compare Einstein and the Eiffel Tower")
    assert result.routing.category == "both"


# ── Tests: chunks returned correctly ─────────────────────────────────────────

def test_einstein_query_returns_einstein_chunks() -> None:
    retriever = _make_retriever([_EINSTEIN_R, _CURIE_R])
    result = retriever.retrieve("Who was Albert Einstein?")
    entity_ids = [c.metadata["entity_id"] for c in result.chunks]
    assert "albert_einstein" in entity_ids
    assert result.scores[0] == pytest.approx(0.88)


def test_eiffel_query_returns_place_chunks() -> None:
    retriever = _make_retriever([_EIFFEL_R, _COLOSSEUM_R])
    result = retriever.retrieve("Where is the Eiffel Tower?")
    for c in result.chunks:
        assert c.metadata["entity_type"] == "place"


def test_tesla_query_returns_tesla_first() -> None:
    retriever = _make_retriever([_TESLA_R, _CURIE_R, _EINSTEIN_R])
    result = retriever.retrieve("What did Nikola Tesla invent?")
    assert result.chunks[0].metadata["entity_id"] == "nikola_tesla"
    assert result.scores[0] == pytest.approx(0.82)


# ── Tests: compare Tesla and Edison ──────────────────────────────────────────

def test_tesla_and_edison_returns_tesla() -> None:
    """Edison is not in corpus; only Tesla chunks are returned."""
    retriever = _make_retriever([_TESLA_R])
    result = retriever.retrieve("Compare Tesla and Edison")
    entity_ids = {c.metadata["entity_id"] for c in result.chunks}
    assert "nikola_tesla" in entity_ids
    assert not result.is_empty


# ── Tests: similarity threshold ───────────────────────────────────────────────

def test_low_score_results_dropped() -> None:
    """Chunks below threshold are filtered out."""
    retriever = _make_retriever([_EINSTEIN_R, _LOW_R], threshold=0.30)
    result = retriever.retrieve("Albert Einstein")
    assert len(result.chunks) == 1
    assert result.below_threshold == 1


def test_all_low_scores_returns_empty() -> None:
    """All chunks below threshold → is_empty = True."""
    retriever = _make_retriever([_LOW_R], threshold=0.30)
    result = retriever.retrieve("Quantum gravity in 2026")
    assert result.is_empty
    assert result.below_threshold == 1
    assert len(result.chunks) == 0


def test_empty_result_max_score_is_zero() -> None:
    retriever = _make_retriever([], threshold=0.30)
    result = retriever.retrieve("Quantum gravity in 2026")
    assert result.max_score == 0.0
    assert result.is_empty


# ── Tests: low confidence flag ────────────────────────────────────────────────

def test_low_confidence_when_best_score_below_mid_threshold() -> None:
    mid_score_r = _qr("person::albert_einstein::000", "albert_einstein",
                       "Albert Einstein", "person", 0.40)
    retriever = _make_retriever([mid_score_r], threshold=0.30)
    result = retriever.retrieve("Albert Einstein")
    assert not result.is_empty
    assert result.low_confidence  # 0.40 < 0.45 (threshold_mid)


def test_not_low_confidence_for_high_score() -> None:
    retriever = _make_retriever([_EINSTEIN_R], threshold=0.30)
    result = retriever.retrieve("Albert Einstein")
    assert not result.low_confidence  # 0.88 >= 0.45


# ── Tests: caching ────────────────────────────────────────────────────────────

def test_cache_returns_same_object() -> None:
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = _unit_vec(seed=1)
    mock_store = MagicMock()
    mock_store.query.return_value = [_EINSTEIN_R]

    retriever = Retriever(
        router=QueryRouter(),
        embedder=mock_embedder,
        store=mock_store,
        enable_cache=True,
    )
    r1 = retriever.retrieve("Who was Albert Einstein?")
    r2 = retriever.retrieve("Who was Albert Einstein?")
    assert r1 is r2
    assert mock_store.query.call_count == 1  # second call served from cache


def test_cache_clear_triggers_new_query() -> None:
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = _unit_vec(seed=1)
    mock_store = MagicMock()
    mock_store.query.return_value = [_EINSTEIN_R]

    retriever = Retriever(
        router=QueryRouter(),
        embedder=mock_embedder,
        store=mock_store,
        enable_cache=True,
    )
    retriever.retrieve("Who was Albert Einstein?")
    retriever.clear_cache()
    retriever.retrieve("Who was Albert Einstein?")
    assert mock_store.query.call_count == 2


# ── Tests: entity coverage augmentation ──────────────────────────────────────

def test_entity_coverage_fires_for_missing_entity() -> None:
    """
    When a named entity is absent from primary results, a secondary query
    should fetch it and used_coverage should be True.
    """
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = _unit_vec(seed=5)

    mock_store = MagicMock()
    # Primary query returns only Einstein; Eiffel Tower is absent
    mock_store.query.side_effect = [
        [_EINSTEIN_R],   # primary query
        [_EIFFEL_R],     # coverage fetch for eiffel_tower
    ]

    retriever = Retriever(
        router=QueryRouter(),
        embedder=mock_embedder,
        store=mock_store,
        top_k=5,
        similarity_threshold=0.30,
    )
    result = retriever.retrieve("Compare Einstein and the Eiffel Tower")
    assert result.used_coverage
    entity_ids = {c.metadata["entity_id"] for c in result.chunks}
    assert "albert_einstein" in entity_ids
    assert "eiffel_tower" in entity_ids


# ── Tests: RetrievalResult helpers ───────────────────────────────────────────

def test_summary_contains_query() -> None:
    retriever = _make_retriever([_EINSTEIN_R])
    result = retriever.retrieve("Who was Albert Einstein?")
    s = result.summary()
    assert "Who was Albert Einstein?" in s
    assert "albert_einstein" in s.lower() or "Albert Einstein" in s


def test_result_is_dataclass_with_expected_fields() -> None:
    retriever = _make_retriever([_EIFFEL_R])
    result = retriever.retrieve("Eiffel Tower height")
    assert isinstance(result, RetrievalResult)
    assert isinstance(result.chunks, list)
    assert isinstance(result.scores, list)
    assert isinstance(result.routing, RoutingDecision)
    assert isinstance(result.query, str)
    assert len(result.chunks) == len(result.scores)
