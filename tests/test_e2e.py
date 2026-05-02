"""
End-to-end evaluation — routing, retrieval, and (optionally) LLM answer quality.

Two test layers:
  1. Retrieval layer  (no LLM needed)  — routing category, entity Hit@5, OOC rejection
  2. Answer layer     (@pytest.mark.llm) — expected keywords in LLM answer, IDK for OOC

Corpus assumption: at minimum albert_einstein, marie_curie, nikola_tesla, eiffel_tower indexed.

Run:
    pytest tests/test_e2e.py                    # retrieval tests only
    pytest tests/test_e2e.py -m llm             # also run LLM answer tests
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline


# ── Test case schema ──────────────────────────────────────────────────────────

@dataclass
class E2ECase:
    query:            str
    routing:          str                  # expected routing category
    entity:           str | None           # entity_id expected in top-5 (None = OOC)
    keywords:         list[str] = field(default_factory=list)  # expected in LLM answer
    expect_idk:       bool = False         # True → OOC; retrieval must be empty / LLM says IDK
    min_score:        float = 0.55         # retrieved max_score must exceed this
    label:            str = ""


# ── Evaluation set (26 queries) ───────────────────────────────────────────────

CASES: list[E2ECase] = [
    # ── Albert Einstein (5) ───────────────────────────────────────────────────
    E2ECase("Who discovered the theory of relativity?",
            "person", "albert_einstein", ["einstein", "relativity"], label="einstein-1"),
    E2ECase("Which physicist won the Nobel Prize for the photoelectric effect?",
            "person", "albert_einstein", ["photoelectric", "einstein"], label="einstein-2"),
    E2ECase("What did Albert Einstein study at university?",
            "person", "albert_einstein", ["einstein"], label="einstein-3"),
    E2ECase("Who published the special theory of relativity in 1905?",
            "person", "albert_einstein", ["einstein", "1905"], label="einstein-4"),
    E2ECase("What is Albert Einstein most famous for?",
            "person", "albert_einstein", ["relativity", "einstein"], label="einstein-5"),

    # ── Marie Curie (5) ───────────────────────────────────────────────────────
    E2ECase("Who discovered polonium and radium?",
            "person", "marie_curie", ["polonium", "radium"], label="curie-1"),
    E2ECase("Which scientist won two Nobel Prizes in different fields?",
            "person", "marie_curie", ["curie", "nobel"], label="curie-2"),
    E2ECase("Tell me about Marie Curie's research on radioactivity.",
            "person", "marie_curie", ["curie", "radioactivity"], label="curie-3"),
    E2ECase("Which female scientist was the first to win the Nobel Prize?",
            "person", "marie_curie", ["curie"], label="curie-4"),
    E2ECase("What country did Marie Curie emigrate to from Poland?",
            "person", "marie_curie", ["france", "paris"], label="curie-5"),

    # ── Nikola Tesla (4) ──────────────────────────────────────────────────────
    E2ECase("Who invented the alternating current electrical system?",
            "person", "nikola_tesla", ["alternating", "tesla"], label="tesla-1"),
    E2ECase("What patents did Nikola Tesla hold?",
            "person", "nikola_tesla", ["tesla"], label="tesla-2"),
    E2ECase("Which inventor worked on wireless power transmission?",
            "person", "nikola_tesla", ["tesla", "wireless"], label="tesla-3"),
    E2ECase("Where was Nikola Tesla born?",
            "person", "nikola_tesla", ["tesla"], label="tesla-4"),

    # ── Eiffel Tower (5) ──────────────────────────────────────────────────────
    E2ECase("How tall is the Eiffel Tower?",
            "place", "eiffel_tower", ["300", "paris"], label="eiffel-1"),
    E2ECase("When was the iron lattice tower in Paris built?",
            "place", "eiffel_tower", ["1889", "eiffel"], label="eiffel-2"),
    E2ECase("What landmark was built for the 1889 World's Fair?",
            "place", "eiffel_tower", ["eiffel"], label="eiffel-3"),
    E2ECase("Who designed the Eiffel Tower?",
            "place", "eiffel_tower", ["gustave", "eiffel"], label="eiffel-4"),
    E2ECase("In which city is the Eiffel Tower located?",
            "place", "eiffel_tower", ["paris"], label="eiffel-5"),

    # ── Cross-entity (2) ──────────────────────────────────────────────────────
    E2ECase("Compare Einstein and the Eiffel Tower",
            "both", "albert_einstein", label="cross-1"),
    E2ECase("What did Tesla invent near the Eiffel Tower era?",
            "both", "nikola_tesla", label="cross-2"),

    # ── Out-of-corpus (5) — must be rejected ──────────────────────────────────
    E2ECase("What is the best recipe for sourdough bread?",
            "unknown", None, expect_idk=True, label="ooc-1"),
    E2ECase("How do I bake sourdough bread at home?",
            "unknown", None, expect_idk=True, label="ooc-2"),
    E2ECase("How do I train a puppy not to bite?",
            "unknown", None, expect_idk=True, label="ooc-3"),
    E2ECase("What are the rules of poker?",
            "unknown", None, expect_idk=True, label="ooc-4"),
    E2ECase("How do I learn to play the guitar?",
            "unknown", None, expect_idk=True, label="ooc-5"),
]

# Separate lists for parametrize IDs
_RETRIEVAL_CASES = CASES
_LLM_CASES = [c for c in CASES if c.keywords]         # only test when keywords defined
_OOC_CASES  = [c for c in CASES if c.expect_idk]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def _retriever():
    return Retriever()


@pytest.fixture(scope="module")
def _pipeline():
    return RAGPipeline()


# ── Layer 1: Routing tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("case", _RETRIEVAL_CASES, ids=[c.label for c in _RETRIEVAL_CASES])
def test_routing(case: E2ECase, _retriever: Retriever) -> None:
    """Router assigns the expected category."""
    result = _retriever.retrieve(case.query)
    assert result.routing.category == case.routing, (
        f"[{case.label}] Expected routing={case.routing!r} "
        f"but got {result.routing.category!r} — "
        f"person_score={result.routing.person_score}, "
        f"place_score={result.routing.place_score}"
    )


# ── Layer 1: Retrieval hit tests ───────────────────────────────────────────────

@pytest.mark.parametrize("case", [c for c in CASES if not c.expect_idk],
                         ids=[c.label for c in CASES if not c.expect_idk])
def test_retrieval_hit(case: E2ECase, _retriever: Retriever) -> None:
    """Expected entity appears in top-5 retrieved chunks."""
    result = _retriever.retrieve(case.query)
    assert not result.is_empty, (
        f"[{case.label}] Retrieval returned 0 chunks — entity index may be missing"
    )
    retrieved_ids = [c.metadata.get("entity_id") for c in result.chunks]
    assert case.entity in retrieved_ids, (
        f"[{case.label}] Expected entity {case.entity!r} not in top-5. "
        f"Got: {retrieved_ids}"
    )


@pytest.mark.parametrize("case", [c for c in CASES if not c.expect_idk],
                         ids=[c.label for c in CASES if not c.expect_idk])
def test_retrieval_score(case: E2ECase, _retriever: Retriever) -> None:
    """Max similarity score meets the minimum threshold."""
    result = _retriever.retrieve(case.query)
    assert result.max_score >= case.min_score, (
        f"[{case.label}] max_score={result.max_score:.3f} < {case.min_score}"
    )


# ── Layer 1: OOC rejection tests ──────────────────────────────────────────────

@pytest.mark.parametrize("case", _OOC_CASES, ids=[c.label for c in _OOC_CASES])
def test_ooc_rejection(case: E2ECase, _retriever: Retriever) -> None:
    """Out-of-corpus queries produce empty retrieval (IDK guard fires)."""
    result = _retriever.retrieve(case.query)
    assert result.is_empty, (
        f"[{case.label}] OOC query was NOT rejected. "
        f"max_score={result.max_score:.3f}, chunks={len(result.chunks)}. "
        f"Lower similarity_threshold_low in src/config.py if this persists."
    )


# ── Layer 2: LLM answer tests ─────────────────────────────────────────────────

@pytest.mark.llm
@pytest.mark.parametrize("case", [c for c in _LLM_CASES if not c.expect_idk],
                         ids=[c.label for c in _LLM_CASES if not c.expect_idk])
def test_answer_keywords(case: E2ECase, _pipeline: RAGPipeline) -> None:
    """LLM answer contains expected keywords (grounded, non-IDK response)."""
    resp = _pipeline.ask(case.query)
    assert not resp.is_idk, (
        f"[{case.label}] Unexpected IDK response for in-corpus query. "
        f"max_score={resp.max_score:.3f}"
    )
    answer_lower = resp.answer.lower()
    for kw in case.keywords:
        assert kw.lower() in answer_lower, (
            f"[{case.label}] Keyword {kw!r} not found in answer:\n{resp.answer[:300]}"
        )


@pytest.mark.llm
@pytest.mark.parametrize("case", _OOC_CASES, ids=[c.label for c in _OOC_CASES])
def test_ooc_answer_idk(case: E2ECase, _pipeline: RAGPipeline) -> None:
    """Pipeline returns IDK for out-of-corpus queries (never calls LLM if guard fires)."""
    from src.llm import IDK_RESPONSE
    resp = _pipeline.ask(case.query)
    assert resp.is_idk, (
        f"[{case.label}] OOC query produced a non-IDK answer:\n{resp.answer[:200]}"
    )
    assert IDK_RESPONSE.lower() in resp.answer.lower() or len(resp.retrieved_chunks) == 0, (
        f"[{case.label}] IDK flag set but answer doesn't contain IDK phrase:\n{resp.answer}"
    )
