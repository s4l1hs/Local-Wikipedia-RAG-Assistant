"""
Anti-hallucination test suite — CRITICAL.

Two categories:

1. IDK-guard tests  (no LLM) — retrieval must be empty for OOC queries.
   If the similarity threshold is set too low, OOC queries pass the guard
   and reach the LLM, which MAY hallucinate.  These tests catch threshold
   miscalibration before the LLM is even invoked.

2. Grounding tests  (@pytest.mark.llm) — queries where:
   (a) the entity IS indexed so retrieval passes the threshold, but
   (b) the specific fact asked about is NOT in any passage.
   The strict 6-rule system prompt must cause the LLM to say IDK or
   explicitly note the information is not in the passages.

   FAIL condition: answer contains fabricated specific facts (names, dates,
   claims) not traceable to the retrieved chunks.

Control queries (marked "control") ask about TRUE facts that ARE in the
passages — these must be answered correctly, not IDK'd.

Run:
    pytest tests/test_no_hallucination.py            # guard tests only
    pytest tests/test_no_hallucination.py -m llm     # also grounding tests
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline
from src.llm import IDK_RESPONSE


# ── IDK-guard test cases ──────────────────────────────────────────────────────

OOC_GUARD_QUERIES: list[str] = [
    "What is the best sourdough bread recipe?",
    "How do I learn to play the guitar?",
    "How do I change a car tire step by step?",
    "How do I train a puppy not to bite?",
    "What is the best exercise routine to lose weight?",
    "How do I write a Python web scraper?",
    "What is the best way to train for a marathon?",
    "What are the rules of poker card game?",
]


@pytest.mark.parametrize("query", OOC_GUARD_QUERIES)
def test_idk_guard_fires(query: str) -> None:
    """Similarity threshold blocks OOC queries before the LLM is called."""
    retriever = Retriever()
    result = retriever.retrieve(query)
    assert result.is_empty, (
        f"OOC query NOT rejected by IDK guard.\n"
        f"Query    : {query!r}\n"
        f"max_score: {result.max_score:.3f}  (threshold_low="
        f"{result.chunks[0].metadata if result.chunks else 'n/a'})\n"
        f"Chunks   : {[c.metadata.get('entity_id') for c in result.chunks]}\n"
        f"Action   : raise similarity_threshold_low in src/config.py "
        f"(current: 0.55, try 0.60)."
    )


# ── Grounding test cases (require Ollama) ─────────────────────────────────────

@dataclass
class GroundingCase:
    query:              str
    # At least one of these must appear in the answer (OR answer is IDK):
    forbidden_claims:   list[str]    # fabricated facts that must NOT appear
    is_control:         bool = False  # True = fact IS in corpus; must answer (not IDK)
    must_contain:       list[str] = field(default_factory=list)  # control: keywords required
    label:              str = ""


GROUNDING_CASES: list[GroundingCase] = [
    # ── Tricky: entity indexed BUT fact not in passages ────────────────────────
    GroundingCase(
        query="What did Einstein say about quantum computing in 2024?",
        forbidden_claims=[
            "said in 2024",
            "stated in 2024",
            "einstein said that quantum",
            "quantum computing in 2024",
            "in his 2024 speech",
        ],
        label="grounding-1",
    ),
    GroundingCase(
        query="Did Tesla invent the iPhone?",
        forbidden_claims=[
            "yes, tesla invented the iphone",
            "tesla invented the iphone",
            "iphone was invented by tesla",
            "tesla created the iphone",
        ],
        label="grounding-2",
    ),
    GroundingCase(
        query="Who is the current owner of the Eiffel Tower as of 2026?",
        forbidden_claims=[
            "currently owned by",        # LLM should hedge, not claim current ownership
            "as of 2026 it is owned",
            "2026 owner is",
        ],
        label="grounding-3",
    ),
    GroundingCase(
        query="What medications did Marie Curie use to treat cancer in 2024?",
        forbidden_claims=[
            "in 2024 she treated",
            "used in 2024",
            "curie treated cancer in 2024",
            "modern treatment",
        ],
        label="grounding-4",
    ),
    GroundingCase(
        query="What did Nikola Tesla say about Elon Musk?",
        forbidden_claims=[
            "said about elon musk",
            "commented on elon musk",
            "musk and tesla discussed",
            "his views on musk",
        ],
        label="grounding-5",
    ),

    # ── Control queries: TRUE facts that ARE in the corpus ────────────────────
    GroundingCase(
        query="Was Marie Curie awarded a Nobel Prize?",
        forbidden_claims=[],           # no forbidden — must answer YES
        is_control=True,
        must_contain=["nobel", "curie"],
        label="control-1",
    ),
    GroundingCase(
        query="Did Einstein win the Nobel Prize for the theory of relativity?",
        forbidden_claims=[
            "yes, for relativity",
            "awarded for relativity",   # FALSE — he won for photoelectric effect
        ],
        is_control=True,
        must_contain=["photoelectric"],  # correct premise correction
        label="control-2",              # also tests false-premise correction (Rule 1)
    ),
    GroundingCase(
        query="Is the Eiffel Tower located in Paris?",
        forbidden_claims=[],
        is_control=True,
        must_contain=["paris"],
        label="control-3",
    ),
]


def _is_idk(answer: str) -> bool:
    return IDK_RESPONSE.lower() in answer.lower()


def _contains_any(answer: str, claims: list[str]) -> tuple[bool, str]:
    """Returns (True, matching_claim) if any forbidden claim is in answer."""
    al = answer.lower()
    for claim in claims:
        if claim.lower() in al:
            return True, claim
    return False, ""


@pytest.mark.llm
@pytest.mark.parametrize(
    "case",
    [c for c in GROUNDING_CASES if not c.is_control],
    ids=[c.label for c in GROUNDING_CASES if not c.is_control],
)
def test_no_hallucination(case: GroundingCase, pipeline: RAGPipeline) -> None:
    """
    LLM must NOT fabricate specific facts that are not in the retrieved passages.
    Acceptable outcomes: IDK response OR answer that avoids all forbidden claims.
    """
    resp = pipeline.ask(case.query)
    answer = resp.answer

    # If IDK guard fired or LLM said IDK → perfect grounding, pass immediately
    if resp.is_idk or _is_idk(answer):
        return

    # Check for fabricated specific facts
    found, claim = _contains_any(answer, case.forbidden_claims)
    assert not found, (
        f"[{case.label}] HALLUCINATION DETECTED.\n"
        f"Query            : {case.query!r}\n"
        f"Forbidden claim  : {claim!r}\n"
        f"Answer           :\n{answer}\n\n"
        f"Action: tighten the system prompt in src/prompts.py or raise "
        f"similarity_threshold_low to prevent this entity from being retrieved."
    )


@pytest.mark.llm
@pytest.mark.parametrize(
    "case",
    [c for c in GROUNDING_CASES if c.is_control],
    ids=[c.label for c in GROUNDING_CASES if c.is_control],
)
def test_control_answered(case: GroundingCase, pipeline: RAGPipeline) -> None:
    """
    Control queries (TRUE facts in corpus) must be answered correctly, not IDK'd.
    Also catches false-premise correction failures.
    """
    resp = pipeline.ask(case.query)
    answer = resp.answer

    # Control queries must NOT be IDK'd — entity is indexed, fact is in passages
    assert not resp.is_idk, (
        f"[{case.label}] Control query incorrectly returned IDK.\n"
        f"Query: {case.query!r}\nAnswer: {answer}"
    )

    # Must contain the required keywords
    answer_lower = answer.lower()
    for kw in case.must_contain:
        assert kw.lower() in answer_lower, (
            f"[{case.label}] Expected keyword {kw!r} missing from answer:\n{answer}"
        )

    # Must not contain forbidden claims (e.g., wrong premise confirmed)
    if case.forbidden_claims:
        found, claim = _contains_any(answer, case.forbidden_claims)
        assert not found, (
            f"[{case.label}] Control answer contains a false/forbidden claim: {claim!r}\n"
            f"Answer:\n{answer}"
        )
