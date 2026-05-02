"""
Router test suite — run with: pytest tests/test_router.py -v

Coverage targets: ≥85% of src/router.py
Test cases: 20 queries covering every decision branch.
"""

from __future__ import annotations

import pytest

from src.router import QueryRouter, RoutingDecision, get_router


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def router() -> QueryRouter:
    return QueryRouter()


# ── Helper ────────────────────────────────────────────────────────────────────

def _cat(router: QueryRouter, query: str) -> str:
    return router.route(query).category


# ── Full entity name matches → single type ────────────────────────────────────

def test_full_name_person_einstein(router: QueryRouter) -> None:
    assert _cat(router, "Who is Albert Einstein?") == "person"


def test_full_name_person_curie(router: QueryRouter) -> None:
    assert _cat(router, "Tell me about Marie Curie") == "person"


def test_full_name_person_tesla(router: QueryRouter) -> None:
    assert _cat(router, "What did Nikola Tesla invent?") == "person"


def test_full_name_place_eiffel(router: QueryRouter) -> None:
    assert _cat(router, "What is the Eiffel Tower?") == "place"


def test_full_name_place_colosseum(router: QueryRouter) -> None:
    assert _cat(router, "Where is the Colosseum located?") == "place"


def test_full_name_place_machu_picchu(router: QueryRouter) -> None:
    assert _cat(router, "Tell me about Machu Picchu") == "place"


# ── Mixed entity matches → both ───────────────────────────────────────────────

def test_both_einstein_and_eiffel(router: QueryRouter) -> None:
    """Person (partial) + place (full) → both."""
    assert _cat(router, "Compare Einstein and the Eiffel Tower") == "both"


def test_both_einstein_and_colosseum(router: QueryRouter) -> None:
    """Person (partial) + place (full) in same query."""
    assert _cat(router, "Einstein visited the Colosseum") == "both"


def test_both_curie_and_taj_mahal(router: QueryRouter) -> None:
    assert _cat(router, "Did Marie Curie ever visit the Taj Mahal?") == "both"


# ── Absorbed-word exclusion: Lincoln Memorial ─────────────────────────────────

def test_lincoln_memorial_is_place(router: QueryRouter) -> None:
    """'lincoln' absorbed by 'lincoln memorial' full match → place only."""
    assert _cat(router, "Tell me about the Lincoln Memorial") == "place"


def test_abraham_lincoln_is_person(router: QueryRouter) -> None:
    assert _cat(router, "Who was Abraham Lincoln?") == "person"


def test_bare_lincoln_is_both(router: QueryRouter) -> None:
    """'lincoln' alone partially matches both Abraham Lincoln (person) and
    Lincoln Memorial (place) — correct fallback is 'both'."""
    result = router.route("Tell me about Lincoln")
    assert result.category == "both"


# ── Keyword-only routing ──────────────────────────────────────────────────────

def test_keyword_person_who_discovered(router: QueryRouter) -> None:
    assert _cat(router, "Who discovered radioactivity?") == "person"


def test_keyword_person_scientist_nobel(router: QueryRouter) -> None:
    assert _cat(router, "What scientist won the Nobel Prize?") == "person"


def test_keyword_place_tallest_monument(router: QueryRouter) -> None:
    assert _cat(router, "Where is the tallest monument?") == "place"


def test_keyword_place_ancient_ruins(router: QueryRouter) -> None:
    assert _cat(router, "What ancient ruins can I visit?") == "place"


# ── Entity match overrides keyword ────────────────────────────────────────────

def test_entity_overrides_where_keyword(router: QueryRouter) -> None:
    """'where' is a place keyword, but 'Einstein' entity match wins → person."""
    assert _cat(router, "Where was Einstein born?") == "person"


# ── No-signal / tied ──────────────────────────────────────────────────────────

def test_empty_query_is_unknown(router: QueryRouter) -> None:
    assert _cat(router, "") == "unknown"


def test_nonsense_query_is_unknown(router: QueryRouter) -> None:
    assert _cat(router, "the quick brown fox") == "unknown"


# ── RoutingDecision structure ─────────────────────────────────────────────────

def test_routing_decision_fields(router: QueryRouter) -> None:
    d = router.route("Who is Marie Curie?")
    assert isinstance(d, RoutingDecision)
    assert d.category == "person"
    assert d.person_score > d.place_score
    assert "marie_curie" in d.matched_entities
    assert d.reasoning != ""


def test_person_score_higher_for_person_query(router: QueryRouter) -> None:
    d = router.route("Albert Einstein developed the theory of relativity")
    assert d.person_score > 0
    assert d.category == "person"


def test_place_score_higher_for_place_query(router: QueryRouter) -> None:
    d = router.route("The Eiffel Tower stands in Paris")
    assert d.place_score > 0
    assert d.category == "place"


# ── intent_to_filter helper ───────────────────────────────────────────────────

def test_intent_to_filter_person(router: QueryRouter) -> None:
    d = router.route("Who is Albert Einstein?")
    f = router.intent_to_filter(d)
    assert f == {"entity_type": "person"}


def test_intent_to_filter_place(router: QueryRouter) -> None:
    d = router.route("Tell me about the Eiffel Tower")
    f = router.intent_to_filter(d)
    assert f == {"entity_type": "place"}


def test_intent_to_filter_both_returns_none(router: QueryRouter) -> None:
    d = router.route("Compare Einstein and the Eiffel Tower")
    f = router.intent_to_filter(d)
    assert f is None


def test_intent_to_filter_unknown_returns_none(router: QueryRouter) -> None:
    d = router.route("")
    f = router.intent_to_filter(d)
    assert f is None


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_get_router_returns_same_instance() -> None:
    r1 = get_router()
    r2 = get_router()
    assert r1 is r2


# ── extract_entity_ids ────────────────────────────────────────────────────────

def test_extract_entity_ids_single(router: QueryRouter) -> None:
    ids = router.extract_entity_ids("Tell me about Marie Curie")
    assert "marie_curie" in ids


def test_extract_entity_ids_multiple(router: QueryRouter) -> None:
    ids = router.extract_entity_ids("Compare Einstein and the Colosseum")
    # einstein partial match + colosseum full match
    assert any("einstein" in i for i in ids)
    assert "colosseum" in ids
