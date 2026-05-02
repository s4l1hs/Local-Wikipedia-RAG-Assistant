"""Query intent classifier — maps a user query to PERSON / PLACE / MIXED."""

from __future__ import annotations

import re
from enum import Enum

from src.config import ENTITY_CATALOG, ENTITY_NAME_LOWER, settings


class Intent(str, Enum):
    PERSON = "PERSON"
    PLACE  = "PLACE"
    MIXED  = "MIXED"   # default: no metadata filter → cast wide net


# ── Keyword sets (lowercase) ──────────────────────────────────────────────────

_PERSON_KEYWORDS: frozenset[str] = frozenset({
    "who", "was born", "discovered", "invented", "wrote",
    "painted", "composed", "won", "died", "studied",
    "developed", "created", "scientist", "artist", "author",
    "athlete", "musician", "politician", "philosopher",
})

_PLACE_KEYWORDS: frozenset[str] = frozenset({
    "where", "located", "city", "country", "monument",
    "built", "height", "visited", "travel", "stands",
    "landmark", "tower", "mountain", "park", "island",
    "temple", "palace", "ruins", "site", "forest",
})

# Map every entity name variant → entity type (built at import time)
_NAME_TO_TYPE: dict[str, str] = {
    name_lower: meta["entity_type"]
    for name_lower, meta in ENTITY_NAME_LOWER.items()
}

# Also index by entity_id for lookup-by-id
_ID_TO_TYPE: dict[str, str] = {
    e["entity_id"]: e["entity_type"] for e in ENTITY_CATALOG
}


def _find_named_entities(query_lower: str) -> list[str]:
    """Return entity types for all catalog members mentioned in the query."""
    return [
        etype
        for name, etype in _NAME_TO_TYPE.items()
        if name in query_lower
    ]


def classify(query: str) -> Intent:
    """
    Classify a user query into PERSON, PLACE, or MIXED.

    Priority order:
      1. Named entity recognition against the catalog.
      2. Keyword heuristics.
      3. Default → MIXED (safe: queries entire corpus).
    """
    q = query.lower()

    # ── 1. Named entity detection ─────────────────────────────────────────────
    found_types = _find_named_entities(q)
    if found_types:
        unique_types = set(found_types)
        if len(unique_types) == 1:
            t = unique_types.pop()
            return Intent.PERSON if t == settings.entity_type_person else Intent.PLACE
        # Multiple entity types mentioned → MIXED
        return Intent.MIXED

    # ── 2. Keyword heuristics ─────────────────────────────────────────────────
    has_person_kw = any(kw in q for kw in _PERSON_KEYWORDS)
    has_place_kw  = any(kw in q for kw in _PLACE_KEYWORDS)

    if has_person_kw and not has_place_kw:
        return Intent.PERSON
    if has_place_kw and not has_person_kw:
        return Intent.PLACE

    # ── 3. Default ────────────────────────────────────────────────────────────
    return Intent.MIXED


def intent_to_chroma_filter(intent: Intent) -> dict | None:
    """Return a ChromaDB `where` clause dict, or None for MIXED (no filter)."""
    if intent == Intent.PERSON:
        return {"entity_type": {"$eq": settings.entity_type_person}}
    if intent == Intent.PLACE:
        return {"entity_type": {"$eq": settings.entity_type_place}}
    return None   # MIXED → no filter


def extract_mentioned_entity_ids(query: str) -> list[str]:
    """Return entity_ids for all catalog members explicitly named in the query."""
    q = query.lower()
    return [
        meta["entity_id"]
        for name, meta in ENTITY_NAME_LOWER.items()
        if name in q
    ]
