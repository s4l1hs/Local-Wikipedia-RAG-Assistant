"""
Query intent router — classifies a user query into person / place / both / unknown.

Algorithm (in priority order):
  1. Full-phrase entity match  (weight 10 per match)
     Absorbed words from full matches are excluded from step 2.
  2. Partial token match against entity names  (weight 3 per unique entity)
     Tokens shorter than MIN_TOKEN_LEN or in GENERIC_TOKENS are skipped.
  3. Keyword scoring  (+1 per keyword hit; de-duped)
  4. Decision rules:
       • Both entity types matched by steps 1–2  → "both"
       • Only person entities matched             → "person"
       • Only place entities matched              → "place"
       • person_kw > place_kw + THRESHOLD         → "person"
       • place_kw > person_kw + THRESHOLD         → "place"
       • Any keyword present (tied or mixed)      → "both"
       • No signal at all                         → "unknown"

The router is fully rule-based — no LLM call, no embedding.
Intended to run in < 1 ms per query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.config import ENTITY_CATALOG

# ── Tuning constants ──────────────────────────────────────────────────────────

_THRESHOLD     = 1    # keyword-only: margin required for decisive routing
_MIN_TOKEN_LEN = 5    # shorter tokens are too ambiguous for partial matching
_WEIGHT_FULL   = 10   # score weight for a full entity name match
_WEIGHT_PARTIAL = 3   # score weight per unique entity matched by partial token

# Generic tokens that appear in many entity names and should not trigger partial
# matches on their own (e.g. "falls" matches Victoria Falls AND Yellowstone).
_GENERIC_TOKENS: frozenset[str] = frozenset({
    "tower", "great", "falls", "saint", "south", "north", "east", "west",
    "lower", "upper", "new", "old", "lake", "mount", "san", "the", "of",
    "national", "park", "wall", "canyon", "island", "valley",
})

# ── Keyword sets ──────────────────────────────────────────────────────────────

_PERSON_KW: frozenset[str] = frozenset({
    "who", "whom", "scientist", "scientists", "inventor", "inventors",
    "physicist", "chemist", "mathematician", "author", "writer", "poet",
    "artist", "painter", "musician", "singer", "actor", "actress",
    "athlete", "footballer", "politician", "president", "emperor",
    "king", "queen", "ruler", "leader", "activist", "philosopher",
    "polymath", "nurse", "doctor", "person", "people", "man", "woman",
    "born", "died", "biography", "discovered", "invented", "wrote",
    "painted", "composed", "nobel", "awarded", "married",
    "father", "mother", "husband", "wife", "family",
})

_PLACE_KW: frozenset[str] = frozenset({
    "where", "located", "location", "monument", "building", "structure",
    "country", "city", "nation", "capital", "museum", "ruins",
    "archaeological", "ancient", "historic", "heritage", "tourist",
    "visit", "constructed", "erected", "built", "height", "tall", "meters",
    "metres", "natural", "landmark", "mountain", "peak", "summit",
    "river", "ocean", "sea", "forest", "rainforest", "jungle", "canyon",
    "valley", "palace", "temple", "cathedral", "park", "garden",
    "tallest", "highest", "island", "region",
})


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    category:         Literal["person", "place", "both", "unknown"]
    person_score:     float
    place_score:      float
    matched_entities: list[str] = field(default_factory=list)  # entity_ids
    reasoning:        str       = ""


# ── Router ────────────────────────────────────────────────────────────────────

class QueryRouter:
    """
    Rule-based query classifier.  No LLM, no embedding — pure string matching.
    Safe to instantiate once and reuse across threads (read-only after __init__).
    """

    def __init__(self) -> None:
        # Index: entity_id → {type, name_lower, name_tokens}
        self._entities: list[dict] = []
        # Reverse index: token → list of entity_ids whose name contains it
        self._token_index: dict[str, list[str]] = {}

        for e in ENTITY_CATALOG:
            eid   = e["entity_id"]
            etype = e["entity_type"]
            name  = e["entity_name"].lower()
            # tokenize name: only alpha tokens, no short/generic ones
            tokens = [
                t for t in re.findall(r"[a-z]+", name)
                if len(t) >= _MIN_TOKEN_LEN and t not in _GENERIC_TOKENS
            ]
            self._entities.append({
                "id":     eid,
                "type":   etype,
                "name":   name,
                "tokens": tokens,
            })
            for tok in tokens:
                self._token_index.setdefault(tok, []).append(eid)

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, query: str) -> RoutingDecision:
        """Classify *query* and return a RoutingDecision."""
        q = query.lower()

        # ── Step 1: Full-phrase entity matching ───────────────────────────────
        absorbed_words: set[str] = set()
        full_person_ids: set[str] = set()
        full_place_ids:  set[str] = set()

        for ent in self._entities:
            if ent["name"] in q:
                # Absorb every word from the matched phrase
                absorbed_words.update(re.findall(r"[a-z]+", ent["name"]))
                if ent["type"] == "person":
                    full_person_ids.add(ent["id"])
                else:
                    full_place_ids.add(ent["id"])

        person_score = _WEIGHT_FULL * len(full_person_ids)
        place_score  = _WEIGHT_FULL * len(full_place_ids)
        matched_ids  = list(full_person_ids | full_place_ids)

        # ── Step 2: Partial token matching (skip absorbed tokens) ─────────────
        q_tokens = {
            t for t in re.findall(r"[a-z]+", q)
            if (
                len(t) >= _MIN_TOKEN_LEN
                and t not in _GENERIC_TOKENS
                and t not in absorbed_words   # absorbed by a full match
            )
        }

        partial_person_ids: set[str] = set()
        partial_place_ids:  set[str] = set()

        for tok in q_tokens:
            for eid in self._token_index.get(tok, []):
                ent = next(e for e in self._entities if e["id"] == eid)
                if eid not in full_person_ids and eid not in full_place_ids:
                    if ent["type"] == "person":
                        partial_person_ids.add(eid)
                    else:
                        partial_place_ids.add(eid)

        person_score += _WEIGHT_PARTIAL * len(partial_person_ids)
        place_score  += _WEIGHT_PARTIAL * len(partial_place_ids)
        matched_ids  += list(partial_person_ids | partial_place_ids)

        # ── Step 3: Keyword scoring ───────────────────────────────────────────
        q_word_set = set(re.findall(r"[a-z]+", q))
        person_kw  = sum(1 for kw in _PERSON_KW if kw in q_word_set)
        place_kw   = sum(1 for kw in _PLACE_KW  if kw in q_word_set)

        # ── Step 4: Decision ──────────────────────────────────────────────────
        has_person_entity = bool(full_person_ids or partial_person_ids)
        has_place_entity  = bool(full_place_ids  or partial_place_ids)

        if has_person_entity and has_place_entity:
            category  = "both"
            reasoning = "matched both person and place entities"

        elif has_person_entity:
            category  = "person"
            reasoning = f"entity match: {sorted(full_person_ids | partial_person_ids)}"

        elif has_place_entity:
            category  = "place"
            reasoning = f"entity match: {sorted(full_place_ids | partial_place_ids)}"

        elif person_kw > 0 and place_kw == 0:
            # Unambiguous: person signal with no competing place signal
            category  = "person"
            reasoning = f"person keywords, no place keywords ({person_kw} vs 0)"

        elif place_kw > 0 and person_kw == 0:
            # Unambiguous: place signal with no competing person signal
            category  = "place"
            reasoning = f"place keywords, no person keywords ({place_kw} vs 0)"

        elif person_kw > place_kw + _THRESHOLD:
            category  = "person"
            reasoning = f"person keywords dominate ({person_kw} vs {place_kw})"

        elif place_kw > person_kw + _THRESHOLD:
            category  = "place"
            reasoning = f"place keywords dominate ({place_kw} vs {person_kw})"

        elif person_kw > 0 or place_kw > 0:
            category  = "both"
            reasoning = f"keywords tied or mixed (person={person_kw}, place={place_kw})"

        else:
            category  = "unknown"
            reasoning = "no signal"

        return RoutingDecision(
            category=category,
            person_score=float(person_score + person_kw),
            place_score=float(place_score + place_kw),
            matched_entities=matched_ids,
            reasoning=reasoning,
        )

    def intent_to_filter(self, decision: RoutingDecision) -> dict | None:
        """Return a ChromaDB `where` clause, or None for both/unknown."""
        if decision.category == "person":
            return {"entity_type": "person"}
        if decision.category == "place":
            return {"entity_type": "place"}
        return None

    def extract_entity_ids(self, query: str) -> list[str]:
        """Return entity_ids explicitly named in *query*."""
        return self.route(query).matched_entities


# ── Backward-compatible thin wrappers ─────────────────────────────────────────
# The old API used Intent enum + top-level classify() / intent_to_chroma_filter().
# Keep them so existing callers don't break while the pipeline is being built.

from typing import Literal as _Literal


class Intent(str):
    PERSON  = "person"
    PLACE   = "place"
    MIXED   = "both"
    UNKNOWN = "unknown"


def classify(query: str) -> str:
    """Legacy shim — returns Intent string value."""
    return get_router().route(query).category


def intent_to_chroma_filter(intent: str) -> dict | None:
    """Legacy shim — maps category string to ChromaDB where clause."""
    if intent == "person":
        return {"entity_type": "person"}
    if intent == "place":
        return {"entity_type": "place"}
    return None


def extract_mentioned_entity_ids(query: str) -> list[str]:
    """Legacy shim."""
    return get_router().extract_entity_ids(query)


# ── Singleton ─────────────────────────────────────────────────────────────────

_router: QueryRouter | None = None


def get_router() -> QueryRouter:
    """Return the shared QueryRouter instance (built once per process)."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
