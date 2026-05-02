"""Pytest smoke tests — run with: pytest tests/ -v"""

import pytest

from src.config import ENTITY_CATALOG, PEOPLE_LIST, PLACES_LIST, settings
from src.router import classify, Intent, intent_to_chroma_filter, extract_mentioned_entity_ids
from src.utils import count_tokens, clean_wikipedia_text, truncate_to_tokens
from src.chunker import split_into_sentences


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_entity_catalog_count(self):
        assert len(ENTITY_CATALOG) == 44   # 22 people + 22 places

    def test_people_count(self):
        assert len(PEOPLE_LIST) == 22

    def test_places_count(self):
        assert len(PLACES_LIST) == 22

    def test_mandatory_people_present(self):
        ids = {e["entity_id"] for e in PEOPLE_LIST}
        mandatory = {
            "albert_einstein", "marie_curie", "leonardo_da_vinci",
            "william_shakespeare", "ada_lovelace", "nikola_tesla",
            "lionel_messi", "cristiano_ronaldo", "taylor_swift", "frida_kahlo",
        }
        assert mandatory.issubset(ids)

    def test_mandatory_places_present(self):
        ids = {e["entity_id"] for e in PLACES_LIST}
        mandatory = {
            "eiffel_tower", "great_wall_of_china", "taj_mahal",
            "grand_canyon", "machu_picchu", "colosseum",
            "hagia_sophia", "statue_of_liberty", "pyramids_of_giza", "mount_everest",
        }
        assert mandatory.issubset(ids)

    def test_all_entities_have_required_fields(self):
        required = {"entity_id", "entity_name", "entity_type", "wikipedia_title"}
        for entity in ENTITY_CATALOG:
            assert required.issubset(entity.keys()), f"Missing fields in {entity}"

    def test_entity_types_valid(self):
        valid_types = {"person", "place"}
        for entity in ENTITY_CATALOG:
            assert entity["entity_type"] in valid_types

    def test_chunk_overlap_smaller_than_size(self):
        assert settings.chunk_overlap < settings.chunk_size

    def test_threshold_low_less_than_mid(self):
        assert settings.similarity_threshold_low < settings.similarity_threshold_mid


# ── Router tests ──────────────────────────────────────────────────────────────

class TestRouter:
    def test_einstein_classifies_as_person(self):
        assert classify("Who was Albert Einstein?") == Intent.PERSON

    def test_eiffel_classifies_as_place(self):
        assert classify("Where is the Eiffel Tower?") == Intent.PLACE

    def test_mixed_query(self):
        assert classify("Compare Albert Einstein and the Eiffel Tower") == Intent.MIXED

    def test_unknown_query_keywords_route_to_person(self):
        # "who" + "president" are both person keywords → "person" routing is correct
        assert classify("Who is the president of Mars?") == Intent.PERSON

    def test_person_filter_returned(self):
        f = intent_to_chroma_filter(Intent.PERSON)
        assert f == {"entity_type": "person"}

    def test_place_filter_returned(self):
        f = intent_to_chroma_filter(Intent.PLACE)
        assert f == {"entity_type": "place"}

    def test_mixed_filter_is_none(self):
        assert intent_to_chroma_filter(Intent.MIXED) is None

    def test_extract_mentioned_entities(self):
        ids = extract_mentioned_entity_ids("Compare Lionel Messi and Cristiano Ronaldo")
        assert "lionel_messi" in ids
        assert "cristiano_ronaldo" in ids


# ── Utils tests ───────────────────────────────────────────────────────────────

class TestUtils:
    def test_count_tokens_nonempty(self):
        assert count_tokens("Hello world") > 0

    def test_truncate_preserves_short_text(self):
        text = "Short text."
        assert truncate_to_tokens(text, 100) == text

    def test_truncate_cuts_long_text(self):
        text = " ".join(["word"] * 1000)
        truncated = truncate_to_tokens(text, 50)
        assert count_tokens(truncated) <= 50

    def test_clean_removes_citation_markers(self):
        raw = "Einstein[1] was born in 1879.[citation needed]"
        cleaned = clean_wikipedia_text(raw)
        assert "[1]" not in cleaned
        assert "[citation needed]" not in cleaned


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_sentence_split_basic(self):
        text = "Hello world. This is a test. And another sentence."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_sentence_split_nonempty(self):
        sentences = split_into_sentences("One sentence only")
        assert len(sentences) == 1
        assert sentences[0] == "One sentence only"
