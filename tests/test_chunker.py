"""Chunking quality tests — run with: pytest tests/test_chunker.py -v"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.chunker import (
    MIN_CHUNK_TOKENS,
    SUB_SPLIT_THRESHOLD,
    chunk_document,
    chunk_text,
    split_into_sentences,
)

# ── Required metadata keys every Chunk must carry ─────────────────────────────

REQUIRED_KEYS = frozenset({
    "chunk_id", "entity_id", "entity_name", "entity_type", "category",
    "section_heading", "is_intro", "chunk_index", "token_count", "word_count",
    "source_url", "wiki_title", "created_at",
})


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def use_word_count(monkeypatch):
    """Replace tiktoken with word count so tests run without the package."""
    monkeypatch.setattr("src.chunker.count_tokens", lambda t: len(t.split()))


@pytest.fixture
def low_min(monkeypatch):
    """Lower MIN_CHUNK_TOKENS to 5 so small test fixtures produce chunks."""
    monkeypatch.setattr("src.chunker.MIN_CHUNK_TOKENS", 5)


def _doc(sections, eid="test_entity", etype="person"):
    full = " ".join(s["text"] for s in sections)
    return {
        "entity_id": eid,
        "name":      "Test Entity",
        "type":      etype,
        "category":  "test",
        "url":       "https://en.wikipedia.org/wiki/Test",
        "wiki_title": "Test",
        "raw_text":  full,
        "sections":  sections,
        "word_count": len(full.split()),
    }


def _text(n_sentences: int = 10, words_per: int = 11) -> str:
    """Return n_sentences distinct sentences of ~words_per words each."""
    return " ".join(
        f"Sentence {i} covers a distinct topic related to science and history."
        for i in range(n_sentences)
    )


# ── split_into_sentences ───────────────────────────────────────────────────────

class TestSplitIntoSentences:
    def test_basic(self):
        s = split_into_sentences("Hello world. This is a test. And another sentence.")
        assert len(s) == 3

    def test_single_sentence(self):
        s = split_into_sentences("One sentence only")
        assert len(s) == 1
        assert s[0] == "One sentence only"

    def test_empty_string(self):
        assert split_into_sentences("") == []

    def test_whitespace_only(self):
        assert split_into_sentences("   ") == []

    def test_abbreviation_guard(self):
        # Dr. and Mr. must not trigger splits; only "Jones. They" should
        s = split_into_sentences("Mr. Smith met Dr. Jones. They discussed U.S. policy.")
        assert len(s) == 2, f"Expected 2 sentences, got {len(s)}: {s}"

    def test_numeric_sentence_ending(self):
        s = split_into_sentences("He published in 1905. The paper changed physics.")
        assert len(s) == 2

    def test_no_empty_fragments(self):
        s = split_into_sentences("One. Two. Three.")
        assert all(f.strip() for f in s)


# ── chunk_text (sliding window) ────────────────────────────────────────────────

class TestChunkText:
    def test_empty_returns_empty(self):
        assert chunk_text("", 400, 80) == []

    def test_short_text_is_single_chunk(self):
        assert len(chunk_text("Short text here.", chunk_size=100, overlap=20)) == 1

    def test_produces_multiple_chunks_for_long_text(self):
        text = _text(20)
        chunks = chunk_text(text, chunk_size=30, overlap=8)
        assert len(chunks) >= 3

    def test_no_empty_chunks(self):
        chunks = chunk_text(_text(20), chunk_size=30, overlap=8)
        assert all(c.strip() for c in chunks)

    def test_size_respected(self):
        # Each sentence ≈ 12 words; chunk_size=30 → ~2 sentences per chunk
        # Allow one-sentence overshoot for the edge case where a single sentence
        # exceeds chunk_size (it still forms its own chunk rather than being dropped)
        chunk_size = 30
        chunks = chunk_text(_text(30), chunk_size=chunk_size, overlap=8)
        for c in chunks:
            assert len(c.split()) <= chunk_size + 15, (
                f"Chunk too large: {len(c.split())} words (limit {chunk_size})"
            )

    def test_overlap_sentence_carried_to_next(self):
        # Short sentences (~6 words); chunk_size=20 fits ~3; overlap=8 carries ~1
        sentences = [f"This is sentence {i} about physics." for i in range(12)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=20, overlap=8)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            last_of_i = split_into_sentences(chunks[i])[-1]
            assert last_of_i in chunks[i + 1], (
                f"Overlap missing between chunk {i} and {i+1}: "
                f"{last_of_i!r} not in {chunks[i+1]!r}"
            )

    def test_single_oversized_sentence_does_not_loop(self):
        # A sentence longer than chunk_size must still produce exactly one chunk
        big = ("word " * 200).strip() + "."
        chunks = chunk_text(big, chunk_size=50, overlap=10)
        assert len(chunks) == 1


# ── section_aware strategy ─────────────────────────────────────────────────────

class TestSectionAware:
    CS, OV = 50, 10  # chunk_size, overlap for all section tests

    def _chunks(self, sections, eid="test_entity", etype="person"):
        return chunk_document(
            _doc(sections, eid=eid, etype=etype),
            strategy="section_aware",
            chunk_size=self.CS,
            overlap=self.OV,
        )

    def test_min_chunk_size_enforced(self):
        # A 1-word stub section must be dropped (1 < MIN_CHUNK_TOKENS=100 when using word count)
        sections = [
            {"heading": "Stub",   "text": "Tiny."},          # 1 word → dropped
            {"heading": "Normal", "text": _text(12)},         # 132 words → kept
        ]
        chunks = self._chunks(sections)
        for c in chunks:
            assert c.token_count >= MIN_CHUNK_TOKENS, (
                f"{c.chunk_id} is below minimum: {c.token_count} tokens"
            )

    def test_empty_sections_produce_no_chunks(self, low_min):
        sections = [
            {"heading": "Empty",  "text": "   "},
            {"heading": "Normal", "text": _text(3)},
        ]
        chunks = self._chunks(sections)
        assert all(c.content.strip() for c in chunks)

    def test_heading_preserved_in_metadata(self, low_min):
        sections = [
            {"heading": "",          "text": _text(2)},
            {"heading": "Early life","text": _text(2)},
            {"heading": "Career",    "text": _text(2)},
        ]
        headings = {c.metadata["section_heading"] for c in self._chunks(sections)}
        assert "" in headings,           "Introduction (empty heading) missing"
        assert "Early life" in headings
        assert "Career" in headings

    def test_heading_prefix_in_content(self, low_min):
        sections = [{"heading": "Scientific work", "text": _text(2)}]
        chunks = self._chunks(sections)
        for c in chunks:
            assert c.content.startswith("Scientific work. "), (
                f"Heading prefix missing: {c.content[:60]!r}"
            )

    def test_intro_flag_correct(self, low_min):
        sections = [
            {"heading": "",       "text": _text(2)},
            {"heading": "Career", "text": _text(2)},
        ]
        chunks = self._chunks(sections)
        assert any(c.metadata["is_intro"] for c in chunks),  "No intro chunk"
        assert any(not c.metadata["is_intro"] for c in chunks), "All chunks marked intro"

    def test_large_section_sub_split(self, monkeypatch, low_min):
        # Patch threshold to 50 so a 100-word section triggers sub-splitting
        monkeypatch.setattr("src.chunker.SUB_SPLIT_THRESHOLD", 50)
        sections = [{"heading": "Long section", "text": _text(10)}]  # ~110 words
        chunks = self._chunks(sections)
        assert len(chunks) > 1, "Section above threshold must produce multiple sub-chunks"
        for c in chunks:
            assert c.token_count <= self.CS + 20, (
                f"Sub-chunk too large: {c.token_count}"
            )

    def test_chunk_indices_sequential(self, low_min):
        sections = [
            {"heading": "A", "text": _text(2)},
            {"heading": "B", "text": _text(2)},
            {"heading": "C", "text": _text(2)},
        ]
        chunks = self._chunks(sections)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_chunk_ids_unique(self, low_min):
        sections = [
            {"heading": "A", "text": _text(2)},
            {"heading": "B", "text": _text(2)},
        ]
        ids = [c.chunk_id for c in self._chunks(sections)]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    def test_chunk_id_format(self, low_min):
        sections = [{"heading": "Test", "text": _text(2)}]
        doc = _doc(sections, eid="nikola_tesla", etype="person")
        chunks = chunk_document(doc, strategy="section_aware",
                                chunk_size=self.CS, overlap=self.OV)
        for c in chunks:
            parts = c.chunk_id.split("::")
            assert len(parts) == 3,        f"Bad ID structure: {c.chunk_id}"
            assert parts[0] == "person",   f"Wrong type: {parts[0]}"
            assert parts[1] == "nikola_tesla"
            assert parts[2].isdigit() and len(parts[2]) == 3, \
                f"Index not zero-padded: {parts[2]}"

    def test_cross_section_boundary_never_crossed(self, low_min):
        # Every chunk must belong to exactly ONE section heading
        sections = [
            {"heading": "Alpha", "text": _text(2)},
            {"heading": "Beta",  "text": _text(2)},
        ]
        chunks = self._chunks(sections)
        for c in chunks:
            h = c.metadata["section_heading"]
            assert h in {"Alpha", "Beta"}, f"Unexpected heading: {h!r}"
            # Content must start with the section's heading prefix
            assert c.content.startswith(f"{h}. "), (
                f"Content not prefixed with its heading: {c.content[:40]!r}"
            )


# ── Metadata integrity ─────────────────────────────────────────────────────────

class TestMetadataIntegrity:
    def _chunks(self, strategy, etype="person"):
        sections = [
            {"heading": "",       "text": _text(12)},
            {"heading": "Career", "text": _text(12)},
        ]
        return chunk_document(
            _doc(sections, etype=etype),
            strategy=strategy, chunk_size=50, overlap=10,
        )

    def test_all_required_keys_section_aware(self):
        for c in self._chunks("section_aware"):
            missing = REQUIRED_KEYS - c.metadata.keys()
            assert not missing, f"Missing keys in {c.chunk_id}: {missing}"

    def test_all_required_keys_fixed_token(self):
        for c in self._chunks("fixed_token"):
            missing = REQUIRED_KEYS - c.metadata.keys()
            assert not missing, f"Missing keys in {c.chunk_id}: {missing}"

    def test_token_count_positive(self):
        assert all(c.token_count > 0 for c in self._chunks("section_aware"))

    def test_word_count_positive(self):
        assert all(c.metadata["word_count"] > 0 for c in self._chunks("section_aware"))

    def test_entity_type_person(self):
        chunks = self._chunks("section_aware", etype="person")
        assert all(c.metadata["entity_type"] == "person" for c in chunks)

    def test_entity_type_place(self):
        chunks = self._chunks("section_aware", etype="place")
        assert all(c.metadata["entity_type"] == "place" for c in chunks)

    def test_source_url_nonempty(self):
        for c in self._chunks("section_aware"):
            assert c.metadata["source_url"], f"source_url empty in {c.chunk_id}"

    def test_created_at_is_timezone_aware(self):
        for c in self._chunks("section_aware"):
            dt = datetime.fromisoformat(c.metadata["created_at"])
            assert dt.tzinfo is not None, f"created_at not timezone-aware: {c.chunk_id}"


# ── Max size: sub-split chunks must respect chunk_size ────────────────────────

class TestChunkSizeBounds:
    def test_fixed_token_chunks_within_size(self):
        chunk_size = 40
        doc = _doc([{"heading": "", "text": _text(40)}])
        chunks = chunk_document(doc, strategy="fixed_token",
                                chunk_size=chunk_size, overlap=8)
        for c in chunks:
            assert c.token_count <= chunk_size + 15, (
                f"fixed_token chunk too large: {c.token_count} > {chunk_size}"
            )

    def test_section_aware_subsplit_within_size(self, monkeypatch):
        monkeypatch.setattr("src.chunker.SUB_SPLIT_THRESHOLD", 50)
        monkeypatch.setattr("src.chunker.MIN_CHUNK_TOKENS", 5)
        chunk_size = 30
        doc = _doc([{"heading": "Long", "text": _text(15)}])
        chunks = chunk_document(doc, strategy="section_aware",
                                chunk_size=chunk_size, overlap=8)
        for c in chunks:
            assert c.token_count <= chunk_size + 15, (
                f"sub-split chunk too large: {c.token_count} > {chunk_size}"
            )


# ── Keyword sanity: entity content survives chunking intact ───────────────────

class TestKeywordSanity:
    """Verify that key terms in synthetic entity text survive through chunking."""

    def _chunk_entity(self, eid, etype, heading, body, cs=200, ov=30):
        doc = {
            "entity_id": eid,
            "name":      eid.replace("_", " ").title(),
            "type":      etype,
            "category":  "test",
            "url":       f"https://en.wikipedia.org/wiki/{eid}",
            "wiki_title": eid,
            "raw_text":  body,
            "sections":  [{"heading": heading, "text": body}],
            "word_count": len(body.split()),
        }
        return chunk_document(doc, strategy="section_aware", chunk_size=cs, overlap=ov)

    def test_relativity_in_einstein_chunks(self):
        body = (
            "Albert Einstein developed the theory of relativity. "
            "Special relativity was published in 1905. "
            "General relativity followed in 1915. "
            "These theories changed our understanding of space and time. "
        ) * 6  # ~144 words — above MIN_CHUNK_TOKENS
        chunks = self._chunk_entity("albert_einstein", "person", "Scientific work", body)
        all_text = " ".join(c.content for c in chunks)
        assert "relativity" in all_text.lower(), "'relativity' missing from Einstein chunks"

    def test_eiffel_in_tower_chunks(self):
        body = (
            "The Eiffel Tower is a wrought-iron lattice tower. "
            "It stands on the Champ de Mars in Paris. "
            "Gustave Eiffel designed the structure in 1887. "
            "The Eiffel Tower was the tallest structure in the world. "
        ) * 6  # ~120 words — above MIN_CHUNK_TOKENS
        chunks = self._chunk_entity("eiffel_tower", "place", "History", body)
        all_text = " ".join(c.content for c in chunks)
        assert "Eiffel" in all_text, "'Eiffel' missing from Eiffel Tower chunks"

    def test_content_not_garbled(self):
        """Chunks must have substantive text beyond just the heading prefix."""
        body = _text(15)  # 165 words > MIN_CHUNK_TOKENS=100
        chunks = self._chunk_entity("test_person", "person", "Overview", body)
        for c in chunks:
            body_only = c.content.replace("Overview. ", "").strip()
            assert len(body_only.split()) >= 5, (
                f"Chunk appears garbled (too little body text): {c.content[:80]!r}"
            )
