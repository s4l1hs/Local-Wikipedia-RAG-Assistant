"""
Wikipedia article chunker — splits processed articles into embeddable units.

Strategy: section_aware (default)
──────────────────────────────────
Wikipedia's editorial section boundaries are used as primary split points.
Each section ≤ SUB_SPLIT_THRESHOLD tokens stays whole (semantic integrity).
Larger sections are sub-divided with a sentence-aware sliding window.
The section heading is prepended to every chunk so the embedding captures
both the topic label and the content — critical for precise retrieval.

Strategy: fixed_token
─────────────────────
Sentence-aware sliding window over the full article text.  Section heading
markers (== … ==) are carried along verbatim in the text and extracted for
metadata heuristically.  Useful as an ablation baseline.

Sizing rationale
────────────────
• nomic-embed-text (primary):  8 192-token context; optimal dense retrieval
  at 300-500 tokens per passage (MTEB embedding benchmarks).
• all-MiniLM-L6-v2 (fallback): 256-token hard limit; chunk_size=400 causes
  truncation, but nomic is primary so we optimise for it.
• LLM context budget: top_k=5 × 400 tokens = 2 000-token context window
  requirement, comfortable inside a 4 096-token LLM context.
• Wikipedia section distribution: most sections run 200-700 tokens, so the
  1 500-token sub-split threshold keeps ≥ 90% of sections as single chunks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.config import settings
from src.utils import count_tokens

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_CHUNK_TOKENS    = 100    # drop chunks shorter than this (stub sections)
SUB_SPLIT_THRESHOLD = 1_500  # tokens; sections above this are sub-divided

# Sentence boundary: terminal punct after TWO lowercase/digit chars, then uppercase.
# Three-char lookbehind means Dr., Mr., St., U.S. won't trigger (capital first char).
_SENT_RE    = re.compile(r'(?<=[a-z0-9][a-z0-9][.!?])\s+(?=[A-Z"])')
_HEADING_RE = re.compile(r"==\s*(.+?)\s*==")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single embeddable text unit with full retrieval metadata."""
    content:  str            # text stored in the vector DB (heading-prefixed)
    metadata: dict[str, Any]

    @property
    def chunk_id(self) -> str:
        return self.metadata["chunk_id"]

    @property
    def token_count(self) -> int:
        return self.metadata["token_count"]


# ── Sentence splitter ─────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """Split text at sentence-terminal boundaries; discard empty fragments."""
    if not text:
        return []
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── Sliding window (low-level) ────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Sentence-aware sliding window over `text`.

    Each chunk is ≤ chunk_size tokens unless a single sentence exceeds the
    limit (that sentence becomes its own chunk).  Consecutive chunks share
    `overlap` tokens of context at sentence granularity.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    results: list[str] = []
    start = 0

    while start < len(sentences):
        end    = start
        tokens = 0
        while end < len(sentences):
            t = count_tokens(sentences[end])
            if tokens + t > chunk_size and end > start:
                break
            tokens += t
            end    += 1

        results.append(" ".join(sentences[start:end]))

        if end >= len(sentences):
            break

        # Walk backwards from `end` to find the overlap start
        back = end
        acc  = 0
        while back > start + 1:
            t = count_tokens(sentences[back - 1])
            if acc + t > overlap:
                break
            acc  += t
            back -= 1

        start = max(back, start + 1)  # always advance to prevent infinite loop

    return [r for r in results if r.strip()]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _base_meta(doc: dict) -> dict:
    return {
        "entity_id":   doc["entity_id"],
        "entity_name": doc["name"],
        "entity_type": doc["type"],
        "category":    doc.get("category", ""),
        "source_url":  doc.get("url", ""),
        "wiki_title":  doc.get("wiki_title", ""),
        "created_at":  datetime.now(timezone.utc).isoformat(),
    }


def _make_chunk(
    content: str,
    idx: int,
    heading: str,
    doc: dict,
    base: dict,
) -> Chunk | None:
    """Build a Chunk; return None if the content is below MIN_CHUNK_TOKENS."""
    t = count_tokens(content)
    if t < MIN_CHUNK_TOKENS:
        return None
    etype = doc["type"]
    eid   = doc["entity_id"]
    return Chunk(
        content=content,
        metadata={
            **base,
            "chunk_id":        f"{etype}::{eid}::{idx:03d}",
            "chunk_index":     idx,
            "section_heading": heading,
            "is_intro":        heading == "",
            "token_count":     t,
            "word_count":      len(content.split()),
        },
    )


# ── Strategy: section_aware ───────────────────────────────────────────────────

def _chunk_section_aware(doc: dict, chunk_size: int, overlap: int) -> list[Chunk]:
    base   = _base_meta(doc)
    chunks: list[Chunk] = []
    idx    = 0

    for section in doc.get("sections", []):
        heading = section.get("heading", "")
        text    = section.get("text", "").strip()
        if not text:
            continue

        # Heading prefix aligns the embedding toward the section's topic
        prefix         = f"{heading}. " if heading else ""
        section_tokens = count_tokens(text)

        if section_tokens <= SUB_SPLIT_THRESHOLD:
            c = _make_chunk(prefix + text, idx, heading, doc, base)
            if c:
                chunks.append(c)
                idx += 1
        else:
            for sub in chunk_text(text, chunk_size, overlap):
                c = _make_chunk(prefix + sub, idx, heading, doc, base)
                if c:
                    chunks.append(c)
                    idx += 1

    return chunks


# ── Strategy: fixed_token ─────────────────────────────────────────────────────

def _chunk_fixed_token(doc: dict, chunk_size: int, overlap: int) -> list[Chunk]:
    raw_text = doc.get("raw_text", "")
    if not raw_text:
        return []

    base   = _base_meta(doc)
    chunks: list[Chunk] = []

    for i, text in enumerate(chunk_text(raw_text, chunk_size, overlap)):
        m       = _HEADING_RE.search(text)
        heading = m.group(1) if m else ""
        c       = _make_chunk(text, i, heading, doc, base)
        if c:
            chunks.append(c)

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(
    doc: dict,
    strategy: str = "section_aware",
    chunk_size: int | None = None,
    overlap:    int | None = None,
) -> list[Chunk]:
    """
    Chunk a processed article document into a list of Chunk objects.

    Args:
        doc:        Processed article JSON dict (data/processed/*.json).
        strategy:   "section_aware" (default) or "fixed_token".
        chunk_size: Max tokens per chunk; defaults to settings.chunk_size (400).
        overlap:    Overlap tokens between chunks; defaults to settings.chunk_overlap (80).
    """
    cs = chunk_size if chunk_size is not None else settings.chunk_size
    ov = overlap    if overlap    is not None else settings.chunk_overlap

    if strategy == "section_aware":
        return _chunk_section_aware(doc, cs, ov)
    if strategy == "fixed_token":
        return _chunk_fixed_token(doc, cs, ov)
    raise ValueError(
        f"Unknown strategy {strategy!r}. Choose 'section_aware' or 'fixed_token'."
    )


def chunk_entity(text: str, entity: dict) -> list[Chunk]:
    """Backward-compatible wrapper: chunk flat text using entity catalog metadata."""
    doc = {
        "raw_text":   text,
        "sections":   [],
        "entity_id":  entity["entity_id"],
        "name":       entity["entity_name"],
        "type":       entity["entity_type"],
        "category":   entity.get("category", ""),
        "url":        f"https://en.wikipedia.org/wiki/{entity['wikipedia_title'].replace(' ', '_')}",
        "wiki_title": entity.get("wikipedia_title", ""),
    }
    return chunk_document(doc, strategy="fixed_token")
