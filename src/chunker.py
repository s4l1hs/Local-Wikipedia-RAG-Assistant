"""Sentence-boundary-aware overlapping text chunker."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

from src.config import settings
from src.utils import count_tokens

logger = logging.getLogger(__name__)

# Sentence boundary: split after . ! ? followed by whitespace or end
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    entity_id: str
    entity_name: str
    entity_type: str
    chunk_index: int
    text: str
    token_count: int
    source_url: str


def split_into_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries; keep non-empty sentences only."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    entity_id: str,
    entity_name: str,
    entity_type: str,
    source_url: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chunk_size: int | None = None,
) -> list[Chunk]:
    """
    Split `text` into overlapping chunks using a sliding window over sentences.

    Algorithm:
      1. Tokenise text into sentences.
      2. Accumulate sentences until the chunk reaches `chunk_size` tokens.
      3. When the limit is hit, close the chunk and back-step by `chunk_overlap`
         tokens worth of sentences (carry them into the next chunk).
      4. Discard any chunk shorter than `min_chunk_size` tokens.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    min_chunk_size = min_chunk_size or settings.min_chunk_size

    sentences = split_into_sentences(text)
    chunks: list[Chunk] = []

    # TODO: implement sliding-window accumulation
    # Hint:
    #   current_sentences: list[str] = []
    #   current_tokens: int = 0
    #   for sentence in sentences:
    #       sent_tokens = count_tokens(sentence)
    #       if current_tokens + sent_tokens > chunk_size and current_sentences:
    #           flush_chunk(current_sentences)
    #           # back-step: find sentences to carry over for overlap
    #           current_sentences, current_tokens = backtrack(current_sentences, chunk_overlap)
    #       current_sentences.append(sentence)
    #       current_tokens += sent_tokens
    #   flush remaining sentences as final chunk

    raise NotImplementedError("chunk_text() — see TODO above")


def chunk_entity(
    text: str,
    entity: dict,
) -> list[Chunk]:
    """Convenience wrapper: chunk one entity using its catalog metadata."""
    source_url = f"https://en.wikipedia.org/wiki/{entity['wikipedia_title'].replace(' ', '_')}"
    return chunk_text(
        text=text,
        entity_id=entity["entity_id"],
        entity_name=entity["entity_name"],
        entity_type=entity["entity_type"],
        source_url=source_url,
    )
