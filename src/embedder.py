"""Local embedding model wrapper — Ollama (nomic-embed-text) with sentence-transformers fallback."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class EmbedderNotReadyError(RuntimeError):
    pass


class LocalEmbedder:
    """
    Thin wrapper that tries Ollama first, falls back to sentence-transformers.

    Usage:
        embedder = LocalEmbedder()
        vector = embedder.embed("Who was Einstein?")
        vectors = embedder.embed_batch(["text1", "text2"])
    """

    def __init__(self) -> None:
        self.model_name: str = ""
        self._backend: str = ""        # "ollama" | "sentence_transformers"
        self._st_model = None          # SentenceTransformer instance if used
        self._initialize()

    def _initialize(self) -> None:
        """Probe Ollama; fall back to sentence-transformers if unavailable."""
        if self._probe_ollama():
            self._backend = "ollama"
            self.model_name = settings.embedding_model
            logger.info("Embedder: using Ollama / %s", self.model_name)
        else:
            self._load_sentence_transformers()
            self._backend = "sentence_transformers"
            self.model_name = settings.embedding_fallback
            logger.info("Embedder: Ollama unavailable, using sentence-transformers / %s", self.model_name)

    def _probe_ollama(self) -> bool:
        """Return True if Ollama is running and the embedding model is available."""
        # TODO:
        #   import ollama
        #   try:
        #       ollama.embeddings(model=settings.embedding_model, prompt="probe")
        #       return True
        #   except Exception:
        #       return False
        raise NotImplementedError

    def _load_sentence_transformers(self) -> None:
        # TODO:
        #   from sentence_transformers import SentenceTransformer
        #   self._st_model = SentenceTransformer(settings.embedding_fallback)
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        """Embed a single string; returns a flat float list."""
        # TODO: dispatch to _embed_ollama or _embed_st based on self._backend
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings; returns a list of float lists."""
        # TODO: batch embed — Ollama doesn't have native batching, loop over texts;
        #       sentence-transformers supports encode(texts) natively
        raise NotImplementedError

    # ── Internal dispatch ─────────────────────────────────────────────────────

    def _embed_ollama(self, text: str) -> list[float]:
        # TODO:
        #   import ollama
        #   response = ollama.embeddings(model=self.model_name, prompt=text)
        #   return response["embedding"]
        raise NotImplementedError

    def _embed_st(self, text: str) -> list[float]:
        # TODO:
        #   vector = self._st_model.encode(text, normalize_embeddings=True)
        #   return vector.tolist()
        raise NotImplementedError


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    """Module-level singleton — import and call this everywhere."""
    return LocalEmbedder()
