"""
Embedding abstraction layer.

Model: BAAI/bge-small-en-v1.5
──────────────────────────────
Decision matrix for this project (≈600 Wikipedia chunks, local machine):

  Model                    MTEB   Dim  Context  Footprint  Setup
  ─────────────────────────────────────────────────────────────────────────
  BAAI/bge-small-en-v1.5   62.17  384   512 tok   130 MB   pip only  ← CHOSEN
  nomic-embed-text          62.39  768  8192 tok   274 MB   Ollama daemon
  all-MiniLM-L6-v2          56.26  384   256 tok    80 MB   pip only
  all-mpnet-base-v2         57.78  768   384 tok   420 MB   pip only

Why bge-small-en-v1.5:
  1. Quality: MTEB 62.17 — within 0.22 points of nomic-embed-text (62.39);
     benchmark noise, not a meaningful difference at our corpus size.
  2. Context: 512-token window fits every chunk we produce (max ~420 tok with
     heading prefix).  MiniLM's 256-token hard limit silently truncates our
     larger chunks and degrades retrieval quality throughout the pipeline.
  3. Storage: 384-dim → 600 × 384 × 4 B ≈ 0.88 MB.  768-dim doubles this to
     1.76 MB — a difference irrelevant at our scale.
  4. Latency: ~3 ms/chunk on CPU; query embedding adds <5 ms — imperceptible
     against the 2–10 s Ollama LLM generation time that follows.
  5. Setup: pure sentence-transformers pip install.  nomic-embed-text requires
     `ollama serve` running as a background daemon; its failure mode at query
     time is an OSError, not a descriptive Python exception.
  6. BGE design: trained specifically for asymmetric dense retrieval (not just
     semantic similarity).  An optional query-side instruction prefix gives a
     retrieval boost on out-of-domain text without retraining.

Normalization: every output vector is L2-normalised so dot product == cosine
similarity — ready for direct use with ChromaDB's cosine distance metric.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH = 32
_CACHE_ROOT   = PROJECT_ROOT / ".embed_cache"

# BGE asymmetric retrieval: queries get this prefix; passages (chunks) do not.
# The bge-small-en-v1.5 "v1.5" update reduced reliance on the prefix, but
# including it on queries still gives a measurable retrieval boost on
# out-of-domain text such as Wikipedia encyclopaedia prose.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Embedder ──────────────────────────────────────────────────────────────────

class Embedder:
    """
    Swappable wrapper around sentence-transformers providing:
    - Lazy model loading (nothing downloaded/loaded until first use)
    - Automatic device selection: CUDA > MPS > CPU
    - Batched inference with an optional tqdm progress bar
    - L2 normalisation (cosine similarity == dot product after normalisation)
    - Optional disk cache: already-embedded texts skip re-encoding, making
      repeated indexing runs during development nearly instant
    """

    def __init__(
        self,
        model_name:   str         = DEFAULT_MODEL,
        batch_size:   int         = DEFAULT_BATCH,
        normalize:    bool        = True,
        cache_dir:    Path | None = _CACHE_ROOT,
        query_prefix: str         = _BGE_QUERY_PREFIX,
    ) -> None:
        self.model_name   = model_name
        self.batch_size   = batch_size
        self.normalize    = normalize
        self.query_prefix = query_prefix

        self._model = None  # loaded on first call to _load()

        # Disk cache: SHA-256(text) → float32 vector
        self._cache_path = (
            (cache_dir / f"{_name_slug(model_name)}.pkl") if cache_dir else None
        )
        self._cache: dict[str, np.ndarray] = _load_cache(self._cache_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts:         List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed passage texts (chunks).  Returns float32 array shape (N, dim).

        Passages are NOT prefixed — BGE convention for asymmetric retrieval.
        Texts already present in the disk cache are never re-encoded.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        results:  list[np.ndarray | None] = [None] * len(texts)
        miss_idx: list[int]               = []
        miss_txt: list[str]               = []

        for i, t in enumerate(texts):
            cached = self._cache.get(_sha256(t))
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_txt.append(t)

        n_cached = len(texts) - len(miss_txt)
        if n_cached:
            logger.debug("Cache hit: %d / %d texts", n_cached, len(texts))

        if miss_txt:
            new_vecs = self._encode(miss_txt, show_progress=show_progress)
            for j, (idx, text) in enumerate(zip(miss_idx, miss_txt)):
                v = new_vecs[j]
                results[idx]               = v
                self._cache[_sha256(text)] = v
            _save_cache(self._cache, self._cache_path)

        return np.stack(results, axis=0).astype(np.float32)  # type: ignore[arg-type]

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single user query.  Returns float32 array shape (dim,).

        Applies the BGE query instruction prefix if configured.
        Queries are never cached — each call is a single forward pass (<5 ms).
        """
        prefixed = f"{self.query_prefix}{text}" if self.query_prefix else text
        vec = self._encode([prefixed], show_progress=False)
        return vec[0]

    @property
    def dim(self) -> int:
        """Embedding dimension — triggers model load on first access."""
        return self._load().get_sentence_embedding_dimension()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self):
        """Lazy-load the SentenceTransformer model on the best available device."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            device = _best_device()
            logger.info(
                "Loading embedding model '%s' on %s …", self.model_name, device
            )
            self._model = SentenceTransformer(self.model_name, device=device)
            logger.info(
                "Model ready  dim=%d  device=%s",
                self._model.get_sentence_embedding_dimension(),
                device,
            )
        return self._model

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        """Run sentence-transformers encode in batches; returns (N, dim) float32."""
        model   = self._load()
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        parts: list[np.ndarray] = []
        bar = tqdm(
            batches,
            desc="Embedding",
            unit="batch",
            disable=not (show_progress and len(batches) > 1),
        )
        for batch in bar:
            vecs = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
            )
            parts.append(vecs.astype(np.float32))

        if parts:
            return np.concatenate(parts, axis=0)
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)


# ── Module-level singleton ────────────────────────────────────────────────────

_default_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the shared Embedder instance (model loaded once per process)."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder(
            model_name=settings.embedding_model,
            batch_size=settings.embed_batch_size,
            cache_dir=settings.embed_cache_dir,
        )
    return _default_embedder


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_device() -> str:
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _name_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _load_cache(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            cache = pickle.load(fh)
        logger.debug("Loaded %d cached vectors from %s", len(cache), path)
        return cache
    except Exception as exc:
        logger.warning("Cache load failed (%s) — starting fresh", exc)
        return {}


def _save_cache(cache: dict[str, np.ndarray], path: Path | None) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.warning("Cache flush failed: %s", exc)
