"""
RAG Pipeline — the single entry point used by CLI, Streamlit UI, and tests.

Pipeline flow for ask():
  1. Route query  →  RoutingDecision
  2. Retrieve     →  RetrievalResult  (may be empty if below threshold)
  3. IDK guard    →  return IDK_RESPONSE immediately, never call LLM
  4. Build prompt →  SYSTEM_PROMPT with numbered passages
  5. LLM.generate →  answer string (non-stream) or token generator (stream)
  6. Citations    →  parse [N] references → map to passage metadata
  7. Log          →  append to data/_query_log.jsonl
  8. History      →  append (query, response) to in-memory list
  9. Return       →  RAGResponse

Streaming variant (ask with stream=True):
  Yields str chunks, then yields the final RAGResponse as the last item.
  Consumer checks isinstance(item, RAGResponse) to detect the sentinel.
  Streamlit does: st.write_stream(chunk for chunk in gen if isinstance(chunk, str))

Citation parsing:
  LLM answers may contain [1], [2], [3] references corresponding to the
  numbered passages in the system prompt.  If references are found, only
  cited passages appear in sources.  If none are found, all retrieved
  passages are included (implicit citation).

Logging schema (data/_query_log.jsonl):
  Each line is a JSON object — append-only, safe for concurrent readers.
  Fields: timestamp, query, routing_category, routing_scores,
          chunk_ids, n_chunks, is_idk, low_confidence, used_coverage,
          latency_ms (route/retrieve/llm/total), answer_length, model.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator, Union

from src.chunker import Chunk
from src.config import PROJECT_ROOT, settings
from src.embedder import Embedder, get_embedder
from src.llm import IDK_RESPONSE, OllamaLLM, get_llm
from src.prompts import (
    IDK_RESPONSE as PROMPTS_IDK,
    build_rag_messages,
    format_passages,
)
from src.retriever import RetrievalResult, Retriever, get_retriever
from src.router import QueryRouter, RoutingDecision, get_router
from src.vector_store import ChromaVectorStore, get_vector_store

logger = logging.getLogger(__name__)

_LOG_PATH = PROJECT_ROOT / "data" / "_query_log.jsonl"
_CITATION_RE = re.compile(r"\[(\d+)\]")


# ── Response type ─────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Full result of a single ask() call."""

    answer:           str
    sources:          list[dict]          # citation metadata dicts
    routing:          RoutingDecision
    retrieved_chunks: list[Chunk]
    scores:           list[float]
    latency_ms:       dict[str, float]    # route / retrieve / llm / total
    is_idk:           bool = False
    low_confidence:   bool = False
    used_coverage:    bool = False        # entity coverage augmentation fired

    @property
    def max_score(self) -> float:
        return max(self.scores, default=0.0)

    def pretty(self) -> str:
        """Single-string debug summary for CLI / smoke tests."""
        lines = [
            f"Routing    : {self.routing.category}  "
            f"(person={self.routing.person_score:.1f}  "
            f"place={self.routing.place_score:.1f})",
            f"Retrieved  : {len(self.retrieved_chunks)} chunk(s)  "
            f"max_score={self.max_score:.3f}",
            f"Latency    : route={self.latency_ms.get('route', 0):.0f}ms  "
            f"retrieve={self.latency_ms.get('retrieve', 0):.0f}ms  "
            f"llm={self.latency_ms.get('llm', 0):.0f}ms  "
            f"total={self.latency_ms.get('total', 0):.0f}ms",
            f"IDK={self.is_idk}  low_conf={self.low_confidence}",
            f"Answer     : {self.answer}",
        ]
        if self.sources:
            lines.append("Sources    :")
            for s in self.sources:
                heading = s.get("section_heading") or "Introduction"
                lines.append(
                    f"  [{s['passage_num']}] {s['entity_name']} § {heading}"
                    f"  (score={s['score']:.3f})"
                )
        return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline.  Designed to be instantiated once and reused
    (embedder and ChromaDB client are expensive to open).

    Args:
        router:    Optional QueryRouter override (default: module singleton).
        embedder:  Optional Embedder override.
        store:     Optional ChromaVectorStore override.
        llm:       Optional OllamaLLM override.
        top_k:     Number of passages to retrieve per query.
        log_path:  Override for the query log file path.
    """

    def __init__(
        self,
        router:   QueryRouter       | None = None,
        embedder: Embedder          | None = None,
        store:    ChromaVectorStore | None = None,
        llm:      OllamaLLM         | None = None,
        top_k:    int                      = settings.top_k,
        log_path: Path              | None = None,
    ) -> None:
        self._retriever = Retriever(
            router=router,
            embedder=embedder,
            store=store,
            top_k=top_k,
        )
        self._llm      = llm or get_llm()
        self._log_path = log_path or _LOG_PATH
        self._history: list[tuple[str, RAGResponse]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(
        self,
        query:  str,
        stream: bool = False,
    ) -> Union[RAGResponse, Generator[Union[str, RAGResponse], None, None]]:
        """
        Run the full RAG pipeline for *query*.

        Args:
            query:  The user question.
            stream: If True, return a generator that yields str tokens followed
                    by one final RAGResponse sentinel.  If False (default),
                    block until complete and return RAGResponse.

        Returns:
            RAGResponse when stream=False.
            Generator[str | RAGResponse, None, None] when stream=True.
        """
        if stream:
            return self._ask_streaming(query)
        return self._ask_blocking(query)

    @property
    def history(self) -> list[tuple[str, RAGResponse]]:
        """Read-only view of conversation history (most recent last)."""
        return list(self._history)

    def clear_history(self) -> None:
        """Reset the in-memory conversation history."""
        self._history.clear()
        logger.debug("Conversation history cleared.")

    # ── Non-streaming path ────────────────────────────────────────────────────

    def _ask_blocking(self, query: str) -> RAGResponse:
        t_start = time.perf_counter()

        # ── Step 1+2: Route + Retrieve ────────────────────────────────────────
        t0 = time.perf_counter()
        retrieval = self._retriever.retrieve(query)
        t_retrieve = (time.perf_counter() - t0) * 1000
        # Route time is embedded inside retrieve(); approximate as 0 since
        # QueryRouter is pure-Python and runs in < 1 ms.
        t_route = 0.0

        latency: dict[str, float] = {
            "route":    t_route,
            "retrieve": t_retrieve,
            "llm":      0.0,
            "total":    0.0,
        }

        # ── Step 3: IDK guard ─────────────────────────────────────────────────
        if retrieval.is_empty:
            latency["total"] = (time.perf_counter() - t_start) * 1000
            resp = self._idk_response(retrieval, latency)
            self._record(query, resp)
            return resp

        # ── Step 4: Build prompt ──────────────────────────────────────────────
        context  = format_passages(retrieval.chunks, retrieval.scores)
        messages = build_rag_messages(
            context,
            query,
            low_confidence=retrieval.low_confidence,
        )

        # ── Step 5: LLM generate ──────────────────────────────────────────────
        t0 = time.perf_counter()
        llm_resp = self._llm.generate_response(
            prompt=messages[-1]["content"],   # user message = plain query
            system=messages[0]["content"],    # system = grounded prompt + context
        )
        latency["llm"] = (time.perf_counter() - t0) * 1000

        # ── Step 6: Citations ─────────────────────────────────────────────────
        sources = _extract_citations(llm_resp.answer, retrieval.chunks, retrieval.scores)

        # ── Step 7: Assemble response ─────────────────────────────────────────
        latency["total"] = (time.perf_counter() - t_start) * 1000
        resp = RAGResponse(
            answer=llm_resp.answer,
            sources=sources,
            routing=retrieval.routing,
            retrieved_chunks=retrieval.chunks,
            scores=retrieval.scores,
            latency_ms=latency,
            is_idk=IDK_RESPONSE.lower() in llm_resp.answer.lower(),
            low_confidence=retrieval.low_confidence,
            used_coverage=retrieval.used_coverage,
        )
        self._record(query, resp)
        return resp

    # ── Streaming path ────────────────────────────────────────────────────────

    def _ask_streaming(
        self,
        query: str,
    ) -> Generator[Union[str, RAGResponse], None, None]:
        """
        Yields str text chunks during LLM generation, then yields the
        completed RAGResponse as the final item.

        Usage (Streamlit):
            for item in pipeline.ask(query, stream=True):
                if isinstance(item, str):
                    placeholder.write(item)
                else:
                    final = item   # RAGResponse sentinel
        """
        t_start = time.perf_counter()

        # Route + Retrieve
        retrieval = self._retriever.retrieve(query)
        t_retrieve = (time.perf_counter() - t_start) * 1000
        latency: dict[str, float] = {"route": 0.0, "retrieve": t_retrieve, "llm": 0.0, "total": 0.0}

        # IDK guard
        if retrieval.is_empty:
            latency["total"] = (time.perf_counter() - t_start) * 1000
            resp = self._idk_response(retrieval, latency)
            self._record(query, resp)
            yield resp
            return

        # Build prompt
        context  = format_passages(retrieval.chunks, retrieval.scores)
        messages = build_rag_messages(
            context,
            query,
            low_confidence=retrieval.low_confidence,
        )

        # Stream LLM tokens
        t_llm_start = time.perf_counter()
        accumulated: list[str] = []

        for token in self._llm.generate(
            prompt=messages[-1]["content"],
            system=messages[0]["content"],
            stream=True,
        ):
            accumulated.append(token)
            yield token

        latency["llm"]   = (time.perf_counter() - t_llm_start) * 1000
        latency["total"] = (time.perf_counter() - t_start) * 1000

        answer  = "".join(accumulated)
        sources = _extract_citations(answer, retrieval.chunks, retrieval.scores)

        resp = RAGResponse(
            answer=answer,
            sources=sources,
            routing=retrieval.routing,
            retrieved_chunks=retrieval.chunks,
            scores=retrieval.scores,
            latency_ms=latency,
            is_idk=IDK_RESPONSE.lower() in answer.lower(),
            low_confidence=retrieval.low_confidence,
            used_coverage=retrieval.used_coverage,
        )
        self._record(query, resp)
        yield resp   # sentinel — consumer detects via isinstance check

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _idk_response(
        self,
        retrieval: RetrievalResult,
        latency:   dict[str, float],
    ) -> RAGResponse:
        return RAGResponse(
            answer=PROMPTS_IDK,
            sources=[],
            routing=retrieval.routing,
            retrieved_chunks=[],
            scores=[],
            latency_ms=latency,
            is_idk=True,
            low_confidence=False,
            used_coverage=retrieval.used_coverage,
        )

    def _record(self, query: str, resp: RAGResponse) -> None:
        """Append to in-memory history and write a log line."""
        self._history.append((query, resp))
        _append_log(query, resp, self._log_path)


# ── Citation parser ───────────────────────────────────────────────────────────

def _extract_citations(
    answer: str,
    chunks: list[Chunk],
    scores: list[float],
) -> list[dict[str, Any]]:
    """
    Parse [N] references from *answer* and return citation metadata dicts.

    If no explicit references are found, all passages are treated as
    implicit citations (the LLM grounded on them but didn't cite explicitly).

    Each dict contains: passage_num, entity_id, entity_name, entity_type,
    section_heading, source_url, score.
    """
    cited: set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(chunks):
            cited.add(n)

    indices = sorted(cited) if cited else list(range(1, len(chunks) + 1))

    sources: list[dict[str, Any]] = []
    for i in indices:
        chunk = chunks[i - 1]
        meta  = chunk.metadata
        sources.append({
            "passage_num":     i,
            "entity_id":       meta.get("entity_id", ""),
            "entity_name":     meta.get("entity_name", ""),
            "entity_type":     meta.get("entity_type", ""),
            "section_heading": meta.get("section_heading") or "Introduction",
            "source_url":      meta.get("source_url", ""),
            "score":           scores[i - 1],
        })
    return sources


# ── Query logger ──────────────────────────────────────────────────────────────

def _append_log(query: str, resp: RAGResponse, path: Path) -> None:
    """Append a single JSON record to the query log (one line per call)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "query":             query,
            "routing_category":  resp.routing.category,
            "routing_person":    resp.routing.person_score,
            "routing_place":     resp.routing.place_score,
            "matched_entities":  resp.routing.matched_entities,
            "n_chunks":          len(resp.retrieved_chunks),
            "chunk_ids":         [c.chunk_id for c in resp.retrieved_chunks],
            "max_score":         resp.max_score,
            "is_idk":            resp.is_idk,
            "low_confidence":    resp.low_confidence,
            "used_coverage":     resp.used_coverage,
            "latency_ms":        resp.latency_ms,
            "answer_length":     len(resp.answer),
            "model":             settings.llm_model,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Query log write failed: %s", exc)


# ── Module-level singleton ────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return the shared RAGPipeline instance (built once per process)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
