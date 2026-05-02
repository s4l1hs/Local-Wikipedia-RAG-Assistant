"""
Performance benchmarks — retrieval and full-pipeline latency.

Retrieval benchmarks run without Ollama.
Pipeline benchmarks require @pytest.mark.llm.

Targets (CPU, bge-small-en-v1.5, llama3.2 3B):
  Retrieval  p50 < 100ms  |  p95 < 250ms   (after first-call model warmup)
  Full pipe  p50 < 3000ms |  p95 < 5000ms  (≈ LLM generation dominates)

Run:
    pytest tests/test_performance.py -m slow           # retrieval benchmarks
    pytest tests/test_performance.py -m "slow and llm" # also pipeline benchmarks
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline

# ── Benchmark query sets ──────────────────────────────────────────────────────

_RETRIEVAL_QUERIES = [
    "Who discovered the theory of relativity?",
    "What did Marie Curie discover about radioactivity?",
    "How tall is the Eiffel Tower?",
    "Who invented the alternating current motor?",
    "Which physicist won the Nobel Prize in 1921?",
    "When was the iron lattice tower in Paris built?",
    "Who discovered polonium and radium?",
    "What is Nikola Tesla famous for?",
    "What was built for the 1889 World's Fair?",
    "Who was Marie Curie and what did she achieve?",
    "How did Einstein develop the theory of relativity?",
    "What is the height of the Eiffel Tower in metres?",
]

_PIPELINE_QUERIES = [
    "What did Einstein win the Nobel Prize for?",
    "What elements did Marie Curie discover?",
    "Where is the Eiffel Tower located?",
    "What is Nikola Tesla famous for inventing?",
    "Who discovered polonium?",
]

# ── Latency targets ────────────────────────────────────────────────────────────

_RETRIEVAL_P50_TARGET_MS  = 100.0
_RETRIEVAL_P95_TARGET_MS  = 250.0
_PIPELINE_P50_TARGET_MS   = 3_000.0
_PIPELINE_P95_TARGET_MS   = 5_000.0


def _percentile(data: list[float], p: int) -> float:
    """Return the p-th percentile of data (0–100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


# ── Retrieval latency benchmark ────────────────────────────────────────────────

@pytest.mark.slow
def test_retrieval_warmup_then_latency() -> None:
    """
    Warm up the embedding model with one ignored query, then benchmark
    the remaining queries and assert p50 / p95 targets.
    """
    r = Retriever()

    # Warmup — discard this time (model load dominates)
    r.retrieve("warm up the embedding model")

    latencies_ms: list[float] = []
    for query in _RETRIEVAL_QUERIES:
        t0 = time.perf_counter()
        result = r.retrieve(query)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        latencies_ms.append(elapsed_ms)
        # Ensure retrieval actually returned something (sanity)
        assert result is not None

    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    avg = statistics.mean(latencies_ms)
    mn  = min(latencies_ms)
    mx  = max(latencies_ms)

    print(
        f"\n  Retrieval latency over {len(latencies_ms)} queries:\n"
        f"    min={mn:.1f}ms  avg={avg:.1f}ms  max={mx:.1f}ms\n"
        f"    p50={p50:.1f}ms  p95={p95:.1f}ms\n"
        f"    target  p50<{_RETRIEVAL_P50_TARGET_MS:.0f}ms  "
        f"p95<{_RETRIEVAL_P95_TARGET_MS:.0f}ms"
    )

    assert p50 <= _RETRIEVAL_P50_TARGET_MS, (
        f"Retrieval p50={p50:.1f}ms exceeds target {_RETRIEVAL_P50_TARGET_MS:.0f}ms"
    )
    assert p95 <= _RETRIEVAL_P95_TARGET_MS, (
        f"Retrieval p95={p95:.1f}ms exceeds target {_RETRIEVAL_P95_TARGET_MS:.0f}ms"
    )


@pytest.mark.slow
def test_retrieval_individual_under_500ms() -> None:
    """No single retrieval call takes longer than 500ms (after warmup)."""
    r = Retriever()
    r.retrieve("warmup")   # discard

    for query in _RETRIEVAL_QUERIES[:6]:
        t0 = time.perf_counter()
        r.retrieve(query)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        assert elapsed_ms < 500, (
            f"Single retrieval call took {elapsed_ms:.1f}ms > 500ms\n"
            f"Query: {query!r}"
        )


# ── Full-pipeline latency benchmark (LLM required) ────────────────────────────

@pytest.mark.slow
@pytest.mark.llm
def test_pipeline_latency() -> None:
    """
    Benchmark end-to-end pipeline latency (retrieval + LLM generation).
    Uses non-streaming mode for deterministic timing.
    """
    p = RAGPipeline()

    latencies_ms: list[float] = []
    for query in _PIPELINE_QUERIES:
        t0 = time.perf_counter()
        resp = p.ask(query, stream=False)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        latencies_ms.append(elapsed_ms)
        assert resp is not None
        assert resp.answer  # must not be empty

    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    avg = statistics.mean(latencies_ms)

    print(
        f"\n  Full-pipeline latency over {len(latencies_ms)} queries:\n"
        f"    avg={avg:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms\n"
        f"    target  p50<{_PIPELINE_P50_TARGET_MS:.0f}ms  "
        f"p95<{_PIPELINE_P95_TARGET_MS:.0f}ms"
    )

    assert p50 <= _PIPELINE_P50_TARGET_MS, (
        f"Pipeline p50={p50:.0f}ms exceeds target {_PIPELINE_P50_TARGET_MS:.0f}ms"
    )
    assert p95 <= _PIPELINE_P95_TARGET_MS, (
        f"Pipeline p95={p95:.0f}ms exceeds target {_PIPELINE_P95_TARGET_MS:.0f}ms"
    )


@pytest.mark.slow
@pytest.mark.llm
def test_pipeline_latency_detail_in_response() -> None:
    """RAGResponse.latency_ms fields are populated and plausible."""
    p = RAGPipeline()
    resp = p.ask("How tall is the Eiffel Tower?")

    lat = resp.latency_ms
    assert lat["total"] > 0,    "total latency should be > 0"
    assert lat["retrieve"] > 0, "retrieve latency should be > 0"
    assert lat["llm"] > 0,      "llm latency should be > 0 for in-corpus query"
    assert lat["total"] >= lat["retrieve"] + lat["llm"] - 10, (
        "total should be ≥ retrieve + llm (within 10ms tolerance)"
    )
