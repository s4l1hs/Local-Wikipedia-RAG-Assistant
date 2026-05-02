"""
Pytest configuration — custom markers and shared fixtures.

Markers:
  @pytest.mark.llm   — requires Ollama to be running; auto-skipped otherwise
  @pytest.mark.slow  — latency / performance benchmarks; run with -m slow

Run just fast tests:   pytest -m "not llm and not slow"
Run all (with LLM):    pytest -m "not slow"        (requires: ollama serve)
Run everything:        pytest                       (requires: ollama serve)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Module-level Ollama check (done once per session) ─────────────────────────

_ollama_ok: bool | None = None


def _check_ollama() -> bool:
    global _ollama_ok
    if _ollama_ok is None:
        try:
            from src.llm import OllamaLLM
            ok, _ = OllamaLLM().health_check()
            _ollama_ok = ok
        except Exception:
            _ollama_ok = False
    return _ollama_ok


# ── Marker registration ───────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "llm: requires Ollama LLM service (auto-skipped if down)")
    config.addinivalue_line("markers", "slow: performance / latency benchmarks")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "llm" in item.keywords and not _check_ollama():
        pytest.skip("Ollama not running — start with: ollama serve && ollama pull llama3.2")


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def retriever():
    from src.retriever import Retriever
    return Retriever()


@pytest.fixture(scope="session")
def pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()


@pytest.fixture(scope="session")
def router():
    from src.router import get_router
    return get_router()


@pytest.fixture(scope="session")
def db_stats():
    from src.vector_store import get_vector_store
    try:
        return get_vector_store().stats()
    except Exception:
        return {"total": 0, "by_type": {}}
