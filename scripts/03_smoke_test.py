#!/usr/bin/env python3
"""
Step 3 — End-to-end smoke test against the built index and live LLM.

Runs 5 representative queries (person / place / both / OOC / ambiguous)
and pretty-prints routing, latency, answer, and sources for each.

Usage:
    python scripts/03_smoke_test.py
    python scripts/03_smoke_test.py --no-llm    # skip LLM, show retrieval only

Prerequisites:
    python scripts/01_fetch_wikipedia.py   # fetch + index entities first
    ollama serve && ollama pull llama3.2   # Ollama running with model pulled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from src.rag_pipeline import RAGPipeline, RAGResponse
from src.retriever import Retriever
from src.utils import configure_logging

console = Console()

# ── Test cases ────────────────────────────────────────────────────────────────

QUERIES: list[dict] = [
    {
        "label":    "PERSON",
        "query":    "What did Einstein win the Nobel Prize for?",
        "expected": "photoelectric",       # substring check (case-insensitive)
        "expect_idk": False,
    },
    {
        "label":    "PLACE",
        "query":    "How tall is the Eiffel Tower and when was it built?",
        "expected": "1889",
        "expect_idk": False,
    },
    {
        "label":    "BOTH",
        "query":    "Compare Albert Einstein and the Eiffel Tower",
        "expected": None,                  # no strict substring — just non-IDK
        "expect_idk": False,
    },
    {
        "label":    "OOC",
        "query":    "What is the best recipe for sourdough bread?",
        "expected": None,
        "expect_idk": True,                # must be IDK
    },
    {
        "label":    "AMBIGUOUS",
        "query":    "Tell me about the most famous discovery in science",
        "expected": None,
        "expect_idk": False,               # should attempt an answer from corpus
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    console.print(f"  [green]✓[/green]  {msg}")

def _fail(msg: str) -> None:
    console.print(f"  [red]✗[/red]  {msg}")


def _print_response(label: str, query: str, resp: RAGResponse) -> None:
    r = resp.routing
    lat = resp.latency_ms

    # Header
    console.print(f"\n  [bold cyan]Q:[/bold cyan] {query}")
    console.print(
        f"  [bold]Routing  :[/bold] {r.category}  "
        f"(person={r.person_score:.1f}  place={r.place_score:.1f})"
        + (f"  entities={r.matched_entities}" if r.matched_entities else "")
    )
    console.print(
        f"  [bold]Retrieved:[/bold] {len(resp.retrieved_chunks)} chunk(s)  "
        f"max_score={resp.max_score:.3f}  "
        f"low_conf={resp.low_confidence}  idk={resp.is_idk}"
    )
    console.print(
        f"  [bold]Latency  :[/bold] "
        f"retrieve={lat.get('retrieve', 0):.0f}ms  "
        f"llm={lat.get('llm', 0):.0f}ms  "
        f"total={lat.get('total', 0):.0f}ms"
    )

    # Answer
    answer_preview = resp.answer.replace("\n", " ")
    console.print(f"  [bold]Answer   :[/bold] {answer_preview}")

    # Sources
    if resp.sources:
        console.print(f"  [bold]Sources  :[/bold]")
        for s in resp.sources:
            heading = s.get("section_heading") or "Introduction"
            console.print(
                f"    [{s['passage_num']}] {s['entity_name']} § {heading}"
                f"  (score={s['score']:.3f})"
            )


def _check(resp: RAGResponse, case: dict) -> bool:
    if case["expect_idk"]:
        if not resp.is_idk:
            _fail(f"Expected IDK but got a real answer")
            return False
        _ok("Correctly returned IDK for out-of-corpus query")
        return True

    if resp.is_idk:
        _fail("Unexpected IDK — retrieval or threshold issue")
        return False

    expected = case.get("expected")
    if expected and expected.lower() not in resp.answer.lower():
        _fail(f"Expected '{expected}' in answer but not found")
        return False

    _ok(f"[{case['label']}] passed")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()
    args = _parse_args()

    console.print(Panel(
        "[bold]RAG Pipeline Smoke Test[/bold]\n"
        "5 query types: PERSON · PLACE · BOTH · OOC · AMBIGUOUS",
        border_style="blue",
    ))

    if args.no_llm:
        console.print("[yellow]--no-llm mode: retrieval only, LLM skipped[/yellow]\n")
        _run_retrieval_only()
        return

    pipeline = RAGPipeline()
    all_passed = True

    for case in QUERIES:
        console.print(Rule(f"[bold]{case['label']}[/bold]"))
        try:
            resp = pipeline.ask(case["query"])
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")
            all_passed = False
            continue

        _print_response(case["label"], case["query"], resp)
        passed = _check(resp, case)
        all_passed &= passed

    # Summary
    console.print()
    if all_passed:
        console.print(Panel(
            "[bold green]✓ All smoke tests passed.[/bold green]\n"
            "RAG pipeline is ready for Phase 6 (Streamlit UI).",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[bold red]✗ Some smoke tests failed.[/bold red]\n"
            "Fix issues above before proceeding to Phase 6.",
            border_style="red",
        ))
        sys.exit(1)


def _run_retrieval_only() -> None:
    """Retrieval-only path — useful when Ollama is not running."""
    retriever = Retriever()
    all_passed = True

    for case in QUERIES:
        console.print(Rule(f"[bold]{case['label']}[/bold]"))
        console.print(f"\n  [bold cyan]Q:[/bold cyan] {case['query']}")
        try:
            result = retriever.retrieve(case["query"])
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")
            all_passed = False
            continue

        r = result.routing
        console.print(
            f"  [bold]Routing  :[/bold] {r.category}  "
            f"(person={r.person_score:.1f}  place={r.place_score:.1f})"
        )
        console.print(
            f"  [bold]Retrieved:[/bold] {len(result.chunks)} chunk(s)  "
            f"max_score={result.max_score:.3f}  "
            f"is_empty={result.is_empty}  low_conf={result.low_confidence}"
        )
        for i, (chunk, score) in enumerate(zip(result.chunks, result.scores), 1):
            meta = chunk.metadata
            console.print(
                f"    [{i}] {meta.get('entity_name', '?')} § "
                f"{meta.get('section_heading') or 'Introduction'}  "
                f"(score={score:.3f})"
            )

        if case["expect_idk"] and not result.is_empty:
            _fail("OOC query not rejected at retrieval stage")
            all_passed = False
        elif case["expect_idk"] and result.is_empty:
            _ok("OOC correctly rejected (is_empty=True)")
        else:
            _ok(f"[{case['label']}] retrieval returned {len(result.chunks)} chunk(s)")

    console.print()
    status = (
        "[bold green]✓ Retrieval smoke tests passed.[/bold green]"
        if all_passed else
        "[bold red]✗ Some retrieval tests failed.[/bold red]"
    )
    console.print(Panel(status, border_style="green" if all_passed else "red"))
    if not all_passed:
        sys.exit(1)


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end RAG pipeline smoke test.")
    p.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM generation; test retrieval path only (no Ollama needed).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
