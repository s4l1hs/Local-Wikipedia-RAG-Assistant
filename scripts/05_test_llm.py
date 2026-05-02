#!/usr/bin/env python3
"""
Step 5 — Smoke-test the Ollama LLM client.

Checks service availability, model presence, non-streaming generation,
streaming generation, and the IDK (empty context) behaviour.

Usage:
    python scripts/05_test_llm.py
    python scripts/05_test_llm.py --model phi3
    python scripts/05_test_llm.py --base-url http://remote:11434

Prerequisites:
    ollama serve          # start Ollama daemon
    ollama pull llama3.2  # download the model (~2 GB)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.llm import IDK_RESPONSE, LLMResponse, OllamaLLM
from src.prompts import IDK_RESPONSE as PROMPTS_IDK, build_rag_messages, format_context
from src.utils import configure_logging

console = Console()

# ── Test data ─────────────────────────────────────────────────────────────────

_CONTEXT = """\
Albert Einstein (1879–1955) was a German-born theoretical physicist who \
developed the theory of relativity, one of the two pillars of modern physics. \
His work is also known for its influence on the philosophy of science. He is \
best known to the general public for his mass–energy equivalence formula E = mc². \
Einstein received the 1921 Nobel Prize in Physics for his discovery of the law \
of the photoelectric effect, a pivotal step in the development of quantum theory.
"""

_QUERY_NORMAL     = "What did Einstein win the Nobel Prize for?"
_QUERY_FALSE_PREM = "What Nobel Prize did Einstein win for his theory of relativity?"
_QUERY_IDK        = "What was Einstein's favourite food?"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    console.print(f"  [green]✓[/green]  {msg}")

def _fail(msg: str) -> None:
    console.print(f"  [red]✗[/red]  {msg}")
    sys.exit(1)

def _info(msg: str) -> None:
    console.print(f"  [dim]{msg}[/dim]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()
    args = _parse_args()

    llm = OllamaLLM(model_name=args.model, base_url=args.base_url)

    console.print(Panel(
        f"[bold]Ollama LLM Smoke Test[/bold]\n"
        f"Model   : [cyan]{args.model}[/cyan]\n"
        f"Base URL: [cyan]{args.base_url}[/cyan]",
        border_style="blue",
    ))

    # ── 1. Health check ───────────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1 — Health Check[/bold]"))
    ok, reason = llm.health_check()
    if not ok:
        console.print(Panel(
            f"[bold red]Ollama health check failed.[/bold red]\n\n{reason}",
            border_style="red",
        ))
        sys.exit(1)
    _ok("Ollama service reachable and model available")

    # ── 2. Non-streaming generation ───────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Step 2 — Non-streaming Generation[/bold]"))
    console.print(f"  Query: [italic]{_QUERY_NORMAL}[/italic]\n")

    messages = build_rag_messages(_CONTEXT, _QUERY_NORMAL)
    resp = llm.generate_response(
        prompt=_QUERY_NORMAL,
        system=messages[0]["content"],   # system prompt already has context
    )
    # Re-run cleanly — use generate_response directly with context baked in
    resp = llm.generate_response(
        prompt=messages[1]["content"],
        system=messages[0]["content"],
    )

    _info(f"Model: {resp.model}  |  tokens: {resp.prompt_tokens}p + {resp.completion_tokens}c")
    console.print(f"\n  [bold]Answer:[/bold] {resp.answer}\n")

    if not resp.answer:
        _fail("Empty response from model")
    _ok("Non-streaming generation returned a non-empty answer")

    # ── 3. False-premise correction ───────────────────────────────────────────
    console.print(Rule("[bold]Step 3 — False Premise Correction[/bold]"))
    console.print(f"  Query: [italic]{_QUERY_FALSE_PREM}[/italic]")
    console.print("  [dim](Correct answer: photoelectric effect, NOT relativity)[/dim]\n")

    msgs2    = build_rag_messages(_CONTEXT, _QUERY_FALSE_PREM)
    resp2 = llm.generate_response(
        prompt=msgs2[1]["content"],
        system=msgs2[0]["content"],
    )
    console.print(f"  [bold]Answer:[/bold] {resp2.answer}\n")

    if not resp2.answer:
        _fail("Empty response for false-premise query")
    _ok("False-premise query returned a response (verify it corrects the assumption)")

    # ── 4. IDK (no context) ───────────────────────────────────────────────────
    console.print(Rule("[bold]Step 4 — IDK on Empty Context[/bold]"))
    console.print(f"  Query: [italic]{_QUERY_IDK}[/italic]\n")

    empty_ctx  = format_context([])
    msgs3      = build_rag_messages(empty_ctx, _QUERY_IDK)
    resp3 = llm.generate_response(
        prompt=msgs3[1]["content"],
        system=msgs3[0]["content"],
    )
    console.print(f"  [bold]Answer:[/bold] {resp3.answer}\n")

    if not resp3.answer:
        _fail("Empty response for IDK query")
    _ok("IDK query handled (verify response says it doesn't know)")

    # ── 5. Streaming generation ───────────────────────────────────────────────
    console.print(Rule("[bold]Step 5 — Streaming Generation[/bold]"))
    console.print(f"  Query: [italic]{_QUERY_NORMAL}[/italic]")
    console.print("  [dim](streaming tokens below…)[/dim]\n  ", end="")

    msgs4    = build_rag_messages(_CONTEXT, _QUERY_NORMAL)
    chunks   = list(llm.generate(
        prompt=msgs4[1]["content"],
        system=msgs4[0]["content"],
        stream=True,
    ))

    console.print("".join(chunks))
    console.print()

    if not chunks:
        _fail("Streaming returned zero chunks")
    _ok(f"Streaming returned {len(chunks)} chunk(s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold green]✓ All LLM smoke tests passed.[/bold green]\n"
        "OllamaLLM is ready for the RAG pipeline.",
        border_style="green",
    ))


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test the Ollama LLM client.")
    p.add_argument("--model",    default="llama3.2",
                   help="Ollama model name (default: llama3.2)")
    p.add_argument("--base-url", default="http://localhost:11434",
                   help="Ollama base URL (default: http://localhost:11434)")
    return p.parse_args()


if __name__ == "__main__":
    main()
