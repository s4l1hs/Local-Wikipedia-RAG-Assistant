"""Terminal chat interface — run with: python main.py or python ui/cli.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.config import ENTITY_CATALOG, settings
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.router import get_router
from src.vector_store import get_vector_store

console = Console()

# ── Command registry ──────────────────────────────────────────────────────────

_HELP = [
    ("/help",          "Show this command list"),
    ("/clear",         "Clear conversation history and screen"),
    ("/sources",       "Show full source detail for the last response"),
    ("/route <query>", "Run router only — debug query classification"),
    ("/stats",         "Show DB chunk counts and entity breakdown"),
    ("/exit",          "Quit  (Ctrl-C also works)"),
]


# ── Formatting helpers ────────────────────────────────────────────────────────

def _welcome(model: str) -> None:
    console.print(Panel(
        f"[bold]Local Wikipedia RAG Assistant[/bold]\n"
        f"Model     : [cyan]{model}[/cyan]\n"
        f"Embedding : [cyan]{settings.embedding_model.split('/')[-1]}[/cyan]  "
        f"· Top-K: [cyan]{settings.top_k}[/cyan]  "
        f"· Threshold: [cyan]{settings.similarity_threshold_low}[/cyan]\n\n"
        f"Type a question or [yellow]/help[/yellow] for commands.  "
        f"[yellow]/exit[/yellow] or Ctrl-C to quit.",
        title="📚 RAG Chat",
        border_style="blue",
    ))


def _print_help() -> None:
    t = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    t.add_column(style="yellow", no_wrap=True)
    t.add_column(style="white")
    for cmd, desc in _HELP:
        t.add_row(cmd, desc)
    console.print(t)


def _print_sources_compact(resp: RAGResponse) -> None:
    """One-line per cited source — shown after every answer."""
    if not resp.sources:
        if resp.is_idk:
            console.print("[dim]  (no sources — query out-of-corpus)[/dim]")
        return
    console.print()
    for src in resp.sources:
        heading = src.get("section_heading") or "Introduction"
        score   = src.get("score", 0.0)
        name    = src.get("entity_name", "?")
        console.print(
            f"  [dim][{src['passage_num']}][/dim] "
            f"[cyan]{name}[/cyan] [dim]§ {heading}[/dim]  "
            f"[dim](score={score:.3f})[/dim]"
        )


def _print_sources_full(resp: RAGResponse) -> None:
    """Detailed source view triggered by /sources command."""
    if not resp.sources:
        console.print("[dim]Last response had no sources.[/dim]")
        return

    console.print(Rule("[bold]Sources — last response[/bold]", style="cyan"))

    r   = resp.routing
    lat = resp.latency_ms
    console.print(
        f"  Routing : [bold]{r.category}[/bold]  "
        f"(person={r.person_score:.1f}  place={r.place_score:.1f})"
        + (f"  entities={r.matched_entities}" if r.matched_entities else "")
    )
    console.print(
        f"  Latency : retrieve=[cyan]{lat.get('retrieve', 0):.0f}ms[/cyan]  "
        f"llm=[cyan]{lat.get('llm', 0):.0f}ms[/cyan]  "
        f"total=[cyan]{lat.get('total', 0):.0f}ms[/cyan]"
    )
    console.print(
        f"  Flags   : idk={resp.is_idk}  "
        f"low_confidence={resp.low_confidence}  "
        f"used_coverage={resp.used_coverage}"
    )
    console.print()

    for src in resp.sources:
        heading = src.get("section_heading") or "Introduction"
        url     = src.get("source_url", "")
        name    = src.get("entity_name", "?")
        score   = src.get("score", 0.0)

        console.print(
            f"[bold cyan][{src['passage_num']}] {name}[/bold cyan] "
            f"[dim]§ {heading}[/dim]  "
            f"score=[green]{score:.3f}[/green]"
        )
        if url:
            console.print(f"  [dim]{url}[/dim]")

        idx = src["passage_num"] - 1
        if idx < len(resp.retrieved_chunks):
            preview = resp.retrieved_chunks[idx].content.strip()
            cutoff  = 300
            console.print(
                f"  [dim]{preview[:cutoff]}"
                + ("…" if len(preview) > cutoff else "") + "[/dim]"
            )
        console.print()


def _print_latency(resp: RAGResponse) -> None:
    lat = resp.latency_ms
    console.print(
        f"[dim]  retrieve={lat.get('retrieve', 0):.0f}ms  "
        f"llm={lat.get('llm', 0):.0f}ms  "
        f"total={lat.get('total', 0):.0f}ms  "
        f"chunks={len(resp.retrieved_chunks)}  "
        f"max_score={resp.max_score:.3f}[/dim]"
    )


def _cmd_route(query: str) -> None:
    """Run router only and display the RoutingDecision."""
    router = get_router()
    dec    = router.route(query)
    console.print(Rule("[bold]Routing debug[/bold]", style="yellow"))
    console.print(f"  Query     : [italic]{query}[/italic]")
    console.print(f"  Category  : [bold cyan]{dec.category}[/bold cyan]")
    console.print(f"  Scores    : person=[yellow]{dec.person_score:.1f}[/yellow]  "
                  f"place=[yellow]{dec.place_score:.1f}[/yellow]")
    if dec.matched_entities:
        console.print(f"  Entities  : {dec.matched_entities}")
    if dec.reasoning:
        console.print(f"  Reasoning : [dim]{dec.reasoning}[/dim]")


def _cmd_stats() -> None:
    """Print DB chunk counts."""
    console.print(Rule("[bold]DB Stats[/bold]", style="cyan"))
    try:
        store = get_vector_store()
        s     = store.stats()
    except Exception as exc:
        console.print(f"[red]Could not read DB: {exc}[/red]")
        return

    if s.get("total", 0) == 0:
        console.print(
            "[yellow]Index is empty.[/yellow]  "
            "Run [cyan]python scripts/02_build_index.py[/cyan] first."
        )
        return

    people = [e for e in ENTITY_CATALOG if e["entity_type"] == "person"]
    places = [e for e in ENTITY_CATALOG if e["entity_type"] == "place"]

    t = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    t.add_column(style="dim")
    t.add_column(style="cyan", justify="right")
    t.add_row("Total chunks",  str(s["total"]))
    t.add_row("👤 People chunks", str(s["by_type"].get("person", 0)))
    t.add_row("🏛️ Places chunks", str(s["by_type"].get("place", 0)))
    t.add_row("Entities in catalog", f"{len(people)} people · {len(places)} places")
    t.add_row("Collection", s.get("collection", "—"))
    console.print(t)


# ── Streaming output ──────────────────────────────────────────────────────────

def _stream_query(pipeline: RAGPipeline, query: str) -> RAGResponse | None:
    """
    Stream LLM tokens directly to stdout (bypassing Rich buffering) then
    return the final RAGResponse sentinel.
    """
    console.print(f"\n[bold green]Assistant[/bold green]")

    accumulated: list[str] = []
    final_resp: RAGResponse | None = None

    try:
        for item in pipeline.ask(query, stream=True):
            if isinstance(item, str):
                accumulated.append(item)
                # Write raw bytes for smooth streaming (no Rich overhead per token)
                sys.stdout.write(item)
                sys.stdout.flush()
            else:
                final_resp = item
    except KeyboardInterrupt:
        # User interrupted mid-stream — still return what we have
        sys.stdout.write("  [interrupted]\n")
        sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write("\n")
        console.print(f"\n[red]⚠ Generation error: {exc}[/red]")
        console.print(
            "[dim]Check that Ollama is running: [cyan]ollama serve[/cyan][/dim]"
        )
        return None

    sys.stdout.write("\n")   # newline after streamed tokens
    sys.stdout.flush()

    # IDK guard responses yield no tokens — print the answer directly
    if not accumulated and final_resp is not None and final_resp.answer:
        console.print(final_resp.answer)

    return final_resp


# ── Main REPL ─────────────────────────────────────────────────────────────────

def run_cli() -> None:
    """Start the interactive REPL."""
    _welcome(settings.llm_model)

    pipeline:  RAGPipeline       = RAGPipeline()
    last_resp: RAGResponse | None = None

    while True:
        # ── Prompt ────────────────────────────────────────────────────────────
        try:
            console.print()
            raw = console.input("[bold blue]>>>[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not raw:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if raw.startswith("/"):
            parts = raw[1:].split(None, 1)     # [cmd, rest?]
            cmd   = parts[0].lower()
            rest  = parts[1] if len(parts) > 1 else ""

            if cmd in ("exit", "quit"):
                console.print("[dim]Goodbye.[/dim]")
                break

            elif cmd == "help":
                _print_help()

            elif cmd == "clear":
                pipeline.clear_history()
                last_resp = None
                console.clear()
                _welcome(settings.llm_model)

            elif cmd == "sources":
                if last_resp is None:
                    console.print("[dim]No response yet — ask a question first.[/dim]")
                else:
                    _print_sources_full(last_resp)

            elif cmd == "route":
                if not rest:
                    console.print("[yellow]Usage: /route <your query>[/yellow]")
                else:
                    _cmd_route(rest)

            elif cmd == "stats":
                _cmd_stats()

            else:
                console.print(
                    f"[red]Unknown command: /{cmd}[/red]  "
                    f"Type [yellow]/help[/yellow] for the command list."
                )
            continue

        # ── RAG query ─────────────────────────────────────────────────────────
        resp = _stream_query(pipeline, raw)

        if resp is not None:
            last_resp = resp
            _print_sources_compact(resp)
            _print_latency(resp)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cli()
