"""Terminal chat interface — run with: python main.py --mode cli"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.config import settings
from src.rag_pipeline import ask, RAGResponse

console = Console()

_COMMANDS = {
    "help":   "Show this help message",
    "clear":  "Clear conversation history",
    "sources":"Toggle source display",
    "debug":  "Toggle debug info",
    "exit":   "Quit",
}


def print_welcome() -> None:
    console.print(Panel(
        f"[bold]Local Wikipedia RAG Assistant[/bold]\n"
        f"Model: [cyan]{settings.llm_model}[/cyan]  |  "
        f"Embedding: [cyan]{settings.embedding_model}[/cyan]\n"
        f"Type [yellow]/help[/yellow] for commands, [yellow]/exit[/yellow] to quit.",
        title="📚 RAG Chat",
        border_style="blue",
    ))


def print_help() -> None:
    t = Table(show_header=False, box=None, padding=(0, 2))
    for cmd, desc in _COMMANDS.items():
        t.add_row(f"[yellow]/{cmd}[/yellow]", desc)
    console.print(t)


def run_cli(show_sources: bool = True, show_debug: bool = False) -> None:
    print_welcome()
    history: list[dict] = []   # kept for display only; not sent to retrieval

    while True:
        try:
            raw = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not raw:
            continue

        # ── Built-in commands ─────────────────────────────────────────────────
        if raw.startswith("/"):
            cmd = raw[1:].lower()
            if cmd in ("exit", "quit"):
                break
            elif cmd == "help":
                print_help()
            elif cmd == "clear":
                history.clear()
                console.clear()
                print_welcome()
            elif cmd == "sources":
                show_sources = not show_sources
                console.print(f"Sources: [{'green' if show_sources else 'red'}]{show_sources}[/]")
            elif cmd == "debug":
                show_debug = not show_debug
                console.print(f"Debug: [{'green' if show_debug else 'red'}]{show_debug}[/]")
            else:
                console.print(f"[red]Unknown command: /{cmd}[/red]")
            continue

        # ── RAG query ─────────────────────────────────────────────────────────
        with console.status("[dim]Retrieving and generating...[/dim]"):
            response: RAGResponse = ask(raw, show_sources=show_sources)

        console.print("\n[bold green]Assistant[/bold green]")
        console.print(Markdown(response.answer))

        if show_sources and response.sources:
            console.print("\n[dim]Sources:[/dim]")
            for i, src in enumerate(response.sources[:3], 1):
                console.print(
                    f"  [dim]{i}.[/dim] [cyan]{src.entity_name}[/cyan] "
                    f"(sim={src.similarity:.3f}) — {src.source_url}"
                )

        if show_debug:
            console.print(
                f"[dim]intent={response.intent} | "
                f"sim_max={response.sim_max:.3f} | "
                f"fallback={response.used_fallback} | "
                f"idk={response.is_idk}[/dim]"
            )

        history.append({"role": "user", "content": raw})
        history.append({"role": "assistant", "content": response.answer})
