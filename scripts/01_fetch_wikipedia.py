#!/usr/bin/env python3
"""
Step 1 — Download Wikipedia articles for all 44 catalog entities.

Usage examples:
    python scripts/01_fetch_wikipedia.py                  # fetch all 44
    python scripts/01_fetch_wikipedia.py --people         # only 22 people
    python scripts/01_fetch_wikipedia.py --places         # only 22 places
    python scripts/01_fetch_wikipedia.py --force          # ignore cache
    python scripts/01_fetch_wikipedia.py --limit 3        # first 3 (smoke test)
    python scripts/01_fetch_wikipedia.py --entity nikola_tesla

Output:
    data/raw/{type}_{entity_id}.json   — structured article JSON
    data/raw/_manifest.json            — index of all fetched articles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.config import ENTITY_CATALOG, PEOPLE_LIST, PLACES_LIST
from src.ingest import (
    ArticleNotFoundError,
    DisambiguationError,
    WikiFetcher,
    is_already_fetched,
    load_manifest,
    save_article,
    update_manifest,
)
from src.utils import configure_logging

console = Console()


# ── Result accumulators ───────────────────────────────────────────────────────

class FetchStats:
    def __init__(self) -> None:
        self.ok:   list[dict] = []    # {entity_id, entity_name, word_count}
        self.skip: list[dict] = []    # {entity_id, entity_name}
        self.fail: list[dict] = []    # {entity_id, entity_name, reason}

    @property
    def total_words(self) -> int:
        return sum(r["word_count"] for r in self.ok)


def _record_failure(
    stats: FetchStats,
    eid: str,
    name: str,
    reason: str,
    display: str,
    color: str = "red",
) -> None:
    stats.fail.append({"entity_id": eid, "entity_name": name, "reason": reason})
    console.print(f"  [{color}]✗[/{color}] {name:42s} [{color}]{display}[/{color}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()
    args = _parse_args()

    # ── Select target entities ────────────────────────────────────────────────
    if args.entity:
        targets = [
            e for e in ENTITY_CATALOG
            if e["entity_id"] == args.entity
        ]
        if not targets:
            console.print(f"[red]Unknown entity_id: '{args.entity}'[/red]")
            console.print("Valid IDs: " + ", ".join(e["entity_id"] for e in ENTITY_CATALOG))
            sys.exit(1)
    elif args.people:
        targets = list(PEOPLE_LIST)
    elif args.places:
        targets = list(PLACES_LIST)
    else:
        targets = list(ENTITY_CATALOG)

    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    # ── Banner ────────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Local Wikipedia RAG — Article Fetcher[/bold]\n"
        f"Targets: [cyan]{len(targets)}[/cyan] entities  |  "
        f"Force: [cyan]{args.force}[/cyan]  |  "
        f"Delay: [cyan]≥0.8 s[/cyan] between requests",
        border_style="blue",
    ))

    stats = FetchStats()
    fetcher = WikiFetcher()

    # ── Progress bar ──────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.fields[entity_name]:40s}[/bold cyan]"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("fetching", total=len(targets), entity_name="starting…")

        for entity in targets:
            eid  = entity["entity_id"]
            name = entity["entity_name"]
            progress.update(task, entity_name=name)

            # ── Skip if cached ────────────────────────────────────────────────
            if not args.force and is_already_fetched(eid):
                stats.skip.append({"entity_id": eid, "entity_name": name})
                console.print(f"  [dim]⊘ {name}[/dim]")
                progress.advance(task)
                continue

            try:
                article = fetcher.fetch(entity)
                save_article(article)
                update_manifest(article)
                stats.ok.append({
                    "entity_id":   eid,
                    "entity_name": name,
                    "entity_type": entity["entity_type"],
                    "word_count":  article.word_count,
                    "sections":    len(article.sections),
                })
                console.print(
                    f"  [green]✓[/green] {name:42s} "
                    f"[dim]{article.word_count:6,d} words  "
                    f"{len(article.sections):2d} sections[/dim]"
                )

            except DisambiguationError:
                _record_failure(stats, eid, name,
                    "DISAMBIGUATION — update wikipedia_title in config.py",
                    "disambiguation page", "yellow")

            except ArticleNotFoundError as exc:
                _record_failure(stats, eid, name, str(exc), "page not found")

            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                _record_failure(stats, eid, name, msg, msg)

            progress.advance(task)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(stats)

    sys.exit(0 if not stats.fail else 1)


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(stats: FetchStats) -> None:
    console.print()
    console.rule("[bold]Fetch Summary[/bold]")

    t = Table.grid(padding=(0, 3))
    t.add_row(
        f"[green]✓ Fetched:[/green]  [bold]{len(stats.ok)}[/bold]",
        f"[dim]⊘ Skipped:[/dim]  {len(stats.skip)}",
        f"[red]✗ Failed:[/red]   {len(stats.fail)}",
        f"[cyan]Total words:[/cyan]  {stats.total_words:,}",
    )
    console.print(t)

    if stats.ok:
        console.print()
        wt = Table(
            "Entity", "Type", "Words", "Sections",
            title="Successfully fetched",
            title_style="bold green",
            header_style="bold",
            show_lines=False,
        )
        for r in sorted(stats.ok, key=lambda x: -x["word_count"]):
            wt.add_row(
                r["entity_name"],
                r.get("entity_type", ""),
                f"{r['word_count']:,}",
                str(r["sections"]),
            )
        console.print(wt)

    if stats.fail:
        console.print()
        ft = Table(
            "Entity", "Reason",
            title="[red]Failed entities[/red]",
            header_style="bold red",
            show_lines=True,
        )
        for r in stats.fail:
            ft.add_row(r["entity_name"], r["reason"])
        console.print(ft)
        console.print(
            "[yellow]Tip:[/yellow] Check wikipedia_title values in src/config.py "
            "for the failed entities above."
        )

    # Manifest location
    manifest = load_manifest()
    n_manifest = len(manifest.get("entities", {}))
    console.print(
        f"\n[dim]Manifest updated: data/raw/_manifest.json "
        f"({n_manifest} total entities recorded)[/dim]"
    )


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia articles for the RAG knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--people",
        action="store_true",
        help="Fetch only the 22 people in the catalog",
    )
    scope.add_argument(
        "--places",
        action="store_true",
        help="Fetch only the 22 places in the catalog",
    )
    scope.add_argument(
        "--entity",
        type=str,
        metavar="ENTITY_ID",
        help="Fetch a single entity by its entity_id (e.g. albert_einstein)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch articles even if they are already cached locally",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Fetch only the first N entities (useful for quick smoke tests)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
