#!/usr/bin/env python3
"""
Step 1 — Download Wikipedia articles for all 40 catalog entities.

Usage:
    python scripts/01_fetch_wikipedia.py
    python scripts/01_fetch_wikipedia.py --force     # re-download existing
    python scripts/01_fetch_wikipedia.py --entity albert_einstein
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.progress import track

from src.config import ENTITY_CATALOG
from src.ingest import ingest_all, fetch_article, save_raw, save_processed
from src.utils import configure_logging

console = Console()


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Fetch Wikipedia articles")
    parser.add_argument("--force",  action="store_true", help="Re-download existing articles")
    parser.add_argument("--entity", type=str, default=None, help="Fetch a single entity_id only")
    args = parser.parse_args()

    if args.entity:
        # Single-entity mode
        meta = next((e for e in ENTITY_CATALOG if e["entity_id"] == args.entity), None)
        if meta is None:
            console.print(f"[red]Unknown entity_id: {args.entity}[/red]")
            sys.exit(1)
        text = fetch_article(meta["wikipedia_title"])
        if text:
            save_raw(meta["entity_id"], text)
            save_processed(meta["entity_id"], text)
            console.print(f"[green]✓[/green] {meta['entity_name']}")
        else:
            console.print(f"[red]✗[/red] {meta['entity_name']} — not found on Wikipedia")
        return

    # Full catalog mode
    console.print(f"[bold]Fetching {len(ENTITY_CATALOG)} Wikipedia articles...[/bold]")
    results = ingest_all(force=args.force)

    success = sum(v for v in results.values())
    failed  = [eid for eid, ok in results.items() if not ok]

    console.print(f"\n[green]✓ {success}/{len(ENTITY_CATALOG)} articles fetched[/green]")
    if failed:
        console.print(f"[red]✗ Failed: {', '.join(failed)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
