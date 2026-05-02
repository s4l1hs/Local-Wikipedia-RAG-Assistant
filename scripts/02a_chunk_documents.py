#!/usr/bin/env python3
"""
Step 2a — Chunk processed Wikipedia articles into embeddable units.

Reads data/processed/*.json, chunks each document, writes all chunks to
data/processed/_chunks.jsonl (one JSON object per line), and prints a
statistics report.

Usage:
    python scripts/02a_chunk_documents.py
    python scripts/02a_chunk_documents.py --strategy fixed_token
    python scripts/02a_chunk_documents.py --preview albert_einstein
    python scripts/02a_chunk_documents.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.chunker import Chunk, chunk_document
from src.ingest import PROCESSED_DIR
from src.utils import configure_logging, ensure_dir

console = Console()

CHUNKS_PATH = PROCESSED_DIR / "_chunks.jsonl"

# Token distribution buckets for histogram
_BUCKETS: list[tuple[int, int, str]] = [
    (0,    100, "  <100"),
    (100,  200, "100-200"),
    (200,  300, "200-300"),
    (300,  400, "300-400"),
    (400,  600, "400-600"),
    (600, 1000, "600-1k "),
    (1000, 10**9, "  1k+  "),
]
_BAR_W = 36


# ── Histogram ─────────────────────────────────────────────────────────────────

def _histogram(values: list[int]) -> str:
    counts = [sum(1 for v in values if lo <= v < hi) for lo, hi, _ in _BUCKETS]
    max_c  = max(counts) or 1
    lines  = []
    for (_, _, label), count in zip(_BUCKETS, counts):
        filled = round(count / max_c * _BAR_W)
        bar    = "█" * filled + "░" * (_BAR_W - filled)
        lines.append(f"  {label} │ {bar} {count:>3}")
    return "\n".join(lines)


# ── Stats printer ─────────────────────────────────────────────────────────────

def _print_stats(all_chunks: list[Chunk], per_doc: dict[str, list[Chunk]]) -> None:
    tokens = [c.token_count for c in all_chunks]

    # Global stats panel
    min_c = min(all_chunks, key=lambda c: c.token_count)
    max_c = max(all_chunks, key=lambda c: c.token_count)
    console.print(Panel(
        "\n".join([
            f"  Total chunks : {len(all_chunks):>5}",
            f"  Mean tokens  : {mean(tokens):>8.1f}",
            f"  Median tokens: {median(tokens):>8.1f}",
            f"  Shortest     : {min_c.token_count:>5}  ({min_c.chunk_id})",
            f"  Longest      : {max_c.token_count:>5}  ({max_c.chunk_id})",
            "",
            _histogram(tokens),
        ]),
        title="[bold]Chunk Statistics[/bold]",
        border_style="cyan",
    ))

    # Per-document breakdown
    console.print()
    t = Table(
        "Entity", "Type", "Chunks", "Mean tok", "Min tok", "Max tok",
        title="Per-Document Breakdown",
        title_style="bold",
        header_style="bold",
        show_lines=False,
    )
    for entity_id, chunks in sorted(per_doc.items(), key=lambda x: x[0]):
        tc  = [c.token_count for c in chunks]
        typ = chunks[0].metadata["entity_type"]
        t.add_row(
            chunks[0].metadata["entity_name"],
            typ,
            str(len(chunks)),
            f"{mean(tc):.0f}",
            str(min(tc)),
            str(max(tc)),
        )
    console.print(t)


# ── Preview printer ───────────────────────────────────────────────────────────

def _print_preview(chunks: list[Chunk], n: int = 3) -> None:
    console.print()
    console.print(Rule(f"[bold]Chunk Preview — {chunks[0].metadata['entity_name']}[/bold]"))
    for chunk in chunks[:n]:
        meta    = chunk.metadata
        heading = meta["section_heading"] or "(introduction)"
        preview = chunk.content[:300].replace("\n", " ")
        if len(chunk.content) > 300:
            preview += "…"

        console.print(f"\n[bold cyan]{chunk.chunk_id}[/bold cyan]")
        console.print(f"  Section : [yellow]{heading}[/yellow]")
        console.print(f"  Tokens  : {meta['token_count']}  |  Words: {meta['word_count']}")
        console.print(f"  Content : {preview}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()
    args = _parse_args()

    doc_files = sorted(
        f for f in PROCESSED_DIR.glob("*.json")
        if not f.name.startswith("_")
    )
    if not doc_files:
        console.print(f"[red]No JSON files in {PROCESSED_DIR}[/red]")
        console.print(
            "Run [bold]scripts/01_fetch_wikipedia.py[/bold] then "
            "[bold]scripts/01b_validate_data.py[/bold] first."
        )
        sys.exit(1)

    console.print(Panel(
        f"[bold]Local Wikipedia RAG — Document Chunker[/bold]\n"
        f"Source: [cyan]{PROCESSED_DIR}[/cyan]  ({len(doc_files)} files)  |  "
        f"Strategy: [cyan]{args.strategy}[/cyan]  |  "
        f"Dry-run: [cyan]{args.dry_run}[/cyan]",
        border_style="blue",
    ))

    all_chunks: list[Chunk]              = []
    per_doc:    dict[str, list[Chunk]]   = {}
    preview_chunks: list[Chunk]          = []

    for path in doc_files:
        doc    = json.loads(path.read_text(encoding="utf-8"))
        chunks = chunk_document(doc, strategy=args.strategy)

        if not chunks:
            console.print(f"  [yellow]⚠[/yellow]  {doc['name']} — 0 chunks produced")
            continue

        eid = doc["entity_id"]
        per_doc[eid] = chunks
        all_chunks.extend(chunks)

        if args.preview and eid == args.preview:
            preview_chunks = chunks

        console.print(
            f"  [green]✓[/green] {doc['name']:42s} "
            f"[dim]{len(chunks):3d} chunks  "
            f"avg {mean(c.token_count for c in chunks):.0f} tok[/dim]"
        )

    if not all_chunks:
        console.print("[red]No chunks produced. Check your processed documents.[/red]")
        sys.exit(1)

    # Write JSONL
    if not args.dry_run:
        ensure_dir(PROCESSED_DIR)
        with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
            for chunk in all_chunks:
                row = {"content": chunk.content, **chunk.metadata}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        console.print(f"\n[dim]Chunks written to: {CHUNKS_PATH}[/dim]")

    # Stats
    console.print()
    _print_stats(all_chunks, per_doc)

    # Preview
    if args.preview:
        if preview_chunks:
            _print_preview(preview_chunks)
        else:
            console.print(
                f"\n[yellow]Entity '{args.preview}' not found in processed documents.[/yellow]"
            )

    console.print(
        f"\n[bold green]Done.[/bold green]  "
        f"{len(all_chunks)} chunks across {len(per_doc)} documents."
    )


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk processed Wikipedia articles into embeddable units.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy",
        choices=["section_aware", "fixed_token"],
        default="section_aware",
        help="Chunking strategy (default: section_aware)",
    )
    parser.add_argument(
        "--preview",
        metavar="ENTITY_ID",
        default=None,
        help="Print 3 example chunks for this entity (e.g. albert_einstein)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute chunks and print stats without writing _chunks.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
