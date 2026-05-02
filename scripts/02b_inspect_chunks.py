#!/usr/bin/env python3
"""
Step 2b — Inspect and sanity-check chunked documents.

Reads data/processed/_chunks.jsonl (produced by 02a_chunk_documents.py),
performs several views, runs keyword sanity checks, and writes a final
chunking report to data/processed/_chunking_report.md.

Usage:
    python scripts/02b_inspect_chunks.py
    python scripts/02b_inspect_chunks.py --entity albert_einstein
    python scripts/02b_inspect_chunks.py --no-report
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.ingest import PROCESSED_DIR

console = Console()

CHUNKS_PATH   = PROCESSED_DIR / "_chunks.jsonl"
REPORT_PATH   = PROCESSED_DIR / "_chunking_report.md"

_PREVIEW_LEN  = 280   # characters of content to show in previews
_TOP_N        = 5
_RANDOM_N     = 5


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        console.print(f"[red]Not found: {CHUNKS_PATH}[/red]")
        console.print(
            "Run [bold]python scripts/02a_chunk_documents.py[/bold] first."
        )
        sys.exit(1)
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ── Formatting helpers ────────────────────────────────────────────────────────

def _preview(content: str, limit: int = _PREVIEW_LEN) -> str:
    flat = content.replace("\n", " ")
    if len(flat) > limit:
        return flat[:limit] + "…"
    return flat


def _print_chunk(chunk: dict, idx: int | None = None) -> None:
    label  = f"[bold cyan]{chunk['chunk_id']}[/bold cyan]"
    if idx is not None:
        label = f"[dim]{idx:>3}.[/dim]  {label}"
    console.print(label)
    heading = chunk.get("section_heading") or "(introduction)"
    console.print(f"      Section : [yellow]{heading}[/yellow]")
    console.print(
        f"      Entity  : {chunk['entity_name']} ({chunk['entity_type']})  "
        f"Tokens: {chunk['token_count']}  Words: {chunk['word_count']}"
    )
    console.print(f"      Content : {_preview(chunk['content'])}")
    console.print()


# ── Views ─────────────────────────────────────────────────────────────────────

def _view_random(chunks: list[dict], n: int = _RANDOM_N) -> None:
    console.print(Rule(f"[bold]Random {n} Chunks[/bold]"))
    sample = random.sample(chunks, min(n, len(chunks)))
    for i, c in enumerate(sample, 1):
        _print_chunk(c, idx=i)


def _view_entity(chunks: list[dict], entity_id: str) -> None:
    subset = [c for c in chunks if c.get("entity_id") == entity_id]
    if not subset:
        console.print(f"[yellow]No chunks for entity_id '{entity_id}'[/yellow]")
        return
    console.print(Rule(
        f"[bold]All chunks for {subset[0]['entity_name']} ({entity_id})[/bold]"
    ))
    for i, c in enumerate(subset):
        _print_chunk(c, idx=i)


def _view_extremes(chunks: list[dict], n: int = _TOP_N) -> None:
    sorted_asc  = sorted(chunks, key=lambda c: c["token_count"])
    sorted_desc = sorted(chunks, key=lambda c: c["token_count"], reverse=True)

    console.print(Rule(f"[bold]Top {n} Shortest Chunks[/bold]"))
    for i, c in enumerate(sorted_asc[:n], 1):
        _print_chunk(c, idx=i)

    console.print(Rule(f"[bold]Top {n} Longest Chunks[/bold]"))
    for i, c in enumerate(sorted_desc[:n], 1):
        _print_chunk(c, idx=i)


# ── Sanity checks ─────────────────────────────────────────────────────────────

def _sanity_checks(chunks: list[dict]) -> list[tuple[str, bool, str]]:
    """Return (description, passed, detail) for each sanity assertion."""
    results: list[tuple[str, bool, str]] = []

    # 1. "relativity" in Einstein chunks
    einstein = [c for c in chunks if "einstein" in c.get("entity_id", "").lower()]
    if einstein:
        combined = " ".join(c["content"] for c in einstein).lower()
        found    = "relativity" in combined
        results.append((
            "'relativity' found in Einstein chunks",
            found,
            f"{len(einstein)} chunks checked",
        ))
    else:
        results.append(("'relativity' in Einstein chunks", False, "entity not found"))

    # 2. "Eiffel" in Eiffel Tower chunks
    eiffel = [c for c in chunks if "eiffel" in c.get("entity_id", "").lower()]
    if eiffel:
        combined = " ".join(c["content"] for c in eiffel)
        found    = "Eiffel" in combined
        results.append((
            "'Eiffel' found in Eiffel Tower chunks",
            found,
            f"{len(eiffel)} chunks checked",
        ))
    else:
        results.append(("'Eiffel' in Eiffel Tower chunks", False, "entity not found"))

    # 3. No chunk below MIN_CHUNK_TOKENS=100
    too_short = [c for c in chunks if c["token_count"] < 100]
    results.append((
        "All chunks ≥ 100 tokens",
        len(too_short) == 0,
        f"{len(too_short)} violating chunks" if too_short else "all clear",
    ))

    # 4. No chunk ID collisions
    ids = [c["chunk_id"] for c in chunks]
    dups = len(ids) - len(set(ids))
    results.append((
        "All chunk IDs unique",
        dups == 0,
        f"{dups} duplicates" if dups else "all unique",
    ))

    # 5. All required metadata keys present
    required = {
        "chunk_id", "entity_id", "entity_name", "entity_type", "category",
        "section_heading", "is_intro", "chunk_index", "token_count", "word_count",
        "source_url", "wiki_title", "created_at",
    }
    bad = [c["chunk_id"] for c in chunks if required - c.keys()]
    results.append((
        "All chunks carry required metadata",
        len(bad) == 0,
        f"{len(bad)} missing keys" if bad else "all complete",
    ))

    # 6. created_at is timezone-aware ISO-8601
    bad_ts = []
    for c in chunks:
        try:
            dt = datetime.fromisoformat(c.get("created_at", ""))
            if dt.tzinfo is None:
                bad_ts.append(c["chunk_id"])
        except ValueError:
            bad_ts.append(c["chunk_id"])
    results.append((
        "created_at timestamps are timezone-aware",
        len(bad_ts) == 0,
        f"{len(bad_ts)} bad timestamps" if bad_ts else "all valid",
    ))

    return results


def _print_sanity(results: list[tuple[str, bool, str]]) -> None:
    console.print(Rule("[bold]Sanity Checks[/bold]"))
    t = Table(show_header=True, header_style="bold", show_lines=False)
    t.add_column("Check", ratio=5)
    t.add_column("Status", justify="center", ratio=1)
    t.add_column("Detail", ratio=3)
    for desc, passed, detail in results:
        icon  = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        style = "" if passed else "red"
        t.add_row(desc, icon, detail, style=style)
    console.print(t)
    console.print()


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _gather_stats(chunks: list[dict]) -> dict:
    by_type:   dict[str, list[dict]] = defaultdict(list)
    by_entity: dict[str, list[dict]] = defaultdict(list)

    for c in chunks:
        by_type[c["entity_type"]].append(c)
        by_entity[c["entity_id"]].append(c)

    return {
        "total":     len(chunks),
        "by_type":   dict(by_type),
        "by_entity": dict(by_entity),
        "tokens":    [c["token_count"] for c in chunks],
    }


def _print_stats_table(stats: dict) -> None:
    tokens = stats["tokens"]
    console.print(Panel(
        "\n".join([
            f"  Total chunks : {stats['total']:>6}",
            f"  Mean tokens  : {mean(tokens):>9.1f}",
            f"  Median tokens: {median(tokens):>9.1f}",
            f"  Min tokens   : {min(tokens):>6}",
            f"  Max tokens   : {max(tokens):>6}",
        ]),
        title="[bold]Global Statistics[/bold]",
        border_style="cyan",
    ))

    console.print()
    t = Table(
        "Type", "Entities", "Chunks", "Avg tok / entity", "Avg tok / chunk",
        title="Breakdown by Type",
        title_style="bold",
        header_style="bold",
        show_lines=False,
    )
    for etype, group in sorted(stats["by_type"].items()):
        n_entities = len({c["entity_id"] for c in group})
        toks       = [c["token_count"] for c in group]
        t.add_row(
            etype,
            str(n_entities),
            str(len(group)),
            f"{sum(toks) / n_entities:.0f}",
            f"{mean(toks):.0f}",
        )
    console.print(t)


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_report(stats: dict, sanity: list[tuple[str, bool, str]]) -> None:
    by_type   = stats["by_type"]
    by_entity = stats["by_entity"]
    tokens    = stats["tokens"]

    # Storage estimate: 384-dim float32 vectors
    EMBED_DIMS  = 384
    BYTES_FLOAT = 4
    storage_mb  = stats["total"] * EMBED_DIMS * BYTES_FLOAT / 1024 / 1024

    lines: list[str] = [
        "# Chunking Quality Report\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}  ",
        f"Source: `data/processed/_chunks.jsonl`\n",
        "## Summary\n",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Total chunks | {stats['total']:,} |",
        f"| Mean tokens / chunk | {mean(tokens):.1f} |",
        f"| Median tokens / chunk | {median(tokens):.1f} |",
        f"| Min tokens | {min(tokens)} |",
        f"| Max tokens | {max(tokens)} |",
        f"| Estimated embedding storage (384-dim float32) | {storage_mb:.2f} MB |",
        "",
    ]

    # Per-type breakdown
    lines.append("## Breakdown by Entity Type\n")
    lines.append("| Type | Entities | Chunks | Avg chunks/entity | Avg tok/chunk |")
    lines.append("|------|:--------:|-------:|------------------:|--------------:|")
    for etype, group in sorted(by_type.items()):
        n_ent      = len({c["entity_id"] for c in group})
        n_chunks   = len(group)
        avg_chunks = n_chunks / n_ent if n_ent else 0
        avg_tok    = mean(c["token_count"] for c in group)
        lines.append(
            f"| {etype} | {n_ent} | {n_chunks} | {avg_chunks:.1f} | {avg_tok:.0f} |"
        )
    lines.append("")

    # Per-entity breakdown
    lines.append("## Per-Entity Detail\n")
    lines.append("| Entity | Type | Chunks | Avg tok | Min tok | Max tok |")
    lines.append("|--------|------|-------:|--------:|--------:|--------:|")
    for eid, group in sorted(by_entity.items(), key=lambda x: x[0]):
        toks  = [c["token_count"] for c in group]
        etype = group[0]["entity_type"]
        ename = group[0]["entity_name"]
        lines.append(
            f"| {ename} | {etype} | {len(group)} "
            f"| {mean(toks):.0f} | {min(toks)} | {max(toks)} |"
        )
    lines.append("")

    # Sanity checks
    lines.append("## Sanity Checks\n")
    lines.append("| Check | Status | Detail |")
    lines.append("|-------|:------:|--------|")
    for desc, passed, detail in sanity:
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {desc} | {status} | {detail} |")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[dim]Report written to: {REPORT_PATH}[/dim]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = _parse_args()
    chunks = _load_chunks()

    console.print(Panel(
        f"[bold]Local Wikipedia RAG — Chunk Inspector[/bold]\n"
        f"Source: [cyan]{CHUNKS_PATH}[/cyan]  "
        f"({len(chunks):,} chunks loaded)",
        border_style="blue",
    ))

    stats = _gather_stats(chunks)
    _print_stats_table(stats)
    console.print()

    if args.entity:
        _view_entity(chunks, args.entity)
    else:
        _view_random(chunks)
        console.print()
        _view_extremes(chunks)

    sanity = _sanity_checks(chunks)
    _print_sanity(sanity)

    if not args.no_report:
        _write_report(stats, sanity)

    n_fail = sum(1 for _, passed, _ in sanity if not passed)
    if n_fail:
        console.print(f"[red]{n_fail} sanity check(s) failed.[/red]")
        sys.exit(1)
    console.print("[bold green]All sanity checks passed.[/bold green]")


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect chunked documents and run sanity checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--entity",
        metavar="ENTITY_ID",
        default=None,
        help="Show all chunks for one entity (e.g. albert_einstein)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing _chunking_report.md",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
