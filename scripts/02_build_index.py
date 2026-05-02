#!/usr/bin/env python3
"""
Step 2 — Embed chunked articles and index them into ChromaDB.

Prerequisites (run in order):
  1. scripts/01_fetch_wikipedia.py     — fetch raw articles
  2. scripts/01b_validate_data.py      — clean → data/processed/*.json
  3. scripts/02a_chunk_documents.py    — chunk → data/processed/_chunks.jsonl

Usage:
    python scripts/02_build_index.py                        # index all
    python scripts/02_build_index.py --reset                # wipe + rebuild
    python scripts/02_build_index.py --entity albert_einstein
    python scripts/02_build_index.py --limit 20             # first 20 chunks
    python scripts/02_build_index.py --batch-size 16        # smaller GPU batches
    python scripts/02_build_index.py --dry-run              # embed only, no write
    python scripts/02_build_index.py --no-quality-check     # skip sample queries
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from src.chunker import Chunk
from src.config import PROJECT_ROOT
from src.embedder import Embedder, get_embedder
from src.ingest import PROCESSED_DIR
from src.utils import configure_logging
from src.vector_store import ChromaVectorStore, QueryResult, get_vector_store

console = Console()

CHUNKS_PATH = PROCESSED_DIR / "_chunks.jsonl"
DB_PATH     = PROJECT_ROOT / "db"

# Sample queries used for end-to-end quality verification
_QUALITY_CHECKS: list[tuple[str, str]] = [
    ("Einstein",                              "albert_einstein"),
    ("Eiffel Tower iron lattice Paris",       "eiffel_tower"),
    ("scientist who discovered radioactivity","marie_curie"),
]


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_chunks(entity_id: str | None, limit: int | None) -> list[Chunk]:
    """Read _chunks.jsonl; optionally filter by entity and/or cap to limit."""
    if not CHUNKS_PATH.exists():
        console.print(f"[red]Not found: {CHUNKS_PATH}[/red]")
        console.print("Run [bold]scripts/02a_chunk_documents.py[/bold] first.")
        sys.exit(1)

    chunks: list[Chunk] = []
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if limit is not None and len(chunks) >= limit:
                break
            row = json.loads(line)
            if entity_id and row["entity_id"] != entity_id:
                continue
            content = row.pop("content")
            chunks.append(Chunk(content=content, metadata=row))

    if entity_id and not chunks:
        console.print(f"[yellow]Entity '{entity_id}' not in {CHUNKS_PATH}[/yellow]")
        sys.exit(1)

    return chunks


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _db_size_mb(path: Path) -> float:
    """Walk path recursively and sum file sizes in MB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _print_perf(n_chunks: int, elapsed: float) -> None:
    rate = n_chunks / elapsed if elapsed > 0 else 0
    db_mb = _db_size_mb(DB_PATH)
    console.print(Panel(
        "\n".join([
            f"  Chunks embedded  : {n_chunks:>6}",
            f"  Elapsed          : {elapsed:>8.2f} s",
            f"  Throughput       : {rate:>8.1f} chunks/s",
            f"  DB size (disk)   : {db_mb:>8.2f} MB",
        ]),
        title="[bold]Performance[/bold]",
        border_style="cyan",
    ))


# ── Quality check ─────────────────────────────────────────────────────────────

def _run_quality_checks(
    store:    ChromaVectorStore,
    embedder: Embedder,
) -> bool:
    """
    Run a fixed set of sample queries and verify top-1 matches the expected
    entity.  Returns True if all checks pass.
    """
    console.print()
    console.print(Rule("[bold]Quality Check — Sample Queries[/bold]"))

    t = Table(
        "Query", "Expected entity", "Top-1 chunk", "Score", "Pass",
        title="End-to-End Verification",
        title_style="bold",
        header_style="bold",
        show_lines=False,
    )

    all_pass = True
    for query_text, expected_eid in _QUALITY_CHECKS:
        # Skip entities not indexed (--entity / --limit may exclude them)
        if store.count() == 0:
            t.add_row(query_text, expected_eid, "—", "—", "[yellow]SKIP[/yellow]")
            continue

        q_vec   = embedder.embed_query(query_text)
        results = store.query(q_vec, top_k=3)

        if not results:
            t.add_row(query_text, expected_eid, "(no results)", "—", "[red]FAIL[/red]")
            all_pass = False
            continue

        top1       = results[0]
        top1_short = top1.chunk_id          # e.g. person::albert_einstein::007
        passed     = top1.entity_id == expected_eid
        icon       = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"

        if not passed:
            all_pass = False

        t.add_row(
            query_text[:45],
            expected_eid,
            top1_short,
            f"{top1.score:.3f}",
            icon,
        )

        # Print top-3 detail for every query regardless of pass/fail
        console.print(f"\n  [bold]Q:[/bold] {query_text}")
        for i, r in enumerate(results[:3], 1):
            bar = "█" * round(r.score * 20)
            console.print(
                f"    #{i} [cyan]{r.chunk_id}[/cyan]  "
                f"score=[{'green' if i==1 else 'dim'}]{r.score:.3f}[/{'green' if i==1 else 'dim'}]  "
                f"[dim]{bar}[/dim]"
            )
            console.print(
                f"       {r.entity_name} / {r.section_heading or '(intro)'}"
            )
            preview = r.content[:120].replace("\n", " ")
            console.print(f"       [dim]{preview}…[/dim]")

    console.print()
    console.print(t)
    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()
    args = _parse_args()

    console.print(Panel(
        f"[bold]Local Wikipedia RAG — Index Builder[/bold]\n"
        f"Source  : [cyan]{CHUNKS_PATH}[/cyan]\n"
        f"Options : reset={args.reset}  "
        f"limit={args.limit}  "
        f"batch_size={args.batch_size}  "
        f"dry_run={args.dry_run}",
        border_style="blue",
    ))

    # ── Load chunks ───────────────────────────────────────────────────────────
    chunks = _load_chunks(args.entity, args.limit)
    if not chunks:
        console.print("[red]No chunks to index.[/red]")
        sys.exit(1)

    n_entities = len({c.metadata["entity_id"] for c in chunks})
    console.print(
        f"  Loaded [bold]{len(chunks)}[/bold] chunks across "
        f"[bold]{n_entities}[/bold] entities\n"
    )

    # ── Setup embedder + store ────────────────────────────────────────────────
    embedder = Embedder(batch_size=args.batch_size) if args.batch_size else get_embedder()
    store    = get_vector_store()

    if args.reset and not args.dry_run:
        store.reset()
        console.print("  [yellow]Collection reset.[/yellow]\n")

    # ── Embed (timed) ─────────────────────────────────────────────────────────
    console.print("  Embedding…")
    t0   = time.perf_counter()
    vecs = embedder.embed_texts([c.content for c in chunks], show_progress=True)
    elapsed_embed = time.perf_counter() - t0

    # ── Index ─────────────────────────────────────────────────────────────────
    if not args.dry_run:
        store.add(chunks, vecs)

        # Per-entity summary
        by_entity: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_entity.setdefault(c.metadata["entity_id"], []).append(c)

        t_sum = Table(
            "Entity", "Type", "Chunks", "Avg tok", "Min tok", "Max tok",
            title="Indexed",
            title_style="bold",
            header_style="bold",
            show_lines=False,
        )
        for eid, ec in sorted(by_entity.items()):
            toks = [c.metadata["token_count"] for c in ec]
            t_sum.add_row(
                ec[0].metadata["entity_name"],
                ec[0].metadata["entity_type"],
                str(len(ec)),
                f"{mean(toks):.0f}",
                str(min(toks)),
                str(max(toks)),
            )
        console.print()
        console.print(t_sum)

        # DB stats
        s = store.stats()
        console.print(
            f"\n  [dim]ChromaDB: {s['total']} total  "
            f"person={s['by_type'].get('person', 0)}  "
            f"place={s['by_type'].get('place', 0)}[/dim]"
        )

    # ── Performance report ────────────────────────────────────────────────────
    console.print()
    _print_perf(len(chunks), elapsed_embed)

    # ── Quality check ─────────────────────────────────────────────────────────
    if not args.dry_run and not args.no_quality_check:
        ok = _run_quality_checks(store, embedder)
        if not ok:
            console.print(
                "\n[red bold]Quality check FAILED.[/red bold]  "
                "Review the results above before proceeding to the next phase."
            )
            sys.exit(1)
        console.print(
            "\n[bold green]✓ All quality checks passed.[/bold green]  "
            "Pipeline is end-to-end verified."
        )
    else:
        suffix = " (dry-run, not written to ChromaDB)" if args.dry_run else ""
        console.print(f"\n[bold green]Done.[/bold green]{suffix}")


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunked articles and index into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--entity",     metavar="ENTITY_ID", default=None,
                        help="Index a single entity only")
    parser.add_argument("--reset",      action="store_true",
                        help="Wipe the collection before indexing")
    parser.add_argument("--limit",      metavar="N", type=int, default=None,
                        help="Index only the first N chunks (quick smoke test)")
    parser.add_argument("--batch-size", metavar="B", type=int, default=None,
                        help="Embedding batch size (overrides settings.embed_batch_size)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Embed only; do not write to ChromaDB")
    parser.add_argument("--no-quality-check", action="store_true",
                        help="Skip sample query verification at the end")
    return parser.parse_args()


if __name__ == "__main__":
    main()
