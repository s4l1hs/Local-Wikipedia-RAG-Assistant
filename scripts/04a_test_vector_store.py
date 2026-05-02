#!/usr/bin/env python3
"""
Step 4a — Smoke-test ChromaVectorStore.

Uses a temporary ChromaDB directory (never touches the production db/).
Embeds 10 hand-crafted sentences with the real Embedder, indexes them,
and runs 4 assertion-backed queries including an empty-store edge case.

Usage:
    python scripts/04a_test_vector_store.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.chunker import Chunk
from src.embedder import Embedder
from src.utils import configure_logging
from src.vector_store import ChromaVectorStore, QueryResult

console = Console()

# ── Test corpus ───────────────────────────────────────────────────────────────
# 5 entities × 2 chunks each = 10 chunks total.
# Designed so queries have a clear "right answer".

_NOW = datetime.now(timezone.utc).isoformat()

def _chunk(eid: str, etype: str, name: str, heading: str, body: str, idx: int) -> Chunk:
    content = f"{heading}. {body}" if heading else body
    return Chunk(
        content=content,
        metadata={
            "chunk_id":        f"{etype}::{eid}::{idx:03d}",
            "entity_id":       eid,
            "entity_name":     name,
            "entity_type":     etype,
            "category":        "test",
            "section_heading": heading,
            "is_intro":        heading == "",
            "chunk_index":     idx,
            "token_count":     len(content.split()),
            "word_count":      len(content.split()),
            "source_url":      f"https://en.wikipedia.org/wiki/{eid}",
            "wiki_title":      eid,
            "created_at":      _NOW,
        },
    )


TEST_CHUNKS: list[Chunk] = [
    # person: albert_einstein
    _chunk("albert_einstein", "person", "Albert Einstein",
           "Early life",
           "Albert Einstein was born in Ulm, Germany in 1879. "
           "He showed an early talent for mathematics and physics.", 0),
    _chunk("albert_einstein", "person", "Albert Einstein",
           "Scientific work",
           "Einstein published the special theory of relativity in 1905. "
           "In 1915 he completed the general theory of relativity.", 1),

    # person: nikola_tesla
    _chunk("nikola_tesla", "person", "Nikola Tesla",
           "Inventions",
           "Nikola Tesla invented the alternating current electrical supply system. "
           "His AC induction motor revolutionised industrial power transmission.", 0),
    _chunk("nikola_tesla", "person", "Nikola Tesla",
           "Later life",
           "Tesla held over 300 patents and worked extensively on wireless transmission. "
           "He demonstrated the first remote-controlled boat in 1898.", 1),

    # person: napoleon_bonaparte
    _chunk("napoleon_bonaparte", "person", "Napoleon Bonaparte",
           "Rise to power",
           "Napoleon Bonaparte seized power in France through the coup of 18 Brumaire. "
           "He was proclaimed Emperor of the French in 1804.", 0),
    _chunk("napoleon_bonaparte", "person", "Napoleon Bonaparte",
           "Military campaigns",
           "Napoleon's Grande Armée conquered much of continental Europe. "
           "His defeat at Waterloo in 1815 ended the Napoleonic era.", 1),

    # place: eiffel_tower
    _chunk("eiffel_tower", "place", "Eiffel Tower",
           "History",
           "The Eiffel Tower was constructed between 1887 and 1889 "
           "as the entrance arch for the 1889 World's Fair in Paris.", 0),
    _chunk("eiffel_tower", "place", "Eiffel Tower",
           "Structure",
           "The iron lattice tower stands 330 metres tall on the Champ de Mars. "
           "It was designed by Gustave Eiffel's engineering company.", 1),

    # place: colosseum
    _chunk("colosseum", "place", "Colosseum",
           "History",
           "The Colosseum is an ancient Roman amphitheatre built between 70 and 80 AD. "
           "It could hold between 50 000 and 80 000 spectators.", 0),
    _chunk("colosseum", "place", "Architecture",
           "Architecture",
           "The Colosseum stands 48 metres high with a base measuring 188 by 156 metres. "
           "Its outer wall used travertine limestone and Roman concrete.", 1),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_results(results: list[QueryResult], label: str) -> None:
    console.print(f"\n[bold cyan]{label}[/bold cyan]")
    if not results:
        console.print("  [dim](no results)[/dim]")
        return
    for i, r in enumerate(results, 1):
        bar = "█" * round(r.score * 20)
        console.print(
            f"  #{i}  [bold]{r.chunk_id}[/bold]  score=[green]{r.score:.3f}[/green]  "
            f"[dim]{bar}[/dim]"
        )
        console.print(f"      {r.entity_name} / {r.section_heading or '(intro)'}")
        preview = r.content[:120].replace("\n", " ")
        console.print(f"      [dim]{preview}…[/dim]")


def _assert(cond: bool, msg: str) -> None:
    if cond:
        console.print(f"  [green]✓[/green]  {msg}")
    else:
        console.print(f"  [red]✗  FAIL: {msg}[/red]")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()

    console.print(Panel(
        "[bold]ChromaVectorStore Smoke Test[/bold]\n"
        "10 chunks  ·  5 entities (3 person + 2 place)  ·  4 queries",
        border_style="blue",
    ))

    embedder = Embedder(cache_dir=None, query_prefix="")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(
            collection_name="smoke_test",
            db_path=Path(tmpdir),
        )

        # ── Edge case: query against empty store ──────────────────────────────
        console.print(Rule("[bold]Edge case: empty store[/bold]"))
        q_vec = embedder.embed_query("alternating current")
        empty_results = store.query(q_vec, top_k=3)
        _assert(empty_results == [], "empty store returns []")
        _assert(store.count() == 0,  "count() == 0 before add()")

        # ── Add all 10 chunks ─────────────────────────────────────────────────
        console.print(Rule("[bold]Indexing 10 test chunks[/bold]"))
        texts     = [c.content for c in TEST_CHUNKS]
        vecs      = embedder.embed_texts(texts, show_progress=False)
        store.add(TEST_CHUNKS, vecs)
        _assert(store.count() == 10, "count() == 10 after add()")

        stats = store.stats()
        console.print(f"  stats: {stats}")
        _assert(stats["by_type"]["person"] == 6, "6 person chunks")
        _assert(stats["by_type"]["place"]  == 4, "4 place chunks")

        # ── Query 1: alternating current → Tesla should dominate ──────────────
        console.print(Rule("[bold]Query 1 — unfiltered[/bold]"))
        q1 = embedder.embed_query("Who invented alternating current electricity?")
        r1 = store.query(q1, top_k=3)
        _print_results(r1, "Q: 'Who invented alternating current electricity?'")
        _assert(len(r1) == 3, "top-3 returned 3 results")
        _assert(
            r1[0].entity_id == "nikola_tesla",
            f"top-1 is Tesla (got {r1[0].entity_id})",
        )
        _assert(r1[0].score > 0.60, f"top-1 score > 0.60 (got {r1[0].score:.3f})")

        # ── Query 2: place filter → no person chunks ──────────────────────────
        console.print(Rule("[bold]Query 2 — filtered by entity_type=place[/bold]"))
        q2 = embedder.embed_query("ancient Roman amphitheatre built in Rome")
        r2 = store.query(q2, top_k=3, filter={"entity_type": "place"})
        _print_results(r2, "Q: 'ancient Roman amphitheatre' (place filter)")
        _assert(all(r.entity_type == "place" for r in r2), "all results are places")
        _assert(
            r2[0].entity_id == "colosseum",
            f"top-1 is Colosseum (got {r2[0].entity_id})",
        )

        # ── Query 3: reranker hook — top_k_fetch > top_k ─────────────────────
        console.print(Rule("[bold]Query 3 — top_k_fetch hook[/bold]"))
        q3 = embedder.embed_query("iron tower Paris France")
        r3 = store.query(q3, top_k=2, top_k_fetch=6)
        _print_results(r3, "Q: 'iron tower Paris' (top_k=2, top_k_fetch=6)")
        _assert(len(r3) == 2, "top_k=2 trims result to 2 items")
        _assert(
            any(r.entity_id == "eiffel_tower" for r in r3),
            "Eiffel Tower in top-2",
        )

        # ── Query 4: relativity → Einstein ───────────────────────────────────
        console.print(Rule("[bold]Query 4 — person filter[/bold]"))
        q4 = embedder.embed_query("theory of relativity space and time")
        r4 = store.query(q4, top_k=3, filter={"entity_type": "person"})
        _print_results(r4, "Q: 'theory of relativity' (person filter)")
        _assert(
            r4[0].entity_id == "albert_einstein",
            f"top-1 is Einstein (got {r4[0].entity_id})",
        )

        # ── Reset and confirm ─────────────────────────────────────────────────
        console.print(Rule("[bold]Reset[/bold]"))
        store.reset()
        _assert(store.count() == 0, "count() == 0 after reset()")
        post_reset = store.query(q1, top_k=3)
        _assert(post_reset == [], "query after reset returns []")

    console.print()
    console.print("[bold green]All assertions passed.[/bold green]")


if __name__ == "__main__":
    main()
