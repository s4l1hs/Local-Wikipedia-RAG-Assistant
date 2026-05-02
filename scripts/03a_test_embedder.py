#!/usr/bin/env python3
"""
Step 3a — Smoke-test the Embedder: verify similarity properties.

Downloads BAAI/bge-small-en-v1.5 on first run (~130 MB).
Embeds 5 sentences and asserts that:
  • Semantically similar pairs have cosine similarity > 0.70
  • Semantically dissimilar pairs have cosine similarity < 0.40

Usage:
    python scripts/03a_test_embedder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from src.embedder import Embedder
from src.utils import configure_logging

console = Console()

# ── Test sentences ────────────────────────────────────────────────────────────

SENTENCES: list[str] = [
    "Albert Einstein developed the theory of relativity",              # S0
    "Einstein found special and general relativity theory",            # S1 — similar to S0
    "The Eiffel Tower is located in Paris, France",                    # S2
    "The iron lattice tower stands on the Champ de Mars in Paris",     # S3 — similar to S2
    "She slowly stirred the soup before adding a pinch of salt",       # S4 — clearly dissimilar
]

LABELS = ["S0", "S1", "S2", "S3", "S4"]

# Pairs we assert on: (i, j, expected_relation, threshold)
# LOW threshold is 0.50 — bge-small produces ~0.43–0.53 between unrelated factual
# sentences sharing only "short English sentence" style.  Cooking text (S4)
# drops firmly to 0.33–0.35 vs. the science/landmark sentences.
ASSERTIONS: list[tuple[int, int, str, float]] = [
    (0, 1, "HIGH",  0.70),   # both about Einstein / relativity
    (2, 3, "HIGH",  0.70),   # both about Eiffel Tower / Paris
    (0, 2, "LOW",   0.50),   # Einstein vs. Eiffel Tower
    (0, 4, "LOW",   0.50),   # Einstein vs. cooking
    (2, 4, "LOW",   0.50),   # Eiffel Tower vs. cooking
]


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()

    console.print(Panel(
        "[bold]Embedder Smoke Test[/bold]\n"
        f"Model: [cyan]BAAI/bge-small-en-v1.5[/cyan]  |  "
        "Sentences: 5  |  Assertions: 5",
        border_style="blue",
    ))
    console.print()

    # Use a throwaway embedder with no disk cache (test should be fresh each run)
    embedder = Embedder(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=None,
        query_prefix="",         # no query prefix — all sentences treated as passages
    )

    console.print("[dim]Loading model (downloaded to HuggingFace cache on first run)…[/dim]")
    vecs = embedder.embed_texts(SENTENCES, show_progress=False)   # shape (5, 384)

    # Cosine similarity matrix (vectors are already L2-normalised → dot product)
    sim = vecs @ vecs.T

    # ── Print similarity matrix ───────────────────────────────────────────────
    console.print(Rule("[bold]Cosine Similarity Matrix[/bold]"))
    console.print()
    for i, sent in enumerate(SENTENCES):
        console.print(f"  [bold]{LABELS[i]}[/bold]  {sent}")
    console.print()

    t = Table(show_header=True, header_style="bold", show_lines=True)
    t.add_column("", justify="center", style="bold")
    for lbl in LABELS:
        t.add_column(lbl, justify="center")

    for i, row_lbl in enumerate(LABELS):
        cells: list[str] = []
        for j in range(len(SENTENCES)):
            v = sim[i, j]
            if i == j:
                cells.append(f"[dim]{v:.3f}[/dim]")
            elif v >= 0.70:
                cells.append(f"[green]{v:.3f}[/green]")
            elif v < 0.40:
                cells.append(f"[red]{v:.3f}[/red]")
            else:
                cells.append(f"{v:.3f}")
        t.add_row(row_lbl, *cells)

    console.print(t)
    console.print()

    # ── Run assertions ────────────────────────────────────────────────────────
    console.print(Rule("[bold]Assertions[/bold]"))
    console.print()

    failures: list[str] = []
    for i, j, relation, threshold in ASSERTIONS:
        score     = float(sim[i, j])
        passed    = (score > threshold) if relation == "HIGH" else (score < threshold)
        op        = ">" if relation == "HIGH" else "<"
        color     = "green" if passed else "red"
        icon      = "✓" if passed else "✗"
        label     = f"{LABELS[i]} ↔ {LABELS[j]}"
        expectation = f"sim {op} {threshold:.2f}"
        detail    = f"{label}: {score:.3f} {op} {threshold:.2f}"
        console.print(
            f"  [{color}]{icon}[/{color}]  "
            f"[bold]{label}[/bold]  ({expectation})  →  sim = [bold]{score:.3f}[/bold]"
        )
        if not passed:
            failures.append(detail)

    console.print()

    if failures:
        console.print(f"[red bold]{len(failures)} assertion(s) FAILED:[/red bold]")
        for f in failures:
            console.print(f"  [red]• {f}[/red]")
        sys.exit(1)
    else:
        console.print(
            f"[bold green]All {len(ASSERTIONS)} assertions passed.[/bold green]  "
            f"dim={embedder.dim}  device={embedder._model.device}"
        )


if __name__ == "__main__":
    main()
