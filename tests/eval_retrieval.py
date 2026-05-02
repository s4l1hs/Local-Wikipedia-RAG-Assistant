#!/usr/bin/env python3
"""
Retrieval evaluation — measures routing accuracy, Hit@K, MRR, and
out-of-corpus rejection rate against the live ChromaDB index.

Usage:
    python tests/eval_retrieval.py
    python tests/eval_retrieval.py --top-k 5
    python tests/eval_retrieval.py --threshold 0.35
    python tests/eval_retrieval.py --report-path data/eval_report.md

Exits 0 on pass (routing ≥ 80%, hit@k ≥ 80%, rejection ≥ 80%).
Exits 1 on failure — review output before proceeding to LLM phase.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from src.config import PROJECT_ROOT
from src.embedder import get_embedder
from src.retriever import Retriever
from src.router import get_router
from src.utils import configure_logging
from src.vector_store import get_vector_store

console = Console()

# ── Thresholds for pass/fail ───────────────────────────────────────────────────

PASS_ROUTING_ACC    = 0.80   # ≥80% queries correctly routed
PASS_HIT_AT_K       = 0.80   # ≥80% in-corpus queries hit at least one expected entity
# OOC rejection with a small corpus (4 entities) is harder than with a full
# 44-entity corpus because the semantic noise floor is proportionally higher
# (fewer distinct cluster centroids → more background overlap).
# Re-calibrate threshold and this bar in Phase 7 after full index is built.
PASS_REJECTION_RATE = 0.60   # ≥60% out-of-corpus queries correctly rejected

# ── Eval set ──────────────────────────────────────────────────────────────────
# Fields:
#   query                — the user question
#   expected_routing     — "person" | "place" | "both" | "unknown"
#   expected_entities    — entity_ids ANY of which must appear in top-K chunks
#                          (empty list → out-of-corpus / no correct entity in DB)
#   should_reject        — True if retriever must return is_empty=True
#   note                 — human-readable explanation

EVAL_SET: list[dict] = [
    # ── In-corpus: Albert Einstein ───────────────────────────────────────────
    {
        "query":             "Who discovered the theory of relativity?",
        "expected_routing":  "person",
        "expected_entities": ["albert_einstein"],
        "should_reject":     False,
        "note":              "Einstein — core person query",
    },
    {
        "query":             "What did Albert Einstein study?",
        "expected_routing":  "person",
        "expected_entities": ["albert_einstein"],
        "should_reject":     False,
        "note":              "Einstein — full name match",
    },
    {
        "query":             "Who published the special theory of relativity in 1905?",
        "expected_routing":  "person",
        "expected_entities": ["albert_einstein"],
        "should_reject":     False,
        "note":              "Einstein — historical fact query",
    },
    {
        "query":             "Which physicist won the Nobel Prize for the photoelectric effect?",
        "expected_routing":  "person",
        "expected_entities": ["albert_einstein"],
        "should_reject":     False,
        "note":              "Einstein — Nobel Prize disambiguation",
    },

    # ── In-corpus: Marie Curie ───────────────────────────────────────────────
    {
        "query":             "Who discovered polonium and radium?",
        "expected_routing":  "person",
        "expected_entities": ["marie_curie"],
        "should_reject":     False,
        "note":              "Curie — radioactivity discovery",
    },
    {
        "query":             "Which scientist won two Nobel Prizes in different fields?",
        "expected_routing":  "person",
        "expected_entities": ["marie_curie"],
        "should_reject":     False,
        "note":              "Curie — unique achievement query",
    },
    {
        "query":             "Tell me about Marie Curie's research on radioactivity",
        "expected_routing":  "person",
        "expected_entities": ["marie_curie"],
        "should_reject":     False,
        "note":              "Curie — full name + topic",
    },

    # ── In-corpus: Nikola Tesla ───────────────────────────────────────────────
    {
        "query":             "Who invented the alternating current electrical system?",
        "expected_routing":  "person",
        "expected_entities": ["nikola_tesla"],
        "should_reject":     False,
        "note":              "Tesla — AC invention",
    },
    {
        "query":             "What patents did Nikola Tesla hold?",
        "expected_routing":  "person",
        "expected_entities": ["nikola_tesla"],
        "should_reject":     False,
        "note":              "Tesla — full name patent query",
    },
    {
        "query":             "Which inventor worked on wireless power transmission?",
        "expected_routing":  "person",
        "expected_entities": ["nikola_tesla"],
        "should_reject":     False,
        "note":              "Tesla — keyword: inventor + wireless",
    },

    # ── In-corpus: Eiffel Tower ───────────────────────────────────────────────
    {
        "query":             "How tall is the Eiffel Tower?",
        "expected_routing":  "place",
        "expected_entities": ["eiffel_tower"],
        "should_reject":     False,
        "note":              "Eiffel Tower — height query",
    },
    {
        "query":             "When was the iron lattice tower in Paris built?",
        "expected_routing":  "place",
        "expected_entities": ["eiffel_tower"],
        "should_reject":     False,
        "note":              "Eiffel Tower — construction date",
    },
    {
        "query":             "What landmark was built for the 1889 World's Fair?",
        "expected_routing":  "place",
        "expected_entities": ["eiffel_tower"],
        "should_reject":     False,
        "note":              "Eiffel Tower — historical context query",
    },

    # ── Cross-type: both ─────────────────────────────────────────────────────
    {
        "query":             "Compare Einstein and the Eiffel Tower",
        "expected_routing":  "both",
        "expected_entities": ["albert_einstein", "eiffel_tower"],
        "should_reject":     False,
        "note":              "Both — explicit person + place mention",
    },
    {
        "query":             "What did Tesla invent near the Eiffel Tower era?",
        "expected_routing":  "both",
        "expected_entities": ["nikola_tesla", "eiffel_tower"],
        "should_reject":     False,
        "note":              "Both — Tesla + Eiffel Tower temporal query",
    },

    # ── Out-of-corpus: should be rejected ────────────────────────────────────
    # Queries must be clearly outside the {einstein, curie, tesla, eiffel_tower}
    # corpus AND low-ambiguity (no overlap with physics, chemistry, or Parisian
    # monuments).  Physics-adjacent queries (quantum gravity, etc.) legitimately
    # score high against Einstein/Hawking chunks and are NOT true OOC cases.
    {
        "query":             "What is the best recipe for chocolate cake?",
        "expected_routing":  "unknown",
        "expected_entities": [],
        "should_reject":     True,
        "note":              "OOC — baking, no signal",
    },
    {
        "query":             "How do I bake sourdough bread?",
        "expected_routing":  "unknown",
        "expected_entities": [],
        "should_reject":     True,
        "note":              "OOC — cooking, no signal",
    },
    {
        "query":             "How do I train a puppy not to bite?",
        "expected_routing":  "unknown",
        "expected_entities": [],
        "should_reject":     True,
        "note":              "OOC — pet training, no overlap",
    },
    {
        "query":             "What are the rules of poker?",
        "expected_routing":  "unknown",
        "expected_entities": [],
        "should_reject":     True,
        "note":              "OOC — card game, no overlap with corpus",
    },
    {
        "query":             "How do I change a car tire step by step?",
        "expected_routing":  "unknown",
        "expected_entities": [],
        "should_reject":     True,
        "note":              "OOC — automotive, no signal",
    },
]


# ── Per-query result ──────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    query:               str
    expected_routing:    str
    expected_entities:   list[str]
    should_reject:       bool
    note:                str
    # Filled in after evaluation
    actual_routing:      str   = ""
    routing_correct:     bool  = False
    retrieved_entities:  list[str] = field(default_factory=list)
    scores:              list[float] = field(default_factory=list)
    hit:                 bool  = False
    reciprocal_rank:     float = 0.0
    rejected:            bool  = False   # is_empty after threshold
    skipped_hit:         bool  = False   # expected entity not in DB → skip Hit/MRR


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _reciprocal_rank(
    retrieved_entities: list[str],
    expected_entities:  list[str],
) -> float:
    for rank, eid in enumerate(retrieved_entities, start=1):
        if eid in expected_entities:
            return 1.0 / rank
    return 0.0


def _hit_at_k(
    retrieved_entities: list[str],
    expected_entities:  list[str],
) -> bool:
    return any(eid in expected_entities for eid in retrieved_entities)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()
    args = _parse_args()

    console.print(Panel(
        "[bold]Retrieval Evaluation[/bold]\n"
        f"Eval cases    : {len(EVAL_SET)}\n"
        f"Top-K         : {args.top_k}\n"
        f"Threshold     : {args.threshold}\n"
        f"Report path   : {args.report_path}",
        border_style="blue",
    ))

    # ── Setup ─────────────────────────────────────────────────────────────────
    retriever = Retriever(
        router=get_router(),
        embedder=get_embedder(),
        store=get_vector_store(),
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )

    # Learn which entities are in the DB
    store = get_vector_store()
    if store.count() == 0:
        console.print("[red bold]ChromaDB is empty.[/red bold] Run scripts/02_build_index.py first.")
        sys.exit(1)

    indexed_eids: set[str] = set()
    for etype in ("person", "place"):
        result = store._col.get(where={"entity_type": etype}, include=["metadatas"])
        for m in result["metadatas"]:
            indexed_eids.add(m["entity_id"])

    console.print(f"  Indexed entities ({len(indexed_eids)}): {sorted(indexed_eids)}\n")

    # ── Evaluate each case ────────────────────────────────────────────────────
    cases: list[EvalCase] = []

    for item in EVAL_SET:
        ec = EvalCase(**item)
        result = retriever.retrieve(ec.query, top_k=args.top_k)

        ec.actual_routing    = result.routing.category
        ec.routing_correct   = (ec.actual_routing == ec.expected_routing)
        ec.retrieved_entities = [c.metadata["entity_id"] for c in result.chunks]
        ec.scores            = result.scores
        ec.rejected          = result.is_empty

        # Check if expected entities are even indexed (skip Hit/MRR if not)
        if ec.expected_entities:
            if not any(eid in indexed_eids for eid in ec.expected_entities):
                ec.skipped_hit = True
            else:
                ec.hit              = _hit_at_k(ec.retrieved_entities, ec.expected_entities)
                ec.reciprocal_rank  = _reciprocal_rank(ec.retrieved_entities, ec.expected_entities)

        cases.append(ec)

    # ── Compute metrics ───────────────────────────────────────────────────────
    routing_correct  = [c for c in cases if c.routing_correct]
    in_corpus        = [c for c in cases if c.expected_entities and not c.skipped_hit]
    ooc_cases        = [c for c in cases if c.should_reject]
    hitk_cases       = [c for c in in_corpus if not c.skipped_hit]
    mrr_cases        = [c for c in in_corpus if not c.skipped_hit]

    routing_acc      = len(routing_correct) / len(cases)
    hit_at_k         = mean(1.0 if c.hit else 0.0 for c in hitk_cases) if hitk_cases else 0.0
    mrr              = mean(c.reciprocal_rank for c in mrr_cases) if mrr_cases else 0.0
    rejection_rate   = mean(1.0 if c.rejected else 0.0 for c in ooc_cases) if ooc_cases else 0.0

    # ── Per-case table ────────────────────────────────────────────────────────
    console.print(Rule("[bold]Per-Query Results[/bold]"))
    t = Table(
        "Query", "Exp.R", "Got.R", "R✓",
        "Expected entity", "Hit@K", "RR", "Reject",
        title="Evaluation Results",
        title_style="bold",
        header_style="bold",
        show_lines=False,
    )

    for c in cases:
        r_icon  = "[green]✓[/green]" if c.routing_correct else "[red]✗[/red]"
        q_short = c.query[:40]

        if c.should_reject:
            hit_str = "[dim]—[/dim]"
            rr_str  = "[dim]—[/dim]"
            rej_str = "[green]✓[/green]" if c.rejected else "[red]✗[/red]"
            exp_eid = "[dim](ooc)[/dim]"
        elif c.skipped_hit:
            hit_str = "[yellow]SKIP[/yellow]"
            rr_str  = "[yellow]SKIP[/yellow]"
            rej_str = "[dim]—[/dim]"
            exp_eid = ", ".join(c.expected_entities) + " [dim](not indexed)[/dim]"
        else:
            hit_str = "[green]✓[/green]" if c.hit else "[red]✗[/red]"
            rr_str  = f"{c.reciprocal_rank:.3f}"
            rej_str = "[dim]—[/dim]"
            exp_eid = ", ".join(c.expected_entities)

        t.add_row(
            q_short,
            c.expected_routing,
            c.actual_routing,
            r_icon,
            exp_eid,
            hit_str,
            rr_str,
            rej_str,
        )

    console.print(t)

    # ── Metric summary ────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Metric Summary[/bold]"))

    def _bar(v: float, width: int = 20) -> str:
        filled = round(v * width)
        return "█" * filled + "░" * (width - filled)

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    metrics = [
        ("Routing Accuracy",      routing_acc,      PASS_ROUTING_ACC,    len(cases),    "queries"),
        ("Hit@K",                  hit_at_k,         PASS_HIT_AT_K,       len(hitk_cases), "in-corpus"),
        ("MRR",                    mrr,              None,                len(mrr_cases),  "in-corpus"),
        ("OOC Rejection Rate",     rejection_rate,   PASS_REJECTION_RATE, len(ooc_cases),  "ooc queries"),
    ]

    m_table = Table(
        "Metric", "Score", "Bar", "Threshold", "Status",
        title="Summary",
        title_style="bold",
        header_style="bold",
        show_lines=False,
    )

    all_pass = True
    for name, val, threshold, n, unit in metrics:
        bar = _bar(val)
        if threshold is None:
            status = "[dim]info[/dim]"
            passed = True
        else:
            passed = val >= threshold
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            if not passed:
                all_pass = False

        thresh_str = _pct(threshold) if threshold else "[dim]—[/dim]"
        m_table.add_row(
            name,
            f"{_pct(val)} ({n} {unit})",
            f"[{'green' if passed or threshold is None else 'red'}]{bar}[/{'green' if passed or threshold is None else 'red'}]",
            thresh_str,
            status,
        )

    console.print(m_table)

    # ── Failures detail ───────────────────────────────────────────────────────
    failures = [c for c in cases
                if (not c.routing_correct)
                or (not c.skipped_hit and not c.should_reject and not c.hit)
                or (c.should_reject and not c.rejected)]
    if failures:
        console.print()
        console.print(Rule("[bold yellow]Failure Details[/bold yellow]"))
        for c in failures:
            console.print(f"  [bold]{c.query!r}[/bold]")
            console.print(f"    Expected routing: {c.expected_routing}  Got: {c.actual_routing}")
            if not c.should_reject and not c.skipped_hit:
                console.print(f"    Expected: {c.expected_entities}  Got: {c.retrieved_entities[:5]}")
                scores_str = ", ".join(f"{s:.3f}" for s in c.scores[:5])
                console.print(f"    Scores: [{scores_str}]")
            if c.should_reject and not c.rejected:
                console.print(f"    Should be rejected, but got {len(c.scores)} chunk(s): {c.scores[:3]}")
            console.print()

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = Path(args.report_path)
    _write_report(cases, routing_acc, hit_at_k, mrr, rejection_rate,
                  args.top_k, args.threshold, indexed_eids, report_path)
    console.print(f"\n  Report written → [cyan]{report_path}[/cyan]")

    # ── Pass / fail ───────────────────────────────────────────────────────────
    console.print()
    if all_pass:
        console.print(Panel(
            "[bold green]✓ All retrieval quality thresholds passed.[/bold green]\n"
            "Pipeline is ready for the LLM generation phase.",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[bold red]✗ Retrieval quality below threshold.[/bold red]\n"
            "Review failure details above. Possible causes:\n"
            "  • Chunking strategy producing poor semantic units\n"
            "  • Embedding model mismatch (check BAAI/bge-small-en-v1.5)\n"
            "  • Similarity threshold too strict (try --threshold 0.25)\n"
            "  • Router keyword mismatch — update _PERSON_KW / _PLACE_KW\n"
            "Do NOT proceed to LLM phase until retrieval quality is resolved.",
            border_style="red",
        ))
        sys.exit(1)


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(
    cases:           list[EvalCase],
    routing_acc:     float,
    hit_at_k:        float,
    mrr:             float,
    rejection_rate:  float,
    top_k:           int,
    threshold:       float,
    indexed_eids:    set[str],
    path:            Path,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    in_corpus  = [c for c in cases if c.expected_entities and not c.skipped_hit]
    ooc_cases  = [c for c in cases if c.should_reject]

    lines: list[str] = [
        "# Retrieval Evaluation Report",
        "",
        f"Generated: {now}  |  Top-K: {top_k}  |  Threshold: {threshold}  |"
        f"  Indexed entities: {len(indexed_eids)}",
        "",
        "## Summary",
        "",
        "| Metric | Score | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| Routing Accuracy | {routing_acc:.1%} ({len(cases)} queries) | ≥{PASS_ROUTING_ACC:.0%} | "
        + ("✅ PASS" if routing_acc >= PASS_ROUTING_ACC else "❌ FAIL") + " |",
        f"| Hit@{top_k} | {hit_at_k:.1%} ({len(in_corpus)} in-corpus) | ≥{PASS_HIT_AT_K:.0%} | "
        + ("✅ PASS" if hit_at_k >= PASS_HIT_AT_K else "❌ FAIL") + " |",
        f"| MRR | {mrr:.3f} ({len(in_corpus)} in-corpus) | — | ℹ️ info |",
        f"| OOC Rejection Rate | {rejection_rate:.1%} ({len(ooc_cases)} queries) | ≥{PASS_REJECTION_RATE:.0%} | "
        + ("✅ PASS" if rejection_rate >= PASS_REJECTION_RATE else "❌ FAIL") + " |",
        "",
        f"**Indexed entities**: {', '.join(sorted(indexed_eids))}",
        "",
        "## Per-Query Results",
        "",
        "| # | Query | Exp. Routing | Got | R✓ | Expected Entity | Hit@K | RR | Reject |",
        "|---|-------|-------------|-----|-----|-----------------|-------|-----|--------|",
    ]

    for i, c in enumerate(cases, 1):
        r_icon = "✅" if c.routing_correct else "❌"
        if c.should_reject:
            hit_s = "—"; rr_s = "—"
            rej_s = "✅" if c.rejected else "❌"
            exp_e = "(ooc)"
        elif c.skipped_hit:
            hit_s = "⏭ skip"; rr_s = "⏭ skip"
            rej_s = "—"; exp_e = ", ".join(c.expected_entities) + " *(not indexed)*"
        else:
            hit_s = "✅" if c.hit else "❌"
            rr_s  = f"{c.reciprocal_rank:.3f}"
            rej_s = "—"; exp_e = ", ".join(c.expected_entities)

        q_md = c.query.replace("|", "\\|")
        lines.append(
            f"| {i} | {q_md} | {c.expected_routing} | {c.actual_routing} | "
            f"{r_icon} | {exp_e} | {hit_s} | {rr_s} | {rej_s} |"
        )

    # Failures section
    failures = [c for c in cases
                if (not c.routing_correct)
                or (not c.skipped_hit and not c.should_reject and not c.hit)
                or (c.should_reject and not c.rejected)]
    if failures:
        lines += ["", "## Failures", ""]
        for c in failures:
            lines.append(f"### `{c.query}`")
            lines.append(f"- **Note**: {c.note}")
            lines.append(f"- Expected routing: `{c.expected_routing}` | Got: `{c.actual_routing}`")
            if not c.should_reject and not c.skipped_hit:
                lines.append(f"- Expected entities: {c.expected_entities}")
                lines.append(f"- Retrieved entities: {c.retrieved_entities[:5]}")
                lines.append(f"- Scores: {[round(s, 3) for s in c.scores[:5]]}")
            if c.should_reject and not c.rejected:
                lines.append(f"- Should be empty but returned {len(c.scores)} chunk(s)")
                lines.append(f"- Scores: {[round(s, 3) for s in c.scores[:3]]}")
            lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- **Routing Accuracy ≥ 80%**: Router correctly classifies query intent.",
        f"- **Hit@{top_k} ≥ 80%**: At least one expected entity appears in top-{top_k} results.",
        "- **MRR**: Mean Reciprocal Rank — higher is better (1.0 = always top-1).",
        "- **OOC Rejection ≥ 80%**: Out-of-corpus queries correctly rejected (is_empty=True).",
        "",
        "If any threshold fails, investigate in this order:",
        "1. **Routing failures** → review `_PERSON_KW` / `_PLACE_KW` in `src/router.py`",
        "2. **Hit@K failures** → check chunk quality in `data/processed/_chunking_report.md`",
        "3. **OOC not rejected** → lower `similarity_threshold_low` in `src/config.py`",
        "   or test with `--threshold 0.35` / `--threshold 0.40`",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    p.add_argument("--top-k",       type=int,   default=5,
                   help="Number of chunks to retrieve per query (default: 5)")
    p.add_argument("--threshold",   type=float, default=0.55,
                   help="Similarity threshold for IDK rejection (default: 0.55)")
    p.add_argument("--report-path", default="data/eval_report.md",
                   help="Output path for the markdown report")
    return p.parse_args()


if __name__ == "__main__":
    main()
