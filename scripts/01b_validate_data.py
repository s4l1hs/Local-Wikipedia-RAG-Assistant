#!/usr/bin/env python3
"""
Step 1b — Validate and clean raw Wikipedia articles.

Reads every JSON in data/raw/, runs quality checks, applies deep cleaning
(HTML decode, NFC normalize, empty-section pruning), writes cleaned output
to data/processed/, and emits a quality report.

Usage:
    python scripts/01b_validate_data.py             # validate + clean + report
    python scripts/01b_validate_data.py --no-clean  # validate + report only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.ingest import (
    ArticleData,
    PROCESSED_DIR,
    RAW_DIR,
    Section,
    clean_article,
    save_article,
)
from src.utils import configure_logging, ensure_dir

console = Console()

QUALITY_REPORT_PATH = PROCESSED_DIR / "_quality_report.md"

_WORD_MIN = 500
_WORD_MAX = 50_000

# ── Issue patterns ────────────────────────────────────────────────────────────

# Common UTF-8-as-Latin1 mojibake sequences
_MOJIBAKE_RE = re.compile(r"Ã[©¨àÃ®°¼½¾]|â€[™œ]|Ã\xa9")

# Table leftovers: line-start pipe or double-pipe cell separator
_TABLE_RE = re.compile(r"(?m)^\s*\||\|\|")

_DISAMBIG_MARKERS = (
    "may refer to:",
    "may mean:",
    "can refer to:",
    "is a disambiguation",
    "refers to multiple",
)


# ── Per-document result ───────────────────────────────────────────────────────

@dataclass
class DocReport:
    entity_id:   str
    entity_name: str
    entity_type: str
    raw_words:   int
    clean_words: int
    sections:    int
    issues:      list[str] = field(default_factory=list)

    @property
    def critical(self) -> bool:
        return any(i.startswith("EMPTY_TEXT") for i in self.issues)


# ── Checkers ──────────────────────────────────────────────────────────────────

def _detect_issues(raw_text: str, word_count: int) -> list[str]:
    issues: list[str] = []
    if not raw_text.strip():
        issues.append("EMPTY_TEXT")
        return issues  # remaining checks are meaningless on empty text
    if word_count < _WORD_MIN:
        issues.append(f"SHORT ({word_count} words)")
    if word_count > _WORD_MAX:
        issues.append(f"LONG ({word_count:,} words)")
    if _MOJIBAKE_RE.search(raw_text):
        issues.append("MOJIBAKE")
    if any(m in raw_text.lower() for m in _DISAMBIG_MARKERS):
        issues.append("DISAMBIGUATION_LINK")
    if _TABLE_RE.search(raw_text):
        issues.append("TABLE_REMNANT")
    return issues


# ── ASCII histogram ───────────────────────────────────────────────────────────

_BUCKETS: list[tuple[int, int, str]] = [
    (0,       1_000, "  0-1k"),
    (1_000,   2_000, "  1-2k"),
    (2_000,   5_000, "  2-5k"),
    (5_000,  10_000, " 5-10k"),
    (10_000, 20_000, "10-20k"),
    (20_000, 50_000, "20-50k"),
    (50_000, 10**9,  "  50k+"),
]
_BAR_WIDTH = 36


def _ascii_histogram(values: list[int]) -> str:
    counts = [sum(1 for v in values if lo <= v < hi) for lo, hi, _ in _BUCKETS]
    max_c  = max(counts) or 1
    lines  = []
    for (_, _, label), count in zip(_BUCKETS, counts):
        filled = round(count / max_c * _BAR_WIDTH)
        bar    = "█" * filled + "░" * (_BAR_WIDTH - filled)
        lines.append(f"  {label} │ {bar} {count:>2}")
    return "\n".join(lines)


# ── Report writers ────────────────────────────────────────────────────────────

def _print_console_report(reports: list[DocReport]) -> None:
    wc = [r.clean_words for r in reports]
    n_issues = sum(1 for r in reports if r.issues)

    console.print(Panel(
        "\n".join([
            f"  Min:    {min(wc):>8,} words",
            f"  Max:    {max(wc):>8,} words",
            f"  Mean:   {mean(wc):>8,.0f} words",
            f"  Median: {median(wc):>8,.0f} words",
            "",
            _ascii_histogram(wc),
        ]),
        title="[bold]Word Count Distribution[/bold]",
        border_style="cyan",
    ))

    console.print()
    t = Table(
        "Document", "Type", "Words", "Sections", "Issues",
        title=f"Quality Checks — {len(reports)} documents · {n_issues} with issues",
        title_style="bold",
        header_style="bold",
        show_lines=False,
    )
    for r in sorted(reports, key=lambda x: x.entity_name):
        style      = "red" if r.critical else ("yellow" if r.issues else "")
        issues_str = ", ".join(r.issues) if r.issues else "[dim]—[/dim]"
        t.add_row(
            r.entity_name,
            r.entity_type,
            f"{r.clean_words:,}",
            str(r.sections),
            issues_str,
            style=style,
        )
    console.print(t)


def _write_md_report(reports: list[DocReport]) -> None:
    ensure_dir(PROCESSED_DIR)
    wc       = [r.clean_words for r in reports]
    n_issues = sum(1 for r in reports if r.issues)

    lines: list[str] = [
        "# Data Quality Report\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}  ",
        f"Source: `data/raw/` — {len(reports)} documents  ",
        f"Issues: {n_issues} / {len(reports)} documents\n",
        "## Word Count Statistics\n",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Min    | {min(wc):>8,} |",
        f"| Max    | {max(wc):>8,} |",
        f"| Mean   | {mean(wc):>8,.0f} |",
        f"| Median | {median(wc):>8,.0f} |",
        "",
        "## Distribution\n",
        "```",
        _ascii_histogram(wc),
        "```\n",
        "## Per-Document Checks\n",
        "| Document | Type | Words | Sections | Issues |",
        "|----------|------|------:|---------:|--------|",
    ]
    for r in sorted(reports, key=lambda x: x.entity_name):
        issues_str = ", ".join(r.issues) if r.issues else "—"
        lines.append(
            f"| {r.entity_name} | {r.entity_type} "
            f"| {r.clean_words:,} | {r.sections} | {issues_str} |"
        )

    QUALITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()
    args = _parse_args()

    raw_files = sorted(f for f in RAW_DIR.glob("*.json") if not f.name.startswith("_"))
    if not raw_files:
        console.print(f"[red]No JSON files in {RAW_DIR}[/red]")
        console.print("Run [bold]scripts/01_fetch_wikipedia.py[/bold] first.")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Local Wikipedia RAG — Data Validator[/bold]\n"
        f"Source: [cyan]{RAW_DIR}[/cyan]  ({len(raw_files)} files)  |  "
        f"Write cleaned: [cyan]{not args.no_clean}[/cyan]",
        border_style="blue",
    ))

    reports: list[DocReport] = []

    for path in raw_files:
        raw       = json.loads(path.read_text(encoding="utf-8"))
        raw_text  = raw.get("raw_text", "")
        raw_words = raw.get("word_count", len(raw_text.split()))
        issues    = _detect_issues(raw_text, raw_words)

        article = ArticleData(
            entity_id   = raw["entity_id"],
            entity_name = raw["name"],
            entity_type = raw["type"],
            category    = raw.get("category", ""),
            wiki_title  = raw.get("wiki_title", ""),
            url         = raw.get("url", ""),
            fetched_at  = raw.get("fetched_at", ""),
            sections    = [
                Section(heading=s["heading"], text=s["text"])
                for s in raw.get("sections", [])
            ],
            word_count  = raw_words,
        )
        cleaned = clean_article(article)

        if not args.no_clean:
            save_article(cleaned, out_dir=PROCESSED_DIR)

        # Log progress for issues
        if issues:
            console.print(
                f"  [yellow]⚠[/yellow] {raw['name']:42s} {', '.join(issues)}"
            )
        else:
            console.print(f"  [green]✓[/green] {raw['name']}")

        reports.append(DocReport(
            entity_id   = raw["entity_id"],
            entity_name = raw["name"],
            entity_type = raw["type"],
            raw_words   = raw_words,
            clean_words = cleaned.word_count,
            sections    = len(cleaned.sections),
            issues      = issues,
        ))

    console.print()
    _print_console_report(reports)
    _write_md_report(reports)

    console.print(f"\n[dim]Report: {QUALITY_REPORT_PATH}[/dim]")
    if not args.no_clean:
        console.print(f"[dim]Cleaned files: {PROCESSED_DIR}[/dim]")

    has_critical = any(r.critical for r in reports)
    sys.exit(1 if has_critical else 0)


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and clean fetched Wikipedia articles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Run checks only; do not write cleaned files to data/processed/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
