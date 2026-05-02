#!/usr/bin/env python3
"""
Step 3 — Quick end-to-end smoke test against the built index.

Runs the 14 example queries from the assignment spec and prints pass/fail.
Does NOT use pytest — intended as a fast manual sanity check.

Usage:
    python scripts/03_smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.rag_pipeline import ask
from src.llm import IDK_RESPONSE

console = Console()

# ── Test cases from assignment spec ───────────────────────────────────────────
IN_SCOPE_QUERIES: list[tuple[str, str]] = [
    # (query, expected_substring_in_answer)  — case-insensitive
    ("Who was Albert Einstein and what is he known for?",    "theory of relativity"),
    ("What did Marie Curie discover?",                       "polonium"),
    ("Why is Nikola Tesla famous?",                          "electricity"),
    ("What is Frida Kahlo known for?",                       "painting"),
    ("Where is the Eiffel Tower located?",                   "paris"),
    ("Why is the Great Wall of China important?",            "china"),
    ("What is Machu Picchu?",                                "inca"),
    ("What was the Colosseum used for?",                     "gladiatorial"),
    ("Where is Mount Everest?",                              "nepal"),
    ("Which famous place is located in Turkey?",             "hagia sophia"),
    ("Which person is associated with electricity?",         "tesla"),
    ("Compare Albert Einstein and Nikola Tesla",             "einstein"),
    ("Compare the Eiffel Tower and the Statue of Liberty",   "eiffel"),
]

OUT_OF_SCOPE_QUERIES: list[str] = [
    "Who is the president of Mars?",
    "Tell me about a random unknown person John Doe",
]


def run_smoke_tests() -> bool:
    table = Table(title="Smoke Test Results", show_lines=True)
    table.add_column("Query",    style="cyan",  max_width=50)
    table.add_column("Result",   style="bold",  width=8)
    table.add_column("sim_max",  width=8)
    table.add_column("Answer preview", max_width=60)

    all_passed = True

    # In-scope: expect non-IDK answers containing the expected substring
    for query, expected in IN_SCOPE_QUERIES:
        resp = ask(query, show_sources=False)
        passed = (not resp.is_idk) and (expected.lower() in resp.answer.lower())
        all_passed &= passed
        table.add_row(
            query[:50],
            "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
            f"{resp.sim_max:.3f}",
            resp.answer[:60].replace("\n", " "),
        )

    # Out-of-scope: expect IDK
    for query in OUT_OF_SCOPE_QUERIES:
        resp = ask(query, show_sources=False)
        passed = resp.is_idk
        all_passed &= passed
        table.add_row(
            query[:50],
            "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
            f"{resp.sim_max:.3f}",
            resp.answer[:60].replace("\n", " "),
        )

    console.print(table)
    status = "[bold green]ALL PASSED[/bold green]" if all_passed else "[bold red]SOME FAILED[/bold red]"
    console.print(f"\nOverall: {status}")
    return all_passed


if __name__ == "__main__":
    ok = run_smoke_tests()
    sys.exit(0 if ok else 1)
