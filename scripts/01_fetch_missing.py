#!/usr/bin/env python3
"""
Fetch the 7 entities that fail with the wikipedia-api library.
Uses a single Wikipedia extracts API call per entity (plain text, full article).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ENTITY_CATALOG, PROJECT_ROOT
from src.ingest import MANIFEST_PATH, RAW_DIR
from src.utils import clean_text_deep, configure_logging

configure_logging()

MISSING = {
    "leonardo_da_vinci",
    "william_shakespeare",
    "ada_lovelace",
    "lionel_messi",
    "cristiano_ronaldo",
    "taylor_swift",
    "frida_kahlo",
}

_EXCLUDED = frozenset({
    "references", "external links", "see also", "further reading", "notes",
    "notes and references", "bibliography", "sources", "citations", "footnotes",
    "works cited", "secondary sources", "primary sources", "literature",
    "selected bibliography", "selected works", "discography", "filmography",
    "awards and honours", "awards", "honours",
})

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = (
    "LocalWikiRAG/1.0 (educational; BLG483E; offline after ingestion; contact:student)"
)

_API = "https://en.wikipedia.org/w/api.php"


def _fetch_full_extract(title: str) -> tuple[str, list[dict]]:
    """
    Fetch the full article as plain text in one API call.
    Returns (url, sections) where sections split on == Heading == markers.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": True,
        "exsectionformat": "wiki",   # includes == Section == markers in plain text
        "inprop": "url",
        "redirects": 1,
        "format": "json",
    }
    r = _SESSION.get(_API, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    pages = data["query"]["pages"]
    page  = next(iter(pages.values()))

    if "missing" in page:
        raise RuntimeError(f"Wikipedia page not found: '{title}'")

    url     = page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
    extract = page.get("extract", "")

    sections = _split_into_sections(extract)
    return url, sections


def _split_into_sections(text: str) -> list[dict]:
    """
    Split plain-text extract (with == Heading == markers) into section dicts.
    Headings are stripped from the text body; excluded sections are dropped.
    """
    import re
    # Pattern: lines that are == Heading ==, === Sub === etc.
    heading_pattern = re.compile(r"^(={2,})\s*(.+?)\s*\1$", re.MULTILINE)

    # Find all headings and their positions
    splits: list[tuple[int, int, str]] = []  # (start, level, heading)
    for m in heading_pattern.finditer(text):
        level   = len(m.group(1))   # 2 = H2, 3 = H3 ...
        heading = m.group(2).strip()
        splits.append((m.start(), m.end(), level, heading))

    sections: list[dict] = []

    # Introduction: text before first heading
    intro_end = splits[0][0] if splits else len(text)
    intro_text = clean_text_deep(text[:intro_end].strip())
    if intro_text:
        sections.append({"heading": "", "text": intro_text})

    # Body sections
    for i, (start, end, level, heading) in enumerate(splits):
        if heading.lower() in _EXCLUDED:
            continue
        if level > 3:
            continue
        # Section body runs until the next heading of same or higher level
        body_start = end
        body_end   = splits[i + 1][0] if i + 1 < len(splits) else len(text)
        body       = text[body_start:body_end].strip()
        # Strip any sub-headings from the body (they'll be their own sections)
        body_clean = clean_text_deep(body)
        if body_clean:
            sections.append({"heading": heading, "text": body_clean})

    return sections


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    entities_to_fetch = [e for e in ENTITY_CATALOG if e["entity_id"] in MISSING]
    print(f"Fetching {len(entities_to_fetch)} missing entities (1 API call each)...")

    for entity in entities_to_fetch:
        eid   = entity["entity_id"]
        name  = entity["entity_name"]
        title = entity["wikipedia_title"]
        etype = entity["entity_type"]
        cat   = entity["category"]

        print(f"  {name} ...", end=" ", flush=True)
        try:
            time.sleep(2.0)   # polite delay — 1 call per entity, no rush
            url, sections = _fetch_full_extract(title)
            full_text = "\n\n".join(
                (f"{s['heading']}\n{s['text']}" if s["heading"] else s["text"])
                for s in sections
                if s["text"]
            )
            word_count = len(full_text.split())

            article = {
                "entity_id":  eid,
                "name":       name,
                "type":       etype,
                "category":   cat,
                "wiki_title": title,
                "url":        url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "raw_text":   full_text,
                "sections":   sections,
                "word_count": word_count,
            }

            fname = f"{etype}_{eid}.json"
            out   = RAW_DIR / fname
            with open(out, "w", encoding="utf-8") as f:
                json.dump(article, f, ensure_ascii=False, indent=2)

            manifest["entities"][eid] = {
                "entity_id":     eid,
                "entity_name":   name,
                "entity_type":   etype,
                "category":      cat,
                "filename":      fname,
                "fetched_at":    article["fetched_at"],
                "word_count":    word_count,
                "section_count": len(sections),
            }

            print(f"OK  ({word_count:,} words, {len(sections)} sections)")

        except Exception as exc:
            print(f"FAILED — {exc}")

    manifest["_updated"] = datetime.now(timezone.utc).isoformat()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total = len(manifest["entities"])
    print(f"\nDone. Manifest now has {total} entities.")


if __name__ == "__main__":
    main()
