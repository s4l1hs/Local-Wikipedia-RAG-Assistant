"""
Wikipedia article fetcher — downloads, cleans, and persists structured article data.

Library choice: wikipedia-api  (pip install Wikipedia-API, import wikipediaapi)
Rationale over the `wikipedia` package:
  • page.sections gives a recursive Section tree → clean filtered extraction
  • page.summary isolates the lede paragraph → introduction section
  • No global state / thread-safe by construction
  • Explicit User-Agent control (Wikipedia bot policy requirement)
  • DisambiguationError detectable without a separate API call
  The plain `wikipedia` package returns undifferentiated text blobs with no section
  structure, making References/External-links filtering fragile and error-prone.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wikipediaapi

from src.config import PROJECT_ROOT, ENTITY_CATALOG
from src.utils import clean_text_deep, clean_wikipedia_text, ensure_dir

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_USER_AGENT = (
    "LocalWikiRAG/1.0 "
    "(educational project; BLG483E Information Retrieval; "
    "fully offline after ingestion; contact: student)"
)

# Sections stripped during extraction — noise that hurts retrieval quality
_EXCLUDED_SECTIONS: frozenset[str] = frozenset({
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "notes and references",
    "bibliography",
    "sources",
    "citations",
    "footnotes",
    "works cited",
    "secondary sources",
    "primary sources",
    "literature",
    "selected bibliography",
    "selected works",
    "discography",        # for musicians — usually just a list of albums
    "filmography",        # similarly list-heavy
    "awards and honours", # typically a table, not prose
    "awards",
    "honours",
})

# Warn if article is shorter than this — may produce too few chunks
_MIN_ARTICLE_WORDS = 500

RAW_DIR:       Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH: Path = RAW_DIR / "_manifest.json"

FETCH_DELAY = 0.8   # minimum seconds between requests (Wikipedia asks for ≥ 0.5s)
MAX_SECTION_DEPTH = 3  # don't recurse deeper than this (avoids junk sub-sub-sections)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Section:
    heading: str   # empty string → Introduction (lede paragraph)
    text: str


@dataclass
class ArticleData:
    entity_id:   str
    entity_name: str
    entity_type: str   # "person" | "place"
    category:    str   # from entity catalog (e.g. "scientist", "monument")
    wiki_title:  str   # exact Wikipedia page title used for the API call
    url:         str   # canonical Wikipedia URL
    fetched_at:  str   # ISO-8601 UTC timestamp
    sections:    list[Section]
    word_count:  int   # computed from filtered full text

    def full_text(self) -> str:
        return _join_sections(self.sections)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the JSON schema expected by downstream modules."""
        return {
            "entity_id":  self.entity_id,
            "name":       self.entity_name,
            "type":       self.entity_type,
            "category":   self.category,
            "wiki_title": self.wiki_title,
            "url":        self.url,
            "fetched_at": self.fetched_at,
            "raw_text":   self.full_text(),
            "sections": [
                {"heading": s.heading, "text": s.text}
                for s in self.sections
            ],
            "word_count": self.word_count,
        }


# ── Exceptions ────────────────────────────────────────────────────────────────

class DisambiguationError(ValueError):
    """Raised when the Wikipedia page is a disambiguation page."""


class ArticleNotFoundError(LookupError):
    """Raised when the Wikipedia page does not exist."""


# ── Fetcher ───────────────────────────────────────────────────────────────────

class WikiFetcher:
    _MAX_RETRIES = 3

    def __init__(self, delay: float = FETCH_DELAY) -> None:
        self._wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent=_USER_AGENT,
        )
        self._delay     = delay
        self._last_call = 0.0   # monotonic time of the last API call

    # ── Public ───────────────────────────────────────────────────────────────

    def fetch(self, entity: dict) -> ArticleData:
        title = entity["wikipedia_title"]
        last_exc: Exception | None = None

        for attempt in range(self._MAX_RETRIES):
            self._throttle()
            try:
                return self._fetch_once(entity, title)
            except (ArticleNotFoundError, DisambiguationError):
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Retry %d/%d for '%s' after %.0fs  (%s: %s)",
                        attempt + 1, self._MAX_RETRIES, title, wait,
                        type(exc).__name__, exc,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"All {self._MAX_RETRIES} attempts failed for '{title}'"
        ) from last_exc

    # ── Internal ─────────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Block until enough time has passed since the last API call."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call = time.monotonic()

    def _fetch_once(self, entity: dict, title: str) -> ArticleData:
        page = self._wiki.page(title)

        if not page.exists():
            raise ArticleNotFoundError(f"Wikipedia page not found: '{title}'")

        if self._is_disambiguation(page):
            raise DisambiguationError(
                f"'{title}' is a disambiguation page — "
                "check entity catalog for a more specific wiki_title"
            )

        sections   = self._build_sections(page)
        text       = _join_sections(sections)
        word_count = len(text.split())

        if word_count < _MIN_ARTICLE_WORDS:
            logger.warning(
                "'%s' article is very short (%d words) — may produce few chunks",
                title, word_count,
            )

        return ArticleData(
            entity_id   = entity["entity_id"],
            entity_name = entity["entity_name"],
            entity_type = entity["entity_type"],
            category    = entity.get("category", ""),
            wiki_title  = title,
            url         = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            fetched_at  = datetime.now(timezone.utc).isoformat(),
            sections    = sections,
            word_count  = word_count,
        )

    def _is_disambiguation(self, page) -> bool:
        """
        Heuristic: is this a Wikipedia disambiguation page?

        Does NOT require an extra API call — checks title and lede only.
        """
        if "(disambiguation)" in page.title.lower():
            return True
        lede = (page.summary or "").lower()[:800]
        markers = (
            "may refer to:",
            "may refer to\n",
            "can refer to:",
            "may mean:",
            "is a disambiguation",
            "refers to multiple",
        )
        return any(m in lede for m in markers)

    def _build_sections(self, page) -> list[Section]:
        """
        Assemble a filtered section list from a wikipedia-api page object.

        Structure:
          [0] Introduction  — page.summary (lede before first heading)
          [1..n] Filtered body sections — page.sections (recursive)
        """
        result: list[Section] = []

        # wikipedia-api returns the lede separately from page.sections; preserve it as section [0]
        intro = clean_wikipedia_text(page.summary)
        if intro:
            result.append(Section(heading="", text=intro))

        result.extend(_extract_sections(page.sections, depth=0))

        return result


# ── Section helpers ───────────────────────────────────────────────────────────

def _extract_sections(
    raw_sections: list,
    depth: int = 0,
) -> list[Section]:
    """Recursively flatten Wikipedia section tree, filtering excluded sections."""
    result: list[Section] = []
    for s in raw_sections:
        heading = s.title.strip()
        if heading.lower() in _EXCLUDED_SECTIONS:
            continue  # noise sections; skip children too to avoid nested junk
        text = clean_wikipedia_text(s.text)
        if text:
            result.append(Section(heading=heading, text=text))
        if depth < MAX_SECTION_DEPTH:
            result.extend(_extract_sections(s.sections, depth + 1))
    return result


def _join_sections(sections: list[Section]) -> str:
    parts: list[str] = []
    for s in sections:
        if s.heading:
            parts.append(f"\n== {s.heading} ==")
        parts.append(s.text)
    return "\n\n".join(filter(None, parts)).strip()


# ── Persistence ───────────────────────────────────────────────────────────────

def article_filename(entity_id: str, entity_type: str) -> str:
    """Canonical filename: {type}_{slug}.json"""
    return f"{entity_type}_{entity_id}.json"


def save_article(article: ArticleData, out_dir: Path = RAW_DIR) -> Path:
    ensure_dir(out_dir)
    path = out_dir / article_filename(article.entity_id, article.entity_type)
    path.write_text(
        json.dumps(article.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Saved: %s (%d words)", path.name, article.word_count)
    return path


def load_article(entity_id: str, entity_type: str) -> ArticleData | None:
    """Load a previously saved article from disk. Returns None if not found."""
    path = RAW_DIR / article_filename(entity_id, entity_type)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ArticleData(
        entity_id   = raw["entity_id"],
        entity_name = raw["name"],
        entity_type = raw["type"],
        category    = raw.get("category", ""),
        wiki_title  = raw["wiki_title"],
        url         = raw["url"],
        fetched_at  = raw["fetched_at"],
        sections    = [
            Section(heading=s["heading"], text=s["text"])
            for s in raw["sections"]
        ],
        word_count  = raw["word_count"],
    )


def clean_article(article: ArticleData) -> ArticleData:
    """Deep-clean all section text and drop any section left empty after cleaning."""
    cleaned = [
        Section(heading=s.heading, text=clean_text_deep(s.text))
        for s in article.sections
    ]
    cleaned = [s for s in cleaned if s.text]
    text = _join_sections(cleaned)
    return ArticleData(
        entity_id   = article.entity_id,
        entity_name = article.entity_name,
        entity_type = article.entity_type,
        category    = article.category,
        wiki_title  = article.wiki_title,
        url         = article.url,
        fetched_at  = article.fetched_at,
        sections    = cleaned,
        word_count  = len(text.split()),
    )


# ── Manifest ──────────────────────────────────────────────────────────────────
# _manifest.json tracks which articles have been fetched and their metadata.
# Used by 02_build_index.py to know which files are ready for embedding.

def load_manifest() -> dict[str, Any]:
    """Return the current manifest dict, or a fresh empty structure."""
    if not MANIFEST_PATH.exists():
        return {"_version": 1, "_updated": "", "entities": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        logger.warning("Manifest file is corrupt — starting fresh.")
        return {"_version": 1, "_updated": "", "entities": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Persist manifest to disk (atomic: write then rename)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest["_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)    # atomic on POSIX; near-atomic on Windows


def update_manifest(article: ArticleData) -> None:
    """Add or update one entity's record in the manifest."""
    manifest = load_manifest()
    manifest["entities"][article.entity_id] = {
        "entity_id":     article.entity_id,
        "entity_name":   article.entity_name,
        "entity_type":   article.entity_type,
        "category":      article.category,
        "filename":      article_filename(article.entity_id, article.entity_type),
        "fetched_at":    article.fetched_at,
        "word_count":    article.word_count,
        "section_count": len(article.sections),
    }
    save_manifest(manifest)


def is_already_fetched(entity_id: str) -> bool:
    """True if the entity is recorded in the manifest AND its file exists."""
    manifest = load_manifest()
    if entity_id not in manifest.get("entities", {}):
        return False
    entry = manifest["entities"][entity_id]
    return (RAW_DIR / entry["filename"]).exists()
