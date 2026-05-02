"""Wikipedia article fetcher — downloads and persists raw article text."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import wikipediaapi

from src.config import PROJECT_ROOT, ENTITY_CATALOG, settings
from src.utils import write_text, clean_wikipedia_text

logger = logging.getLogger(__name__)

_WIKI = wikipediaapi.Wikipedia(
    language="en",
    user_agent="LocalWikiRAG/1.0 (educational project; BLG483E)",
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def fetch_article(wikipedia_title: str) -> str | None:
    """Return full plain-text of a Wikipedia article, or None if not found."""
    # TODO: implement retry with exponential backoff for HTTP errors
    page = _WIKI.page(wikipedia_title)
    if not page.exists():
        logger.warning("Wikipedia page not found: %s", wikipedia_title)
        return None
    return page.text


def save_raw(entity_id: str, text: str) -> Path:
    path = RAW_DIR / f"{entity_id}.txt"
    write_text(path, text)
    return path


def save_processed(entity_id: str, text: str) -> Path:
    path = PROCESSED_DIR / f"{entity_id}.txt"
    write_text(path, clean_wikipedia_text(text))
    return path


def article_already_fetched(entity_id: str) -> bool:
    return (RAW_DIR / f"{entity_id}.txt").exists()


def ingest_all(force: bool = False, delay_seconds: float = 1.0) -> dict[str, bool]:
    """
    Fetch and persist raw + processed text for every entity in the catalog.

    Returns a mapping of entity_id → success.
    Skips entities already fetched unless force=True.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}

    for entity in ENTITY_CATALOG:
        eid = entity["entity_id"]
        title = entity["wikipedia_title"]

        if not force and article_already_fetched(eid):
            logger.info("[SKIP] %s (already fetched)", eid)
            results[eid] = True
            continue

        logger.info("[FETCH] %s → Wikipedia: '%s'", eid, title)
        text = fetch_article(title)

        if text is None:
            results[eid] = False
            continue

        save_raw(eid, text)
        save_processed(eid, text)
        results[eid] = True

        # Polite delay to respect Wikipedia rate limits (FR-1)
        time.sleep(delay_seconds)

    return results


def load_processed_text(entity_id: str) -> str:
    """Load cleaned text for a single entity from disk."""
    path = PROCESSED_DIR / f"{entity_id}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed text not found for '{entity_id}'. Run ingest_all() first."
        )
    return path.read_text(encoding="utf-8")
