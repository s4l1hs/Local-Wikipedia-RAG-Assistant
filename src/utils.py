"""Shared utility helpers — logging, token counting, path management."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# ── Token counting ────────────────────────────────────────────────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")  # proxy for LLaMA-family models


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_wikipedia_text(raw: str) -> str:
    """Strip markup artifacts left over from Wikipedia plain-text extraction."""
    # Remove citation markers like [1], [citation needed]
    text = re.sub(r"\[\d+\]", "", raw)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are only punctuation / whitespace
    lines = [l for l in text.splitlines() if l.strip() and not re.match(r"^[=\-\s]+$", l)]
    return "\n".join(lines).strip()


# ── File I/O helpers ──────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── Timing decorator ──────────────────────────────────────────────────────────

def timed(fn):
    """Log wall-clock time of any function call."""
    def wrapper(*args, **kwargs) -> Any:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug("%s completed in %.2fs", fn.__name__, elapsed)
        return result
    wrapper.__name__ = fn.__name__
    return wrapper


# ── Logging setup ─────────────────────────────────────────────────────────────

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
