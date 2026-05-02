"""Shared utility helpers — logging, token counting, path management."""

from __future__ import annotations

import html as _html
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Token counting ────────────────────────────────────────────────────────────
# Lazy-loaded so the module can be imported without tiktoken (e.g. in tests).
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken  # noqa: PLC0415
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc    = _get_tokenizer()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_wikipedia_text(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"\[\d+\]", "", raw)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[nb \d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [l for l in text.splitlines() if l.strip() and not re.match(r"^[=\-\s]+$", l)]
    return "\n".join(lines).strip()


def clean_text_deep(raw: str) -> str:
    """HTML-decode + NFC-normalize + markup strip. Safe to call on already-cleaned text."""
    if not raw:
        return ""
    text = _html.unescape(raw)
    text = unicodedata.normalize("NFC", text)
    return clean_wikipedia_text(text)


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
