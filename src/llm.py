"""
Ollama LLM client — context-grounded generation with retry and streaming.

Design decisions:
  • OllamaLLM.generate() accepts prompt + system separately so prompt
    construction lives in src/prompts.py (separation of concerns).
  • stream=True returns a Generator[str, None, None]; the Streamlit UI
    consumes it with st.write_stream().
  • Retry (max 2) is applied only to non-streaming calls; streaming
    generators cannot be safely retried mid-stream.
  • temperature=0.1 is the default — low temperature keeps the model in
    retrieval/summarisation mode and reduces creative confabulation.
  • Token counts are available only on non-streaming responses (Ollama
    includes eval_count in the final response object, not per-chunk).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator

import ollama

from src.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

IDK_RESPONSE = "I don't know based on the available information."

_RETRY_DELAY_BASE = 1.0   # seconds; multiplied by attempt number


# ── Response type ─────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Non-streaming response with token-usage metadata."""

    answer:            str
    model:             str
    prompt_tokens:     int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ── Main client ───────────────────────────────────────────────────────────────

class OllamaLLM:
    """
    Wrapper around the official `ollama` Python client.

    Thread-safe after construction (all state read-only except the
    underlying ollama.Client which is also thread-safe per their docs).
    """

    def __init__(
        self,
        model_name:  str = settings.llm_model,
        base_url:    str = settings.ollama_base_url,
        max_retries: int = 2,
    ) -> None:
        self._model       = model_name
        self._base_url    = base_url
        self._max_retries = max_retries
        self._client      = ollama.Client(host=base_url)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:      str,
        system:      str | None = None,
        temperature: float      = settings.llm_temperature,
        max_tokens:  int        = settings.llm_max_tokens,
        stream:      bool       = False,
    ) -> str | Generator[str, None, None]:
        """
        Generate a response from the LLM.

        Args:
            prompt:      The user-facing question / instruction.
            system:      Optional system prompt (prepended as role=system).
            temperature: Sampling temperature (0.0–1.0).  Default 0.1.
            max_tokens:  Maximum tokens to generate.
            stream:      If True, return a generator yielding text chunks.

        Returns:
            str when stream=False; Generator[str, None, None] when stream=True.

        Raises:
            ollama.RequestError:  Ollama service unreachable.
            ollama.ResponseError: Model not found or server error.
        """
        messages = self._build_messages(prompt, system)
        options  = {"temperature": temperature, "num_predict": max_tokens}

        if stream:
            return self._stream(messages, options)
        return self._generate_with_retry(messages, options)

    def generate_response(
        self,
        prompt:      str,
        system:      str | None = None,
        temperature: float      = settings.llm_temperature,
        max_tokens:  int        = settings.llm_max_tokens,
    ) -> LLMResponse:
        """
        Non-streaming generate that returns an LLMResponse with token counts.
        Convenience wrapper for the RAG pipeline.
        """
        messages = self._build_messages(prompt, system)
        options  = {"temperature": temperature, "num_predict": max_tokens}
        raw      = self._chat_with_retry(messages, options, stream=False)

        return LLMResponse(
            answer=raw["message"]["content"].strip(),
            model=raw.get("model", self._model),
            prompt_tokens=raw.get("prompt_eval_count", 0),
            completion_tokens=raw.get("eval_count", 0),
        )

    def health_check(self) -> tuple[bool, str]:
        """
        Verify service reachability and model availability.

        Returns:
            (True, "OK") on success.
            (False, reason_str) on failure — reason_str contains actionable
            instructions for the user.
        """
        # ── 1. Check service ──────────────────────────────────────────────────
        try:
            result = self._client.list()
        except ollama.RequestError as e:
            return False, (
                f"Ollama service is not reachable at {self._base_url}.\n"
                f"  Start it with:  ollama serve\n"
                f"  Error: {e}"
            )
        except Exception as e:
            return False, f"Unexpected error contacting Ollama: {e}"

        # ── 2. Check model ────────────────────────────────────────────────────
        models = result.get("models", [])
        pulled = [m.get("name", "") for m in models]
        # ollama stores names as "llama3.2:latest" — match prefix
        model_available = any(
            m == self._model or m.startswith(self._model + ":")
            for m in pulled
        )
        if not model_available:
            pulled_str = ", ".join(pulled) if pulled else "(none)"
            return False, (
                f"Model '{self._model}' is not pulled.\n"
                f"  Pull it with:  ollama pull {self._model}\n"
                f"  Available models: {pulled_str}"
            )

        return True, "OK"

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(prompt: str, system: str | None) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _generate_with_retry(
        self,
        messages: list[dict],
        options:  dict,
    ) -> str:
        raw = self._chat_with_retry(messages, options, stream=False)
        return raw["message"]["content"].strip()

    def _chat_with_retry(
        self,
        messages: list[dict],
        options:  dict,
        stream:   bool,
    ) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.chat(
                    model=self._model,
                    messages=messages,
                    options=options,
                    stream=stream,
                )
            except (ollama.RequestError, ollama.ResponseError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = _RETRY_DELAY_BASE * (attempt + 1)
                    logger.warning(
                        "Ollama call failed (attempt %d/%d): %s  retrying in %.1fs",
                        attempt + 1, self._max_retries + 1, exc, delay,
                    )
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _stream(
        self,
        messages: list[dict],
        options:  dict,
    ) -> Generator[str, None, None]:
        """Yield text chunks from a streaming Ollama response."""
        try:
            for chunk in self._client.chat(
                model=self._model,
                messages=messages,
                options=options,
                stream=True,
            ):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except (ollama.RequestError, ollama.ResponseError) as exc:
            logger.error("Streaming error: %s", exc)
            yield f"\n[Error: {exc}]"


# ── Prompt helpers (kept here for backwards compatibility) ────────────────────
# Full template library lives in src/prompts.py (added in Prompt 16).

def build_prompt_messages(context: str, user_query: str) -> list[dict]:
    """Return Ollama message list for the grounded RAG prompt."""
    from src.prompts import build_rag_messages   # lazy import avoids circular dep
    return build_rag_messages(context, user_query)


# ── Module-level singleton ────────────────────────────────────────────────────

_llm: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    """Return the shared OllamaLLM instance (built once per process)."""
    global _llm
    if _llm is None:
        _llm = OllamaLLM()
    return _llm
