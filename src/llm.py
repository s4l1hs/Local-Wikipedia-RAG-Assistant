"""Ollama LLM client — context-grounded generation with IDK fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import settings

logger = logging.getLogger(__name__)

# ── System prompt template (FR-10) ───────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise factual assistant. Your ONLY knowledge source is the context provided below.

Rules you must follow without exception:
1. Answer ONLY using information explicitly present in the context.
2. Do not use any external knowledge, training data, or general facts not in the context.
3. If the context does not contain enough information to answer the question, respond with \
exactly: "I don't know based on the available data."
4. If asked to compare two entities, use ONLY facts stated in the context for each entity.
5. Be concise — one to three paragraphs maximum. Do not speculate.

Context:
{context}
"""

IDK_RESPONSE = "I don't know based on the available data."


@dataclass
class LLMResponse:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int


def build_prompt_messages(context: str, user_query: str) -> list[dict]:
    """Return Ollama-compatible message list."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT.format(context=context)},
        {"role": "user",   "content": user_query},
    ]


def generate(context: str, user_query: str) -> LLMResponse:
    """
    Call local Ollama with the grounded system prompt.
    Returns LLMResponse with the generated answer.
    """
    # TODO:
    #   import ollama
    #   messages = build_prompt_messages(context, user_query)
    #   response = ollama.chat(
    #       model=settings.llm_model,
    #       messages=messages,
    #       options={
    #           "temperature": settings.llm_temperature,
    #           "num_predict": settings.llm_max_tokens,
    #       },
    #   )
    #   answer = response["message"]["content"].strip()
    #   return LLMResponse(
    #       answer=answer,
    #       model=settings.llm_model,
    #       prompt_tokens=response.get("prompt_eval_count", 0),
    #       completion_tokens=response.get("eval_count", 0),
    #   )
    raise NotImplementedError


def generate_streaming(context: str, user_query: str):
    """
    Streaming variant — yields answer tokens one by one.
    Used by the Streamlit UI for progressive display.
    """
    # TODO:
    #   import ollama
    #   messages = build_prompt_messages(context, user_query)
    #   stream = ollama.chat(model=settings.llm_model, messages=messages, stream=True)
    #   for chunk in stream:
    #       yield chunk["message"]["content"]
    raise NotImplementedError


def is_ollama_running() -> bool:
    """Health check — returns True if Ollama server is reachable."""
    # TODO:
    #   import httpx
    #   try:
    #       r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
    #       return r.status_code == 200
    #   except Exception:
    #       return False
    raise NotImplementedError
