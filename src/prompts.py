"""
Prompt templates for the Local Wikipedia RAG Assistant.

Design rationale (see ULTRATHINK in session notes):
─────────────────────────────────────────────────────
All rules are *operational*, not abstract:
  • "Trace every fact to a passage sentence" > "answer from context"
    → Forces per-statement provenance-checking rather than a global feeling
      of grounding that the model can satisfy while confabulating.
  • IDK trigger is phrased as "if answering requires ANY knowledge beyond the
    passages — even knowledge you are confident is true" — explicitly makes
    training-data confidence irrelevant.
  • False-premise correction is Rule 1 (not Rule 5): it is the most
    counterintuitive behaviour for a helpfulness-trained model and must be
    front-loaded to activate before the model starts generating.
  • The low-confidence variant extends the grounded variant (adds Rule 7);
    it does NOT replace Rules 4–6 as the previous stub did.
  • Passages are surrounded by delimiter lines to prevent user-injected fake
    passages and to clearly mark the knowledge boundary.
  • Anti-sycophancy rule suppresses "Certainly!" / "Great question!" openers
    that trigger agreeable (= less grounded) generation modes.
  • No cross-entity contamination rule: "when passages describe multiple
    entities, keep each entity's facts from that entity's passages only" —
    critical for comparison queries that mix person + place passages.

Threshold linkage:
  similarity_threshold_low = 0.55 → pipeline returns IDK_RESPONSE directly,
    LLM is never called (no hallucination risk, no token cost).
  similarity_threshold_mid = 0.70 → pipeline passes low_confidence=True,
    activating SYSTEM_PROMPT_LOW_CONFIDENCE with hedging Rule 7.
  score ≥ 0.70 → SYSTEM_PROMPT used (full confidence, no hedging).
"""

from __future__ import annotations

from src.chunker import Chunk

# ── String constants ──────────────────────────────────────────────────────────

IDK_RESPONSE = "I don't know based on the available information."

LOW_CONFIDENCE_PREFIX = "Based on the available information, "


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Wikipedia-based factual assistant. Your knowledge is strictly \
limited to the passages provided below — you have no other information source.

Follow these rules without exception:

1. CORRECT FALSE PREMISES FIRST: If the question contains an incorrect \
factual assumption, your first sentence must correct it using evidence from \
the passages. Then provide the accurate answer. Do not confirm false \
assumptions, even partially.

2. TRACE EVERY FACT: Before writing any factual statement, identify the exact \
sentence in the passages that supports it. If no passage sentence supports it, \
do not write it — even if you are confident it is true from your training.

3. SAY "I DON'T KNOW" WHEN NECESSARY: If answering the question would require \
ANY knowledge beyond what is written in the passages — even knowledge you are \
certain is correct — respond with exactly:
"{idk}"

4. KEEP ENTITIES SEPARATE: When passages describe multiple entities, treat \
each entity's passages as independent. Do not attribute a fact from one \
entity's passages to a different entity.

5. CONCISE AND ENCYCLOPEDIC: Respond in one to three paragraphs. Use neutral, \
factual language. Do not speculate, infer, or extrapolate beyond the text.

6. NO META-COMMENTARY: Do not start with "Certainly!", "Of course!", \
"Great question!", or similar phrases. Do not mention that you are an AI \
or that you are summarising passages. Do not add a concluding summary paragraph.

--- PASSAGES ---

{{context}}

--- END OF PASSAGES ---\
""".format(idk=IDK_RESPONSE)

SYSTEM_PROMPT_LOW_CONFIDENCE = """\
You are a Wikipedia-based factual assistant. Your knowledge is strictly \
limited to the passages provided below — you have no other information source. \
Note: the retrieved passages may be only partially relevant to this question.

Follow these rules without exception:

1. CORRECT FALSE PREMISES FIRST: If the question contains an incorrect \
factual assumption, your first sentence must correct it using evidence from \
the passages. Then provide the accurate answer.

2. TRACE EVERY FACT: Before writing any factual statement, identify the exact \
sentence in the passages that supports it. If no passage sentence supports it, \
do not write it — even if you are confident it is true from your training.

3. SAY "I DON'T KNOW" WHEN NECESSARY: If answering the question would require \
ANY knowledge beyond what is written in the passages — even knowledge you are \
certain is correct — respond with exactly:
"{idk}"

4. KEEP ENTITIES SEPARATE: When passages describe multiple entities, treat \
each entity's passages as independent. Do not attribute a fact from one \
entity's passages to a different entity.

5. CONCISE AND ENCYCLOPEDIC: Respond in one to three paragraphs. Use neutral, \
factual language. Do not speculate, infer, or extrapolate beyond the text.

6. NO META-COMMENTARY: Do not start with "Certainly!", "Of course!", \
"Great question!", or similar phrases. Do not mention that you are an AI \
or that you are summarising passages. Do not add a concluding summary paragraph.

7. ACKNOWLEDGE UNCERTAINTY: Because the retrieved passages may be only \
partially relevant, begin your response with exactly: "{prefix}" \
If the passages do not address a specific aspect of the question, state this \
explicitly rather than guessing.

--- PASSAGES ---

{{context}}

--- END OF PASSAGES ---\
""".format(idk=IDK_RESPONSE, prefix=LOW_CONFIDENCE_PREFIX)


# ── Passage formatter ─────────────────────────────────────────────────────────

def format_passages(chunks: list[Chunk], scores: list[float]) -> str:
    """
    Format retrieved chunks into a numbered passage block.

    Each passage header shows entity name, section heading, and relevance
    score so the LLM can attribute facts and gauge confidence.

    Args:
        chunks: Retrieved Chunk objects in rank order (highest score first).
        scores: Corresponding cosine similarity scores.

    Returns:
        Multi-line string suitable for insertion into SYSTEM_PROMPT {context}.
    """
    if not chunks:
        return "(No relevant passages retrieved.)"

    parts: list[str] = []
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        meta    = chunk.metadata
        entity  = meta.get("entity_name", "Unknown")
        section = meta.get("section_heading") or "Introduction"
        header  = f"[{i}] {entity} — {section}  (score: {score:.2f})"
        parts.append(f"{header}\n{chunk.content.strip()}")

    return "\n\n".join(parts)


def format_context(chunks_with_scores: list[tuple[str, float]]) -> str:
    """
    Backward-compatible formatter that accepts raw (text, score) pairs.
    Use format_passages() for new code that has Chunk objects.
    """
    if not chunks_with_scores:
        return "(No relevant passages retrieved.)"

    parts: list[str] = []
    for i, (content, score) in enumerate(chunks_with_scores, 1):
        parts.append(f"[{i}]  (score: {score:.2f})\n{content.strip()}")
    return "\n\n".join(parts)


# ── Message builders ──────────────────────────────────────────────────────────

def build_rag_messages(
    context:        str,
    query:          str,
    low_confidence: bool = False,
) -> list[dict]:
    """
    Build an Ollama-compatible message list for a grounded RAG turn.

    Args:
        context:        Pre-formatted passage block from format_passages().
        query:          Raw user question (passed as-is to the model).
        low_confidence: If True, use the hedging system prompt variant.
                        Set this when RetrievalResult.low_confidence is True
                        (i.e. max_score < settings.similarity_threshold_mid).
    """
    template = SYSTEM_PROMPT_LOW_CONFIDENCE if low_confidence else SYSTEM_PROMPT
    system   = template.format(context=context)
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": query},
    ]


def build_rag_prompt(
    chunks:         list[Chunk],
    scores:         list[float],
    query:          str,
    low_confidence: bool = False,
) -> list[dict]:
    """
    One-call convenience: format passages then build message list.

    Args:
        chunks:         Retrieved Chunk objects.
        scores:         Corresponding cosine similarity scores.
        query:          Raw user question.
        low_confidence: Passed through to build_rag_messages().

    Returns:
        Ollama message list ready for OllamaLLM.generate() or .generate_response().
    """
    context = format_passages(chunks, scores)
    return build_rag_messages(context, query, low_confidence=low_confidence)
