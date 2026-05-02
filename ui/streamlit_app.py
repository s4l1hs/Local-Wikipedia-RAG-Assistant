"""Streamlit chat UI — run with: streamlit run ui/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import ENTITY_CATALOG, settings
from src.rag_pipeline import ask, RAGResponse
from src.vector_store import collection_count

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Wikipedia RAG",
    page_icon="📚",
    layout="wide",
)

# ── Session state initialisation ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": str, "content": str, "meta": dict}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    show_sources = st.toggle("Show Sources", value=True)
    show_debug   = st.toggle("Show Debug Info", value=False)

    st.divider()
    st.subheader("📋 Entity Catalog")

    people = [e for e in ENTITY_CATALOG if e["entity_type"] == "person"]
    places = [e for e in ENTITY_CATALOG if e["entity_type"] == "place"]

    with st.expander(f"👤 People ({len(people)})", expanded=False):
        for p in people:
            st.markdown(f"- {p['entity_name']}")

    with st.expander(f"🏛️ Places ({len(places)})", expanded=False):
        for p in places:
            st.markdown(f"- {p['entity_name']}")

    st.divider()
    # TODO: display collection_count() when index is available
    st.caption(f"Model: {settings.llm_model} · Embedding: {settings.embedding_model}")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("📚 Local Wikipedia RAG Assistant")
st.caption("Ask me anything about the 40 indexed people and places — running 100% locally.")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg["role"] == "assistant" and msg.get("meta", {}).get("sources"):
            _render_sources(msg["meta"]["sources"])
        if show_debug and msg["role"] == "assistant":
            meta = msg.get("meta", {})
            st.caption(
                f"sim_max={meta.get('sim_max', 0):.3f} | "
                f"intent={meta.get('intent', '-')} | "
                f"fallback={meta.get('used_fallback', False)}"
            )

# Input
if prompt := st.chat_input("Ask a question about a famous person or place..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # TODO: replace with generate_streaming() for progressive display
            response: RAGResponse = ask(prompt, show_sources=show_sources)

        st.markdown(response.answer)

        if show_sources and response.sources:
            _render_sources(response.sources)

        if show_debug:
            st.caption(
                f"sim_max={response.sim_max:.3f} | "
                f"intent={response.intent} | "
                f"fallback={response.used_fallback}"
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "meta": {
            "sources":      response.sources,
            "sim_max":      response.sim_max,
            "intent":       response.intent,
            "used_fallback": response.used_fallback,
        },
    })


# ── Helper ────────────────────────────────────────────────────────────────────
def _render_sources(sources) -> None:
    if not sources:
        return
    with st.expander("📎 Retrieved Sources", expanded=False):
        for i, src in enumerate(sources[:3], 1):
            st.markdown(
                f"**{i}. {src.entity_name}** — "
                f"[Wikipedia]({src.source_url}) · sim={src.similarity:.3f}"
            )
            st.caption(src.text[:200] + "…")
