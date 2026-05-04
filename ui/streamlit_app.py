"""Streamlit chat UI — run with: streamlit run ui/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import ENTITY_CATALOG, settings
from src.llm import OllamaLLM
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.router import RoutingDecision
from src.vector_store import get_vector_store

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Wikipedia RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
_EXAMPLE_QUERIES = [
    "What did Einstein win the Nobel Prize for?",
    "How tall is the Eiffel Tower and when was it built?",
    "Compare Marie Curie and Nikola Tesla",
    "What was the Colosseum used for in ancient Rome?",
]
_MODELS = ["llama3.2", "phi3", "mistral"]

_PEOPLE = [e for e in ENTITY_CATALOG if e["entity_type"] == "person"]
_PLACES = [e for e in ENTITY_CATALOG if e["entity_type"] == "place"]


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG pipeline…")
def _get_pipeline(model: str, top_k: int) -> RAGPipeline:
    """One RAGPipeline instance per (model, top_k) combination."""
    llm = OllamaLLM(model_name=model)
    return RAGPipeline(llm=llm, top_k=top_k)


@st.cache_data(ttl=30, show_spinner=False)
def _get_db_stats() -> dict:
    try:
        store = get_vector_store()
        return store.stats()
    except Exception as exc:
        return {"total": 0, "by_type": {}, "error": str(exc)}


# ── Session state init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    # Each message: {role, content, _resp (RAGResponse | None)}
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox(
        "Model",
        _MODELS,
        index=0,
        help="Ollama model used for answer generation.",
    )

    top_k = st.slider(
        "Top-K passages",
        min_value=3, max_value=10,
        value=settings.top_k,
        help="Number of passages retrieved per query.",
    )

    st.markdown("**Similarity thresholds**")
    threshold_low = st.slider(
        "IDK threshold",
        min_value=0.0, max_value=1.0,
        value=float(settings.similarity_threshold_low),
        step=0.05,
        help="Queries scoring below this return IDK without calling the LLM.",
    )
    threshold_mid = st.slider(
        "Low-confidence threshold",
        min_value=0.0, max_value=1.0,
        value=float(settings.similarity_threshold_mid),
        step=0.05,
        help="Queries between IDK and this threshold get a hedged response.",
    )
    if threshold_low >= threshold_mid:
        st.warning("IDK threshold must be below low-confidence threshold.")

    st.divider()

    # Ollama health indicator
    st.subheader("🔌 Ollama Status")
    col_h1, col_h2 = st.columns([1.5, 1])
    if col_h2.button("Check", use_container_width=True):
        with st.spinner("Checking…"):
            try:
                llm_check = OllamaLLM(model_name=model)
                ok, reason = llm_check.health_check()
                st.session_state.ollama_status = (ok, reason)
            except Exception as exc:
                st.session_state.ollama_status = (False, str(exc))

    status = st.session_state.ollama_status
    if status is None:
        col_h1.caption("Press Check to test connection.")
    elif status[0]:
        col_h1.success("🟢 Running")
    else:
        col_h1.error("🔴 Down")
        with st.expander("Details", expanded=False):
            st.code(status[1], language=None)

    st.divider()

    # DB stats
    st.subheader("📊 DB Stats")
    db_stats = _get_db_stats()
    if db_stats.get("error") or db_stats.get("total", 0) == 0:
        st.warning(
            "Index not built.  \n"
            "Run `python scripts/02_build_index.py` first."
        )
    else:
        st.metric("Total chunks", db_stats["total"])
        c1, c2 = st.columns(2)
        c1.metric("👤 People", db_stats["by_type"].get("person", 0))
        c2.metric("🏛️ Places", db_stats["by_type"].get("place", 0))

    st.divider()

    # Entity catalog expanders
    with st.expander(f"👤 People ({len(_PEOPLE)})", expanded=False):
        for p in _PEOPLE:
            st.markdown(f"- {p['entity_name']}")

    with st.expander(f"🏛️ Places ({len(_PLACES)})", expanded=False):
        for p in _PLACES:
            st.markdown(f"- {p['entity_name']}")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        try:
            _get_pipeline(model, top_k).clear_history()
        except Exception:
            pass
        st.rerun()

    st.caption(
        f"Embed: `{settings.embedding_model.split('/')[-1]}`  ·  "
        f"Temp: `{settings.llm_temperature}`"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_sources(resp: RAGResponse) -> None:
    """Render 📚 View Sources expander for one assistant turn."""
    with st.expander("📚 View Sources", expanded=False):
        # Routing + latency
        r = resp.routing
        lat = resp.latency_ms
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"**Routing:** `{r.category}`  \n"
                f"person={r.person_score:.1f}  ·  place={r.place_score:.1f}"
                + (f"  \nentities: {', '.join(r.matched_entities)}" if r.matched_entities else "")
            )
            st.markdown(
                f"**Flags:** "
                f"idk=`{resp.is_idk}`  "
                f"low_conf=`{resp.low_confidence}`  "
                f"coverage=`{resp.used_coverage}`"
            )
        with col2:
            st.markdown(
                f"**Latency (ms)**  \n"
                f"retrieve: `{lat.get('retrieve', 0):.0f}`  \n"
                f"llm: `{lat.get('llm', 0):.0f}`  \n"
                f"total: `{lat.get('total', 0):.0f}`"
            )

        if not resp.sources:
            st.info("No sources — query was out-of-corpus or no passages met the threshold.")
            return

        st.divider()
        for src in resp.sources:
            heading = src.get("section_heading") or "Introduction"
            url     = src.get("source_url", "")
            name    = src.get("entity_name", "Unknown")
            score   = src.get("score", 0.0)
            num     = src["passage_num"]

            link_md = f"[Wikipedia ↗]({url})" if url else ""
            st.markdown(
                f"**[{num}] {name} § {heading}**  "
                f"score=`{score:.3f}`  {link_md}"
            )

            idx = num - 1
            if idx < len(resp.retrieved_chunks):
                preview = resp.retrieved_chunks[idx].content.strip()
                cutoff  = 400
                st.caption(preview[:cutoff] + ("…" if len(preview) > cutoff else ""))


def _stream_response(pipeline: RAGPipeline, query: str) -> RAGResponse | None:
    """
    Stream tokens into the current chat_message context.
    Returns the final RAGResponse (or a minimal error response on failure).
    """
    placeholder = st.empty()
    accumulated: list[str] = []
    final_resp: RAGResponse | None = None

    try:
        for item in pipeline.ask(query, stream=True):
            if isinstance(item, str):
                accumulated.append(item)
                placeholder.markdown("".join(accumulated) + "▌")
            else:
                final_resp = item
    except Exception as exc:
        # Graceful degradation: show error, return synthetic IDK response
        msg = str(exc)
        if "connection refused" in msg.lower() or "connect" in msg.lower():
            msg = "Ollama is not running. Start it with: `ollama serve`"
        err_text = f"⚠️ **Generation error:** {msg}"
        placeholder.error(err_text)
        final_resp = RAGResponse(
            answer=err_text,
            sources=[],
            routing=RoutingDecision(
                category="unknown", person_score=0.0, place_score=0.0,
                matched_entities=[], reasoning="error",
            ),
            retrieved_chunks=[],
            scores=[],
            latency_ms={"route": 0.0, "retrieve": 0.0, "llm": 0.0, "total": 0.0},
            is_idk=True,
        )
        return final_resp

    # Remove cursor from final render
    placeholder.markdown("".join(accumulated))
    return final_resp


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("📚 Local Wikipedia RAG Assistant")
st.caption(
    f"Fully local RAG over **{len(ENTITY_CATALOG)} Wikipedia entities** "
    f"({len(_PEOPLE)} people · {len(_PLACES)} places).  "
    f"Powered by **{model}** via Ollama."
)

# Empty state — show example query buttons before any conversation
if not st.session_state.messages:
    st.markdown("#### Try one of these:")
    ex_cols = st.columns(len(_EXAMPLE_QUERIES))
    for col, example in zip(ex_cols, _EXAMPLE_QUERIES):
        if col.button(example, use_container_width=True, key=f"ex_{example[:20]}"):
            st.session_state.pending_query = example
            st.rerun()
    st.markdown("---")

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("_resp") is not None:
            _render_sources(msg["_resp"])

# Determine active query (chat input OR example button)
active_query: str | None = None

if st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input := st.chat_input("Ask a question about a famous person or place…"):
    active_query = user_input

# ── Process new query ─────────────────────────────────────────────────────────

if active_query:
    # Guard: index must be built
    fresh_stats = _get_db_stats()
    if fresh_stats.get("error") or fresh_stats.get("total", 0) == 0:
        st.error(
            "**Index not found.**  "
            "Run `python scripts/02_build_index.py` to build the vector index first."
        )
        st.stop()

    # Apply threshold overrides (Retriever reads settings at query time)
    settings.similarity_threshold_low = threshold_low
    settings.similarity_threshold_mid = threshold_mid

    # Show user turn immediately
    st.session_state.messages.append({"role": "user", "content": active_query, "_resp": None})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Get (or create) pipeline
    try:
        pipeline = _get_pipeline(model, top_k)
    except Exception as exc:
        st.error(
            f"**Failed to initialize pipeline:** {exc}  \n\n"
            "Make sure Ollama is running: `ollama serve`"
        )
        st.stop()

    # Stream assistant response
    with st.chat_message("assistant"):
        resp = _stream_response(pipeline, active_query)
        if resp is not None:
            _render_sources(resp)
            lat = resp.latency_ms
            st.caption(
                f"retrieve={lat.get('retrieve', 0):.0f}ms  ·  "
                f"llm={lat.get('llm', 0):.0f}ms  ·  "
                f"total={lat.get('total', 0):.0f}ms  ·  "
                f"chunks={len(resp.retrieved_chunks)}  ·  "
                f"max_score={resp.max_score:.3f}"
            )

    # Persist to history
    if resp is not None:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": resp.answer,
            "_resp":   resp,
        })
