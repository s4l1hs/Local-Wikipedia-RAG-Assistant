# Product Requirements Document
## Local Wikipedia RAG Assistant

| Field | Value |
|-------|-------|
| **Document Version** | 1.0 |
| **Date** | 2026-05-02 |
| **Course** | BLG483E — Information Retrieval and Search Engines |
| **Project** | 3 — Build a Local Wikipedia RAG Assistant |
| **Status** | Approved |

---

## Table of Contents

1. [Vision and Goal](#1-vision-and-goal)
2. [Target User Persona](#2-target-user-persona)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [User Stories](#5-user-stories)
6. [Success Metrics](#6-success-metrics)
7. [Out of Scope](#7-out-of-scope)
8. [Risk Register](#8-risk-register)
9. [Acceptance Criteria](#9-acceptance-criteria)
10. [Appendix A — Entity Catalog](#appendix-a--entity-catalog)
11. [Appendix B — System Architecture](#appendix-b--system-architecture)

---

## 1. Vision and Goal

### 1.1 The Real Problem This System Solves

Large Language Models (LLMs) suffer from a fundamental architectural tension: their knowledge
is **parametric** — encoded in billions of neural network weights during a training process that
ended at a fixed cutoff date. This architectural property causes three compounding failure modes
that make pure LLMs unsuitable for factual Q&A on specific entities:

**Problem 1 — Hallucination.** When a user asks "When was Ada Lovelace born?", the LLM
generates a token sequence that *maximizes plausibility given prior context* — not a sequence
that *reflects ground truth*. The model may answer "December 10, 1815" (correct) or
"October 3, 1812" (invented) with identical confidence and zero indication of uncertainty.
There is no mechanism in a pure LLM to distinguish a recalled fact from a confabulated one.

**Problem 2 — Temporal staleness.** Facts change. Messi's club affiliation, Taylor Swift's
discography count, and Hagia Sophia's UNESCO designation status are all facts that were
different at various training cutoffs. A pure LLM cannot know what it doesn't know, and it
cannot report "this information may be outdated."

**Problem 3 — Unverifiability and privacy cost.** Cloud LLM answers cannot be traced to a
source document — there is no audit trail. Additionally, sending queries to cloud APIs creates
ongoing costs, leaks query patterns to third-party servers, and creates a hard dependency on
external availability. For educational, research, and air-gapped environments, this is
structurally unacceptable.

### 1.2 Why RAG Solves These Problems

Retrieval-Augmented Generation (RAG) resolves the tension by **decoupling knowledge storage
from reasoning**:

- **Knowledge lives in a verifiable document store** (local Wikipedia articles, indexed in a
  vector database). Every fact has a known provenance.
- **Reasoning is performed by the LLM**, but it is *constrained* to the retrieved context via
  a system prompt. The LLM becomes a language processor, not a knowledge oracle.
- **Every claim is structurally traceable** to a specific Wikipedia chunk. Answers that
  reference a retrieved passage are grounded; answers that do not trigger fallback behavior.

The net result: a system where users receive factual answers about 40 specific entities, those
answers are derived from verifiable source documents, the entire pipeline runs on localhost
with zero external API calls, and the system honestly acknowledges when a question lies
outside its knowledge corpus.

### 1.3 Mission Statement

> Build a fully local, privacy-preserving, retrieval-augmented question-answering system that
> accurately answers natural language questions about 20 famous people and 20 famous places,
> using only Wikipedia data, a local embedding model, a local LLM, and a local vector database —
> with principled fallback when the query lies outside the known corpus.

### 1.4 Strategic Goals

| ID | Goal |
|----|------|
| G-1 | Ingest Wikipedia articles for ≥ 40 entities (20 people + 20 places) and persist them locally |
| G-2 | Enable fast semantic search over the knowledge base to retrieve relevant text chunks |
| G-3 | Generate grounded, factual answers using a local LLM constrained strictly to retrieved context |
| G-4 | Detect and gracefully handle out-of-scope queries with honest "I don't know" responses |
| G-5 | Run entirely on localhost — zero outbound network calls at query time |
| G-6 | Provide a usable chat-style interface (Streamlit UI and CLI mode) |

---

## 2. Target User Persona

### Persona A — The CS/AI Student (Primary User)

**Profile:** Undergraduate or graduate student in a computer science or AI program. Comfortable
with Python and basic ML concepts. Has used cloud LLM APIs before but wants to understand what
happens *inside* a RAG pipeline, not just call it from a library.

**Goals:**
- Understand how RAG works at the systems level (chunking strategy, embedding geometry,
  vector similarity, prompt engineering for grounding)
- Build a working AI application from scratch using language-native functionality
- Demonstrate the measurable difference between pure LLM generation (hallucination-prone)
  and retrieval-augmented generation (context-constrained)
- Complete a graded assignment with specific deliverables that an instructor can reproduce

**Pain Points:**
- API costs eating into experimentation budgets
- Black-box cloud services that obscure the mechanics being studied
- High-level libraries (LangChain, LlamaIndex) that abstract away the core concepts the
  course intends students to implement

**How this system helps:** Every component (embedding, cosine similarity, threshold logic,
prompt template, vector store query) is implemented at a level where the student understands
and can explain each decision — which is itself a graded evaluation criterion.

---

### Persona B — The Privacy-Conscious Researcher (Secondary User)

**Profile:** Academic or industry researcher working in domains where sending queries to
cloud LLM providers is problematic (sensitive topics, institutional policy, network
restrictions).

**Goals:**
- Query a bounded, known knowledge base without leaking query patterns to external services
- Maintain full audit trail linking each generated claim to a source document
- Operate in offline or restricted-network environments

**How this system helps:** Zero-egress architecture at query time. The ingestion step requires
internet access (to fetch Wikipedia), but all subsequent operation is fully air-gapped.

---

### Persona C — The Instructor / Evaluator

**Profile:** Course instructor who will evaluate the submitted project by cloning the
repository and following the README.

**Goals:**
- Verify the system runs on their machine following only README instructions
- Confirm retrieval is genuinely occurring (not the LLM using parametric memory)
- Assess the quality of architectural decisions, tradeoffs, and implementation

**How this system helps:** Well-documented codebase, observable retrieval (source chunks
displayable in UI), all assignment example queries pass, and the similarity threshold
mechanism makes retrieval failures transparent and diagnosable.

---

## 3. Functional Requirements

### 3.1 Data Ingestion

**FR-1: Wikipedia Article Fetching**

The system shall fetch the full plain-text content of each Wikipedia article for all 40
entities in the Entity Catalog (Appendix A) using the Wikipedia REST API. Fetching shall be
idempotent — re-running the ingestion script shall not re-download articles that already exist
locally. Raw article text shall be saved to `./data/raw/{entity_id}.txt`.

```
Endpoint: https://en.wikipedia.org/api/rest_v1/page/summary/{title}
          https://en.wikipedia.org/w/api.php?action=query&prop=extracts&...
```

Between requests the script shall pause for ≥ 1 second to respect Wikipedia's rate limits
(FR-1 is coupled with R-5 mitigation in the Risk Register).

---

**FR-2: Entity Metadata Persistence**

For each ingested article, the system shall insert a record into the `entities` table in a
local SQLite database (`./data/metadata.db`):

| Column | Type | Description |
|--------|------|-------------|
| `entity_id` | TEXT PK | URL-safe slug, e.g. `albert_einstein` |
| `entity_name` | TEXT | Display name |
| `entity_type` | TEXT | `"person"` or `"place"` |
| `wikipedia_url` | TEXT | Full URL of the Wikipedia article |
| `ingested_at` | TEXT | ISO-8601 timestamp |
| `article_word_count` | INTEGER | Word count of raw text |
| `embedding_model` | TEXT | Name of embedding model used (for compatibility checks) |

The `embedding_model` field is critical: if the user later switches embedding models, the
system detects the mismatch and prevents stale embeddings from being queried.

---

### 3.2 Text Chunking

**FR-3: Sentence-Boundary-Aware Chunking**

The system shall split each Wikipedia article into overlapping text chunks. Default parameters:

| Parameter | Default | Config Key |
|-----------|---------|------------|
| Chunk size | 400 tokens | `CHUNK_SIZE` |
| Overlap | 80 tokens | `CHUNK_OVERLAP` |
| Minimum chunk size | 50 tokens | `MIN_CHUNK_SIZE` |

**Chunking algorithm:**
1. Tokenize the article into sentences using `re.split(r'(?<=[.!?])\s+', text)`
2. Accumulate sentences into a chunk until `token_count(chunk) >= CHUNK_SIZE`
3. When a chunk boundary is reached, back-step by `CHUNK_OVERLAP` tokens (carry the last
   N tokens into the next chunk)
4. Never split mid-sentence — if adding the next sentence would exceed the limit, close
   the current chunk first

**Why 400/80 tokens:**
400 tokens (≈ 300 words) is large enough to contain a coherent biographical or geographical
paragraph with full contextual meaning. It is small enough that a single embedding vector
represents a focused semantic unit rather than blending multiple topics. The 20% overlap
(80 tokens) ensures that facts spanning chunk boundaries appear completely in at least one
chunk, preventing information loss at seams.

---

**FR-4: Chunk Metadata Storage**

Every chunk shall be stored in the `chunks` table in SQLite:

| Column | Type | Description |
|--------|------|-------------|
| `chunk_id` | TEXT PK | `{entity_id}_chunk_{index}` |
| `entity_id` | TEXT FK | References `entities.entity_id` |
| `entity_type` | TEXT | Denormalized for fast filtering |
| `chunk_index` | INTEGER | Position within the article |
| `raw_text` | TEXT | Full text of the chunk |
| `token_count` | INTEGER | Approximate token count |
| `source_url` | TEXT | Wikipedia URL |

This table serves as a ground-truth audit log independent of the vector store, and as a
recovery source if ChromaDB needs to be rebuilt.

---

### 3.3 Embedding and Vector Storage

**FR-5: Local Embedding Generation**

The system shall generate vector embeddings for all chunks using a **local** embedding model
with zero external API calls. The supported options, in priority order:

1. **Primary:** `nomic-embed-text` via Ollama (`ollama pull nomic-embed-text`)
   — 768-dimensional embeddings, strong semantic quality
2. **Fallback:** `sentence-transformers` with `all-MiniLM-L6-v2`
   — 384-dimensional embeddings, pure pip install, no Ollama required

The system shall check at startup which model is available and select accordingly. The
chosen model name is written to `entities.embedding_model` and checked on each query.

---

**FR-6: Single Vector Store with Metadata (Option B)**

The system shall use a **single** ChromaDB persistent collection named `wikipedia_rag`
with metadata filtering, rather than two separate collections (Option A).

Each document in the collection shall include:

```python
{
  "id": "albert_einstein_chunk_003",
  "embedding": [...],          # vector from FR-5
  "document": "chunk raw text",
  "metadata": {
    "entity_id": "albert_einstein",
    "entity_name": "Albert Einstein",
    "entity_type": "person",         # enables metadata filtering
    "chunk_index": 3,
    "source_url": "https://en.wikipedia.org/wiki/Albert_Einstein"
  }
}
```

**Design rationale (Option B over Option A):**

| Criterion | Option A (Two stores) | Option B (One store + metadata) |
|-----------|----------------------|----------------------------------|
| Cross-entity queries | Requires two sequential queries + manual merge | Single query with `where={}` or no filter |
| Comparison queries ("Compare Messi and Ronaldo") | Must query both stores and merge | Single query, rank by similarity |
| Code complexity | Two collection objects, two query paths | One collection, one query path |
| Future extensibility | Adding a third type requires a new collection | Adding a new type requires zero code change |
| ChromaDB metadata filter overhead | N/A | O(1), negligible at this scale |

Option B is strictly superior for this use case. The assignment's comparison queries
("Compare the Eiffel Tower and the Statue of Liberty", "Compare Messi and Ronaldo") would
be awkward to implement cleanly under Option A.

---

### 3.4 Query Processing and Retrieval

**FR-7: Intent Classification (Person / Place / Mixed)**

The system shall classify each incoming user query into an intent category to enable
metadata-filtered retrieval:

| Intent | ChromaDB Filter | Trigger Condition |
|--------|----------------|-------------------|
| `PERSON` | `{"entity_type": "person"}` | Query contains known person name OR person-indicator keywords |
| `PLACE` | `{"entity_type": "place"}` | Query contains known place name OR place-indicator keywords |
| `MIXED` | No filter (retrieve from all) | Query spans both types, or classification is uncertain |

**Classification algorithm (rule-based, in priority order):**

1. If query contains an exact entity name from the catalog → use that entity's type
2. If query mentions multiple entity names of different types → `MIXED`
3. If query contains person-indicator words: `who`, `was born`, `discovered`, `invented`,
   `wrote`, `painted`, `played for`, `won`, `died` → `PERSON`
4. If query contains place-indicator words: `where`, `located`, `city`, `country`,
   `monument`, `built`, `height`, `visited`, `travel` → `PLACE`
5. Default → `MIXED`

Misclassifying as `MIXED` is always safe (it just retrieves more broadly). This logic is
intentionally simple and extensible; it does not use an LLM for classification.

---

**FR-8: Semantic Retrieval with Two-Stage Similarity Threshold**

The system shall embed the user query with the same model used during ingestion, then query
ChromaDB for the top-k most similar chunks using L2 or cosine distance. Default `k = 5`.

The retrieved results carry similarity scores. The following threshold decision tree governs
whether generation proceeds:

```
sim_max = max cosine similarity score among top-k retrieved chunks

Interpretation for nomic-embed-text (768-dim, cosine similarity):
  ─────────────────────────────────────────────────────────────
  sim_max < 0.30  →  OUT OF SCOPE
                     → Return "I don't know based on available data"
                     → Do NOT call LLM (saves compute)

  0.30 ≤ sim_max < 0.45  →  LOW CONFIDENCE
                             → Call LLM but prefix response with:
                               "Based on limited available information..."

  sim_max ≥ 0.45  →  IN SCOPE
                     → Call LLM with full context, no qualifier
  ─────────────────────────────────────────────────────────────
```

**Threshold calibration protocol:** Run all "failure case" queries from the assignment spec
("Who is the president of Mars?", "Tell me about John Doe") and record their `sim_max`
values. Then run all in-scope example queries and record theirs. Set `RAG_LOW_THRESHOLD` to
sit above the maximum out-of-scope `sim_max` with a safety margin. If the distributions
overlap, prefer the lower threshold (accept some false negatives over hallucinating answers
for out-of-scope queries). Thresholds are configurable via environment variables
`RAG_LOW_THRESHOLD` and `RAG_MID_THRESHOLD`.

---

**FR-9: Context Assembly**

The system shall assemble a context string from the top-k retrieved chunks as follows:

1. Deduplicate: remove any chunks sharing the same `entity_id` and `chunk_index`
2. Sort by similarity score, descending
3. Truncate: if total token count exceeds `MAX_CONTEXT_TOKENS` (default: 2000), drop
   the lowest-scoring chunks until the limit is met
4. Format each chunk as `[Source: {entity_name} — {source_url}]\n{chunk_text}`

The 2000-token context budget is chosen to stay within the effective context window of
llama3.2 3B (4096 tokens) while leaving headroom for the system prompt (~200 tokens) and
the generated answer (~500 tokens).

---

### 3.5 Answer Generation

**FR-10: Grounded Generation via Local LLM**

The system shall send the assembled context and user query to the locally-running Ollama LLM.
The system prompt enforces context-only answering:

```
SYSTEM PROMPT:
──────────────────────────────────────────────────────────────────
You are a precise factual assistant. Your ONLY knowledge source is
the context provided below. You must follow these rules without exception:

1. Answer ONLY using information explicitly present in the context.
2. Do not use any external knowledge, training data, or general facts
   not found in the context.
3. If the context does not contain enough information to fully answer
   the question, respond with:
   "I don't know based on the available data."
4. If asked to compare two entities, use ONLY facts stated in the context
   for each entity. Do not invent differences or similarities.
5. Be concise. One to three paragraphs maximum.
6. Do not speculate, extrapolate, or add caveats about what might be true.

Context:
{assembled_context}
──────────────────────────────────────────────────────────────────

User question: {user_query}
Answer:
```

**Why two-sentence rule #3?** Instructing the LLM with the exact string it should output when
context is insufficient reduces the probability of a hybrid response that partly halluculates.
A model told to say "I don't know based on the available data" will be more likely to output
that exact phrase than if given a vague instruction to "decline to answer."

---

**FR-11: Two-Layer "I Don't Know" Mechanism**

The system guarantees an "I don't know" response through two independent layers:

| Layer | Mechanism | Trigger Condition |
|-------|-----------|-------------------|
| **Layer 1 — Retrieval** | Similarity threshold check (FR-8) | `sim_max < RAG_LOW_THRESHOLD` |
| **Layer 2 — Generation** | System prompt instruction (FR-10) | LLM determines context insufficient |

**Why two layers?**

Layer 1 catches clearly out-of-scope queries before wasting LLM compute
(e.g., "Who is the president of Mars?" will have `sim_max ≈ 0.10–0.20`).

Layer 2 handles edge cases where retrieved chunks are topically adjacent but
non-answering — for example, a query about "Einstein's favorite food" may retrieve
Einstein chunks with moderate similarity (because the query mentions Einstein), but
those chunks likely contain no information about food preferences. Layer 1 cannot catch
this; the LLM must recognize the gap.

**Mathematical grounding of Layer 1:**

If the embedding space is well-calibrated, queries about topics *not* represented in the
corpus will have low cosine similarity to all stored embeddings because their semantic
neighborhood is not populated. The threshold is an empirically-derived boundary on the
cosine similarity distribution:

```
P("I don't know" | query topic ∉ corpus) → 1  as  sim_max → 0
P("I don't know" | query topic ∈ corpus) → 0  as  sim_max → 1

The threshold θ = 0.30 is set such that:
  For all queries Q_out about topics outside the 40-entity corpus:
    P(sim_max(Q_out) < θ) > 0.95   (empirically validated during calibration)
  
  For all queries Q_in about topics within the 40-entity corpus:
    P(sim_max(Q_in) ≥ θ) > 0.95   (empirically validated during calibration)
```

---

**FR-12: Source Attribution Display**

The system shall optionally display the top-3 retrieved source chunks alongside each answer.
Each displayed source shall include:
- Entity name (e.g., "Albert Einstein")
- Wikipedia URL
- Similarity score
- The chunk text excerpt (first 150 characters)

This feature is toggled via a "Show Sources" checkbox in the UI (FR-13) or `--show-sources`
flag in CLI mode (FR-14).

---

### 3.6 Chat Interface

**FR-13: Streamlit Chat UI**

The system shall provide a Streamlit-based web interface accessible at
`http://localhost:8501`. Required UI elements:

- Text input at the bottom of the screen (chat-style)
- Scrollable conversation history showing alternating user/assistant messages
- "Show Sources" toggle in the sidebar (collapses/expands source chunks below each answer)
- "Clear Conversation" button that resets `st.session_state.messages = []`
- Sidebar panel listing all 40 indexed entities with their type badge

---

**FR-14: CLI Interface**

The system shall also support a terminal-based interface:

```bash
python main.py --mode cli [--show-sources] [--model llama3.2]
```

CLI mode shall display a `> ` prompt, accept readline input, print the answer to stdout,
and optionally print source chunks below the answer. The command `clear` or `reset` clears
the conversation history. `exit` or `quit` terminates the session.

---

**FR-15: Conversation Session State**

The system shall maintain the last N = 5 conversation turns in memory for display purposes.
**Only the current user query** is used for vector retrieval — the conversation history is
not appended to the embedding query. This prevents context drift where early turns corrupt
the retrieval signal for later questions.

---

## 4. Non-Functional Requirements

### 4.1 Performance

**NFR-1: End-to-End Response Latency**

The system shall return a complete answer within 30 seconds for a single-turn query on
consumer hardware (8 GB RAM, Apple M1 or equivalent x86_64 CPU at 2.5 GHz+):

| Step | Budget |
|------|--------|
| Query embedding | ≤ 1 s |
| ChromaDB retrieval | ≤ 0.5 s |
| LLM generation (llama3.2 3B) | ≤ 25 s |
| UI rendering | ≤ 0.5 s |
| **Total** | **≤ 30 s** |

**NFR-2: Ingestion Throughput**

The full ingestion pipeline (fetch + chunk + embed + store) for all 40 entities shall
complete within 30 minutes on the hardware profile above. Ingestion is a one-time offline
process; this budget does not apply to query time.

**NFR-3: Memory Footprint**

The running system (Ollama model in VRAM/RAM + ChromaDB + Streamlit) shall operate within
8 GB of total system RAM, enabling use on a standard student laptop.

---

### 4.2 Privacy and Security

**NFR-4: Zero Egress at Query Time**

After initial ingestion, the system shall make **zero outbound network requests** during
query processing. This shall be verifiable via OS-level network monitoring. Specifically:
no DNS queries, no HTTP calls, no socket connections to external addresses.

**NFR-5: Data Locality**

All ingested articles, embeddings, and conversation history shall be stored under
`./data/` within the project directory. No data shall be written outside the project tree
except Ollama model weights (managed by Ollama's own storage).

---

### 4.3 Portability

**NFR-6: Cross-Platform Compatibility**

The system shall run on:
- macOS 12+ (Apple Silicon and x86_64)
- Linux (Ubuntu 20.04+ or equivalent)
- Windows 10+ with WSL2

All Python dependencies shall be installable via `pip install -r requirements.txt`.
The system shall not require Docker, conda, or any runtime beyond Python 3.10+, pip,
and Ollama.

**NFR-7: Reproducible Setup**

A user with no prior knowledge of the project shall be able to complete setup by
following only the README instructions, in under 20 minutes excluding model download time.
The instructor grading criterion "runnable by the instructor by following only the README"
is the acceptance test for this requirement.

---

### 4.4 Maintainability

**NFR-8: Entity Extensibility**

Adding a new entity to the knowledge base shall require only: (1) appending the entity name
to the configuration list in `config.py`, (2) running `python ingest.py`. No other code
changes shall be required.

**NFR-9: Configurability**

The following parameters shall be modifiable in `config.py` or `.env` without source code
changes:

```
CHUNK_SIZE          = 400
CHUNK_OVERLAP       = 80
RETRIEVAL_K         = 5
MAX_CONTEXT_TOKENS  = 2000
RAG_LOW_THRESHOLD   = 0.30
RAG_MID_THRESHOLD   = 0.45
LLM_MODEL           = "llama3.2"
EMBEDDING_MODEL     = "nomic-embed-text"
```

---

## 5. User Stories

**US-1 — Factual Person Query**

> As a student researching historical scientists, I want to ask "What did Marie Curie
> discover?" and receive an accurate, concise answer derived from her Wikipedia article,
> so that I can verify key facts without reading the full article.

*Acceptance:* The system retrieves chunks from Marie Curie's article, generates a response
mentioning polonium and radium as her discoveries, and cites the Wikipedia source if "Show
Sources" is enabled.

---

**US-2 — Factual Place Query**

> As a history enthusiast, I want to ask "What was the Colosseum used for?" and receive
> a historically accurate answer about its function as a Roman amphitheater for gladiatorial
> games, so that I understand the site's historical significance before visiting Rome.

*Acceptance:* The system retrieves chunks from the Colosseum article, generates a response
grounded in those chunks that mentions gladiatorial games, public spectacles, and Roman
Empire context — without adding invented details.

---

**US-3 — Comparison Query**

> As a sports analyst, I want to ask "Compare Lionel Messi and Cristiano Ronaldo" and receive
> a structured comparison of their careers based solely on their respective Wikipedia articles,
> so that I can see what factual differences the encyclopedia documents.

*Acceptance:* The system classifies the intent as `PERSON` (both are people), retrieves chunks
from both Messi and Ronaldo articles, and generates a comparison that references only facts
present in those chunks. The answer does not invent statistics or opinions not found in Wikipedia.

---

**US-4 — Out-of-Scope Query (Graceful Failure)**

> As a user testing the system's limits, I want to ask "Who is the president of Mars?" and
> receive an honest "I don't know" response rather than a fabricated answer, so that I can
> trust that the system does not hallucinate for topics outside its knowledge base.

*Acceptance:* The retrieval similarity score `sim_max` falls below `RAG_LOW_THRESHOLD`, the
system triggers Layer 1 "I don't know" without calling the LLM, and the response clearly
states the information is not available in the knowledge base.

---

**US-5 — Geographically Anchored Mixed Query**

> As a curious user, I want to ask "Which famous place is located in Turkey?" and receive
> an answer correctly identifying the Hagia Sophia, so that the system demonstrates it can
> handle location-anchored queries without me specifying the entity name directly.

*Acceptance:* The system classifies intent as `PLACE`, retrieves chunks from the Hagia Sophia
article (which contains "Istanbul, Turkey"), and generates an answer identifying the Hagia
Sophia as the indexed famous place in Turkey.

---

**US-6 — Source Transparency**

> As a researcher who values academic rigor, I want to toggle "Show Sources" and see which
> specific Wikipedia chunks were used to generate each answer, so that I can verify the answer
> against the original text and assess retrieval quality directly.

*Acceptance:* When "Show Sources" is enabled, each answer is followed by the top-3 retrieved
chunks with entity name, Wikipedia URL, similarity score, and a text excerpt.

---

**US-7 — Ambiguity Between City and Landmark**

> As a curious user, I want to ask "Tell me about Istanbul" and receive a response about the
> city itself, even though the Hagia Sophia (a famous place in Istanbul) is also indexed, so
> that the system correctly distinguishes between an entity and its containing geography.

*Acceptance:* The system retrieves chunks from the Istanbul article (which is indexed) as
the highest-similarity match for the city query, and the Hagia Sophia chunks (if retrieved)
appear as lower-ranked context rather than the primary answer.

---

**US-8 — Session Management**

> As a user who wants to start a fresh topic, I want to click "Clear Conversation" and have
> all previous messages removed from the display immediately, so that previous conversation
> context does not affect the visual presentation of my new questions.

*Acceptance:* Clicking "Clear Conversation" empties `st.session_state.messages`. The
conversation display shows an empty state. The vector store is unaffected.

---

## 6. Success Metrics

### 6.1 The Three Critical Metrics

The following three metrics are the most direct measures of whether the system fulfills its
core purpose. If any one of these falls below its target, the system is not considered
successful regardless of other qualities.

---

#### Metric 1 — Retrieval Hit Rate @ k=5

**Why it is the most critical metric:** The entire RAG contract depends on retrieval
succeeding. If the relevant chunk does not appear in the top-5 results, the LLM has no
access to the correct information. No prompt engineering, no model quality, and no
downstream tuning can compensate for a retrieval miss. This is the structural foundation.

**Formal definition:**

```
Let B = benchmark set of N query-goldchunk pairs: {(q_i, c_i*)}
Let R(q, k) = set of top-k chunk IDs retrieved for query q

Hit Rate@k = |{i : c_i* ∈ R(q_i, k)}| / N
```

**Measurement protocol:**
1. Manually create 30 (query, gold_chunk_id) pairs by writing questions from the assignment
   spec and identifying which specific chunk in the vector store contains the answer
2. Run retrieval for each query with k=5
3. Record whether `gold_chunk_id ∈ top-5 results`

**Targets:**

| Query Category | Target |
|----------------|--------|
| Person queries | ≥ 87% |
| Place queries | ≥ 85% |
| Mixed/comparison queries | ≥ 80% |
| **Overall** | **≥ 85%** |

**Interpretation:** A Hit Rate of 85% on 30 queries means at most 4-5 queries fail at
retrieval. These should be documented as known limitations.

---

#### Metric 2 — Answer Grounding Rate

**Why it is the second most critical metric:** Even when retrieval succeeds, a small LLM
may ignore the retrieved context and generate from its parametric memory instead — what
researchers call "parametric override." This metric measures whether the RAG's grounding
constraint is actually holding. A system with 100% retrieval hit rate but 50% grounding
rate is not a RAG system; it is an LLM with a retrieval decoration.

**Formal definition:**

```
For a generated answer A with factual claims {f_1, f_2, ..., f_m}:
  A claim f_i is "grounded" if it can be identified as a substring or
  paraphrase of text in one of the k retrieved chunks.
  A claim f_i is "ungrounded" if it cannot be traced to any retrieved chunk.

Grounding Rate = (count of grounded claims) / (total factual claims)
across a manually audited sample of answers.
```

**Measurement protocol:**
1. Select 20 generated answers from in-scope queries where retrieval succeeded
2. For each answer, list all factual claims (names, dates, locations, achievements)
3. For each claim, search the retrieved chunks for the supporting text passage
4. Score = grounded_claims / total_claims

**Target:** ≥ 90%

**Interpretation:** A 90% grounding rate means ≤ 10% of factual claims are unverifiable
against the retrieved context. These unverifiable claims *may* be correct (the LLM has
accurate parametric knowledge about very famous entities like Einstein), but they cannot
be trusted without independent verification. The goal is to push this number as high as
possible via system prompt engineering.

---

#### Metric 3 — "I Don't Know" Precision and False Negative Rate

**Why it is the third most critical metric:** Confident hallucination on out-of-scope queries
is the most dangerous failure mode. A user who asks "Who is the president of Mars?" and
receives a fabricated name with a confident tone has been actively misled. The IDK mechanism
is the system's honesty guarantee. However, it must not over-trigger: refusing to answer
legitimate in-scope queries ("Who was Albert Einstein?") is a usability failure.

**Formal definitions:**

```
IDK Precision:
  Let O = set of N_out clearly out-of-scope queries
  IDK Precision = |{q ∈ O : system returns IDK}| / N_out

IDK False Negative Rate (over-refusal):
  Let I = set of N_in clearly in-scope queries
  IDK FNR = |{q ∈ I : system incorrectly returns IDK}| / N_in
```

**Measurement protocol:**
- Out-of-scope set (20 queries): "Who is the president of Mars?", "Tell me about John Doe",
  "What is the best pizza recipe?", "Who invented the internet?" (not in corpus), etc.
- In-scope set (20 queries): all 14 example queries from the assignment spec + 6 additional
  single-entity queries about catalog members

**Targets:**

| Measure | Target | Reasoning |
|---------|--------|-----------|
| IDK Precision | ≥ 95% | At most 1 out of 20 out-of-scope queries produces a hallucinated answer |
| IDK False Negative Rate | ≤ 5% | At most 1 out of 20 in-scope queries is wrongly refused |

---

### 6.2 Secondary Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| End-to-end latency (p50) | ≤ 30 s | Time 20 queries, record median |
| Ingestion time (40 entities) | ≤ 30 min | Wall-clock time of `python ingest.py` |
| Peak RAM usage at runtime | ≤ 8 GB | `psutil.Process().memory_info()` during 10-query session |
| Fresh setup time | ≤ 20 min | Timed on a machine with Python and Ollama pre-installed |
| ChromaDB collection size | ≥ 400 chunks | `collection.count()` after ingestion |

---

## 7. Out of Scope

The following capabilities are explicitly excluded from this project. This section exists to
prevent scope creep, clarify expectations for the instructor, and identify potential areas
for future work.

| # | Excluded Feature | Justification |
|---|-----------------|---------------|
| OOS-1 | **Multi-language support** | System operates in English only. Non-English queries are not handled or translated. Wikipedia articles are fetched from English Wikipedia exclusively. |
| OOS-2 | **Real-time or live Wikipedia updates** | The knowledge base is static after ingestion. Wikipedia edits made after ingestion are not reflected. This is a known limitation, not a bug. |
| OOS-3 | **Image, audio, or multimodal content** | The system is text-only. Wikipedia images, audio descriptions, and infobox data are not ingested or displayed. |
| OOS-4 | **Any external API calls** | No OpenAI, Anthropic, Google, Hugging Face Inference API, or any other external service. This is a hard constraint enforced by the assignment. |
| OOS-5 | **User authentication or multi-user sessions** | Single-user localhost application. No login, no session isolation, no authorization. |
| OOS-6 | **Entities outside the 40-entity catalog** | Queries about entities not in the catalog (Barack Obama, Mount Fuji, etc.) will trigger "I don't know" — this is correct behavior, not a bug. |
| OOS-7 | **Web search or live internet access at query time** | The system does not perform web searches during query processing. The ingestion step requires internet; query time does not. |
| OOS-8 | **LLM fine-tuning or weight modification** | The local LLM is used as-is via Ollama's inference endpoint. No training, fine-tuning, or quantization beyond what Ollama provides. |
| OOS-9 | **Production deployment or cloud hosting** | The system is localhost-only. Production deployment recommendations are documented in `recommendation.md`. |
| OOS-10 | **Automatic knowledge base refresh or staleness detection** | Re-ingestion is a manual process. No scheduled jobs, no Wikipedia change detection, no TTL-based invalidation. |
| OOS-11 | **Automated evaluation harness with ground-truth QA datasets** | Evaluation is performed manually using the benchmark protocol in Section 6. No integration with BEIR, RAGAS, or similar automated evaluation frameworks. |
| OOS-12 | **Semantic caching or query deduplication** | While listed as an optional extension in the assignment, caching is not a base requirement and is not in scope for the initial implementation. |

---

## 8. Risk Register

| ID | Risk Description | Likelihood | Impact | Mitigation Strategy |
|----|-----------------|-----------|--------|---------------------|
| **R-1** | **Small LLM quality insufficient for complex queries.** llama3.2 3B and phi3 may produce incoherent, grammatically broken, or logically inconsistent answers for comparison queries or multi-fact questions. This would cause Metric 2 (Grounding Rate) to appear high but actual answer quality to be low. | Medium | High | Test all three model options (llama3.2, phi3, mistral) on the full set of assignment example queries before committing to one. Select based on observed output quality. Document the comparison in `recommendation.md`. Expose model selection as a startup parameter so the instructor can switch models. |
| **R-2** | **Embedding model mismatch after re-ingestion.** If the user runs ingestion with `nomic-embed-text` but later queries with `sentence-transformers/all-MiniLM-L6-v2` (or vice versa), the vector space is incompatible — cosine similarities will be meaningless, retrieval will silently fail. | High | High | Store the embedding model name in `entities.embedding_model` in SQLite. At startup, compare the currently configured model against the stored value. If they differ, raise an error: `"Embedding model mismatch: ingested with X, configured for Y. Re-run ingest.py or set EMBEDDING_MODEL=X."` Never silently mix embeddings. |
| **R-3** | **Similarity thresholds miscalibrated for the chosen embedding model.** The default thresholds (0.30, 0.45) are calibrated for `nomic-embed-text` cosine similarity. If `sentence-transformers/all-MiniLM-L6-v2` is used instead, the similarity distribution is different and these thresholds may cause all out-of-scope queries to pass (threshold too low) or all in-scope queries to fail (threshold too high). | Medium | High | After ingestion, run a threshold calibration script that: (1) queries all failure-case queries and records `sim_max`; (2) queries all 14 assignment example queries and records `sim_max`; (3) recommends threshold values based on the observed gap. Print the recommended values and let the user confirm before updating `config.py`. |
| **R-4** | **LLM context window overflow on phi3 or long multi-entity queries.** phi3 has a 2048-token context window. With 5 chunks × 400 tokens + system prompt (~250 tokens) + user query (~50 tokens) = ~2300 tokens, this exceeds phi3's limit and will cause a generation error or silent truncation. | Medium | Medium | Implement a context length guard in the generation step: `if total_tokens > model_max_tokens: reduce k to k-1 and retry`. Default k values per model: llama3.2 → k=5; phi3 → k=3; mistral → k=5. Expose `MODEL_MAX_CONTEXT` as a config parameter. |
| **R-5** | **Wikipedia rate limiting during bulk ingestion.** Fetching 40 articles in rapid succession without delays may trigger HTTP 429 (Too Many Requests) responses, causing incomplete ingestion or script failure. | Low | Medium | Add `time.sleep(1.0)` between article fetch calls. Implement exponential backoff: on 429, wait 5s, 10s, 20s before retrying. Cache raw article text locally before processing, so a failed ingestion run can resume from the last successfully saved article rather than re-fetching all. |
| **R-6** | **LLM ignores system prompt (parametric override).** Small LLMs trained on large corpora about famous entities (Einstein, Shakespeare, Messi) may override the system prompt's context restriction and answer from training memory rather than retrieved chunks. This directly undermines the RAG contract and makes Metric 2 hard to pass. | Medium | High | Use the strongest possible grounding language in the system prompt (FR-10). Add a post-generation verification step: check if key entities/dates in the answer appear in the retrieved chunks. If not, append a disclaimer. Consider using a lower temperature (0.1–0.3) to reduce creative deviation from the provided context. |
| **R-7** | **ChromaDB persistence corruption across versions.** ChromaDB's on-disk format changes between versions. A pinned version in `requirements.txt` that differs from a previously-created database will cause import errors or silent data corruption. | Low | High | Pin the ChromaDB version exactly in `requirements.txt` (e.g., `chromadb==0.4.22`). Document the recovery procedure: `rm -rf ./data/chroma/ && python ingest.py`. Keep the SQLite `chunks` table as a complete recovery source — the entire ChromaDB collection can be rebuilt from it without re-fetching Wikipedia. |
| **R-8** | **Chunking splits critical facts across boundaries.** If a key biographical fact ("She was awarded the Nobel Prize in Chemistry in 1911") spans a chunk boundary, it may be split such that no single chunk contains the complete fact. Retrieval can then return a chunk with "Nobel Prize" but without "1911", causing the LLM to generate an incorrect year. | Medium | Medium | Use sentence-boundary-aware chunking (split on `[.!?]` rather than character count) combined with 20% overlap. Test by manually checking that 5 known key facts for 5 entities each appear complete in at least one chunk. If any fact is found split, reduce chunk size or increase overlap until all test facts are complete. |

---

## 9. Acceptance Criteria

The project is **complete and ready for submission** when ALL of the following criteria are
satisfied. Criteria are grouped by functional area.

---

### AC-1: Ingestion Complete and Verified

- [ ] All 40 entities from Appendix A are ingested and present in the SQLite `entities` table
- [ ] All 10 mandatory people from the assignment spec are confirmed present
- [ ] All 10 mandatory places from the assignment spec are confirmed present
- [ ] ChromaDB collection contains ≥ 400 document chunks (`collection.count() >= 400`)
- [ ] Each entity has ≥ 5 chunks (no entity reduced to fewer than 5 chunks due to a short article)
- [ ] Re-running `python ingest.py` with all articles already present completes without errors
  and does not create duplicate entries

---

### AC-2: Retrieval Meets Minimum Quality

- [ ] Retrieval Hit Rate@5 ≥ 85% on the 30-query benchmark set (defined in Section 6.1)
- [ ] Person intent queries retrieve person-type chunks in the top-2 results for ≥ 90% of
  clearly person-targeted queries
- [ ] Place intent queries retrieve place-type chunks in the top-2 results for ≥ 90% of
  clearly place-targeted queries
- [ ] All 14 example queries from the assignment spec return at least one relevant chunk with
  `sim_max ≥ 0.40`

---

### AC-3: Answers Are Grounded and Correct

- [ ] Answer Grounding Rate ≥ 90% across a 20-answer manual audit (Section 6.1, Metric 2)
- [ ] All 14 example queries from the assignment spec produce factually correct, non-empty
  answers (verified against Wikipedia)
- [ ] Zero generated answers for out-of-scope test queries contain specific invented proper
  nouns (names, dates, locations) that are not in the retrieved chunks

---

### AC-4: "I Don't Know" Behavior Is Correct

- [ ] "Who is the president of Mars?" → system returns IDK response, does NOT call LLM
- [ ] "Tell me about John Doe" → system returns IDK response
- [ ] IDK Precision ≥ 95% (≥ 19 out of 20 out-of-scope queries return IDK)
- [ ] IDK False Negative Rate ≤ 5% (≥ 19 out of 20 in-scope queries return substantive answers)

---

### AC-5: All Assignment Example Queries Pass

All of the following must produce factually correct, non-empty, non-hallucinated answers:

**People:**
- [ ] "Who was Albert Einstein and what is he known for?"
- [ ] "What did Marie Curie discover?"
- [ ] "Why is Nikola Tesla famous?"
- [ ] "Compare Lionel Messi and Cristiano Ronaldo"
- [ ] "What is Frida Kahlo known for?"

**Places:**
- [ ] "Where is the Eiffel Tower located?"
- [ ] "Why is the Great Wall of China important?"
- [ ] "What is Machu Picchu?"
- [ ] "What was the Colosseum used for?"
- [ ] "Where is Mount Everest?"

**Mixed:**
- [ ] "Which famous place is located in Turkey?" → correctly identifies Hagia Sophia
- [ ] "Which person is associated with electricity?" → correctly identifies Tesla (and/or Edison if indexed)
- [ ] "Compare Albert Einstein and Nikola Tesla"
- [ ] "Compare the Eiffel Tower and the Statue of Liberty"

**Failure Cases (must return IDK):**
- [ ] "Who is the president of Mars?" → IDK
- [ ] "Tell me about a random unknown person John Doe" → IDK

---

### AC-6: System Runs Fully Locally (Hard Constraint)

- [ ] Zero outbound network connections occur during query processing, verified via OS-level
  network monitoring (`netstat` or equivalent)
- [ ] All LLM inference served by local Ollama instance at `http://localhost:11434`
- [ ] All embeddings generated by local model (nomic-embed-text or sentence-transformers)
- [ ] System starts and operates correctly with no internet connection after ingestion

---

### AC-7: Interface Requirements Satisfied

- [ ] Streamlit UI accessible at `http://localhost:8501` and displays chat-style conversation
- [ ] CLI mode functional via `python main.py --mode cli`
- [ ] "Clear Conversation" in UI empties chat history without restarting the server
- [ ] Source chunks visible in UI when "Show Sources" is toggled on
- [ ] Sidebar (or equivalent) lists all 40 indexed entities

---

### AC-8: Repository Completeness

- [ ] Public GitHub repository with all required files:
  - [ ] `README.md` — installation, model setup, ingestion, startup, example queries
  - [ ] `product_prd.md` — this document
  - [ ] `recommendation.md` — production deployment analysis
  - [ ] `requirements.txt` — all dependencies with pinned versions
  - [ ] `ingest.py` — ingestion script
  - [ ] `main.py` — application entry point
  - [ ] `config.py` — configurable parameters
- [ ] Repository is reproducibly runnable by the instructor following only `README.md`
- [ ] Demo video (≥ 5 minutes) covering: system overview, live ingestion + Q&A demo, model
  choice rationale, retrieval method explanation, tradeoffs, and potential improvements
- [ ] Demo video link present in `README.md`

---

## Appendix A — Entity Catalog

### A.1 Selection Rationale

Entities were selected to maximize coverage across five dimensions:

1. **Wikipedia article depth** — articles with ≥ 5,000 words produce ≥ 12 chunks, enabling
   meaningful retrieval quality testing
2. **Domain diversity** — science, arts, sports, politics, philosophy, music ensure a broad
   semantic space for the embedding model
3. **Temporal diversity** — ancient (300 BCE) through contemporary (1989–) prevents era-clustering
4. **Geographic diversity** — entities from Europe, Americas, Asia, Africa, and the Middle East
5. **Deliberate ambiguity** — certain entity pairs share semantic neighborhoods
   (Istanbul + Hagia Sophia, Einstein + Tesla for electricity) to test retrieval precision

### A.2 People (20)

| # | Name | Era | Domain | Notes |
|---|------|-----|--------|-------|
| 1 | Albert Einstein | 1879–1955 | Physics | **Mandatory.** Physics disambiguation target with Tesla. |
| 2 | Marie Curie | 1867–1934 | Chemistry/Physics | **Mandatory.** First woman to win a Nobel Prize; gender diversity. |
| 3 | Leonardo da Vinci | 1452–1519 | Art/Science/Engineering | **Mandatory.** Archetypal polymath; cross-domain ambiguity test. |
| 4 | William Shakespeare | 1564–1616 | Literature | **Mandatory.** Canonical literary figure; long Wikipedia article (~9k words). |
| 5 | Ada Lovelace | 1815–1852 | Computing | **Mandatory.** First programmer narrative; useful contrast with Turing (not indexed). |
| 6 | Nikola Tesla | 1856–1943 | Electrical Engineering | **Mandatory.** Electricity association test with Einstein. |
| 7 | Lionel Messi | 1987– | Football | **Mandatory.** Living person; contemporary facts test temporal edge. |
| 8 | Cristiano Ronaldo | 1985– | Football | **Mandatory.** Paired with Messi for comparison queries. |
| 9 | Taylor Swift | 1989– | Music | **Mandatory.** Contemporary popular culture; long Wikipedia article. |
| 10 | Frida Kahlo | 1907–1954 | Visual Art | **Mandatory.** Mexican painter; non-European Western perspective. |
| 11 | Stephen Hawking | 1942–2018 | Theoretical Physics | Cosmology contrast with Einstein; disability representation. |
| 12 | Napoleon Bonaparte | 1769–1821 | Military/Politics | Historical political figure; ~15k word Wikipedia article. |
| 13 | Mahatma Gandhi | 1869–1948 | Activism/Politics | South Asian context; non-violence philosophy. |
| 14 | Nelson Mandela | 1918–2013 | Activism/Politics | African context; civil rights; Nobel Peace Prize. |
| 15 | Cleopatra VII | 69–30 BCE | Royalty/Politics | Ancient history; Egyptian context; name ambiguity (multiple queens). |
| 16 | Isaac Newton | 1643–1727 | Physics/Mathematics | Pre-modern physics; contrast with Einstein; gravity/calculus. |
| 17 | Wolfgang Amadeus Mozart | 1756–1791 | Classical Music | 18th century; contrast with Taylor Swift for music disambiguation. |
| 18 | Martin Luther King Jr. | 1929–1968 | Civil Rights | American activism; pairs with Gandhi and Mandela. |
| 19 | Pablo Picasso | 1881–1973 | Visual Art | Spanish painter; Cubism; pairs with Frida Kahlo for art queries. |
| 20 | Aristotle | 384–322 BCE | Philosophy/Science | Ancient Greek; foundational Western thinker; very long Wikipedia article. |

### A.3 Places (20)

| # | Name | Location | Type | Notes |
|---|------|----------|------|-------|
| 1 | Eiffel Tower | Paris, France | Monument | **Mandatory.** Comparison target with Statue of Liberty. |
| 2 | Great Wall of China | Northern China | Fortification | **Mandatory.** UNESCO; Chinese geographic context. |
| 3 | Taj Mahal | Agra, India | Mausoleum | **Mandatory.** UNESCO; South Asian context. |
| 4 | Grand Canyon | Arizona, USA | Natural Wonder | **Mandatory.** Natural geography; different from built monuments. |
| 5 | Machu Picchu | Cusco Region, Peru | Archaeological Site | **Mandatory.** Inca civilization; South American context. |
| 6 | Colosseum | Rome, Italy | Amphitheater | **Mandatory.** Ancient Roman; function/purpose query test. |
| 7 | Hagia Sophia | Istanbul, Turkey | Religious/Museum | **Mandatory.** Turkey-location query target; Byzantine/Ottoman duality. |
| 8 | Statue of Liberty | New York, USA | Monument | **Mandatory.** American symbol; comparison with Eiffel Tower. |
| 9 | Pyramids of Giza | Giza, Egypt | Ancient Monument | **Mandatory.** Oldest ancient wonder; Egypt context; links to Cleopatra. |
| 10 | Mount Everest | Nepal/Tibet border | Natural Wonder | **Mandatory.** Geographic superlative; highest point on Earth. |
| 11 | Stonehenge | Wiltshire, UK | Prehistoric Monument | Prehistoric era; mystery/archaeology angle; British context. |
| 12 | Angkor Wat | Siem Reap, Cambodia | Temple Complex | Southeast Asian context; largest religious monument; medieval era. |
| 13 | Petra | Ma'an Governorate, Jordan | Rock-carved City | Middle Eastern context; Nabataean civilization; unique geological architecture. |
| 14 | Amazon Rainforest | South America | Natural Region | Non-built landmark; biodiversity; different type from monuments. |
| 15 | Yellowstone National Park | Wyoming, USA | National Park | Geothermal features; second US natural site alongside Grand Canyon. |
| 16 | Venice | Veneto, Italy | City | Unique urban geography; canal city; long Wikipedia article (~8k words). |
| 17 | Acropolis of Athens | Athens, Greece | Archaeological Site | Ancient Greek context; pairs with Aristotle for Greece-anchored queries. |
| 18 | Tokyo | Japan | Megacity | Modern city; Asian urban context; very long Wikipedia article. |
| 19 | Istanbul | Turkey | City | **Deliberate ambiguity test:** Hagia Sophia is in Istanbul. Both indexed. Tests hierarchical geographic disambiguation. |
| 20 | Pompeii | Campania, Italy | Archaeological Site | Ancient Roman; Vesuvius eruption narrative; historical contrast with living cities. |

---

## Appendix B — System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║               INGESTION PIPELINE  (one-time, offline)               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Wikipedia REST API                                                  ║
║        │                                                             ║
║        ▼                                                             ║
║  [ FR-1 ] Raw article text → ./data/raw/{entity_id}.txt             ║
║        │                                                             ║
║        ▼                                                             ║
║  [ FR-3 ] Sentence-boundary chunker                                 ║
║           chunk_size=400 tokens, overlap=80 tokens                  ║
║        │                                                             ║
║        ├──────────────────────────────────────────────────────┐      ║
║        ▼                                                      ▼      ║
║  [ FR-4 ] SQLite chunks table               [ FR-5 ] Embedding Model ║
║           (audit log + recovery)                   (nomic-embed-text ║
║                                                     or MiniLM-L6-v2) ║
║                                                      │               ║
║                                                      ▼               ║
║                                             [ FR-6 ] ChromaDB        ║
║                                                    collection:       ║
║                                                    wikipedia_rag     ║
║                                                    (+ metadata)      ║
╚══════════════════════════════════════════════════════════════════════╝
                              │   persisted to ./data/
                              │
╔══════════════════════════════════════════════════════════════════════╗
║               QUERY PIPELINE  (interactive, zero-egress)            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  User Query                                                          ║
║      │                                                               ║
║      ▼                                                               ║
║  [ FR-7 ] Intent Classifier                                          ║
║           person / place / mixed                                     ║
║      │                                                               ║
║      ▼                                                               ║
║  [ FR-5 ] Query Embedding (same model as ingestion)                 ║
║      │                                                               ║
║      ▼                                                               ║
║  [ FR-8 ] ChromaDB Retrieval                                         ║
║           top-k=5, metadata filter by entity_type                   ║
║      │                                                               ║
║      ▼                                                               ║
║  Similarity Threshold Check                                          ║
║  sim_max < 0.30?  ──────────────────────────────────► IDK Response   ║
║      │ (sim_max ≥ 0.30)                                              ║
║      ▼                                                               ║
║  [ FR-9 ] Context Assembler                                          ║
║           dedup + rank + truncate to 2000 tokens                    ║
║      │                                                               ║
║      ▼                                                               ║
║  [ FR-10 ] Ollama LLM                                                ║
║            system prompt: "answer only from context"                ║
║            model: llama3.2 | phi3 | mistral                         ║
║      │                                                               ║
║      ▼                                                               ║
║  [ FR-11 ] Layer 2 IDK check (prompt-level)                          ║
║      │                                                               ║
║      ▼                                                               ║
║  [ FR-12 ] Answer + Sources                                          ║
║      │                                                               ║
║      ▼                                                               ║
║  [ FR-13 ] Streamlit UI  /  [ FR-14 ] CLI                            ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

*This document is intended to serve as the specification for both human developers and AI
code generation tools. All functional requirements are written at sufficient specificity
to be directly implementable without additional clarification.*

*Document end. Version 1.0 — 2026-05-02*
