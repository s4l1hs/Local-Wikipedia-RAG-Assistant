# Vector Store Architecture Recommendation
## Option A vs Option B — Design Decision Report

| Field | Value |
|-------|-------|
| **Document Version** | 1.0 |
| **Date** | 2026-05-02 |
| **Project** | Local Wikipedia RAG Assistant (BLG483E — HW3) |
| **Decision Status** | **FINAL — Option B Selected** |
| **Supersedes** | N/A (initial design decision) |

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Option A — Two Separate Vector Stores](#2-option-a--two-separate-vector-stores)
3. [Option B — Single Vector Store with Metadata Filtering](#3-option-b--single-vector-store-with-metadata-filtering)
4. [Comparison Matrix](#4-comparison-matrix)
5. [Hybrid Approach Analysis](#5-hybrid-approach-analysis)
6. [Final Decision and Rationale](#6-final-decision-and-rationale)
7. [Impact on Implementation](#7-impact-on-implementation)
8. [Appendix — Mathematical Scale Analysis](#appendix--mathematical-scale-analysis)

---

## 1. Problem Definition

### 1.1 Why This Decision Matters

The vector store architecture is the single most consequential design choice in this project
because it dictates the shape of three other components: the ingestion pipeline, the query
router, and the retrieval logic. A wrong choice here propagates complexity into every
downstream module.

The assignment explicitly offers two options:

- **Option A:** Two separate ChromaDB collections — one named `people`, one named `places`
- **Option B:** One ChromaDB collection named `wikipedia_rag` with a metadata field
  `entity_type` that is either `"person"` or `"place"`

This sounds like a minor implementation detail. It is not. The choice determines how the
system handles its hardest class of queries — those that span both entity types — and whether
routing errors cause silent, unrecoverable retrieval failures.

### 1.2 The Routing Dependency

Before evaluating the two options, we must understand what the vector store architecture is
coupled to: **query intent classification**.

When a user submits a query, the system must decide *where to look*. The intent classifier
maps a natural language query to a retrieval target:

```
Query: "Why is Nikola Tesla famous?"
       ↓ intent classifier
Intent: PERSON
       ↓ routing
Target: people collection (Option A) | entity_type="person" filter (Option B)
```

The intent classifier uses a simple rule-based approach (see FR-7 in the PRD). It is not
a neural classifier. It is a heuristic. **It will be wrong in some cases.**

The central question of this decision: *when the classifier is wrong, which architecture
fails more gracefully?*

### 1.3 Scope of the Knowledge Base

To evaluate the options fairly, we must first establish the concrete scale of the system,
because the user asks explicitly whether this choice matters at this scale.

**Corpus composition:**
- 40 Wikipedia articles: 20 people + 20 places
- Average Wikipedia article length for our entity set: ~7,500 words
  (Einstein: ~10,000 words; Frida Kahlo: ~4,500 words; Hagia Sophia: ~7,000 words)
- Total raw corpus: ~300,000 words

**Chunk estimation (400 tokens ≈ 300 words, 20% overlap):**

```
Per-article chunks = ceil(words_per_article / (chunk_size_words × (1 - overlap_fraction)))
                   ≈ ceil(7,500 / (300 × 0.80))
                   ≈ ceil(7,500 / 240)
                   ≈ 32 chunks per article

With overlap recycling the math becomes:
  effective_step = chunk_size × (1 - overlap) = 300 × 0.8 = 240 words/step
  chunks = ceil(7,500 / 240) ≈ 32

Total chunks:
  Conservative (short articles only): 40 articles × 15 = 600 chunks
  Nominal:                             40 articles × 32 = 1,280 chunks
  Maximum (long articles, small step): 40 articles × 50 = 2,000 chunks

Realistic working estimate: ~1,000–1,400 total chunks
```

Under Option A, this splits approximately evenly: ~700 people chunks + ~700 place chunks.

**This scale is the foundation for the performance analysis in Section 4 and Appendix.**

---

## 2. Option A — Two Separate Vector Stores

### 2.1 Architecture

```
╔══════════════════════════════════════════════════════════╗
║               OPTION A — TWO COLLECTIONS                 ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Ingestion:                                              ║
║  ┌──────────────┐     entity_type == "person"            ║
║  │  Wikipedia   │ ──────────────────────────────────┐    ║
║  │  Articles    │                                   ▼    ║
║  └──────────────┘              ┌─────────────────────┐   ║
║                                │  ChromaDB           │   ║
║                                │  Collection:        │   ║
║                                │  "people"           │   ║
║                                │  (~700 chunks)      │   ║
║                                └─────────────────────┘   ║
║                                                          ║
║                     entity_type == "place"               ║
║  ┌──────────────┐ ──────────────────────────────────┐    ║
║  │  Wikipedia   │                                   ▼    ║
║  │  Articles    │              ┌─────────────────────┐   ║
║  └──────────────┘              │  ChromaDB           │   ║
║                                │  Collection:        │   ║
║                                │  "places"           │   ║
║                                │  (~700 chunks)      │   ║
║                                └─────────────────────┘   ║
╚══════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║               QUERY ROUTING (OPTION A)                   ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  User Query                                              ║
║      │                                                   ║
║      ▼                                                   ║
║  Intent Classifier                                       ║
║      │                                                   ║
║      ├─── PERSON ──────────► Query "people" only         ║
║      │                            │                     ║
║      ├─── PLACE ───────────► Query "places" only         ║
║      │                            │                     ║
║      └─── MIXED ──────────► Query BOTH collections       ║
║                              people_results + place_results
║                                   │                     ║
║                              MERGE & RE-RANK             ║
║                              (cross-collection sort)     ║
║                                   │                     ║
║                              Context Assembly            ║
╚══════════════════════════════════════════════════════════╝
```

### 2.2 Advantages of Option A

**A1 — Conceptual separation matches the domain model.**
The distinction between "person" and "place" is a first-class concept in the assignment
specification. Two collections make this separation physically explicit at the storage layer,
which aligns the data model with the domain.

**A2 — Smaller index per query (for PERSON or PLACE intents).**
A PERSON query searches only ~700 vectors instead of ~1,400. At very large scale (millions of
vectors), partitioned HNSW indices provide meaningful ANN search speedup. The index for each
subcollection is smaller and has better cache locality.

**A3 — Independent collection management.**
Each collection can be independently backed up, deleted, or rebuilt without touching the other.
If the people data becomes corrupted, re-running ingestion for people only does not affect
the places collection.

**A4 — Reduced false positives for single-type queries.**
A query about "Frida Kahlo" sent exclusively to the people collection cannot accidentally
surface a chunk from the Colosseum article. The collection boundary is a hard filter with
zero false-positive risk for typed queries.

**A5 — Independent tuning per domain.**
In a more advanced system, you might tune the embedding model, chunk size, or k separately for
people vs places (biographies benefit from different chunking strategies than architectural
descriptions). Option A makes this possible without a schema change.

### 2.3 Disadvantages of Option A

**A-D1 — Cross-collection distance scores are semantically incommensurable for merge ranking.**
This is the most technically serious problem with Option A. When a MIXED query requires results
from both collections, the system retrieves ranked results from each and must merge them into a
single ordered context. The problem: the distance scores from two separate ChromaDB collections
are not on the same scale.

Concretely:
```
people_results:  [(einstein_chunk_04, dist=0.18), (tesla_chunk_02, dist=0.31), ...]
places_results:  [(eiffel_chunk_01, dist=0.21), (colosseum_chunk_07, dist=0.28), ...]

How do you merge these? Sorting by distance naively may produce a biased ranking.
The people collection's HNSW index may have a different internal normalization than
the places collection's HNSW index, making 0.18 in people ≠ 0.18 in places.
```

In practice, because we use the same embedding model for both collections, the raw cosine
distances are on the same mathematical scale. But this reasoning requires understanding
ChromaDB internals, and the merge step still adds code complexity with no benefit over
Option B's native single-collection ranking.

**A-D2 — MIXED query path requires two network/disk calls and a manual merge step.**
Option A's MIXED query requires:
1. Embed the query (1 operation)
2. Query collection `people` (1 call)
3. Query collection `places` (1 call)
4. Merge, deduplicate, sort results (1 merge operation)

Option B's equivalent:
1. Embed the query (1 operation)
2. Query `wikipedia_rag` with no filter (1 call)

For every MIXED query, Option A does 2× the ChromaDB I/O and requires a non-trivial merge
step. Given that comparison queries are explicitly required by the assignment spec ("Compare
Messi and Ronaldo", "Compare the Eiffel Tower and the Statue of Liberty", "Compare Einstein
and Tesla"), MIXED queries are not edge cases — they are a required feature class.

**A-D3 — Routing misclassification causes hard, unrecoverable retrieval failure for one type.**
If the classifier incorrectly routes "Which scientist is associated with the construction of
the Eiffel Tower?" as PERSON (because it contains "scientist"), the places collection is never
queried. The system retrieves scientist chunks, finds no relevant context about the Eiffel
Tower, and either hallucinates or returns IDK — with no indication that the routing was the
cause.

With Option B, if the same query is sent with `where={"entity_type": "person"}` filter, we can
implement a graceful fallback:

```python
results = collection.query(q, where={"entity_type": "person"})
if max_similarity(results) < LOW_THRESHOLD:
    # Routing may have been wrong — retry without filter
    results = collection.query(q, where=None)
```

This two-step fallback is architecturally impossible in Option A without an explicit re-query
to the other collection — which is, again, re-implementing Option B.

**A-D4 — Entity-targeted forced retrieval is more complex.**
For comparison queries, the system needs to guarantee that at least one chunk from each named
entity appears in the context (otherwise the LLM cannot compare them). With Option A, if the
query "Compare Newton and the Colosseum" is classified as MIXED, forcing one Newton chunk
requires knowing that Newton is in the `people` collection and one Colosseum chunk requires
knowing it is in the `places` collection, then making two targeted queries. With Option B,
a single `where={"entity_id": {"$eq": "isaac_newton"}}` query works regardless of type.

**A-D5 — Increased operational surface area.**
Two collections means two persistent directories on disk, two ChromaDB indexes to maintain,
and two objects to initialize, validate, and reference throughout the codebase. For a
~1,400-chunk dataset, this operational overhead has no performance return.

### 2.4 Implementation Complexity Assessment

```python
# Option A — Initialization
people_collection = chroma_client.get_or_create_collection(
    name="people",
    metadata={"hnsw:space": "cosine"}
)
places_collection = chroma_client.get_or_create_collection(
    name="places",
    metadata={"hnsw:space": "cosine"}
)

# Option A — Ingestion routing
def ingest(entity_id, entity_type, chunks, embeddings):
    target = people_collection if entity_type == "person" else places_collection
    target.add(ids=[...], embeddings=[...], documents=[...], metadatas=[...])

# Option A — Query routing
def retrieve(query_embedding, intent, k=5):
    if intent == "PERSON":
        return people_collection.query(query_embeddings=[query_embedding], n_results=k)
    elif intent == "PLACE":
        return places_collection.query(query_embeddings=[query_embedding], n_results=k)
    else:  # MIXED
        p_results = people_collection.query(query_embeddings=[query_embedding], n_results=k//2+1)
        pl_results = places_collection.query(query_embeddings=[query_embedding], n_results=k//2+1)
        return merge_and_rerank(p_results, pl_results, total_k=k)

# The merge_and_rerank function must be implemented — it does not exist in ChromaDB
def merge_and_rerank(p_results, pl_results, total_k):
    all_items = []
    for i, doc in enumerate(p_results["documents"][0]):
        all_items.append({
            "document": doc,
            "distance": p_results["distances"][0][i],
            "metadata": p_results["metadatas"][0][i]
        })
    for i, doc in enumerate(pl_results["documents"][0]):
        all_items.append({
            "document": doc,
            "distance": pl_results["distances"][0][i],
            "metadata": pl_results["metadatas"][0][i]
        })
    # Sort by distance ascending (lower = more similar)
    all_items.sort(key=lambda x: x["distance"])
    return all_items[:total_k]
    # Problem: how many results to request from each collection?
    # k//2 + 1 may not be optimal — if the best 5 results are all from one collection,
    # this artificially limits to k//2 from the dominant collection.
```

The `merge_and_rerank` function introduces a subtle bias: it pre-limits each collection to
`k//2 + 1` results, meaning that even if the 5 most relevant chunks in the entire corpus are
all people chunks, the MIXED result will only contain `k//2 + 1` of them, wasting context
window space on less-relevant place chunks.

---

## 3. Option B — Single Vector Store with Metadata Filtering

### 3.1 Architecture

```
╔══════════════════════════════════════════════════════════╗
║               OPTION B — SINGLE COLLECTION               ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Ingestion:                                              ║
║  ┌──────────────┐                                        ║
║  │  Wikipedia   │  (all 40 entities)                     ║
║  │  Articles    │                                        ║
║  └──────────────┘                                        ║
║         │                                                ║
║         ▼                                                ║
║  ┌─────────────────────────────────────────────────┐     ║
║  │  ChromaDB Collection: "wikipedia_rag"           │     ║
║  │  (~1,400 chunks total)                          │     ║
║  │                                                 │     ║
║  │  einstein_chunk_00 │ entity_type: "person"  │   │     ║
║  │  einstein_chunk_01 │ entity_type: "person"  │   │     ║
║  │  eiffel_chunk_00   │ entity_type: "place"   │   │     ║
║  │  eiffel_chunk_01   │ entity_type: "place"   │   │     ║
║  │  ...               │ ...                    │   │     ║
║  └─────────────────────────────────────────────────┘     ║
╚══════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║               QUERY ROUTING (OPTION B)                   ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  User Query                                              ║
║      │                                                   ║
║      ▼                                                   ║
║  Intent Classifier                                       ║
║      │                                                   ║
║      ├─ PERSON ──► where={"entity_type": "person"}       ║
║      │                  │                               ║
║      ├─ PLACE ───► where={"entity_type": "place"}        ║
║      │                  │                               ║
║      └─ MIXED ───► where=None  (no filter, all corpus)   ║
║                         │                               ║
║              collection.query(                          ║
║                query_embeddings=[q_emb],                ║
║                n_results=k,                             ║
║                where=where_clause   ← single call       ║
║              )                                          ║
║                         │                               ║
║                    Results ranked by                    ║
║                    cosine similarity                    ║
║                    across all filtered                  ║
║                    chunks (native ChromaDB)             ║
║                         │                               ║
║              ┌──────────▼──────────────────────┐        ║
║              │  OPTIONAL FALLBACK               │        ║
║              │  if sim_max < LOW_THRESHOLD:     │        ║
║              │      retry with where=None       │        ║
║              └─────────────────────────────────┘        ║
╚══════════════════════════════════════════════════════════╝
```

### 3.2 Metadata Schema

Each document inserted into the `wikipedia_rag` collection carries the following metadata:

```python
{
    "id": "albert_einstein_chunk_003",   # globally unique chunk identifier

    "embedding": [...],                  # 768-dim vector (nomic-embed-text)
                                         # or 384-dim (MiniLM-L6-v2)

    "document": "Einstein was born on 14 March 1879...",  # raw chunk text

    "metadata": {
        # ── Primary filter fields ──────────────────────────────────────
        "entity_type":    "person",      # "$eq" filter for PERSON/PLACE routing

        # ── Entity identification ──────────────────────────────────────
        "entity_id":      "albert_einstein",   # URL-safe slug, FK to SQLite
        "entity_name":    "Albert Einstein",   # human-readable display name

        # ── Chunk provenance ───────────────────────────────────────────
        "chunk_index":    3,             # sequential position in article
        "source_url":     "https://en.wikipedia.org/wiki/Albert_Einstein",

        # ── Optional quality signals ───────────────────────────────────
        "article_section": "Early life", # Wikipedia section header (if parseable)
        "token_count":     412           # approximate chunk token count
    }
}
```

**Why each field:**

| Field | Purpose |
|-------|---------|
| `entity_type` | Primary routing filter: `where={"entity_type": {"$eq": "person"}}` |
| `entity_id` | Entity-targeted retrieval: `where={"entity_id": {"$eq": "albert_einstein"}}` — used for comparison query coverage enforcement |
| `entity_name` | Source attribution display: "Retrieved from: Albert Einstein (Wikipedia)" |
| `chunk_index` | Deduplication and ordering within an article; also enables adjacent-chunk expansion if needed |
| `source_url` | Citation display and provenance audit |
| `article_section` | Optional context signal for the LLM and for debugging retrieval quality |
| `token_count` | Context window budget tracking during context assembly |

### 3.3 ChromaDB Filter Syntax Examples

ChromaDB's `where` clause uses a JSON filter syntax. All examples below are for the
`wikipedia_rag` collection:

```python
# ── Typed routing filters ────────────────────────────────────────────────────

# All person chunks (PERSON intent)
where = {"entity_type": {"$eq": "person"}}

# All place chunks (PLACE intent)
where = {"entity_type": {"$eq": "place"}}

# No filter (MIXED intent — returns from entire corpus)
where = None  # or omit the where parameter entirely

# ── Entity-targeted retrieval (for comparison coverage enforcement) ──────────

# Force at least one chunk from Albert Einstein
where = {"entity_id": {"$eq": "albert_einstein"}}

# Force one chunk from Eiffel Tower (used to fill gaps in comparison queries)
where = {"entity_id": {"$eq": "eiffel_tower"}}

# ── Compound filters (ChromaDB AND syntax) ───────────────────────────────────

# Retrieve only from Einstein AND Tesla chunks (for a comparison query)
where = {
    "$or": [
        {"entity_id": {"$eq": "albert_einstein"}},
        {"entity_id": {"$eq": "nikola_tesla"}}
    ]
}

# Retrieve only places with more than 5 tokens (quality filter)
where = {
    "$and": [
        {"entity_type": {"$eq": "place"}},
        {"token_count": {"$gte": 50}}
    ]
}

# ── Fallback query (misclassification recovery) ──────────────────────────────

def retrieve_with_fallback(query_embedding, intent, k=5, low_threshold=0.30):
    where_clause = get_filter_for_intent(intent)  # returns dict or None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    sim_max = 1.0 - min(results["distances"][0])  # ChromaDB returns L2 distances

    if sim_max < low_threshold and where_clause is not None:
        # Routing may have been wrong — retry without type filter
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=None
        )

    return results
```

### 3.4 Advantages of Option B

**B1 — Globally optimal retrieval ranking for all query types.**
A single ChromaDB collection query returns the k most similar chunks from the *entire corpus*,
ranked by cosine similarity on a unified scale. There is no merge step, no cross-collection
distance normalization problem, and no artificial capping of results per category. The
similarity scores are directly comparable because they are computed within the same HNSW
index against the same vector distribution.

**B2 — Misclassification is self-healing via the fallback pattern.**
When intent classification fails, the system can retry without a filter at negligible extra
cost (one additional ChromaDB query taking < 1ms at this corpus scale). This graceful
degradation is structurally impossible in Option A without explicitly querying the other
collection — which defeats the purpose of having separate collections.

**B3 — Entity-targeted coverage enforcement is simple and uniform.**
Comparison queries ("Compare Einstein and Tesla") require guaranteed representation from each
named entity. With Option B, the entity coverage check is:

```python
# Elegant: entity_id is a metadata field on all chunks
missing_entities = named_entities_in_query - entities_in_retrieved_chunks
for entity_id in missing_entities:
    extra = collection.query(q_emb, k=2, where={"entity_id": {"$eq": entity_id}})
    results.extend(extra)
```

With Option A, this requires knowing which collection each missing entity belongs to — adding
a lookup table or if-else branching that grows with the complexity of the entity catalog.

**B4 — Single I/O path regardless of query intent.**
Every query type (PERSON, PLACE, MIXED) follows the same code path: one embedding call, one
ChromaDB query call, one result set. The routing decision only changes the `where` parameter.
This reduces the testable surface area and eliminates an entire class of routing bugs.

**B5 — Future-proof extensibility at zero cost.**
Adding a third entity type (e.g., "event", "organization", "artwork") requires:
- Option A: create a new collection, update all routing code, add a new code path for queries
  spanning the new type
- Option B: add `entity_type="event"` to new chunk metadata; update the intent classifier
  to recognize the new type; zero other changes

**B6 — Simpler operational maintenance.**
One collection to back up, one directory to delete when rebuilding, one `collection.count()`
to monitor. When the instructor runs the project on their machine, there is one fewer failure
mode: no possibility of accidentally having one collection stale while the other is fresh.

### 3.5 Disadvantages of Option B

**B-D1 — Type-specific query scans the full collection metadata.**
A PERSON query applies the `where={"entity_type": "person"}` filter to all ~1,400 chunks to
extract the ~700 person chunks before running ANN similarity search. Option A would have the
person index pre-partitioned. At our scale this is sub-millisecond. At millions of vectors
this could matter (see Appendix for full analysis).

**B-D2 — No hard isolation between entity types.**
If a bug in the ingestion pipeline accidentally tags all Einstein chunks as `entity_type="place"`,
a PERSON query will miss them entirely without a clear error message. Option A would store
them in the wrong collection, but the structural separation might make the bug more visible
during debugging.

**B-D3 — Metadata schema must be agreed upon upfront.**
The `entity_type` field in the metadata is the contract between ingestion and retrieval. If the
field name changes (e.g., renamed to `type` or `category`), all queries break silently because
ChromaDB does not enforce schema. Option A's routing by collection name is slightly more robust
to metadata field naming changes.

**B-D4 — Slightly higher memory footprint for the HNSW index.**
A single HNSW index over 1,400 vectors uses more memory than two HNSW indices over 700 vectors
each, because HNSW's neighborhood graph grows non-linearly with collection size. In practice,
at this scale (~1,400 vectors × 768 dimensions × 4 bytes ≈ 4.3 MB of raw vector data), the
HNSW overhead difference is a few megabytes at most — irrelevant.

---

## 4. Comparison Matrix

Each criterion is scored 1–5 (5 = best performance on that criterion).

| Criterion | Option A | Option B | Winner | Notes |
|-----------|----------|----------|--------|-------|
| **Query latency (PERSON/PLACE)** | 5 | 4 | A | Option A searches ~700 vectors vs ~1,400. Difference: sub-millisecond at this scale. |
| **Query latency (MIXED)** | 2 | 5 | B | Option A requires 2 queries + merge. Option B: 1 query. |
| **Cross-type result ranking** | 2 | 5 | B | Option A distances must be manually merged; Option B has native unified ranking. |
| **Misclassification resilience** | 1 | 5 | B | Option A: routing error → silent hard failure. Option B: fallback to no-filter query. |
| **Code complexity** | 2 | 5 | B | Option A: 2 collection objects, 3 query paths, merge function. Option B: 1 collection, 1 query path. |
| **Entity-targeted retrieval** | 3 | 5 | B | Option A requires knowing target collection per entity. Option B: uniform `entity_id` filter. |
| **Operational simplicity** | 3 | 5 | B | Option A: 2 indexes to maintain, backup, rebuild. Option B: 1. |
| **Scalability (> 1M vectors)** | 5 | 3 | A | At very large scale, partitioned indexes provide meaningful HNSW speedup. |
| **Future extensibility** | 2 | 5 | B | Adding a 3rd entity type requires new collection + routing code in A. In B: metadata value only. |
| **Debugging visibility** | 4 | 3 | A | Collection-level isolation makes entity type errors visible at the collection level. |
| **Assignment example query coverage** | 3 | 5 | B | All 4 comparison queries in the spec require MIXED retrieval — Option A handles these poorly. |
| **Implementation correctness risk** | 2 | 4 | B | Option A's merge step is a correctness hazard. Option B has no merge. |

**Totals (sum of scores):**

| Option | Total Score | Average |
|--------|-------------|---------|
| Option A | 34 / 60 | 2.83 |
| **Option B** | **54 / 60** | **4.50** |

---

## 5. Hybrid Approach Analysis

### 5.1 "Can We Have Both?"

A natural question: can we combine the benefits of both options by using Option A's physical
separation for single-type queries while maintaining a unified view for MIXED queries?

Three hybrid strategies exist. All have fatal flaws at this project's scale and complexity.

---

**Hybrid H1 — Option A + Manual Union View**

```
Two separate collections + a Python-level "union query" function that queries both
and merges results for MIXED intents.
```

This is exactly Option A with extra steps. The merge problem from A-D1 and A-D2 is not
solved — it is given a name. The result is Option A's full complexity (two collections,
two code paths) plus the merge function that Option B makes unnecessary.

**Verdict: Rejected. Strictly worse than Option B in all dimensions.**

---

**Hybrid H2 — Option B + Separate Full-Corpus Index for MIXED**

```
One unified collection (Option B) + a second "fast lookup" collection for typed queries.
Data is duplicated across collections.
```

This doubles storage and ingestion complexity without any retrieval quality improvement.
The filtering overhead in Option B at ~1,400 vectors is negligible, so the "fast lookup"
collection provides no measurable benefit.

**Verdict: Rejected. The premise (filtering overhead is significant) is false at this scale.**

---

**Hybrid H3 — SQLite Full-Text Search for Typed Queries + ChromaDB for Semantic**

```
Use SQLite FTS5 for keyword-based typed queries (exact name matches).
Use ChromaDB for semantic queries.
```

This is a reasonable production-grade hybrid search architecture (BM25 + vector). However:
- It introduces a second search technology (FTS5) alongside ChromaDB
- It requires query routing to decide which search backend to use
- The assignment explicitly requires ChromaDB as the vector store
- At this corpus size, the precision improvement from hybrid search is marginal

**Verdict: Out of scope for this project. Valid for recommendation.md's production section
as a future enhancement.**

---

### 5.2 The "BOTH" Query Problem and Its Correct Solution

The most challenging query class is not MIXED (which just means "query everything") but
**comparison queries with named entities from different types**. For example:

```
"Did Einstein ever visit the Eiffel Tower?"
```

This query requires:
1. Retrieval from Einstein's article (person)
2. Retrieval from the Eiffel Tower's article (place)
3. The LLM to synthesize whether these two entities are connected

Neither Option A nor Option B handles this automatically — the vector similarity alone will
not guarantee that both entities appear in the top-k results. The query embedding for
"Einstein visit Eiffel Tower" may have higher cosine similarity to Einstein chunks (because
the biographical content is semantically richer) and return 4 Einstein chunks and 1 Eiffel
Tower chunk — or vice versa.

**The correct solution (applicable under both options, but easier to implement in Option B):**

```python
def retrieve_comparison(query_embedding, named_entities, k=5):
    """
    Ensure at least one chunk from each named entity in context.
    """
    # Step 1: standard semantic retrieval
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    # Step 2: check entity coverage
    covered = {m["entity_id"] for m in results["metadatas"][0]}
    missing = set(named_entities) - covered

    # Step 3: fill gaps with entity-targeted queries
    for entity_id in missing:
        gap_fill = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            where={"entity_id": {"$eq": entity_id}}
        )
        if gap_fill["documents"][0]:
            results["documents"][0].append(gap_fill["documents"][0][0])
            results["metadatas"][0].append(gap_fill["metadatas"][0][0])
            results["distances"][0].append(gap_fill["distances"][0][0])

    # Step 4: re-sort by similarity, keep top k+len(missing)
    combined = list(zip(results["distances"][0], results["documents"][0], results["metadatas"][0]))
    combined.sort(key=lambda x: x[0])
    return combined[:k + len(missing)]
```

This pattern works elegantly in Option B because `entity_id` is a first-class metadata field
on all chunks. In Option A, the gap-fill step for a missing entity requires knowing which
collection it belongs to and querying that specific collection.

---

## 6. Final Decision and Rationale

### THE DECISION: **Option B — Single Vector Store with Metadata Filtering**

This is not a conditional recommendation. It is not "Option B for now, reconsider at scale."
At the defined scope of this project (40 entities, ~1,400 chunks, single-user localhost), and
given the specific query requirements of the assignment specification, **Option B is the
correct architectural choice**. Option A would be a mistake.

### 6.1 The Decisive Arguments

**Argument 1: The assignment's required queries demand cross-type retrieval.**

Four of the fourteen example queries in the assignment specification explicitly require
retrieving chunks from entities of different types, or from both collections simultaneously:

- "Which person is associated with electricity?" — person query but involves a concept
  (electricity) that could also be associated with places (power plants, Tesla's lab)
- "Compare Albert Einstein and Nikola Tesla" — two people, same type, but needs both
  entities represented in results
- "Compare the Eiffel Tower and the Statue of Liberty" — two places, same type
- "Which famous place is located in Turkey?" — place query that requires semantic search
  (no explicit entity name given)

And the hardest class, the comparison edge cases:
- "Compare Albert Einstein and the Eiffel Tower" (implied by the assignment's broader
  comparison query category)
- "Did Einstein visit the Colosseum?" (mixed-type cross-entity query)

Under Option A, every MIXED-intent query requires two collection queries plus a merge step.
This merge step has a subtle correctness bug (pre-limiting each collection to k//2 results
rather than letting the similarity score determine the best k results globally). Option B
handles all of these with a single query at no extra cost.

**Argument 2: Misclassification in Option A causes silent, unrecoverable failures.**

The intent classifier is rule-based. It will misclassify queries. Consider:

```
Query: "Which scientist built the most famous tower in France?"
Naive classification: PERSON (keyword "scientist")
Correct retrieval: needs both Gustave Eiffel (not in our corpus) and Eiffel Tower chunks
Actual retrieval: person chunks only → IDK response

Option A behavior: hard failure, no fallback possible within the query path
Option B behavior: sim_max on person-filtered results is low → fallback to no-filter → 
                   Eiffel Tower chunk surfaces (Eiffel Tower Wikipedia mentions Gustave Eiffel) → 
                   partial answer returned
```

The fallback pattern in Option B (`if sim_max < threshold: retry without filter`) transforms
a hard failure into a graceful degradation. This pattern is architecturally impossible under
Option A without explicitly querying the other collection — which means Option A without
this fallback is strictly less reliable than Option B.

**Argument 3: At ~1,400 chunks, Option A's performance advantage is mathematically zero.**

This is proven in the Appendix with concrete numbers. Summary:
- ChromaDB uses brute-force search for collections under ~10,000 vectors
- Brute-force comparison: ~700 vectors × 768 float32 multiplications ≈ 0.54M operations
  for Option A vs ~1,400 × 768 ≈ 1.08M operations for Option B
- On modern hardware (AVX2 SIMD): ~10 billion float ops/second → difference is **0.054ms**
- This is below the system's measurement noise floor

Anyone who chooses Option A for "performance" at this scale has made a premature optimization
error. The performance argument for Option A only becomes valid at ~100,000+ vectors, a scale
that is explicitly out of scope for this project.

**Argument 4: Option B scores 54/60 vs Option A's 34/60 across all architectural criteria.**

The comparison matrix in Section 4 evaluated 12 criteria. Option B wins on 9 of them.
Option A only scores higher on scalability (irrelevant at our scale), debugging visibility
(marginal), and single-type query latency (a difference of 0.054ms).

### 6.2 Addressing the Edge Case: "Einstein Eyfel Kulesi'ni Ziyaret Etti Mi?"

This is the user's specifically posed edge case: "Did Einstein visit the Eiffel Tower?"

This query contains two named entities from different types (Einstein = person, Eiffel Tower
= place). The intent classifier will likely classify it as PERSON (the query begins with
"Einstein" and contains no place-indicator keywords except the proper noun itself).

**Under Option A:**
1. Classifier → PERSON
2. Query people collection → returns Einstein chunks (correct) but no Eiffel Tower chunks
3. LLM receives only Einstein context → cannot answer whether Einstein visited the tower
4. Either hallucinates (bad) or returns IDK (technically correct but unsatisfying)
5. No architectural fallback available without code modification

**Under Option B:**
1. Classifier → PERSON (same misclassification)
2. Query with `where={"entity_type": "person"}` → returns Einstein chunks
3. sim_max for Eiffel Tower-related content in Einstein chunks: likely low (0.25-0.35)
   because Einstein's Wikipedia article probably doesn't mention the Eiffel Tower
4. **Fallback trigger**: sim_max < threshold → retry without filter
5. No-filter query returns best 5 chunks from entire corpus for this query
6. "Eiffel Tower" chunks are retrieved alongside Einstein chunks
7. LLM can now answer: "Einstein's Wikipedia article does not mention a visit to the
   Eiffel Tower, but the Eiffel Tower's article was completed in 1889 during Einstein's
   lifetime (1879-1955)."

The fallback is not guaranteed to produce a perfect answer, but it is guaranteed to be more
robust than Option A's hard failure. The edge case that breaks Option A is handled gracefully
by Option B.

### 6.3 The One Scenario Where Option A Would Be Correct

Option A is the right choice if and only if ALL of the following are true simultaneously:
1. The corpus has > 500,000 vectors per partition (where HNSW speedup is meaningful)
2. Cross-partition queries are rare or absent from the query workload
3. Intent classification accuracy is ≥ 99% (misclassification is negligible)
4. The development team has the bandwidth to implement and test the merge function correctly

None of these conditions hold for this project. Condition 1 is off by three orders of
magnitude. Conditions 2 and 3 are violated by the assignment's explicit cross-type comparison
queries and rule-based classifier. Condition 4 is violated by the single-developer, course
project context.

---

## 7. Impact on Implementation

### 7.1 Ingestion Script (`ingest.py`)

Under Option B, the ingestion script initializes a single collection and tags every chunk
with `entity_type` metadata:

```python
import chromadb
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,            # "wikipedia_rag"
    metadata={"hnsw:space": "cosine"}
)

def ingest_entity(entity_id, entity_name, entity_type, chunks, embeddings, source_url):
    """
    entity_type must be "person" or "place" — this is the routing contract.
    """
    ids = [f"{entity_id}_chunk_{i:03d}" for i in range(len(chunks))]
    metadatas = [
        {
            "entity_id":      entity_id,
            "entity_name":    entity_name,
            "entity_type":    entity_type,   # PRIMARY ROUTING FIELD
            "chunk_index":    i,
            "source_url":     source_url,
            "token_count":    len(chunk.split())
        }
        for i, chunk in enumerate(chunks)
    ]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
```

**Key constraint:** `entity_type` must be exactly `"person"` or `"place"` (case-sensitive).
Any deviation (e.g., `"Person"`, `"human"`, `"people"`) will break all metadata filters
silently. A constant should be defined in `config.py`:

```python
ENTITY_TYPE_PERSON = "person"
ENTITY_TYPE_PLACE  = "place"
```

### 7.2 Retrieval Logic (`retriever.py`)

The retrieval module maps intent to a ChromaDB `where` clause and implements the fallback:

```python
from config import (
    ENTITY_TYPE_PERSON, ENTITY_TYPE_PLACE,
    RETRIEVAL_K, RAG_LOW_THRESHOLD
)

INTENT_FILTERS = {
    "PERSON": {"entity_type": {"$eq": ENTITY_TYPE_PERSON}},
    "PLACE":  {"entity_type": {"$eq": ENTITY_TYPE_PLACE}},
    "MIXED":  None,   # no filter
    "UNKNOWN": None,  # no filter — default safe
}

def retrieve(query_embedding, intent, k=RETRIEVAL_K):
    where_clause = INTENT_FILTERS.get(intent, None)

    results = _query(query_embedding, k, where_clause)
    sim_max = compute_sim_max(results)

    # Misclassification recovery: if filtered result is low-confidence,
    # retry without the type filter to cast a wider net
    if sim_max < RAG_LOW_THRESHOLD and where_clause is not None:
        results = _query(query_embedding, k, where=None)
        sim_max = compute_sim_max(results)

    return results, sim_max

def _query(query_embedding, k, where):
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"]
    }
    if where is not None:
        kwargs["where"] = where
    return collection.query(**kwargs)

def compute_sim_max(results):
    if not results["distances"][0]:
        return 0.0
    # ChromaDB returns L2 or cosine distance (lower = more similar)
    # For cosine distance: similarity = 1 - distance
    min_dist = min(results["distances"][0])
    return 1.0 - min_dist
```

### 7.3 Intent Classifier Behavior Under Option B

Because the fallback pattern handles misclassification gracefully, the intent classifier
can be implemented conservatively — defaulting to `MIXED` (no filter) whenever uncertain:

```python
PERSON_KEYWORDS = {"who", "was born", "discovered", "invented", "wrote",
                   "painted", "won", "died", "studied", "developed"}
PLACE_KEYWORDS  = {"where", "located", "city", "country", "monument",
                   "built", "height", "visited", "travel", "tall", "stands"}

KNOWN_PEOPLE = {e["entity_id"] for e in ENTITY_CATALOG if e["type"] == "person"}
KNOWN_PLACES = {e["entity_id"] for e in ENTITY_CATALOG if e["type"] == "place"}

def classify_intent(query: str) -> str:
    q_lower = query.lower()
    
    # Priority 1: named entity detection
    person_mentioned = any(name in q_lower for name in KNOWN_PEOPLE)
    place_mentioned  = any(name in q_lower for name in KNOWN_PLACES)
    
    if person_mentioned and place_mentioned:
        return "MIXED"
    if person_mentioned:
        return "PERSON"
    if place_mentioned:
        return "PLACE"
    
    # Priority 2: keyword heuristics
    has_person_kw = any(kw in q_lower for kw in PERSON_KEYWORDS)
    has_place_kw  = any(kw in q_lower for kw in PLACE_KEYWORDS)
    
    if has_person_kw and not has_place_kw:
        return "PERSON"
    if has_place_kw and not has_person_kw:
        return "PLACE"
    
    return "MIXED"  # default: cast wide net, let fallback handle it
```

Note: under Option A, defaulting to `MIXED` has a real performance cost (two queries).
Under Option B, defaulting to `MIXED` has no performance cost — it's just a query without
a `where` clause.

### 7.4 Summary of Implementation Delta

| Component | Option A would require | Option B requires |
|-----------|----------------------|-------------------|
| Collection initialization | 2 collection objects | 1 collection object |
| Ingestion routing | `if entity_type == "person": people_col.add(...)` | Uniform: `collection.add(..., metadata={"entity_type": ...})` |
| Query routing | 3 code paths (person/place/mixed), merge function for mixed | 1 code path, `where` param changes only |
| Fallback on misclassification | Explicit re-query of other collection | Change `where=None` — same query structure |
| Entity-targeted gap fill | Lookup which collection holds entity, then query that collection | `collection.query(where={"entity_id": ...})` uniformly |
| Backup & recovery | `rm -rf ./data/chroma/people/ && python ingest.py --type person` | `rm -rf ./data/chroma/ && python ingest.py` |

---

## Appendix — Mathematical Scale Analysis

### A.1 Does Option A vs B Actually Matter at This Scale?

**Question posed by the project brief:** "At 40 documents and ~2,000–5,000 chunks, does this
choice actually matter?"

**Short answer:** The performance difference is zero. The correctness and reliability
difference is significant.

### A.2 ChromaDB's Search Algorithm at This Scale

ChromaDB uses two search strategies:

| Collection size | Search algorithm | Complexity |
|----------------|-----------------|------------|
| < ~10,000 vectors | Brute-force exact search | O(n × d) |
| ≥ ~10,000 vectors | HNSW approximate search | O(log n × d) |

At our corpus size (~1,400 vectors), ChromaDB uses **brute-force search**. There is no
approximation, no graph traversal, no HNSW neighborhood lookup. Every query computes the
cosine similarity between the query vector and every stored vector.

### A.3 Latency Calculation

```
Vector dimension:     d = 768  (nomic-embed-text)
Option A corpus:      n_A = ~700 vectors per collection
Option B corpus:      n_B = ~1,400 vectors total

Cosine similarity computation per vector:
  dot_product:    d multiplications + (d-1) additions ≈ 2d operations
  norm_query:     computed once, shared across all n vectors
  Total per comparison: ≈ 2 × 768 = 1,536 float operations

Total operations:
  Option A (single-type query): 700 × 1,536 = 1,075,200 ops
  Option A (MIXED query):      1,400 × 1,536 = 2,150,400 ops + merge overhead
  Option B (filtered query):   1,400 × 1,536 = 2,150,400 ops + 1,400 boolean checks
  Option B (MIXED query):      1,400 × 1,536 = 2,150,400 ops

Modern CPU throughput with AVX2 (256-bit SIMD, 8 float32/cycle at 3 GHz):
  throughput ≈ 8 × 3 × 10^9 = 2.4 × 10^10 float ops/second

Latency:
  Option A single-type:  1.075 × 10^6 / 2.4 × 10^10 ≈ 0.045 ms
  Option B filtered:     2.150 × 10^6 / 2.4 × 10^10 ≈ 0.090 ms
  Difference:            0.045 ms

In the context of the end-to-end query latency budget:
  LLM generation:     ~20,000 ms  (llama3.2 3B on CPU)
  ChromaDB latency:   ~0.045–0.090 ms
  Performance ratio:  ChromaDB latency = 0.0002% of total query time
```

**Conclusion:** The performance difference between Option A and Option B at this scale is
0.045 milliseconds — approximately 0.0002% of the total query latency budget. Any
architectural decision motivated by this difference is a premature optimization error.

### A.4 The Break-Even Point for Option A Performance Advantage

For HNSW to engage (and provide an asymptotic speedup over Option B's brute-force):

```
ChromaDB HNSW threshold: ~10,000 vectors

At what corpus size does Option A's single-collection search become meaningfully faster?
  Option A (typed) searches n/2 vectors in HNSW mode: O(log(n/2) × d)
  Option B (filtered) searches n vectors in HNSW mode: O(log(n) × d)
  Speedup ratio: log(n) / log(n/2) = log(n) / (log(n) - 1)

For n = 10,000:  ratio = 4.000 / 3.000 = 1.33x  (33% faster for Option A)
For n = 100,000: ratio = 5.000 / 4.699 = 1.06x  (6% faster for Option A)
For n = 1,000,000: ratio = 6.000 / 5.699 = 1.05x (5% faster for Option A)

Paradoxically, Option A's relative speedup DECREASES at larger n due to HNSW's
logarithmic scaling. The absolute latency difference at large n would still be meaningful
(e.g., 5ms vs 50ms at 1M vectors), but the relative advantage shrinks.
```

To reach the ChromaDB HNSW threshold for both collections simultaneously, the project would
need:
```
10,000 vectors per collection = 10,000 / 25 chunks per article ≈ 400 articles
× 2 entity types = 800 Wikipedia articles
```

The current project has 40 articles. To justify Option A on performance grounds, the corpus
would need to be **20× larger** than the current scope.

### A.5 Metadata Filter Overhead in Option B

One concern about Option B: does the `where={"entity_type": "person"}` filter add meaningful
overhead to brute-force search?

```
Brute-force with filter (Option B):
  For each of the n=1,400 stored vectors:
    1. Compute cosine similarity: ~1,536 float ops
    2. Check metadata: `metadata["entity_type"] == "person"` = 1 string comparison

String comparison overhead per vector:
  Modern Python string equality for short strings: ~50 ns

Total filter overhead: 1,400 × 50 ns = 70,000 ns = 0.07 ms

This is less than the brute-force compute time (~0.09 ms) and is not the bottleneck.
```

ChromaDB's implementation processes metadata filters inline with the vector scan — it does
not first filter the metadata and then compute similarities (which would require two passes).
The overhead is therefore purely the boolean predicate evaluation, which is negligible.

---

*This document constitutes the final, binding architectural decision for the Local Wikipedia
RAG Assistant project. Option B (single vector store with metadata filtering) is selected.
All implementation files (`ingest.py`, `retriever.py`, `config.py`) must conform to the
metadata schema defined in Section 3.2.*

*Document end. Version 1.0 — 2026-05-02*
