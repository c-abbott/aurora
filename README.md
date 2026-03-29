# Aurora Q&A

A question-answering service for Aurora's premium concierge platform. Takes natural language questions about members and returns precise, source-cited answers grounded in member data (messages, calendar, Spotify, Whoop).

**Live URL:** https://aurora-1052079892687.europe-west1.run.app

## Architecture

```
                    +----------------------------+
                    |     Aurora Messages API     |
                    |  /messages  /calendar       |
                    |  /spotify   /whoop   /me    |
                    +-------------+--------------+
                                  |
                           (startup fetch)
                                  |
                                  v
+-----------------------------------------------------------+
|                     FastAPI on Cloud Run                   |
|                                                           |
|  +------------------+    +-----------------------------+  |
|  |   Data Store     |    |        POST /ask            |  |
|  |   (in-memory)    |    |                             |  |
|  |                  |    |  1. Fuzzy entity resolution  |  |
|  |  members:        |    |  2. Embed question           |  |
|  |   name ->        +--->|  3. Retrieve top-25 items    |  |
|  |    messages[]    |    |     (source-diverse)         |  |
|  |    calendar[]    |    |  4. Gemini 2.5 Flash call    |  |
|  |    spotify[]     |    |  5. Structured JSON response |  |
|  |    whoop[]       |    +-----------------------------+  |
|  |    vectors[]     |                                     |
|  +------------------+                                     |
+-----------------------------------------------------------+
```

### Key design decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Retrieval** | RAG with `text-embedding-005` | Context-stuffing all data caused 30% JSON parse failures and 4-18s latency. Embedding + top-25 retrieval brings latency under 2s with source-type diversity. |
| **LLM** | Gemini 2.5 Flash on Vertex AI | Fast inference, structured JSON output mode, same SDK for embeddings and generation. No API key management via Application Default Credentials. |
| **Data store** | In-memory, re-fetched on startup | Full dataset is ~500KB. No external datastore needed at this scale. |
| **Entity resolution** | Pre-LLM fuzzy matching + self-reference fallback | Resolves member before RAG retrieval so only relevant data enters the context window. Handles partial names, prefixes, possessives, and "my/me" self-references. |
| **Confidence** | LLM self-assessment with rubric | No logprobs available. Rubric (1.0 = direct quote, 0.7-0.9 = inferred, 0.3-0.6 = weak, 0.0 = no data) ensures directional consistency. |
| **Output** | `response_mime_type="application/json"` with schema | More reliable than function calling with large contexts. Schema-enforced fields: answer, confidence, sources, reasoning. |

Full rationale in [`docs/design-decisions.md`](docs/design-decisions.md).

### Data flow

1. **Startup** (~20s): Fetch all data from Aurora API, group by member, embed all items with `text-embedding-005` in batches of 250, L2-normalize vectors.
2. **Request** (~1-2s): Fuzzy-match member name -> embed question -> retrieve top-25 items with source-type diversity -> single Gemini 2.5 Flash call -> structured JSON response.
3. **Fallback**: If embedding fails at query time, falls back to full context-stuffing for the resolved member.

## Running locally

```bash
# Prerequisites: Python 3.12+, uv, gcloud CLI authenticated
# (needs Vertex AI access on GCP project aurora-491618)

# Install dependencies
uv sync

# Set environment (optional, defaults shown)
export GOOGLE_CLOUD_PROJECT=aurora-491618
export GOOGLE_CLOUD_LOCATION=europe-west1

# Run
uv run uvicorn main:app --port 8080

# Test
uv run pytest
```

## API

### `POST /ask`

```bash
curl -X POST https://aurora-1052079892687.europe-west1.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Amira'\''s favorite restaurant in Paris?"}'
```

```json
{
  "answer": "Based on her messages, Amina's favorite restaurant in Paris is Chez Janou.",
  "confidence": 0.95,
  "sources": ["msg_id_1", "msg_id_2"],
  "metadata": {
    "reasoning": "Resolved 'Amira' to Amina Van Den Berg via prefix match. Found two messages mentioning Chez Janou in Paris."
  }
}
```

### `GET /health`

Returns `200 {"status": "healthy", "members": 10}` when data is loaded, `503` otherwise.

## Production Readiness

> *If you were to scale this to 100,000 members with 10 years of history each, what would be the first architectural change?*

At 100K members with 10 years of history, the dataset grows from ~4K items to roughly **1 billion items** (~10K items/member). Three things break simultaneously:

### 1. Replace in-memory vectors with a dedicated vector database

The first change. At 1B items with 768-dim vectors, the index alone is ~3TB of memory. Move to a purpose-built vector store (pgvector, Pinecone, Qdrant, or Vertex AI Vector Search) that supports:

- **Approximate nearest-neighbor (ANN) search** for sub-100ms retrieval at billion scale, replacing the current O(n) brute-force scan.
- **Pre-filtered search** — filter by `member_id` before similarity search, so retrieval cost scales with per-member item count (~10K), not total corpus size.
- **Incremental indexing** — add new items without re-embedding the entire corpus.

### 2. Streaming ingestion replaces startup fetch

Re-fetching 1B items on every startup is not viable. Move to an event-driven pipeline: new messages arrive via webhook or change stream, get embedded incrementally, and are upserted into the vector store. The application server becomes stateless — it reads from the vector store and calls the LLM, with no startup cost.

### 3. Entity resolution becomes a retrieval problem

With 100K members, fuzzy prefix matching on names is O(n) and collision-prone. Replace with an embedding-based entity resolution step: embed the question, search a member-name index, and use the LLM only for ambiguous cases. This also enables resolving members by context ("the person who booked Le Cinq last week") rather than just by name.

These three changes — vector database, streaming ingestion, embedding-based entity resolution — form a natural first milestone. They address the immediate scaling bottlenecks while keeping the core architecture (embed, retrieve, generate) intact.
