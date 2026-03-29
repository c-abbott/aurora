# Aurora Q&A

A question-answering service for Aurora's concierge platform. Takes natural language questions about members and returns precise, source-cited answers grounded in member data (messages, calendar, Spotify, Whoop).

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

| Decision | Choice | Why |
|----------|--------|-----|
| **Retrieval** | RAG with `text-embedding-005` | Context-stuffing all data caused 30% JSON parse failures and 4-18s latency. Embedding + top-25 retrieval brings latency under 2s with source-type diversity. |
| **LLM** | Gemini 2.5 Flash on Vertex AI | Fast structured JSON output, same SDK for embeddings and generation. No API key management — Application Default Credentials on Cloud Run. |
| **Entity resolution** | Pre-LLM fuzzy matching + self-reference fallback | Resolves member before retrieval so only relevant data enters the context window. Handles partial names, prefixes, possessives, and "my/me" self-references. |
| **Confidence** | LLM self-assessment with rubric | No logprobs available. Rubric (1.0 = direct quote, 0.7-0.9 = inferred, 0.3-0.6 = weak, 0.0 = no data) ensures directional consistency. |

Full rationale in [`docs/design-decisions.md`](docs/design-decisions.md).

## Try it

```bash
# Health check
curl https://aurora-1052079892687.europe-west1.run.app/health

# Precision — specific facts from dense message history
curl -s -X POST https://aurora-1052079892687.europe-west1.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What specific wine did Sophia ask about?"}' | python3 -m json.tool

# Cross-source — Spotify data matched to the right member
curl -s -X POST https://aurora-1052079892687.europe-west1.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What music does James listen to while working out?"}' | python3 -m json.tool

# Fuzzy matching — "Soph" resolves to "Sophia Al-Farsi"
curl -s -X POST https://aurora-1052079892687.europe-west1.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Has Soph mentioned any restaurants?"}' | python3 -m json.tool

# Self-reference — "my" resolves to the concierge (James Fletcher)
curl -s -X POST https://aurora-1052079892687.europe-west1.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many steps did I take last week?"}' | python3 -m json.tool

# No data — correct 0.0 confidence, no hallucination
curl -s -X POST https://aurora-1052079892687.europe-west1.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Sophia'\''s blood type?"}' | python3 -m json.tool

# Unknown member — correct 0.0 confidence
curl -s -X POST https://aurora-1052079892687.europe-west1.run.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Zara like to eat?"}' | python3 -m json.tool
```

## Running locally

```bash
# Python 3.12+, uv, gcloud CLI authenticated with Vertex AI access
uv sync
uv run uvicorn main:app --port 8080
uv run pytest
```

## Production Readiness

> *If you were to scale this to 100,000 members with 10 years of history each, what would be the first architectural change?*

**Replace the in-memory vector index with a dedicated vector database.**

At 100K members with 10 years of history, the dataset grows from ~4K items to roughly 1 billion (~10K items/member). The in-memory index — currently ~500KB — becomes ~3TB of 768-dim vectors. That doesn't fit in a single process.

A purpose-built vector store (pgvector, Qdrant, or Vertex AI Vector Search) gives you:

- **ANN search** for sub-100ms retrieval at billion scale, replacing the current O(n) brute-force scan
- **Pre-filtered search** — filter by `member_id` first, so retrieval cost scales with per-member item count (~10K), not total corpus size
- **Incremental indexing** — embed and upsert new items without reprocessing the full corpus

This is the first change because it unblocks everything else. Once the vector store exists, startup fetch becomes streaming ingestion (webhooks/change streams upsert into the store, the app server becomes stateless), and the application server scales horizontally with no shared state.

Entity resolution stays simple: in production the concierge is authenticated, so self-references resolve from the session. Third-party references ("What's Sophia's schedule?") match against the concierge's member portfolio (~10-20 people) — prefix matching works fine at that cardinality.
