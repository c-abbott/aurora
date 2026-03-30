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
| **Retrieval** | RAG with `text-embedding-005` | Context-stuffing all data caused 30% JSON parse failures and 4-18s latency. Embedding + top-25 retrieval with source-type diversity eliminated parse failures and cut latency to 1.5-4s. |
| **LLM** | Gemini 2.5 Flash on Vertex AI | Fast structured JSON output, same SDK for embeddings and generation. No API key management — Application Default Credentials on Cloud Run. |
| **Entity resolution** | Pre-LLM fuzzy matching + self-reference fallback | Resolves member before retrieval so only relevant data enters the context window. Handles partial names, prefixes, possessives, and "my/me" self-references. |
| **Latency** | 1.5-4s (precision over speed) | 92-97% of response time is the Gemini LLM call. We tested thinking budgets of 0, 512, and 1024 — lower budgets didn't reduce latency but did reduce answer quality. Chose to prioritise precision. |

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

**Replace the in-memory vector index with a dedicated vector database.** This is the first change because it unblocks everything else — but at Aurora's scale, the Q&A service is one component of a broader member intelligence system. Here's how the architecture evolves.

### The member context graph

The core problem Aurora is solving is **taste and judgment** — knowing that Lorenzo prefers pasta over sushi, that Sophia wants Château Latour not "a nice red," that Fatima's travel style is adventurous not resort-based. That knowledge lives in a context graph with layers optimised for different access patterns:

- **Hot layer** (Redis cluster or a feature store like Feast) — current location, calendar state, latest wearable readings. Sub-ms reads, always included in every LLM context window. Updated via event streams — calendar webhooks, wearable API polling, mobile location pings. ~1KB per member, fits entirely in memory
- **Cold layer** (Postgres with read replicas) — structured preferences, dietary restrictions, loyalty memberships, the derived preference "brief" (see Compounding memory below). Indexed SQL queries, no embeddings needed. "What's Lorenzo's dietary preference?" is a `WHERE member_id = ?` lookup, not a vector search. ~50KB per member at scale
- **Deep layer** (pgvector on Postgres to start, Pinecone or Qdrant when index size exceeds single-node RAM) — full conversation history and interaction logs. This is where embeddings earn their keep — finding "that Italian restaurant they loved in Rome last March" from thousands of interactions. Pre-filtered ANN search scoped by `member_id` partition, HNSW indexing, incremental upserts

Item growth is non-linear — long-tenured members accumulate more data types and message volume compounds. At 20-50K items per member, that's **2-5 billion vectors at 768 dimensions = 6-15TB of index**. Initial embedding cost: ~$2-5K one-time using a batch embedding API. Incremental cost is negligible — a few hundred new items per member per month.

### Streaming ingestion

Member signals (messages, calendar changes, wearable syncs, location updates) publish to a durable message queue (Kafka for ordering guarantees, or SQS/Pub/Sub for simplicity). A consumer service:

1. Writes the raw event to the interaction log (Postgres, append-only)
2. Embeds the text payload via a batch-aware embedding service (buffers for ~100ms to amortise API calls, flushes on timeout or batch-full)
3. Upserts the vector into the deep layer, partitioned by `member_id`
4. Updates the hot layer if the event affects real-time state (location, calendar)

Consistency model: **eventually consistent**, typically <500ms end-to-end from event to queryable vector. The hot layer is updated synchronously on the write path — a query immediately after a calendar change will see the new state. The deep layer has a short lag from the embedding step. A batch backfill pipeline handles historical data, re-embedding after model upgrades, and recovery from consumer failures (replay from the queue's retention window).

### Agentic retrieval and model routing

At 20-50K items per member, the current flat top-25 retrieval captures <0.1% of history. The retrieval layer becomes an [orchestration agent](https://www.anthropic.com/engineering/building-effective-agents) that reasons about *how* to search:

- **Query classification** (lightweight model or structured output from a fast LLM call, ~50ms) — determines query type and routes to the right storage layer. Simple factual queries ("dietary preference?") hit the cold layer directly and skip embeddings entirely. Temporal queries ("last Tuesday") apply date-range filters as pre-filters on the ANN index. Cross-source queries ("how did sleep affect meetings?") issue parallel retrievals across data types with temporal alignment
- **Two-stage retrieval** on the deep layer — coarse ANN returns top-100 candidates (~20ms with HNSW), a cross-encoder re-ranks to top-25 (~100ms). Total retrieval budget: <200ms
- **Model routing** by classified query complexity:
  - Simple factual → fast model, minimal thinking budget, **<1s end-to-end**
  - Multi-evidence reasoning → mid-tier model, moderate budget, **1-2s**
  - Cross-source synthesis → frontier model, full thinking budget, **2-4s**
  - This solves today's 1.5-4s uniform latency by making the common case (estimated ~60% of queries) fast

### Compounding memory

Every interaction produces signal — explicit ("I don't like spicy food"), implicit (consistently chooses window seats), and negative (ignores repeated cocktail bar suggestions). This is what makes a concierge better over time:

- **Raw interaction log** (Postgres, append-only) — every message, recommendation, booking, acceptance, rejection, rating. The source of truth. Partitioned by member, retained indefinitely
- **Derived preference model** — a structured member "brief" (~500-1000 tokens) that summarises tastes, patterns, and constraints. What a new human concierge would read before their first shift. Rebuilt periodically by an LLM job: "given these N recent interactions, update this member's preference brief." Decay-weighted so recent signals outweigh historical ones. Stored in the cold layer, always retrieved into the LLM context window
- **Active override layer** — explicit corrections from the member or curator that take immediate effect and persist until the next preference rebuild incorporates them

The preference brief is the critical abstraction — it compresses years of interaction history into a retrievable document that fits in a context window. The LLM provides reasoning; taste comes from what you put in front of it.

### What stays the same

The core architecture — embed, retrieve, generate — remains intact. Entity resolution becomes trivial: in production the concierge is authenticated, so self-references resolve from the session and third-party lookups are scoped to the concierge's member portfolio (~10-20 people).
