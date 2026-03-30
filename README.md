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

# Self-reference — "my" resolves to the primary member (James Fletcher)
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

- **Hot layer** (Redis cluster or a feature store like Feast) — the real-time state of a member: current location, calendar for today, latest wearable readings. These are small (~1KB per member), fit entirely in memory, and are read in under a millisecond. They're always included in every LLM prompt because they're cheap to fetch and provide essential context (e.g., "the member is currently in Tokyo" changes which restaurants to suggest). Updated continuously via event streams — calendar webhooks, wearable API polling, mobile location pings
- **Cold layer** (Postgres with read replicas) — structured, queryable member data: dietary restrictions, loyalty memberships, airline preferences, the derived preference "brief" (see Compounding memory below). These don't need embeddings — "What's Lorenzo's dietary preference?" is a `WHERE member_id = ?` SQL query, not a similarity search. ~50KB per member at scale
- **Deep layer** (pgvector on Postgres to start, migrating to Pinecone or Qdrant when the index exceeds single-node RAM) — the full conversation history and interaction logs, where every message and event is stored alongside its embedding vector. This is where semantic search earns its keep — when a concierge needs to find "that Italian restaurant they loved in Rome last March" from thousands of interactions, exact keyword matching fails but vector similarity finds it. Queries are scoped to a single member's partition using pre-filtered ANN (approximate nearest neighbour) search with HNSW indexing, and new items are upserted incrementally as they arrive

Not every query needs to touch every layer. A simple factual question hits the cold layer and returns in milliseconds. A nuanced question about past experiences searches the deep layer. The query classifier (see Agentic retrieval) routes to the right layer.

Item growth is non-linear — long-tenured members accumulate more data types over time and message volume compounds as trust builds. A conservative estimate is 20-50K items per member, giving **2-5 billion vectors at 768 dimensions = 6-15TB of index**. Initial embedding cost: ~$2-5K one-time using a batch embedding API. Incremental cost is negligible — a few hundred new items per member per month.

### Streaming ingestion

The current system fetches all data on startup — viable at 4K items, impossible at 2-5 billion. Instead, member signals flow through a durable message queue (Kafka if you need strict ordering guarantees, SQS or Pub/Sub for simpler operations). When a member sends a message, updates their calendar, or syncs their wearable, that event publishes to the queue. A consumer service then:

1. Writes the raw event to the interaction log (Postgres, append-only) — this is the permanent record
2. Embeds the text content by batching events together (~100ms buffer window, flushed on timeout or batch-full) to amortise the cost of embedding API calls
3. Upserts the resulting vector into the deep layer, partitioned by `member_id`
4. Updates the hot layer if the event affects real-time state (e.g., a new calendar event or a location change)

The consistency model is **eventually consistent** — typically <500ms from event to queryable vector. The hot layer updates synchronously (a query immediately after a calendar change sees the new state), while the deep layer has a short lag from the embedding step. For failure recovery, the queue retains events for a configurable window, allowing the consumer to replay and backfill. The same backfill pipeline handles historical data imports and re-embedding when the embedding model is upgraded.

### Agentic retrieval and model routing

At 20-50K items per member, the current flat top-25 retrieval captures <0.1% of a member's history. The retrieval layer needs to become smarter — instead of always running the same vector search, it should reason about *how* to search based on what's being asked. This follows the [orchestrator-worker pattern](https://www.anthropic.com/engineering/building-effective-agents) where a lightweight orchestrator classifies the query and dispatches the right retrieval strategy:

- **Query classification** (~50ms, using a fast LLM call with structured output or a fine-tuned classifier) determines the query type and routes to the appropriate storage layer:
  - Simple factual ("What's Lorenzo's dietary preference?") → cold layer SQL lookup, no embeddings, <50ms
  - Temporal ("What did Amina do last Tuesday?") → date-range filter applied *before* the vector search on the deep layer, narrowing candidates before similarity matching
  - Cross-source ("How did James's sleep affect his meeting performance?") → parallel retrievals across wearable and calendar data types, aligned by date, then merged
- **Two-stage retrieval** for deep layer queries — a coarse ANN pass returns the top-100 candidates quickly (~20ms with HNSW indexing), then a cross-encoder (a model that scores query-document pairs more accurately than embedding similarity alone) re-ranks them to select the best 25 (~100ms). This dramatically improves recall without increasing the LLM context window. Total retrieval budget: <200ms
- **Model routing** — the same query classification determines which LLM to use. Not every question needs a frontier model with a large thinking budget:
  - Simple factual → fast model, minimal thinking, **<1s end-to-end**
  - Multi-evidence reasoning → mid-tier model, moderate budget, **1-2s**
  - Cross-source synthesis → frontier model, full thinking budget, **2-4s**
  - This solves today's 1.5-4s uniform latency by making the common case (estimated ~60% of queries) fast

### Compounding memory

A concierge that doesn't learn from past interactions is just a search engine. Every interaction produces signal — explicit ("I don't like spicy food"), implicit (consistently chooses window seats over aisle), and negative (ignores repeated cocktail bar suggestions). The architecture needs to capture all three and make them retrievable:

- **Raw interaction log** (Postgres, append-only, partitioned by member) — every message, recommendation, booking, acceptance, rejection, and rating. This is the immutable source of truth, retained indefinitely
- **Derived preference model** — a structured member "brief" (~500-1000 tokens) that distils years of interactions into a summary of tastes, patterns, and constraints. Think of it as the document a new human concierge would read before their first shift with a member. It's rebuilt periodically by an LLM job ("given these N recent interactions, update this member's preference brief"), with decay weighting so recent signals matter more than historical ones — a member who went vegan six months ago shouldn't still get steakhouse recommendations. Stored in the cold layer, always included in the LLM context window
- **Active override layer** — explicit corrections from the member ("actually I've started eating meat again") or curator that take immediate effect and persist until the next preference rebuild incorporates them

The preference brief is the critical abstraction — it compresses years of interaction history into a document that fits in a context window. The LLM provides reasoning; taste comes from what you put in front of it.

### What stays the same

The core architecture — embed, retrieve, generate — remains intact