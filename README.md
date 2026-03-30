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

**Replace the in-memory vector index with a dedicated vector database.** This is the first change because it unblocks everything else — but at Aurora's scale, the Q&A service is really one component of a broader member intelligence system. Here's how the architecture evolves to support that.

### The member context graph

The core problem Aurora is solving is **taste and judgment** — knowing that Lorenzo prefers pasta over sushi, that Sophia wants Château Latour not "a nice red," that Fatima's travel style is adventurous not resort-based. That knowledge lives in a context graph with layers optimised for different access patterns:

- **Hot layer** (Redis or a feature store) — current location, calendar state, latest wearable readings. Sub-ms reads, always included in context. Updated continuously via event streams from mobile, calendar webhooks, wearable APIs
- **Cold layer** (Postgres) — structured preferences, dietary restrictions, loyalty memberships. SQL queries, no embeddings. "What's Lorenzo's dietary preference?" never needs a vector store
- **Deep layer** (pgvector to start, Pinecone or Qdrant at scale) — full conversation history and interaction logs. This is where embeddings earn their keep — finding "that Italian restaurant they loved in Rome last March" from thousands of interactions. Pre-filtered ANN search by `member_id`, incremental indexing

Item growth is non-linear — long-tenured members accumulate more data types and message volume compounds. At 20-50K items per member, that's 2-5 billion items and 6-15TB of vectors. Member signals (messages, calendar changes, wearable syncs, location updates) flow into the graph via a message queue (Kafka, SQS, Pub/Sub), embedded and upserted in real-time.

### Agentic retrieval and model routing

At 20-50K items per member, the current flat top-25 retrieval captures <0.1% of history. The retrieval layer becomes an [orchestration agent](https://www.anthropic.com/engineering/building-effective-agents) that reasons about *how* to search:

- A query classifier routes to the right layer — temporal queries apply date filters before semantic search, simple factual queries hit the cold layer directly, cross-source queries ("how did sleep affect meeting performance?") retrieve with temporal alignment
- Two-stage retrieval on the deep layer — coarse ANN returns top-100, a cross-encoder re-ranks to 25
- Model routing by query complexity: simple factual → fast model, <1s; multi-evidence → mid-tier; cross-source synthesis → frontier model with full thinking budget. This solves today's 1.5-4s latency by making the common case fast

### Compounding memory

Every interaction produces signal — explicit ("I don't like spicy food"), implicit (consistently chooses window seats), and negative (ignores repeated cocktail bar suggestions). This is what makes a concierge better over time, and the architecture needs to support it:

- **Raw interaction log** — append-only, the source of truth
- **Derived preference model** — periodically rebuilt from the log with decay weighting (recent > historical). This is a member "brief" — what a new human concierge would read before their first shift. The rebuild is itself an LLM job: "given these 50 recent interactions, update this member's preference brief"
- **Active override layer** — explicit corrections from the member or curator that take immediate effect

The preference brief is what gets retrieved into the context window. The LLM provides reasoning; taste comes from what you put in front of it.

### What stays the same

The core pattern — embed, retrieve, generate — remains intact. Entity resolution stays simple: in production the concierge is authenticated, so self-references resolve from the session and third-party lookups are scoped to ~10-20 members.
