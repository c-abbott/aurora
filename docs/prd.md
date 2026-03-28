# Aurora Q&A Service — PRD

## Problem Statement

Aurora provides a premium concierge service. Members interact via messages requesting bookings, preferences, and arrangements. To serve members well, concierge staff need instant, accurate recall of a member's full history — their preferences, past requests, and patterns — even when that history is dense and spans months.

The task is to build a question-answering API that ingests member data from Aurora's Messages API and answers natural language questions with precision, traceability, and sub-2-second latency.

## Solution

A single FastAPI service deployed to Google Cloud Run that:

1. Pre-fetches and indexes all member data on startup (messages, calendar, spotify, whoop)
2. Groups data into per-member profiles
3. Exposes a `POST /ask` endpoint that resolves the target member, context-stuffs their full history into a single Claude Haiku 3.5 call, and returns a structured answer with confidence score, source message IDs, and reasoning trace

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data ingestion | Pre-fetch all on startup, hold in memory | Dataset is small (~500KB total). No external datastore needed. |
| Retrieval strategy | Filter by member + context-stuff all messages | 3,349 messages / 10 users = ~335 per user (~8K tokens). Fits in context. Lossless retrieval maximizes precision. |
| Entity resolution | LLM resolves member name in same call as answer | Saves a round trip. Handles partial names, misspellings, contextual references. |
| LLM | Claude Haiku 3.5 via tool use | Fast (~1s), cheap, sufficient for reading comprehension over small context. Tool use guarantees structured output. |
| Confidence scoring | LLM self-assessment with rubric | No logprobs available. Rubric ensures directional correlation (high = direct evidence, low = no data). |
| Multi-source routing | Include all non-empty sources per member | Data-driven — no hardcoded names. Calendar/spotify/whoop included only for members who have that data. |
| Edge cases | Handled in prompt | No-match, no-data, and ambiguous members all handled via prompt instructions. No pre-LLM filtering. |
| Deployment | Google Cloud Run | Personal GCP project. `min-instances=1` avoids cold starts during eval. |

## User Stories

1. As an evaluator, I want to ask a question about a member's preferences and get the correct answer, so that I can verify the system's precision
2. As an evaluator, I want the response to include source message IDs, so that I can trace the answer back to raw data
3. As an evaluator, I want the `metadata.reasoning` field to show the logical path from data to conclusion, so that I can assess traceability
4. As an evaluator, I want a meaningful confidence score, so that I can verify the system knows when it doesn't know
5. As an evaluator, I want a response in under 2 seconds, so that the service meets the latency requirement
6. As an evaluator, I want the system to handle "no data" queries gracefully, so that I can verify reliability
7. As an evaluator, I want the system to handle ambiguous or partial member names, so that I can verify robustness
8. As an evaluator, I want the system to handle questions spanning multiple data sources (messages + calendar + health), so that I can verify cross-source reasoning
9. As an evaluator, I want to read a clear README explaining the architecture, so that I can assess engineering quality
10. As an evaluator, I want a "Production Readiness" section describing scaling strategy, so that I can assess architectural thinking

## Architecture

```
                    ┌─────────────────────────┐
                    │     Aurora Messages API  │
                    │  /messages  /calendar    │
                    │  /spotify   /whoop  /me  │
                    └────────────┬────────────┘
                                 │
                          (startup fetch)
                                 │
                                 v
┌──────────────────────────────────────────────────────┐
│                   FastAPI Service                     │
│                                                      │
│  ┌──────────────┐    ┌───────────────────────────┐   │
│  │  Data Store   │    │     POST /ask             │   │
│  │  (in-memory)  │    │                           │   │
│  │               │    │  1. Receive question       │   │
│  │  members:     │───>│  2. Build prompt with      │   │
│  │   user_name → │    │     member list + data     │   │
│  │    messages[]  │    │  3. Single Haiku 3.5 call  │   │
│  │    calendar[]  │    │     (tool use)             │   │
│  │    spotify[]   │    │  4. Return structured      │   │
│  │    whoop[]     │    │     response               │   │
│  └──────────────┘    └───────────────────────────┘   │
│                                                      │
└──────────────────────────────────────────────────────┘
                         │
                    Google Cloud Run
```

### Data Flow

1. **Startup**: Paginate through all API endpoints, group data by member
2. **Request**: Parse question → build prompt with member list → single LLM call resolves member + answers → return structured response
3. **Response schema** (enforced via tool use):
   - `answer`: concise answer string
   - `confidence`: float, rubric-guided (1.0 = direct quote, 0.7-0.9 = inferred, 0.3-0.6 = weak, 0.0 = no data)
   - `sources`: list of message IDs cited
   - `metadata.reasoning`: step-by-step trace (member resolution → evidence scan → conclusion)

### Prompt Design

Single system prompt that instructs Haiku to:

1. Resolve the member from the question against the provided member list
2. If no match or ambiguous → low confidence response
3. Scan the member's data for relevant evidence
4. Answer using only the provided data — no fabrication
5. Rate confidence per the rubric
6. Cite source IDs for every claim

## Testing Strategy

### Unit Tests
- **Data ingestion**: Verify pagination fetches all records and groups by member correctly
- **Member profile construction**: Verify all non-empty sources are included per member
- **Response schema**: Verify Pydantic models accept/reject correct/malformed shapes
- **Prompt building**: Verify the prompt includes the right data for a given member

### Integration Tests
- **`POST /ask` happy path**: Known question → assert response has correct shape, valid source IDs, non-empty reasoning
- **`POST /ask` no-data**: Question about nonexistent topic → assert low confidence, appropriate answer
- **`POST /ask` unknown member**: Question about "Zara" → assert graceful handling

### Not In Scope (for hackathon)
- Load testing
- E2E browser tests
- Coverage thresholds

## Deliverables

1. **GitHub repository** (public) with:
   - `main.py` — FastAPI app with `POST /ask`
   - `data.py` — startup data fetching and member profile construction
   - `prompt.py` — prompt building and LLM interaction
   - `models.py` — Pydantic request/response schemas
   - `tests/` — unit and integration tests
   - `Dockerfile` — for Cloud Run deployment
   - `README.md` — architecture overview + Production Readiness section
2. **Live URL** — Cloud Run deployment
3. **README** with:
   - Architecture diagram
   - How to run locally
   - Production Readiness: at 100K members with 10 years of history, first change is vector embeddings (e.g., pgvector or Pinecone) for per-member retrieval, replacing context-stuffing with semantic search over chunked message history

## Out of Scope

- Authentication / API keys on the `/ask` endpoint
- Caching of LLM responses
- Streaming responses
- Multi-turn conversation (each question is independent)
- Updating member data after startup (no hot-reload)
- Admin UI or dashboard
- Rate limiting

## Further Notes

- The Messages API has no auth and no search/filter — only pagination with `skip`/`limit`
- The `/hackathon/me/` endpoint returns a single profile (James Fletcher) — this is context for the concierge's own identity, not a member lookup
- "Amira" in the example question likely maps to "Amina Van Den Berg" — expect the eval to test fuzzy name matching
- Total data: 3,349 messages (10 users), 154 calendar events, 338 spotify streams, 31 whoop records
