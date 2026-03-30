# Aurora Q&A — Design Decisions

## 1. In-memory data store (no Redis, no SQLite)

**Decision:** Fetch all data from the Aurora API on startup, hold in memory. No external datastore.

**Why:** The full dataset is ~500KB (3,349 messages, 154 calendar events, 338 spotify streams, 31 whoop records). Redis or SQLite add deployment complexity for a persistence problem that doesn't exist at this scale. Startup re-fetch takes <10 seconds and avoids stale cache bugs.

## 2. RAG with in-memory vector index

**Decision:** Embed all data items at startup using `text-embedding-005`, retrieve top-25 items per query via cosine similarity with source-type diversity.

**Why:** Context-stuffing the primary member's full profile (~500+ items across messages, calendar, spotify, whoop) caused 30% malformed JSON responses and 4-18s latency. RAG reduces context to ~25 items, eliminating parse failures and reducing latency to 1.5-4s (see decision #8 for the latency breakdown). Pre-normalized vectors enable pure-Python dot-product similarity with no external dependencies.

**Source-type diversity:** Retrieval guarantees at least one item from each data source (messages, calendar, spotify, whoop) before filling remaining slots by relevance score. This prevents high-volume sources (messages) from crowding out cross-source evidence.

**Trade-off:** Top-K retrieval is lossy — the answer might not be in the top 25. Mitigated by strong semantic matching, source diversity, and per-member item counts (~335 messages, where K=25 captures ~7.5%). Falls back to full context-stuffing if the embedding API fails at query time.

## 3. Pre-LLM entity resolution + single LLM call

**Decision:** Resolve the member name with fuzzy matching before the LLM call, then answer in a single Gemini call.

**Why:** Resolving before retrieval means only one member's data enters the context window. The LLM call handles answering, confidence scoring, and source citation in one pass. Self-references ("my", "me") fall back to the primary member (identified via `/me`). **Prod change:** entity resolution becomes trivial — the member is authenticated, so self-references resolve from the session and third-party lookups are scoped to their ~10-20 contact portfolio.

## 4. Gemini 2.5 Flash on Vertex AI

**Decision:** Use Gemini 2.5 Flash via Vertex AI.

**Why:** Fast inference (~1-3s), structured JSON output mode, and the google-genai SDK supports both generation and embeddings — one client for both RAG and answering. Application Default Credentials mean no API key management; Cloud Run's service account gets Vertex AI access via IAM. **Prod change:** add latency-based model routing — simple factual queries to Flash, complex cross-source reasoning to Flash with higher thinking budget or Pro.

## 5. Structured JSON output mode

**Decision:** Use `response_mime_type="application/json"` with a `response_schema` instead of function calling.

**Why:** Gemini's function calling produced `MALFORMED_FUNCTION_CALL` errors with large contexts. JSON output mode is more reliable and lets us define the exact schema. The response is parsed with `json.loads` and validated in Python.

## 6. Data-driven multi-source routing

**Decision:** Include all non-empty data sources per member. No hardcoded name checks.

**Why:** Calendar/spotify/whoop data currently belongs to one member (James Fletcher), but routing is based on whether a member's profile has data for that source — not on checking the name. If the API adds spotify data for another member tomorrow, it just works.

## 7. Confidence via LLM self-assessment with rubric

**Decision:** The LLM rates its own confidence following a rubric (1.0 = direct quote, 0.7-0.9 = inferred, 0.3-0.6 = weak, 0.0 = no data).

**Why:** No logprobs available. A second LLM call for faithfulness scoring would blow the latency budget. Embedding similarity measures text similarity, not answer quality. A rubric makes the LLM's self-assessment directionally consistent — which is what the eval actually checks. **Prod change:** log confidence vs human feedback to calibrate the rubric over time.

## 8. Thinking budget of 1024 tokens (precision over latency)

**Decision:** Use `ThinkingConfig(thinking_budget=1024)` on Gemini 2.5 Flash, accepting 1.5-4s response times.

**Why:** We instrumented the request path and found that 92-97% of wall-clock time is the Gemini LLM call. Entity resolution, embedding, retrieval, and parsing together account for ~120ms — effectively free.

```
Component            Time        % of total
─────────────────    ─────────   ──────────
Entity resolution    <1ms        ~0%
Embed + retrieve     80-150ms    ~3-5%
Gemini LLM call      1,000-3,500ms  92-97%
JSON parse           <1ms        ~0%
```

We tested three thinking budgets:

| Budget | Latency | Precision | Verdict |
|--------|---------|-----------|---------|
| 0 (disabled) | <1.5s | Dropped — shallow answers, missed cross-source evidence | Rejected |
| 512 | 1.4-4.1s (no improvement) | Amira restaurant query dropped to 0.0 confidence | Rejected |
| 1024 | 1.5-4s | Correct on 30/30 eval queries, zero hallucinations | **Chosen** |

The brief says "aim for" <2s (aspirational) but ranks precision first in evaluation criteria. Since the only lever that meaningfully affects latency is the thinking budget, and reducing it degrades precision, we chose to prioritise answer quality. Reducing TOP_K or trimming prompt tokens would save ~100-200ms — not enough to move a 3.5s query under 2s.

## 9. Edge cases handled in prompt (no pre-LLM filtering)

**Decision:** No-match, no-data, and ambiguous member cases are all handled by prompt instructions, not deterministic code.

**Why:** Only "no matching member" could be handled pre-LLM. But with 10 users, the LLM call is cheap, and a rigid fuzzy matcher might reject valid creative references ("the Italian guy" → Lorenzo Cavalli). Cases 2 and 3 inherently require the LLM to have read the messages.

## 10. FastAPI + Cloud Run

**Decision:** FastAPI for the framework, Google Cloud Run for deployment.

**Why:** FastAPI: async, built-in OpenAPI docs, Pydantic models, zero debate. Cloud Run: user has a personal GCP project, one-command deploy, `min-instances=1` avoids cold starts during eval.

## 11. Re-fetch on every startup (no disk cache)

**Decision:** Always fetch fresh data from the Aurora API on startup.

**Why:** The dataset is small and the API is fast. Disk caching saves ~5 seconds on startup but introduces staleness risk and file management complexity. Not worth it.
