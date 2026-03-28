# Aurora Q&A — Design Decisions

## 1. In-memory data store (no Redis, no SQLite)

**Decision:** Fetch all data from the Aurora API on startup, hold in memory. No external datastore.

**Why:** The full dataset is ~500KB (3,349 messages, 154 calendar events, 338 spotify streams, 31 whoop records). Redis or SQLite add deployment complexity for a persistence problem that doesn't exist at this scale. Startup re-fetch takes <10 seconds and avoids stale cache bugs.

## 2. Context-stuffing over embeddings/RAG

**Decision:** Filter messages by member, pass their entire history to the LLM. No vector embeddings.

**Why:** ~335 messages per user = ~8K tokens. Fits comfortably in context. Embeddings are a lossy retrieval step — top-K can miss the message containing the answer. The eval scores **precision**, so giving the LLM every message maximizes the chance of a correct answer. Embeddings are discussed in the README as the scaling strategy for 100K+ members.

## 3. Single LLM call (entity resolution + answer combined)

**Decision:** One Haiku call resolves the member name and answers the question.

**Why:** Two sequential calls would consume ~2 seconds (the full latency budget). Combining them into one call halves latency. The member list is only 10 names (~50 tokens of overhead). The reasoning trace captures the resolution step for traceability.

## 4. Claude Haiku 3.5

**Decision:** Use Haiku 3.5 over Sonnet/Opus/GPT-4o.

**Why:** This is reading comprehension over ~8K tokens, not complex reasoning. Haiku is fast (~1s), cheap, and sufficient. Leaves headroom in the 2-second latency budget for network overhead.

## 5. Tool use for structured output

**Decision:** Define the response schema as a tool, forcing Haiku to return structured JSON.

**Why:** Guarantees valid JSON every time. No parsing logic, no retry on malformed output. Native to the Anthropic API.

## 6. Data-driven multi-source routing

**Decision:** Include all non-empty data sources per member. No hardcoded name checks.

**Why:** Calendar/spotify/whoop data currently belongs to one member (James Fletcher), but routing is based on whether a member's profile has data for that source — not on checking the name. If the API adds spotify data for another member tomorrow, it just works.

## 7. Confidence via LLM self-assessment with rubric

**Decision:** The LLM rates its own confidence following a rubric (1.0 = direct quote, 0.7-0.9 = inferred, 0.3-0.6 = weak, 0.0 = no data).

**Why:** Logprobs aren't available on Haiku. A second LLM call for faithfulness scoring would blow the latency budget. Embedding similarity measures text similarity, not answer quality. A rubric makes the LLM's self-assessment directionally consistent — which is what the eval actually checks.

## 8. Edge cases handled in prompt (no pre-LLM filtering)

**Decision:** No-match, no-data, and ambiguous member cases are all handled by prompt instructions, not deterministic code.

**Why:** Only "no matching member" could be handled pre-LLM. But with 10 users, the LLM call is cheap, and a rigid fuzzy matcher might reject valid creative references ("the Italian guy" → Lorenzo Cavalli). Cases 2 and 3 inherently require the LLM to have read the messages.

## 9. FastAPI + Cloud Run

**Decision:** FastAPI for the framework, Google Cloud Run for deployment.

**Why:** FastAPI: async, built-in OpenAPI docs, Pydantic models, zero debate. Cloud Run: user has a personal GCP project, one-command deploy, `min-instances=1` avoids cold starts during eval.

## 10. Re-fetch on every startup (no disk cache)

**Decision:** Always fetch fresh data from the Aurora API on startup.

**Why:** The dataset is small and the API is fast. Disk caching saves ~5 seconds on startup but introduces staleness risk and file management complexity. Not worth it.
