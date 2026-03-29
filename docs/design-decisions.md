# Aurora Q&A — Design Decisions

## 1. In-memory data store (no Redis, no SQLite)

**Decision:** Fetch all data from the Aurora API on startup, hold in memory. No external datastore.

**Why:** The full dataset is ~500KB (3,349 messages, 154 calendar events, 338 spotify streams, 31 whoop records). Redis or SQLite add deployment complexity for a persistence problem that doesn't exist at this scale. Startup re-fetch takes <10 seconds and avoids stale cache bugs.

## 2. RAG with in-memory vector index

**Decision:** Embed all data items at startup using `text-embedding-005`, retrieve top-30 items per query via cosine similarity.

**Why:** Context-stuffing the concierge member's full profile (~500+ items across messages, calendar, spotify, whoop) caused 30% malformed JSON responses and 4-18s latency. RAG reduces context to ~30 items, bringing latency under 2s and eliminating parse failures. Pre-normalized vectors enable pure-Python dot-product similarity with no external dependencies.

**Trade-off:** Top-K retrieval is lossy — the answer might not be in the top 30. Mitigated by strong semantic matching and per-member item counts (~335 messages, where K=30 captures ~9%). Falls back to full context-stuffing if the embedding API fails at query time.

## 3. Single LLM call (entity resolution + answer combined)

**Decision:** One Haiku call resolves the member name and answers the question.

**Why:** Two sequential calls would consume ~2 seconds (the full latency budget). Combining them into one call halves latency. The member list is only 10 names (~50 tokens of overhead). The reasoning trace captures the resolution step for traceability.

## 4. Gemini 2.5 Flash on Vertex AI

**Decision:** Use Gemini 2.5 Flash via Vertex AI over Claude Haiku / GPT-4o.

**Why:** Fast inference (~1s for small contexts), free tier on Vertex AI, and the google-genai SDK supports both generation and embeddings — one client for both RAG and answering. Application Default Credentials mean no API key management; Cloud Run's service account gets Vertex AI access via IAM.

## 5. Structured JSON output mode

**Decision:** Use `response_mime_type="application/json"` with a `response_schema` instead of function calling.

**Why:** Gemini's function calling produced `MALFORMED_FUNCTION_CALL` errors with large contexts. JSON output mode is more reliable and lets us define the exact schema. The response is parsed with `json.loads` and validated in Python.

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
