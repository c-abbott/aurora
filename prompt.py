"""Prompt construction and LLM interaction for Aurora Q&A."""

import json
import logging
import os

from google import genai
from google.genai import types as genai_types
from google.genai.types import GenerateContentConfig, ThinkingConfig

from data import DataItem, DataStore, MemberProfile, normalize
from models import AskResponse, ResponseMetadata

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-005"
TOP_K = 15
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "aurora-491618")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1")

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "1-2 sentence answer.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score 0.0-1.0 per the rubric.",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
            "description": "Up to 5 most relevant source IDs.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief trace (1-3 sentences): member resolution -> evidence -> conclusion.",
        },
    },
    "required": ["answer", "confidence", "sources", "reasoning"],
}


_STOPWORDS = frozenset({
    "the", "and", "for", "but", "not", "you", "all", "can", "had", "her",
    "was", "one", "our", "his", "has", "are", "who", "how", "what", "when",
    "where", "why", "does", "did", "been", "have", "this", "that", "with",
    "from", "they", "will", "would", "could", "should", "about", "which",
    "their", "there", "than", "then", "them", "each", "make", "like",
    "just", "over", "such", "take", "also", "most", "into", "some",
})


def _resolve_member(question: str, member_names: list[str]) -> str | None:
    """Best-effort fuzzy match of a member name from the question."""
    words = [
        w.removesuffix("'s").removesuffix("\u2019s").strip(".,!?;:").lower()
        for w in question.split()
    ]
    words = [w for w in words if len(w) >= 3 and w not in _STOPWORDS]

    best, best_score = None, 0
    for name in member_names:
        for part in name.lower().split():
            part = part.strip("-")
            for w in words:
                if w == part:
                    score = 3
                elif len(w) >= 3 and len(part) >= 3 and w[:3] == part[:3]:
                    score = 2
                else:
                    continue
                if score > best_score:
                    best, best_score = name, score
    return best if best_score >= 2 else None


def _build_system_prompt(concierge_summary: str) -> str:
    return f"""You are Aurora's concierge assistant. Answer questions about members using ONLY the provided data.

## Concierge Context
{concierge_summary}

## Instructions
1. The user message tells you which member was resolved and provides their data, filtered for relevance to the question. The resolution uses fuzzy name matching, so the name in the question may differ slightly from the member's actual name (e.g., "Amira" matches "Amina"). Trust the resolution and answer using the resolved member's data.
2. The provided data items are the most relevant to the question, retrieved via semantic search. Search them carefully for evidence.
3. Answer using ONLY the provided data. Never fabricate information.
4. Rate confidence per this rubric:
   - 1.0 = direct quote or explicit statement in the data
   - 0.7-0.9 = inferred from multiple data points
   - 0.3-0.6 = weak or indirect evidence
   - 0.0 = no matching member or no relevant data
5. Cite source IDs (message IDs, event IDs, stream IDs) for every claim.
6. Keep reasoning to 1-3 sentences: member resolution -> evidence -> conclusion.

## Edge Cases
- If the user message says no member was resolved: confidence 0.0, explain in answer.
- Member found but no relevant data for the question: confidence 0.0, explain.

Always use the answer_question tool to respond."""


def _format_member_data(member: MemberProfile) -> str:
    """Format a member's full data as compact text for context-stuffing."""
    sections = [f"## Data for {member.user_name}"]

    if member.messages:
        sections.append(f"\n### Messages ({len(member.messages)})")
        for m in member.messages:
            sections.append(f"[{m['id']}] {m['timestamp']}: {m['message']}")

    if member.calendar:
        sections.append(f"\n### Calendar ({len(member.calendar)})")
        for e in member.calendar:
            attendees = ", ".join(e.get("attendees", []))
            sections.append(
                f"[{e['id']}] {e['start']} - {e['end']} | {e['title']}"
                f" | {e.get('location', '')} | attendees: {attendees}"
                f" | {e.get('notes', '')}"
            )

    if member.spotify:
        sections.append(f"\n### Spotify ({len(member.spotify)})")
        for s in member.spotify:
            sections.append(
                f"[{s['stream_id']}] {s['timestamp']}"
                f" | {s['title']} | {s.get('artist_or_show', '')}"
                f" | {s.get('context', '')}"
            )

    if member.whoop:
        sections.append(f"\n### Health/Whoop ({len(member.whoop)})")
        for w in member.whoop:
            r = w.get("recovery", {})
            sl = w.get("sleep", {})
            st = w.get("strain", {})
            sections.append(
                f"[whoop_{w['date']}] Recovery: {r.get('score')}"
                f" | HRV: {r.get('hrv_ms')}ms | RHR: {r.get('rhr_bpm')}bpm"
                f" | Sleep: {sl.get('duration_hours')}h (quality: {sl.get('quality_score')})"
                f" | Strain: {st.get('score')} | Steps: {st.get('steps')}"
            )

    if not any([member.messages, member.calendar, member.spotify, member.whoop]):
        sections.append("\nNo data available for this member.")

    return "\n".join(sections)


def _dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


async def _retrieve(
    question: str,
    member: MemberProfile,
    client: genai.Client,
) -> list[DataItem]:
    """Embed the question and return the top-K most relevant member items."""
    response = await client.aio.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vec = normalize(response.embeddings[0].values)

    scored = [(item, _dot_product(query_vec, item.vector)) for item in member.items]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored[:TOP_K]]


def _format_retrieved_data(member_name: str, items: list[DataItem]) -> str:
    """Format a retrieved subset of items for the LLM prompt."""
    sections = [f"## Retrieved data for {member_name} ({len(items)} items by relevance)"]
    for item in items:
        sections.append(item.text)
    if not items:
        sections.append("\nNo data available for this member.")
    return "\n".join(sections)


async def ask(question: str, store: DataStore) -> AskResponse:
    """Answer a question using member data and a single Gemini call."""
    member_names = list(store.members.keys())
    resolved = _resolve_member(question, member_names)
    member = store.members.get(resolved) if resolved else None

    system = _build_system_prompt(store.concierge.summary)

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    user_parts = [f"Question: {question}"]
    if member:
        user_parts.append(f"Resolved member: {member.user_name}")
        if member.items:
            try:
                retrieved = await _retrieve(question, member, client)
                user_parts.append(_format_retrieved_data(member.user_name, retrieved))
            except Exception:
                logger.warning("Embedding query failed, falling back to full context", exc_info=True)
                user_parts.append(_format_member_data(member))
        else:
            user_parts.append(_format_member_data(member))
    else:
        names = ", ".join(sorted(member_names))
        user_parts.append(
            f"No member could be resolved from the question."
            f" Known members: {names}"
        )
    try:
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents="\n\n".join(user_parts),
            config=GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                max_output_tokens=8192,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception:
        logger.exception("LLM call failed")
        return AskResponse(
            answer="Sorry, I couldn't process that question right now.",
            confidence=0.0,
            sources=[],
            metadata=ResponseMetadata(reasoning="Internal error processing the question."),
        )

    try:
        text = response.text
        args = json.loads(text)
        return AskResponse(
            answer=args["answer"],
            confidence=max(0.0, min(1.0, float(args["confidence"]))),
            sources=list(args.get("sources", [])),
            metadata=ResponseMetadata(reasoning=args["reasoning"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse LLM response: %s", exc)
        return AskResponse(
            answer="Unable to process the question.",
            confidence=0.0,
            sources=[],
            metadata=ResponseMetadata(reasoning="LLM returned an unparseable response."),
        )
