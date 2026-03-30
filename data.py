import asyncio
import logging
import math
from dataclasses import dataclass, field

import httpx
from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

BASE_URL = "https://november7-730026606190.europe-west1.run.app"
PAGE_SIZE = 1000


@dataclass
class DataItem:
    id: str
    source: str
    text: str
    member: str
    vector: list[float] = field(default_factory=list)


@dataclass
class MemberProfile:
    user_name: str
    messages: list[dict] = field(default_factory=list)
    calendar: list[dict] = field(default_factory=list)
    spotify: list[dict] = field(default_factory=list)
    whoop: list[dict] = field(default_factory=list)
    items: list[DataItem] = field(default_factory=list)


@dataclass
class PrimaryMember:
    name: str
    date_of_birth: str
    summary: str


@dataclass
class DataStore:
    members: dict[str, MemberProfile] = field(default_factory=dict)
    primary_member: PrimaryMember | None = None


async def _fetch_paginated(client: httpx.AsyncClient, path: str) -> list[dict]:
    items: list[dict] = []
    skip = 0
    while True:
        resp = await client.get(
            f"{BASE_URL}{path}", params={"skip": skip, "limit": PAGE_SIZE}
        )
        resp.raise_for_status()
        page = resp.json()
        batch = page["items"]
        items.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        skip += PAGE_SIZE
    return items


async def _fetch_me(client: httpx.AsyncClient) -> PrimaryMember:
    resp = await client.get(f"{BASE_URL}/hackathon/me/")
    resp.raise_for_status()
    data = resp.json()
    return PrimaryMember(**data)


async def load_all() -> DataStore:
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                primary_member, messages, calendar, spotify, whoop = await asyncio.gather(
                    _fetch_me(client),
                    _fetch_paginated(client, "/messages/"),
                    _fetch_paginated(client, "/hackathon/calendar-events/"),
                    _fetch_paginated(client, "/hackathon/spotify/"),
                    _fetch_paginated(client, "/hackathon/whoop/"),
                )
            break
        except (httpx.HTTPError, httpx.TimeoutException):
            if attempt == 2:
                raise
            logger.warning("Data fetch attempt %d failed, retrying...", attempt + 1)
            await asyncio.sleep(1)

    store = DataStore(primary_member=primary_member)

    for msg in messages:
        name = msg["user_name"]
        if name not in store.members:
            store.members[name] = MemberProfile(user_name=name)
        store.members[name].messages.append(msg)

    # Calendar, Spotify, and Whoop belong to the primary member (identified via /me).
    if primary_member.name not in store.members:
        store.members[primary_member.name] = MemberProfile(user_name=primary_member.name)
    store.members[primary_member.name].calendar = calendar
    store.members[primary_member.name].spotify = spotify
    store.members[primary_member.name].whoop = whoop

    logger.info(
        "Loaded %d members, %d messages, %d calendar events, %d spotify streams, %d whoop records",
        len(store.members),
        len(messages),
        len(calendar),
        len(spotify),
        len(whoop),
    )

    return store


def _render_items(member: MemberProfile) -> list[DataItem]:
    """Convert a member's raw data dicts into DataItem objects with formatted text."""
    items: list[DataItem] = []
    for m in member.messages:
        items.append(DataItem(
            id=m["id"],
            source="messages",
            text=f"[{m['id']}] {m['timestamp']}: {m['message']}",
            member=member.user_name,
        ))
    for e in member.calendar:
        attendees = ", ".join(e.get("attendees", []))
        items.append(DataItem(
            id=e["id"],
            source="calendar",
            text=(
                f"[{e['id']}] {e['start']} - {e['end']} | {e['title']}"
                f" | {e.get('location', '')} | attendees: {attendees}"
                f" | {e.get('notes', '')}"
            ),
            member=member.user_name,
        ))
    for s in member.spotify:
        items.append(DataItem(
            id=s["stream_id"],
            source="spotify",
            text=(
                f"[{s['stream_id']}] {s['timestamp']}"
                f" | {s['title']} | {s.get('artist_or_show', '')}"
                f" | {s.get('context', '')}"
            ),
            member=member.user_name,
        ))
    for w in member.whoop:
        r = w.get("recovery", {})
        sl = w.get("sleep", {})
        st = w.get("strain", {})
        items.append(DataItem(
            id=f"whoop_{w['date']}",
            source="whoop",
            text=(
                f"[whoop_{w['date']}] Recovery: {r.get('score')}"
                f" | HRV: {r.get('hrv_ms')}ms | RHR: {r.get('rhr_bpm')}bpm"
                f" | Sleep: {sl.get('duration_hours')}h (quality: {sl.get('quality_score')})"
                f" | Strain: {st.get('score')} | Steps: {st.get('steps')}"
            ),
            member=member.user_name,
        ))
    return items


def normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Returns the original if zero-length."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


EMBED_MODEL = "text-embedding-005"
EMBED_BATCH_SIZE = 250


async def build_index(store: DataStore, client: genai.Client) -> None:
    """Embed all data items and attach pre-normalized vectors to the DataStore."""
    all_items: list[DataItem] = []
    for member in store.members.values():
        member.items = _render_items(member)
        all_items.extend(member.items)

    texts = [item.text for item in all_items]
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = await client.aio.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        all_vectors.extend(e.values for e in response.embeddings)

    for item, vec in zip(all_items, all_vectors):
        item.vector = normalize(vec)

    logger.info(
        "Indexed %d items (%d embedding API calls)",
        len(all_items),
        math.ceil(len(texts) / EMBED_BATCH_SIZE) if texts else 0,
    )
