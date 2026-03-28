import asyncio
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://november7-730026606190.europe-west1.run.app"
PAGE_SIZE = 1000


@dataclass
class MemberProfile:
    user_name: str
    messages: list[dict] = field(default_factory=list)
    calendar: list[dict] = field(default_factory=list)
    spotify: list[dict] = field(default_factory=list)
    whoop: list[dict] = field(default_factory=list)


@dataclass
class ConciergeProfile:
    name: str
    date_of_birth: str
    summary: str


@dataclass
class DataStore:
    members: dict[str, MemberProfile] = field(default_factory=dict)
    concierge: ConciergeProfile | None = None


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


async def _fetch_me(client: httpx.AsyncClient) -> ConciergeProfile:
    resp = await client.get(f"{BASE_URL}/hackathon/me/")
    resp.raise_for_status()
    data = resp.json()
    return ConciergeProfile(**data)


async def load_all() -> DataStore:
    async with httpx.AsyncClient(timeout=30.0) as client:
        concierge, messages, calendar, spotify, whoop = await asyncio.gather(
            _fetch_me(client),
            _fetch_paginated(client, "/messages/"),
            _fetch_paginated(client, "/hackathon/calendar-events/"),
            _fetch_paginated(client, "/hackathon/spotify/"),
            _fetch_paginated(client, "/hackathon/whoop/"),
        )

    store = DataStore(concierge=concierge)

    for msg in messages:
        name = msg["user_name"]
        if name not in store.members:
            store.members[name] = MemberProfile(user_name=name)
        store.members[name].messages.append(msg)

    # Calendar, Spotify, and Whoop belong to the concierge (identified via /me).
    if concierge.name not in store.members:
        store.members[concierge.name] = MemberProfile(user_name=concierge.name)
    store.members[concierge.name].calendar = calendar
    store.members[concierge.name].spotify = spotify
    store.members[concierge.name].whoop = whoop

    logger.info(
        "Loaded %d members, %d messages, %d calendar events, %d spotify streams, %d whoop records",
        len(store.members),
        len(messages),
        len(calendar),
        len(spotify),
        len(whoop),
    )

    return store
