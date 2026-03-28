from unittest.mock import AsyncMock

import httpx
import pytest

from data import DataStore, MemberProfile, _fetch_paginated, load_all


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://fake"),
    )


# -- Pagination --


@pytest.mark.asyncio
async def test_fetch_paginated_single_page():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response(
        {"total": 2, "items": [{"id": "1"}, {"id": "2"}]}
    )

    items = await _fetch_paginated(client, "/things/")

    assert items == [{"id": "1"}, {"id": "2"}]
    assert client.get.call_count == 1


@pytest.mark.asyncio
async def test_fetch_paginated_multiple_pages(monkeypatch):
    monkeypatch.setattr("data.PAGE_SIZE", 2)

    call_count = 0

    async def mock_get(url, params=None):
        nonlocal call_count
        call_count += 1
        skip = params["skip"]
        if skip == 0:
            return _mock_response({"total": 3, "items": [{"id": "1"}, {"id": "2"}]})
        return _mock_response({"total": 3, "items": [{"id": "3"}]})

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = mock_get

    items = await _fetch_paginated(client, "/things/")

    assert [i["id"] for i in items] == ["1", "2", "3"]
    assert call_count == 2


@pytest.mark.asyncio
async def test_fetch_paginated_empty():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response({"total": 0, "items": []})

    items = await _fetch_paginated(client, "/things/")

    assert items == []
    assert client.get.call_count == 1


# -- load_all --


def _make_messages(*names: str) -> list[dict]:
    return [
        {"id": f"msg_{i}", "user_id": f"uid_{i}", "user_name": name, "timestamp": "2025-01-01T00:00:00", "message": f"Hello from {name}"}
        for i, name in enumerate(names)
    ]


@pytest.fixture
def mock_api(monkeypatch):
    """Patch httpx.AsyncClient to return controlled API responses."""
    messages = _make_messages("Alice", "Alice", "Bob")
    calendar = [{"id": "evt_1", "title": "Standup"}]
    spotify = [{"stream_id": "sp_1", "title": "Song"}]
    whoop = [{"date": "2025-01-01", "recovery": {}}]
    me = {"name": "Alice", "date_of_birth": "1990-01-01", "summary": "Test user"}

    async def mock_get(url, params=None):
        path = str(url)
        if "/hackathon/me/" in path:
            return _mock_response(me)
        if "/messages/" in path:
            return _mock_response({"total": len(messages), "items": messages})
        if "/calendar-events/" in path:
            return _mock_response({"total": len(calendar), "items": calendar})
        if "/spotify/" in path:
            return _mock_response({"total": len(spotify), "items": spotify})
        if "/whoop/" in path:
            return _mock_response({"total": len(whoop), "items": whoop})
        raise ValueError(f"Unexpected URL: {url}")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.side_effect = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("data.httpx.AsyncClient", lambda **kwargs: mock_client)


@pytest.mark.asyncio
async def test_load_all_groups_messages_by_member(mock_api):
    store = await load_all()

    assert len(store.members["Alice"].messages) == 2
    assert len(store.members["Bob"].messages) == 1


@pytest.mark.asyncio
async def test_load_all_assigns_personal_data_to_concierge(mock_api):
    store = await load_all()

    alice = store.members["Alice"]
    assert len(alice.calendar) == 1
    assert len(alice.spotify) == 1
    assert len(alice.whoop) == 1


@pytest.mark.asyncio
async def test_load_all_other_members_have_no_personal_data(mock_api):
    store = await load_all()

    bob = store.members["Bob"]
    assert bob.calendar == []
    assert bob.spotify == []
    assert bob.whoop == []


@pytest.mark.asyncio
async def test_load_all_sets_concierge_profile(mock_api):
    store = await load_all()

    assert store.concierge.name == "Alice"
    assert store.concierge.date_of_birth == "1990-01-01"


@pytest.mark.asyncio
async def test_load_all_returns_fresh_datastore(mock_api):
    store1 = await load_all()
    store2 = await load_all()

    assert store1 is not store2
