"""Integration tests for the POST /ask endpoint."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from data import ConciergeProfile, DataStore, MemberProfile
from main import app


def _make_store() -> DataStore:
    store = DataStore(
        concierge=ConciergeProfile(name="Alice", date_of_birth="1990-01-01", summary="Test"),
    )
    store.members["Alice"] = MemberProfile(
        user_name="Alice",
        messages=[
            {"id": "msg_1", "timestamp": "2025-01-01", "message": "Book me a table at Chez Janou in Paris"},
            {"id": "msg_2", "timestamp": "2025-01-15", "message": "The dinner at Chez Janou was great"},
        ],
    )
    store.members["Bob"] = MemberProfile(
        user_name="Bob",
        messages=[{"id": "msg_3", "timestamp": "2025-02-01", "message": "Hello, can you arrange a taxi?"}],
    )
    return store


def _mock_json_response(answer, confidence, sources, reasoning):
    resp = MagicMock()
    resp.text = json.dumps({
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "reasoning": reasoning,
    })
    return resp


@pytest.fixture
def store():
    return _make_store()


@pytest.fixture(autouse=True)
def inject_store(store):
    """Inject the test DataStore into app.state, bypassing the lifespan loader."""
    app.state.data = store


@pytest.mark.asyncio
async def test_ask_happy_path(store):
    mock_response = _mock_json_response(
        answer="Chez Janou in Paris.",
        confidence=0.9,
        sources=["msg_1", "msg_2"],
        reasoning="Resolved Alice. Found two restaurant mentions.",
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("prompt.genai.Client", return_value=mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "What is Alice's favorite restaurant?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Chez Janou in Paris."
    assert body["confidence"] == 0.9
    assert body["sources"] == ["msg_1", "msg_2"]
    assert body["metadata"]["reasoning"] != ""


@pytest.mark.asyncio
async def test_ask_unknown_member(store):
    mock_response = _mock_json_response(
        answer="No member named 'Zara' was found.",
        confidence=0.0,
        sources=[],
        reasoning="Could not resolve 'Zara' to any known member.",
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("prompt.genai.Client", return_value=mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "What does Zara like?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == 0.0


@pytest.mark.asyncio
async def test_ask_no_relevant_data(store):
    mock_response = _mock_json_response(
        answer="No information about Bob's music preferences was found.",
        confidence=0.0,
        sources=[],
        reasoning="Resolved Bob. Searched messages but found no music-related data.",
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("prompt.genai.Client", return_value=mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "What music does Bob listen to?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == 0.0
    assert body["metadata"]["reasoning"] != ""


@pytest.mark.asyncio
async def test_ask_missing_question():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ask", json={})

    assert resp.status_code == 422
