from unittest.mock import AsyncMock, patch

import pytest

from data import ConciergeProfile, DataStore, MemberProfile
from models import AskResponse, ResponseMetadata
from prompt import _build_system_prompt, _format_member_data, _resolve_member, ask, ANSWER_TOOL

MEMBERS = [
    "Sophia Al-Farsi",
    "Fatima El-Tahir",
    "Amina Van Den Berg",
    "James Fletcher",
    "Lorenzo Cavalli",
]


# -- Entity resolution --


def test_resolve_exact_first_name():
    assert _resolve_member("What does Sophia like?", MEMBERS) == "Sophia Al-Farsi"


def test_resolve_exact_last_name():
    assert _resolve_member("Tell me about Cavalli", MEMBERS) == "Lorenzo Cavalli"


def test_resolve_full_name():
    assert _resolve_member("What about James Fletcher?", MEMBERS) == "James Fletcher"


def test_resolve_possessive():
    assert _resolve_member("What is Sophia's favorite food?", MEMBERS) == "Sophia Al-Farsi"


def test_resolve_fuzzy_prefix():
    # "Amira" shares 3-char prefix "ami" with "Amina"
    assert _resolve_member("What is Amira's favorite restaurant?", MEMBERS) == "Amina Van Den Berg"


def test_resolve_no_match():
    assert _resolve_member("What is the weather today?", MEMBERS) is None


def test_resolve_short_words_ignored():
    # "is", "an" are < 3 chars and should not match
    assert _resolve_member("Is an apple red?", MEMBERS) is None


def test_resolve_case_insensitive():
    assert _resolve_member("what does SOPHIA think?", MEMBERS) == "Sophia Al-Farsi"


def test_resolve_stopwords_ignored():
    # "the" should not match "El-Tahir", "can" should not match "Cavalli"
    assert _resolve_member("Can the person have this?", MEMBERS) is None


# -- Prompt formatting --


def test_format_member_data_messages():
    member = MemberProfile(
        user_name="Alice",
        messages=[{"id": "msg_1", "timestamp": "2025-01-01T00:00:00", "message": "Hello"}],
    )
    text = _format_member_data(member)
    assert "## Data for Alice" in text
    assert "[msg_1]" in text
    assert "Hello" in text


def test_format_member_data_empty():
    member = MemberProfile(user_name="Ghost")
    text = _format_member_data(member)
    assert "No data available" in text


def test_format_member_data_calendar():
    member = MemberProfile(
        user_name="James",
        calendar=[{
            "id": "evt_1",
            "start": "2026-01-01T09:00",
            "end": "2026-01-01T10:00",
            "title": "Standup",
            "location": "Zoom",
            "attendees": ["Alice", "Bob"],
            "notes": "",
        }],
    )
    text = _format_member_data(member)
    assert "[evt_1]" in text
    assert "Standup" in text
    assert "Alice, Bob" in text


def test_format_member_data_spotify():
    member = MemberProfile(
        user_name="James",
        spotify=[{
            "stream_id": "sp_1",
            "timestamp": "2026-01-01T08:00",
            "title": "Feel It Still",
            "artist_or_show": "Portugal. The Man",
            "context": "commute",
        }],
    )
    text = _format_member_data(member)
    assert "[sp_1]" in text
    assert "Feel It Still" in text


def test_format_member_data_whoop():
    member = MemberProfile(
        user_name="James",
        whoop=[{
            "date": "2026-01-01",
            "recovery": {"score": 80, "hrv_ms": 65.0, "rhr_bpm": 50.0},
            "sleep": {"duration_hours": 7.5, "quality_score": 75},
            "strain": {"score": 10.0, "steps": 8000},
        }],
    )
    text = _format_member_data(member)
    assert "[whoop_2026-01-01]" in text
    assert "Recovery: 80" in text


# -- System prompt --


def test_system_prompt_includes_rubric():
    prompt = _build_system_prompt("Test concierge")
    assert "1.0" in prompt
    assert "0.0" in prompt


def test_system_prompt_includes_concierge_context():
    prompt = _build_system_prompt("Founder and CEO")
    assert "Founder and CEO" in prompt


# -- ask() integration (mocked LLM) --


def _make_store() -> DataStore:
    store = DataStore(
        concierge=ConciergeProfile(name="Alice", date_of_birth="1990-01-01", summary="Test"),
    )
    store.members["Alice"] = MemberProfile(
        user_name="Alice",
        messages=[{"id": "msg_1", "timestamp": "2025-01-01", "message": "Book me a table at Chez Janou"}],
    )
    store.members["Bob"] = MemberProfile(
        user_name="Bob",
        messages=[{"id": "msg_2", "timestamp": "2025-01-02", "message": "Hello"}],
    )
    return store


def _mock_tool_response(answer: str, confidence: float, sources: list[str], reasoning: str):
    """Create a mock Anthropic response with a tool_use block."""
    block = AsyncMock()
    block.type = "tool_use"
    block.input = {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "reasoning": reasoning,
    }
    response = AsyncMock()
    response.content = [block]
    return response


@pytest.mark.asyncio
async def test_ask_returns_structured_response():
    store = _make_store()
    mock_response = _mock_tool_response(
        answer="Chez Janou",
        confidence=0.95,
        sources=["msg_1"],
        reasoning="Resolved Alice. Found restaurant mention.",
    )

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    with patch("prompt.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await ask("What is Alice's favorite restaurant?", store)

    assert isinstance(result, AskResponse)
    assert result.answer == "Chez Janou"
    assert result.confidence == 0.95
    assert result.sources == ["msg_1"]
    assert "Alice" in result.metadata.reasoning


@pytest.mark.asyncio
async def test_ask_unknown_member():
    store = _make_store()
    mock_response = _mock_tool_response(
        answer="No matching member found.",
        confidence=0.0,
        sources=[],
        reasoning="Could not resolve 'Zara' to any known member.",
    )

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    with patch("prompt.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await ask("What does Zara like?", store)

    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_ask_clamps_confidence():
    store = _make_store()
    mock_response = _mock_tool_response(
        answer="Test", confidence=1.5, sources=[], reasoning="Test",
    )

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    with patch("prompt.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await ask("What does Alice like?", store)

    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_ask_handles_llm_error():
    store = _make_store()

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
    with patch("prompt.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await ask("What does Alice like?", store)

    assert result.confidence == 0.0
    assert "Internal error" in result.metadata.reasoning


@pytest.mark.asyncio
async def test_ask_no_tool_use_fallback():
    store = _make_store()
    block = AsyncMock()
    block.type = "text"
    response = AsyncMock()
    response.content = [block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=response)
    with patch("prompt.anthropic.AsyncAnthropic", return_value=mock_client):
        result = await ask("What does Alice like?", store)

    assert result.confidence == 0.0
    assert "did not return a tool call" in result.metadata.reasoning
