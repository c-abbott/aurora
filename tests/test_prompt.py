import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from data import PrimaryMember, DataItem, DataStore, MemberProfile, normalize
from models import AskResponse, ResponseMetadata
from prompt import (
    RESPONSE_SCHEMA,
    _SELF_REFERENCES,
    _build_system_prompt,
    _dot_product,
    _format_member_data,
    _format_retrieved_data,
    _resolve_member,
    ask,
)

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


def test_resolve_hyphenated_last_name():
    assert _resolve_member("Tell me about El-Tahir", MEMBERS) == "Fatima El-Tahir"


def test_resolve_hyphenated_partial():
    # "Tahir" alone should match the "tahir" part of "El-Tahir"
    assert _resolve_member("What does Tahir want?", MEMBERS) == "Fatima El-Tahir"


def test_resolve_stopwords_ignored():
    # "the" should not match "El-Tahir", "can" should not match "Cavalli"
    assert _resolve_member("Can the person have this?", MEMBERS) is None


def test_resolve_self_reference_my():
    assert _resolve_member("What is my sleep score?", MEMBERS, primary_member_name="James Fletcher") == "James Fletcher"


def test_resolve_self_reference_me():
    assert _resolve_member("Tell me about me", MEMBERS, primary_member_name="James Fletcher") == "James Fletcher"


def test_resolve_name_takes_priority_over_self_reference():
    # "my" is a self-reference, but "Sophia" is an explicit name match
    assert _resolve_member("What is my friend Sophia's favorite?", MEMBERS, primary_member_name="James Fletcher") == "Sophia Al-Farsi"


def test_resolve_self_reference_without_primary_member():
    assert _resolve_member("What is my sleep score?", MEMBERS) is None


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
    prompt = _build_system_prompt("Test primary member")
    assert "1.0" in prompt
    assert "0.0" in prompt


def test_system_prompt_includes_primary_member_context():
    prompt = _build_system_prompt("Founder and CEO")
    assert "Founder and CEO" in prompt


# -- ask() integration (mocked LLM) --


def _make_store() -> DataStore:
    store = DataStore(
        primary_member=PrimaryMember(name="Alice", date_of_birth="1990-01-01", summary="Test"),
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


def _mock_json_response(answer: str, confidence: float, sources: list[str], reasoning: str):
    """Create a mock Gemini response with structured JSON output."""
    response = MagicMock()
    response.text = json.dumps({
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "reasoning": reasoning,
    })
    return response


@pytest.mark.asyncio
async def test_ask_returns_structured_response():
    store = _make_store()
    mock_response = _mock_json_response(
        answer="Chez Janou",
        confidence=0.95,
        sources=["msg_1"],
        reasoning="Resolved Alice. Found restaurant mention.",
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    result = await ask("What is Alice's favorite restaurant?", store, client=mock_client)

    assert isinstance(result, AskResponse)
    assert result.answer == "Chez Janou"
    assert result.confidence == 0.95
    assert result.sources == ["msg_1"]
    assert "Alice" in result.metadata.reasoning


@pytest.mark.asyncio
async def test_ask_unknown_member():
    store = _make_store()
    mock_response = _mock_json_response(
        answer="No matching member found.",
        confidence=0.0,
        sources=[],
        reasoning="Could not resolve 'Zara' to any known member.",
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    result = await ask("What does Zara like?", store, client=mock_client)

    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_ask_clamps_confidence():
    store = _make_store()
    mock_response = _mock_json_response(
        answer="Test", confidence=1.5, sources=[], reasoning="Test",
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    result = await ask("What does Alice like?", store, client=mock_client)

    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_ask_handles_llm_error():
    store = _make_store()

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API down"))
    result = await ask("What does Alice like?", store, client=mock_client)

    assert result.confidence == 0.0
    assert "Internal error" in result.metadata.reasoning


@pytest.mark.asyncio
async def test_ask_handles_malformed_json():
    store = _make_store()

    response = MagicMock()
    response.text = "not valid json"

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=response)
    result = await ask("What does Alice like?", store, client=mock_client)

    assert result.confidence == 0.0
    assert "unparseable" in result.metadata.reasoning


# -- RAG retrieval --


def test_dot_product():
    assert _dot_product([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _dot_product([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs(_dot_product([0.6, 0.8], [0.6, 0.8]) - 1.0) < 1e-9


def test_format_retrieved_data():
    items = [
        DataItem(id="msg_1", source="messages", text="[msg_1] 2025-01-01: Hello", member="Alice"),
        DataItem(id="evt_1", source="calendar", text="[evt_1] Meeting", member="Alice"),
    ]
    text = _format_retrieved_data("Alice", items)
    assert "Retrieved data for Alice" in text
    assert "[msg_1]" in text
    assert "[evt_1]" in text


def test_format_retrieved_data_empty():
    text = _format_retrieved_data("Ghost", [])
    assert "No data available" in text


def _make_rag_store() -> DataStore:
    """DataStore with embedded items for RAG testing."""
    store = DataStore(
        primary_member=PrimaryMember(name="Alice", date_of_birth="1990-01-01", summary="Test"),
    )
    items = [
        DataItem(
            id="msg_1", source="messages",
            text="[msg_1] 2025-01-01: Book me a table at Chez Janou",
            member="Alice",
            vector=normalize([1.0, 0.0, 0.0]),
        ),
        DataItem(
            id="msg_2", source="messages",
            text="[msg_2] 2025-01-02: I love Italian food",
            member="Alice",
            vector=normalize([0.0, 1.0, 0.0]),
        ),
    ]
    store.members["Alice"] = MemberProfile(
        user_name="Alice",
        messages=[
            {"id": "msg_1", "timestamp": "2025-01-01", "message": "Book me a table at Chez Janou"},
            {"id": "msg_2", "timestamp": "2025-01-02", "message": "I love Italian food"},
        ],
        items=items,
    )
    return store


@pytest.mark.asyncio
async def test_ask_uses_retrieval_when_items_exist():
    store = _make_rag_store()
    mock_response = _mock_json_response(
        answer="Chez Janou",
        confidence=0.95,
        sources=["msg_1"],
        reasoning="Found restaurant booking.",
    )

    mock_embedding = MagicMock()
    mock_embedding.values = [1.0, 0.0, 0.0]
    mock_embed_response = MagicMock()
    mock_embed_response.embeddings = [mock_embedding]

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    mock_client.aio.models.embed_content = AsyncMock(return_value=mock_embed_response)

    result = await ask("What is Alice's favorite restaurant?", store, client=mock_client)

    assert result.answer == "Chez Janou"
    mock_client.aio.models.embed_content.assert_called_once()


@pytest.mark.asyncio
async def test_ask_falls_back_on_embed_failure():
    store = _make_rag_store()
    mock_response = _mock_json_response(
        answer="Chez Janou",
        confidence=0.95,
        sources=["msg_1"],
        reasoning="Found restaurant booking.",
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    mock_client.aio.models.embed_content = AsyncMock(side_effect=RuntimeError("Embed API down"))

    result = await ask("What is Alice's favorite restaurant?", store, client=mock_client)

    assert result.answer == "Chez Janou"
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_ask_prompt_contains_member_data():
    """Verify the prompt sent to the LLM includes resolved member and their data."""
    store = _make_store()
    mock_response = _mock_json_response(
        answer="Test", confidence=0.5, sources=["msg_1"], reasoning="Test",
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    await ask("What is Alice's favorite restaurant?", store, client=mock_client)

    call_args = mock_client.aio.models.generate_content.call_args
    prompt_content = call_args.kwargs.get("contents") or call_args[1].get("contents", "")
    assert "Resolved member: Alice" in prompt_content
    assert "Chez Janou" in prompt_content
    assert "msg_1" in prompt_content
