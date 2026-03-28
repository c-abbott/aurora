import pytest
from pydantic import ValidationError

from models import AskRequest, AskResponse, ResponseMetadata


# -- AskRequest --


def test_ask_request_valid():
    req = AskRequest(question="What is Amira's favorite restaurant?")
    assert req.question == "What is Amira's favorite restaurant?"


def test_ask_request_missing_question():
    with pytest.raises(ValidationError):
        AskRequest()


# -- AskResponse --


def test_ask_response_valid():
    resp = AskResponse(
        answer="Le Jules Verne",
        confidence=0.85,
        sources=["msg_1", "msg_2"],
        metadata=ResponseMetadata(reasoning="Found in messages."),
    )
    assert resp.answer == "Le Jules Verne"
    assert resp.confidence == 0.85
    assert resp.sources == ["msg_1", "msg_2"]


def test_ask_response_confidence_lower_bound():
    with pytest.raises(ValidationError):
        AskResponse(
            answer="test",
            confidence=-0.1,
            sources=[],
            metadata=ResponseMetadata(reasoning="test"),
        )


def test_ask_response_confidence_upper_bound():
    with pytest.raises(ValidationError):
        AskResponse(
            answer="test",
            confidence=1.1,
            sources=[],
            metadata=ResponseMetadata(reasoning="test"),
        )


def test_ask_response_missing_metadata():
    with pytest.raises(ValidationError):
        AskResponse(answer="test", confidence=0.5, sources=[])


def test_ask_response_empty_sources_allowed():
    resp = AskResponse(
        answer="No data found.",
        confidence=0.0,
        sources=[],
        metadata=ResponseMetadata(reasoning="No match."),
    )
    assert resp.sources == []
