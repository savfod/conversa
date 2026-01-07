"""Tests for conversa.features.llm.

These tests monkeypatch a fake `openai` module to avoid network calls.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from conversa.features import llm_api


class TestResponse(BaseModel):
    """Test response format for structured output."""

    message: str
    confidence: float


@patch("conversa.features.llm_api.openai")
def test_answer_with_fake_openai(mock_openai, monkeypatch):
    def create(model, input, temperature):
        # simple assertion to ensure the wrapper passed expected model
        assert "gpt" in model
        # ensure our user message is present
        assert any(m.get("role") == "user" and m.get("content") == "hi" for m in input)
        res = MagicMock()
        res.output_text = "Hello back"
        return res

    # Attach ChatCompletion with a create method
    mock_openai.OpenAI().responses.create.side_effect = create

    res = llm_api.call_llm(
        "hi",
        history=[{"role": "user", "content": "previous"}],
        sys_prompt="You are a helpful assistant.",
    )
    assert isinstance(res, str)
    assert "Hello back" in res


@patch("conversa.features.llm_api.openai")
def test_answer_structured_with_fake_openai(mock_openai, monkeypatch):
    def parse(model, input, text_format):
        # simple assertion to ensure the wrapper passed expected model
        assert "gpt" in model
        # ensure our user message is present
        assert any(
            m.get("role") == "user" and m.get("content") == "test query" for m in input
        )
        # ensure the format is passed through
        assert text_format == TestResponse
        res = MagicMock()
        res.output_parsed = TestResponse(message="Structured response", confidence=0.95)
        return res

    # Attach responses.parse method
    mock_openai.OpenAI().responses.parse.side_effect = parse

    res = llm_api.call_llm_structured(
        "test query",
        history=[{"role": "user", "content": "previous"}],
        sys_prompt="You are a helpful assistant.",
        answer_format=TestResponse,
    )
    assert isinstance(res, TestResponse)
    assert res.message == "Structured response"
    assert res.confidence == 0.95


@pytest.mark.slow
def test_answer_with_real_openai():
    """Test the llm_api.answer function with a real OpenAI call."""
    res = llm_api.call_llm(
        "Hi! What is the value of 2+2?", sys_prompt="You are a helpful assistant."
    )
    assert isinstance(res, str)
    assert len(res) > 0
    assert "4" in res


@pytest.mark.slow
def test_answer_structured_with_real_openai():
    """Test the llm_api.call_llm_structured function with a real OpenAI call."""
    res = llm_api.call_llm_structured(
        "Hi! What is the value of 2+2?",
        sys_prompt="You are a helpful assistant.",
        answer_format=TestResponse,
    )
    assert isinstance(res, TestResponse)
    assert isinstance(res.message, str)
    assert len(res.message) > 0
    assert "4" in res.message
    assert isinstance(res.confidence, float)
    assert 0 <= res.confidence <= 1
