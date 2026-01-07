"""Tests for conversa.features.answer module."""

from unittest.mock import patch

from conversa.features.answer import teacher_answer


@patch("conversa.features.answer.call_llm")
def test_teacher_answer_with_history(mock_call_llm):
    """Test teacher_answer with conversation history."""
    mock_call_llm.return_value = "That's interesting! What else?"

    history = [
        {"role": "user", "content": "I like learning languages."},
        {"role": "assistant", "content": "That's wonderful!"},
    ]

    result = teacher_answer("I study Spanish.", history=history)

    assert isinstance(result, str)
    assert result == "That's interesting! What else?"

    # Verify the function was called with correct parameters
    call_args = mock_call_llm.call_args
    assert call_args.kwargs["query"] == "I study Spanish."
    assert call_args.kwargs["history"] == history
    assert "You are a language teacher" in call_args.kwargs.get("sys_prompt", "")
