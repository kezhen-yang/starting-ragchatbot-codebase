"""
Tests for AIGenerator in ai_generator.py.

Covers:
- Direct (non-tool) responses
- Single-round tool use
- Two sequential tool-call rounds
- Max-rounds termination (synthesis after 2 tool rounds)
- Tool execution errors (round 1 and round 2)
- Conversation history injected into system prompt
- Model name validity (a misconfigured model silently breaks every query)
"""

import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator

# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic response objects
# ---------------------------------------------------------------------------

def _make_direct_response(text: str):
    """Simulate a Claude response that contains plain text (no tool use)."""
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [content_block]
    return response


def _make_tool_use_response(tool_name: str, tool_id: str, tool_input: dict):
    """Simulate a Claude response that requests a tool call."""
    content_block = MagicMock()
    content_block.type = "tool_use"
    content_block.name = tool_name
    content_block.id = tool_id
    content_block.input = tool_input
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [content_block]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    with patch("ai_generator.anthropic.Anthropic"):
        gen = AIGenerator(api_key="test-key", model="claude-sonnet-4-6")
    return gen


# ---------------------------------------------------------------------------
# generate_response() — no tool use
# ---------------------------------------------------------------------------

def test_generate_response_returns_text_when_no_tool_use(generator):
    """generate_response() returns the assistant text directly when stop_reason != tool_use."""
    generator.client.messages.create.return_value = _make_direct_response("Paris")
    result = generator.generate_response("What is the capital of France?")
    assert result == "Paris"


def test_generate_response_with_tools_but_no_tool_call(generator, mock_tool_manager):
    """Even when tools are provided, a non-tool response is returned as-is."""
    generator.client.messages.create.return_value = _make_direct_response("General answer")
    result = generator.generate_response(
        "Hello", tools=[{"name": "search_course_content"}], tool_manager=mock_tool_manager
    )
    assert result == "General answer"
    mock_tool_manager.execute_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Single-round tool use
# ---------------------------------------------------------------------------

def test_single_round_tool_use_executes_tool_and_returns_synthesis(generator, mock_tool_manager):
    """Claude calls a tool once; result is fed back and Claude produces a final answer."""
    tool_response = _make_tool_use_response(
        "search_course_content", "tu_001", {"query": "Python decorators"}
    )
    final_response = _make_direct_response("Decorators wrap functions.")
    generator.client.messages.create.side_effect = [tool_response, final_response]

    result = generator.generate_response(
        "Explain Python decorators",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )
    assert result == "Decorators wrap functions."
    assert generator.client.messages.create.call_count == 2


def test_single_round_calls_tool_manager_with_correct_args(generator, mock_tool_manager):
    """execute_tool() is called with the tool name and input from Claude's response."""
    tool_response = _make_tool_use_response(
        "search_course_content", "tu_002", {"query": "async await"}
    )
    final_response = _make_direct_response("Async/await manages concurrency.")
    generator.client.messages.create.side_effect = [tool_response, final_response]

    generator.generate_response(
        "Explain async/await",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    mock_tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="async await"
    )


def test_single_round_tool_result_is_in_second_api_call(generator, mock_tool_manager):
    """The tool result is included as a tool_result message in the follow-up API call."""
    mock_tool_manager.execute_tool.return_value = "Found: async/await tutorial"
    tool_response = _make_tool_use_response(
        "search_course_content", "tu_003", {"query": "async await"}
    )
    final_response = _make_direct_response("Here is the explanation.")
    generator.client.messages.create.side_effect = [tool_response, final_response]

    generator.generate_response(
        "Explain async/await",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    second_call_kwargs = generator.client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]
    tool_result_messages = [
        m for m in messages
        if isinstance(m.get("content"), list)
        and any(c.get("type") == "tool_result" for c in m["content"])
    ]
    assert len(tool_result_messages) == 1
    tool_result_content = tool_result_messages[0]["content"][0]
    assert tool_result_content["content"] == "Found: async/await tutorial"
    assert tool_result_content["tool_use_id"] == "tu_003"


def test_generate_response_tool_use_without_tool_manager_returns_something(generator):
    """If stop_reason is tool_use but no tool_manager is given, falls through gracefully."""
    tool_response = _make_tool_use_response("search_course_content", "tu_004", {"query": "x"})
    generator.client.messages.create.return_value = tool_response
    result = generator.generate_response("What is X?")
    assert result is not None


# ---------------------------------------------------------------------------
# Two sequential tool-call rounds
# ---------------------------------------------------------------------------

def test_two_sequential_rounds_calls_each_tool_and_returns_synthesis(generator, mock_tool_manager):
    """Claude calls a tool in round 1, then another tool in round 2, then synthesizes."""
    round1_tool = _make_tool_use_response("get_course_outline", "tu_010", {"course": "Python 101"})
    round2_tool = _make_tool_use_response("search_course_content", "tu_011", {"query": "lesson 4"})
    synthesis = _make_direct_response("Here is the complete answer.")
    generator.client.messages.create.side_effect = [round1_tool, round2_tool, synthesis]

    result = generator.generate_response(
        "Find a course about the same topic as lesson 4 of Python 101",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert result == "Here is the complete answer."
    assert generator.client.messages.create.call_count == 3
    assert mock_tool_manager.execute_tool.call_count == 2


def test_two_sequential_rounds_synthesis_call_has_no_tools(generator, mock_tool_manager):
    """After 2 tool rounds the final synthesis call must not include tools or tool_choice."""
    round1_tool = _make_tool_use_response("get_course_outline", "tu_012", {"course": "X"})
    round2_tool = _make_tool_use_response("search_course_content", "tu_013", {"query": "y"})
    synthesis = _make_direct_response("Answer.")
    generator.client.messages.create.side_effect = [round1_tool, round2_tool, synthesis]

    generator.generate_response(
        "Question",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    final_call_kwargs = generator.client.messages.create.call_args_list[2][1]
    assert "tools" not in final_call_kwargs
    assert "tool_choice" not in final_call_kwargs


def test_two_sequential_rounds_message_order(generator, mock_tool_manager):
    """After 2 tool rounds the final messages list has the correct alternating structure."""
    mock_tool_manager.execute_tool.side_effect = ["outline result", "search result"]
    round1_tool = _make_tool_use_response("get_course_outline", "tu_014", {"course": "X"})
    round2_tool = _make_tool_use_response("search_course_content", "tu_015", {"query": "y"})
    synthesis = _make_direct_response("Answer.")
    generator.client.messages.create.side_effect = [round1_tool, round2_tool, synthesis]

    generator.generate_response(
        "Question",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    final_messages = generator.client.messages.create.call_args_list[2][1]["messages"]
    roles = [m["role"] for m in final_messages]
    # user query → assistant (round 1) → user (round 1 results) →
    # assistant (round 2) → user (round 2 results)
    assert roles == ["user", "assistant", "user", "assistant", "user"]

    # Verify each tool_result message carries the right content
    round1_results = final_messages[2]["content"]
    assert round1_results[0]["content"] == "outline result"
    assert round1_results[0]["tool_use_id"] == "tu_014"

    round2_results = final_messages[4]["content"]
    assert round2_results[0]["content"] == "search result"
    assert round2_results[0]["tool_use_id"] == "tu_015"


def test_max_rounds_stops_at_two_tool_calls(generator, mock_tool_manager):
    """Even if Claude keeps returning tool_use, only 2 rounds are executed."""
    tool_resp = _make_tool_use_response("search_course_content", "tu_020", {"query": "x"})
    synthesis = _make_direct_response("Done.")
    # Three tool-use responses available, but only 2 should be consumed
    generator.client.messages.create.side_effect = [tool_resp, tool_resp, synthesis]

    generator.generate_response(
        "Question",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    # 2 tool rounds + 1 synthesis = 3 total API calls
    assert generator.client.messages.create.call_count == 3
    assert mock_tool_manager.execute_tool.call_count == 2


# ---------------------------------------------------------------------------
# Tool error handling
# ---------------------------------------------------------------------------

def test_tool_error_in_round1_appends_error_result_and_synthesizes(generator, mock_tool_manager):
    """A tool error in round 1 appends an is_error result and triggers synthesis."""
    mock_tool_manager.execute_tool.side_effect = RuntimeError("DB unavailable")
    tool_response = _make_tool_use_response("search_course_content", "tu_030", {"query": "x"})
    synthesis = _make_direct_response("Sorry, the tool failed.")
    generator.client.messages.create.side_effect = [tool_response, synthesis]

    result = generator.generate_response(
        "Question",
        tools=[{"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert result == "Sorry, the tool failed."
    assert generator.client.messages.create.call_count == 2

    synthesis_messages = generator.client.messages.create.call_args_list[1][1]["messages"]
    tool_result_msgs = [
        m for m in synthesis_messages
        if isinstance(m.get("content"), list)
        and any(c.get("type") == "tool_result" for c in m["content"])
    ]
    assert len(tool_result_msgs) == 1
    error_block = tool_result_msgs[0]["content"][0]
    assert error_block.get("is_error") is True
    assert "DB unavailable" in error_block["content"]


def test_tool_error_in_round2_terminates_loop_and_synthesizes(generator, mock_tool_manager):
    """A tool error in round 2 stops the loop and the synthesis call receives both results."""
    mock_tool_manager.execute_tool.side_effect = ["good result", RuntimeError("timeout")]
    round1_tool = _make_tool_use_response("get_course_outline", "tu_031", {"course": "X"})
    round2_tool = _make_tool_use_response("search_course_content", "tu_032", {"query": "y"})
    synthesis = _make_direct_response("Partial answer.")
    generator.client.messages.create.side_effect = [round1_tool, round2_tool, synthesis]

    result = generator.generate_response(
        "Question",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=mock_tool_manager,
    )

    assert result == "Partial answer."
    assert generator.client.messages.create.call_count == 3

    final_messages = generator.client.messages.create.call_args_list[2][1]["messages"]
    tool_result_msgs = [
        m for m in final_messages
        if isinstance(m.get("content"), list)
        and any(c.get("type") == "tool_result" for c in m["content"])
    ]
    # Both round 1 (success) and round 2 (error) results must be present
    assert len(tool_result_msgs) == 2
    assert tool_result_msgs[0]["content"][0]["content"] == "good result"
    assert tool_result_msgs[1]["content"][0].get("is_error") is True


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def test_conversation_history_appended_to_system_prompt(generator):
    """When conversation_history is provided it is injected into the system prompt."""
    generator.client.messages.create.return_value = _make_direct_response("Answer")
    generator.generate_response("New question", conversation_history="User: hi\nAI: hello")

    call_kwargs = generator.client.messages.create.call_args[1]
    assert "User: hi" in call_kwargs["system"]
    assert "AI: hello" in call_kwargs["system"]


def test_no_history_uses_plain_system_prompt(generator):
    """Without conversation_history the system prompt is used unchanged."""
    generator.client.messages.create.return_value = _make_direct_response("Answer")
    generator.generate_response("Question")

    call_kwargs = generator.client.messages.create.call_args[1]
    assert call_kwargs["system"] == AIGenerator.SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Configuration — model name validity
# ---------------------------------------------------------------------------

# Known valid Anthropic model IDs as of the current assistant knowledge cutoff.
# If the configured model is not in this list the API call will fail with a
# model-not-found error, which surfaces as a 500 "Query failed" in the UI.
KNOWN_VALID_MODELS = {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
}


def test_configured_model_name_is_valid():
    """
    The model name in config.py must be a known valid Anthropic model ID.

    An invalid model causes every API call to fail, producing the
    'Query failed' error users see in the UI.
    """
    from config import config

    assert config.ANTHROPIC_MODEL in KNOWN_VALID_MODELS, (
        f"\n\nINVALID MODEL: '{config.ANTHROPIC_MODEL}' is not a known valid Anthropic model.\n"
        f"Valid models: {sorted(KNOWN_VALID_MODELS)}\n"
        f"Fix: update ANTHROPIC_MODEL in backend/config.py to a valid model ID.\n"
        f"Recommended: 'claude-sonnet-4-6'"
    )
