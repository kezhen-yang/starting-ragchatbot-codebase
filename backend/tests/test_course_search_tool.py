"""
Tests for CourseSearchTool.execute() in search_tools.py.

Covers:
- Normal result formatting
- Error propagation from VectorStore
- Empty result messages (with/without filters)
- Filter parameters forwarded to VectorStore.search()
- Source tracking (last_sources) and lesson link inclusion
"""

import pytest
from unittest.mock import MagicMock, call
from search_tools import CourseSearchTool
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool(mock_vector_store):
    return CourseSearchTool(mock_vector_store)


# ---------------------------------------------------------------------------
# execute() — happy path
# ---------------------------------------------------------------------------

def test_execute_returns_formatted_content(tool):
    """execute() formats document text together with course/lesson header."""
    result = tool.execute(query="Python basics")
    assert "Python Course" in result
    assert "Lesson 1" in result
    assert "Content about Python basics" in result


def test_execute_forwards_query_to_vector_store(tool, mock_vector_store):
    """execute() passes the query string to VectorStore.search()."""
    tool.execute(query="decorators")
    mock_vector_store.search.assert_called_once()
    _, kwargs = mock_vector_store.search.call_args
    assert kwargs["query"] == "decorators"


def test_execute_forwards_course_name_filter(tool, mock_vector_store):
    """execute() passes course_name to VectorStore.search()."""
    tool.execute(query="closures", course_name="Python Course")
    _, kwargs = mock_vector_store.search.call_args
    assert kwargs["course_name"] == "Python Course"


def test_execute_forwards_lesson_number_filter(tool, mock_vector_store):
    """execute() passes lesson_number to VectorStore.search()."""
    tool.execute(query="closures", lesson_number=3)
    _, kwargs = mock_vector_store.search.call_args
    assert kwargs["lesson_number"] == 3


# ---------------------------------------------------------------------------
# execute() — error / empty paths
# ---------------------------------------------------------------------------

def test_execute_returns_store_error_message(tool, mock_vector_store):
    """execute() surfaces the error string from VectorStore when search fails."""
    mock_vector_store.search.return_value = SearchResults.empty("Search error: index is empty")
    result = tool.execute(query="something")
    assert "Search error: index is empty" in result


def test_execute_returns_no_results_message_when_empty(tool, mock_vector_store):
    """execute() returns a 'no relevant content' message for empty results."""
    mock_vector_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    result = tool.execute(query="something")
    assert "No relevant content found" in result


def test_execute_empty_message_includes_course_filter(tool, mock_vector_store):
    """Empty-result message mentions the course filter when one was given."""
    mock_vector_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    result = tool.execute(query="something", course_name="MCP Course")
    assert "MCP Course" in result


def test_execute_empty_message_includes_lesson_filter(tool, mock_vector_store):
    """Empty-result message mentions the lesson number filter when one was given."""
    mock_vector_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    result = tool.execute(query="something", lesson_number=2)
    assert "lesson 2" in result.lower()


# ---------------------------------------------------------------------------
# _format_results() — source tracking
# ---------------------------------------------------------------------------

def test_format_results_sets_last_sources(tool):
    """After execute(), last_sources contains one entry per result."""
    tool.execute(query="Python basics")
    assert len(tool.last_sources) == 1


def test_format_results_source_label_contains_course_and_lesson(tool):
    """Source label carries both course title and lesson number."""
    tool.execute(query="Python basics")
    source = tool.last_sources[0]
    assert "Python Course" in source
    assert "1" in source  # lesson number


def test_format_results_source_wraps_link_in_anchor_tag(tool, mock_vector_store):
    """When a lesson link exists the source becomes an <a> element."""
    mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
    tool.execute(query="Python basics")
    source = tool.last_sources[0]
    assert "<a " in source
    assert "https://example.com/lesson/1" in source


def test_format_results_source_is_plain_text_when_no_link(tool, mock_vector_store):
    """When no lesson link is available the source is plain text (no anchor)."""
    mock_vector_store.get_lesson_link.return_value = None
    tool.execute(query="Python basics")
    source = tool.last_sources[0]
    assert "<a " not in source
    assert "Python Course" in source


def test_format_results_source_omits_lesson_when_no_lesson_number(tool, mock_vector_store):
    """Results without a lesson_number get a source label with only the course title."""
    mock_vector_store.search.return_value = SearchResults(
        documents=["Course intro text"],
        metadata=[{"course_title": "Python Course", "lesson_number": None}],
        distances=[0.3],
    )
    tool.execute(query="Python basics")
    source = tool.last_sources[0]
    assert "Python Course" in source
    assert "Lesson" not in source
