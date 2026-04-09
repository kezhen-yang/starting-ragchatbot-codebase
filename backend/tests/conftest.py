import sys
import os
from unittest.mock import MagicMock
import pytest

# Make backend/ importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search.return_value = SearchResults(
        documents=["Content about Python basics"],
        metadata=[{"course_title": "Python Course", "lesson_number": 1}],
        distances=[0.4],
    )
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_link.return_value = "https://example.com/course"
    return store


@pytest.fixture
def mock_tool_manager():
    mgr = MagicMock()
    mgr.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    mgr.execute_tool.return_value = "Python is a high-level programming language."
    mgr.get_last_sources.return_value = ["Python Course - Lesson 1"]
    return mgr
