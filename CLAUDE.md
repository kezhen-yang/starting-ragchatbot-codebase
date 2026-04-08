# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Quick start
./run.sh

# Manual (from repo root)
cd backend && uv run uvicorn app:app --reload --port 8000
```

Requires a `.env` file in the repo root:
```
ANTHROPIC_API_KEY=your_key_here
```

App runs at `http://localhost:8000`, API docs at `http://localhost:8000/docs`.

## Dependency Management

Uses `uv`. To add/update dependencies:
```bash
uv add <package>
uv sync
```

Always use `uv run` to execute any Python file (e.g. `uv run script.py`). Never invoke `python` or `python3` directly. Never use `pip`; always use `uv add` or `uv sync` for dependencies.

## Architecture

This is a RAG chatbot that answers questions about course transcripts using Claude + ChromaDB.

**Request flow:**
1. `POST /api/query` → `RAGSystem.query()` in `rag_system.py`
2. `AIGenerator.generate_response()` calls Claude with the `search_course_content` tool
3. Claude invokes the tool → `CourseSearchTool.execute()` → `VectorStore.search()` queries ChromaDB
4. Tool results are fed back to Claude in a second API call → final answer returned

**Key design decisions:**
- All backend modules live flat in `backend/` (no sub-packages); imports are relative by name (e.g. `from vector_store import VectorStore`)
- ChromaDB uses two collections: `course_catalog` (course-level metadata) and `course_content` (text chunks). Course title is the document ID in the catalog.
- Conversation history is stored in-memory in `SessionManager` (lost on restart). History is injected into the system prompt as a formatted string, not as structured messages.
- The tool loop is single-depth: Claude may call `search_course_content` once; `AIGenerator` handles the tool result and makes a final non-tool call.

**Adding a new tool:**
1. Subclass `Tool` in `search_tools.py`, implement `get_tool_definition()` and `execute()`
2. Register it with `tool_manager.register_tool(your_tool)` in `RAGSystem.__init__()`

**Document format** (`docs/*.txt`):
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<content>
```
Documents are loaded on startup and deduplicated by course title. To force a reload, clear `backend/chroma_db/`.
