# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Application Structure

```
.
├── backend/
│   ├── app.py               # FastAPI app; serves static frontend, defines /api/query endpoint
│   ├── rag_system.py        # RAGSystem — top-level orchestrator; loads docs, handles queries
│   ├── ai_generator.py      # AIGenerator — calls Claude API, manages tool loop
│   ├── search_tools.py      # Tool base class + CourseSearchTool; extensible tool registry
│   ├── vector_store.py      # VectorStore — wraps ChromaDB; ingests and searches embeddings
│   ├── document_processor.py# Parses .txt course docs into chunks for indexing
│   ├── session_manager.py   # In-memory conversation history per session
│   ├── models.py            # Pydantic request/response models
│   ├── config.py            # App configuration (model name, DB path, etc.)
│   └── chroma_db/           # Persisted ChromaDB vector data (auto-created)
├── frontend/
│   ├── index.html           # Chat UI
│   ├── script.js            # Sends queries to /api/query, renders responses
│   └── style.css            # Styles
├── docs/                    # Course transcript .txt files (ingested on startup)
├── run.sh                   # Quick-start script
└── pyproject.toml           # uv/Python project config
```

## Request Flow

```
Browser
  │
  └─ POST /api/query
        │
        ▼
   app.py → RAGSystem.query()
                │
                ▼
         AIGenerator.generate_response()
           │  ① First Claude API call (with search_course_content tool available)
           │
           ▼
     Claude invokes tool
           │
           ▼
     CourseSearchTool.execute()
           │
           ▼
     VectorStore.search() → ChromaDB query
           │
           ▼
     Tool result returned to Claude
           │
           ▼
          ② Second Claude API call (tool result in context)
           │
           ▼
     Final answer streamed back to browser
```

**Key design decisions:**
- ChromaDB holds two collections: `course_catalog` (course-level metadata) and `course_content` (text chunks).
- Conversation history lives in `SessionManager` (in-memory, lost on restart); it is injected as a formatted string into the system prompt.
- The tool loop is single-depth: Claude may call `search_course_content` once per query.

**Adding a new tool:**
1. Subclass `Tool` in `search_tools.py` and implement `get_tool_definition()` and `execute()`.
2. Register it with `tool_manager.register_tool(your_tool)` in `RAGSystem.__init__()`.

