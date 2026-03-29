# OpenRAG — Multi-Strategy RAG MCP Server

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An open-source MCP server for RAG over personal documents. Supports **three parallel strategies** — Traditional, Contextual, and Graph RAG — with all data stored locally for privacy.

---

## Quick Start (Docker)

No Python or conda required.

```bash
# 1. Clone the repository
git clone https://github.com/janik96v/OpenRAG.git
cd OpenRAG

# 2. Copy and adjust the config (change NEO4J_PASSWORD etc.)
cp .env.docker.example .env

# 3. Start everything
docker compose up -d
```

Neo4j and the MCP server start automatically. ChromaDB data is persisted in `./chroma_db`, Neo4j data in a Docker volume.

For Contextual and Graph RAG, Ollama must run on your host:

```bash
brew install ollama        # macOS — Linux/Windows: https://ollama.ai/download
ollama serve
ollama pull llama3.2:3b
```

> **Traditional RAG only?** Set `CONTEXTUAL_ENABLED=false` and `GRAPH_ENABLED=false` in `.env` — no Ollama needed.

See [docs/installation.md](docs/installation.md) for the full setup guide including configuration options and troubleshooting.

---

## Configure Claude Code / Claude Desktop

### Claude Code (CLI)

```bash
claude mcp add openrag -- docker compose \
  -f /absolute/path/to/OpenRAG/docker-compose.yml \
  run --rm -i openrag
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent config file on your OS:

```json
{
  "mcpServers": {
    "openrag": {
      "command": "docker",
      "args": [
        "compose",
        "-f", "/absolute/path/to/OpenRAG/docker-compose.yml",
        "run", "--rm", "-i", "openrag"
      ]
    }
  }
}
```

Replace `/absolute/path/to/OpenRAG/docker-compose.yml` with your actual path. Restart Claude Desktop after saving.

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `ingest_text` | Ingest raw text content. Traditional RAG is immediate; Contextual and Graph RAG run in the background. |
| `query_documents` | Semantic search. Choose `rag_type` ("traditional", "contextual", "graph") and optionally `max_hops` (1–5) for Graph RAG. |
| `list_documents` | List all ingested documents with metadata. |
| `delete_document` | Remove a document from all RAG collections. |
| `get_stats` | System statistics: collection sizes, configuration, background task status. |

---

## RAG Strategy Comparison

| | Traditional | Contextual | Graph |
|--|-------------|------------|-------|
| **Speed** | Fastest | Background | Background |
| **Accuracy** | Good | Better | Best for relationships |
| **Use Case** | Direct facts | Complex queries | Multi-hop reasoning |
| **Dependencies** | ChromaDB | + Ollama | + Ollama + Neo4j |
| **Collections** | 1 | 2 | 3 |

---

## Architecture

```
src/openrag/
├── server.py                      # MCP server entry point
├── config.py                      # Centralized configuration (pydantic-settings)
├── core/
│   ├── chunker.py                 # Token-aware text chunking (tiktoken)
│   ├── embedder.py                # Sentence-transformers embeddings
│   ├── vector_store.py            # Traditional RAG (ChromaDB)
│   ├── contextual_processor.py    # Context generation (Ollama)
│   ├── contextual_vector_store.py # Dual-collection management
│   ├── graph_processor.py         # Entity extraction (Ollama + Neo4j)
│   ├── graph_vector_store.py      # Triple-collection + graph traversal
│   └── ollama_client.py           # Ollama API client
├── tools/                         # MCP tool implementations
├── models/                        # Pydantic data models
└── utils/                         # Logging, validation, background tasks
```

All three RAG types share the same ChromaDB instance via separate collections:
- Traditional → `documents`
- Contextual → `documents` + `documents_contextual`
- Graph → `documents` + `documents_contextual` + `documents_graph` + Neo4j

---

## Smoke Tests

Manual end-to-end scripts for verifying each RAG strategy are in [`tests/quick/`](tests/quick/):

```bash
# Traditional RAG (no external dependencies)
python tests/quick/test_normalRAG.py

# Contextual RAG (requires Ollama)
python tests/quick/test_contextualRAG.py

# Graph RAG (requires Ollama + Neo4j, takes 3–5 min)
python tests/quick/test_graphRAG.py
```

For automated unit/integration tests, run `pytest tests/`.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/installation.md](docs/installation.md) | Docker setup, configuration, MCP wiring, troubleshooting |
| [docs/quick-start.md](docs/quick-start.md) | Step-by-step usage guide |
| [docs/architecture.md](docs/architecture.md) | System design deep-dive |

---

## License

MIT License — see [LICENSE](LICENSE)
