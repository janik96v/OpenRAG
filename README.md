# OpenRAG - Multi-Strategy RAG MCP Server

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An open-source MCP (Model Context Protocol) server for Retrieval Augmented Generation (RAG) over personal documents. Supports **three RAG strategies** running in parallel: Traditional RAG, Contextual RAG, and Graph RAG. Built with ChromaDB, Neo4j, sentence-transformers, and designed for local-first, privacy-preserving document search.

**Perfect for use with Claude Code and Claude Desktop.**

## Features

### Core Features
- **🔒 Privacy-First**: All data stored locally using ChromaDB and Neo4j
- **📄 Document Ingestion**: Process .txt files OR ingest text directly (no file required)
- **🔍 Semantic Search**: Natural language queries with similarity scoring
- **⚡ Fast Embeddings**: Configurable models (all-mpnet-base-v2 default)
- **🎯 Token-Aware Chunking**: Uses tiktoken for accurate token counting
- **🔧 Configurable**: Chunk size, overlap, embedding model all customizable
- **🧪 Well-Tested**: Comprehensive test suite

### Three RAG Strategies

**1. Traditional RAG** 🎯
- Vector-based semantic search
- Fast and efficient for straightforward queries
- Ideal for direct factual retrieval

**2. Contextual RAG** 📚
- Adds document-level context to each chunk
- Improved accuracy for complex queries
- Background processing for performance

**3. Graph RAG** 🕸️ *(New!)*
- Entity extraction and relationship mapping
- Multi-hop reasoning across knowledge graphs
- Neo4j-powered graph traversal
- Perfect for discovering connections and relationships
- Combines vector similarity with graph structure

## Quick Start

### 1. Install OpenRAG (5 minutes)

```bash
# Clone or navigate to OpenRAG directory
cd /path/to/OpenRAG

# Run automated setup (creates conda environment "OpenRAG" with Python 3.12)
./setup_environment.sh

# Activate environment
conda activate OpenRAG
```

### 2. Configure for Claude Code (3 minutes)

Edit `~/.claude/config.json`:

```json
{
  "mcpServers": {
    "openrag": {
      "command": "/path/to/anaconda3/envs/OpenRAG/bin/python",
      "args": ["-m", "openrag.server"],
      "env": {
        "CHROMA_DB_PATH": "/absolute/path/to/your/chroma_db",
        "EMBEDDING_MODEL": "all-mpnet-base-v2"
      }
    }
  }
}
```

Find your Python path: `conda activate OpenRAG && which python`

**Important**: Use absolute paths, not relative paths!

### 3. Test in Claude Code (2 minutes)

Restart Claude Code, then try:

```
Ingest this text with name "test.txt":
OpenRAG supports traditional, contextual, and graph RAG strategies.

Query OpenRAG for "RAG strategies"
```

Done! See [Quick Start Guide](docs/quick-start.md) for Contextual and Graph RAG setup.

## Documentation

**Essential Reading**:
- **[Installation Guide](docs/installation.md)** - Complete setup and MCP configuration
- **[Quick Start Guide](docs/quick-start.md)** - Get started in 15 minutes
- **[Architecture Overview](docs/architecture.md)** - System design and components

**Development**:
- **[CLAUDE.md](CLAUDE.md)** - Development conventions (read this first!)
- **[Lab Journal](docs/lab_journal.md)** - Research notes and decisions

## MCP Tools

OpenRAG exposes 5 MCP tools:

1. **ingest_text**: Ingest text content directly
   - Automatically processes with all enabled RAG strategies
   - Traditional RAG: immediate
   - Contextual RAG: background processing
   - Graph RAG: background entity extraction and graph building

2. **query_documents**: Semantic search over documents
   - `rag_type`: Choose "traditional", "contextual", or "graph"
   - `max_hops`: For Graph RAG, control graph traversal depth (1-5)
   - Returns ranked results with similarity scores

3. **list_documents**: List all ingested documents with metadata

4. **delete_document**: Remove documents from all RAG collections

5. **get_stats**: System statistics including collection sizes and configuration

## Configuration

Create `.env` file or set environment variables:

```bash
# Core Configuration
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=60
LOG_LEVEL=INFO

# Contextual RAG (Optional - requires Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CONTEXT_MODEL=llama3.2:3b

# Graph RAG (Optional - requires Ollama + Neo4j)
GRAPH_ENABLED=true
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GRAPH_ENTITY_MODEL=llama3.2:3b
GRAPH_MAX_HOPS=2
```

### RAG Strategy Setup

| Strategy | Requirements | Setup Time |
|----------|--------------|------------|
| **Traditional** | None | 0 min (works out of the box) |
| **Contextual** | + Ollama | +5 min ([guide](docs/quick-start.md#step-4-enable-contextual-rag-optional-5-minutes)) |
| **Graph** | + Ollama + Neo4j | +10 min ([guide](docs/quick-start.md#step-5-enable-graph-rag-optional-10-minutes)) |

## Development

```bash
# Run tests
pytest tests/ -v --cov=src/openrag

# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/
```

## Architecture

```
src/openrag/
├── server.py                      # MCP server entry point
├── config.py                      # Centralized configuration
├── core/                          # Core RAG components
│   ├── chunker.py                 # Token-aware text chunking
│   ├── embedder.py                # Sentence-transformers embeddings
│   ├── vector_store.py            # Traditional RAG (ChromaDB)
│   ├── contextual_processor.py    # Contextual RAG processor
│   ├── contextual_vector_store.py # Dual collection management
│   ├── graph_processor.py         # Entity extraction (Ollama + Neo4j)
│   ├── graph_vector_store.py      # Tri-collection + graph traversal
│   └── ollama_client.py           # Ollama LLM client
├── tools/                         # MCP tool implementations
│   ├── ingest.py                  # Document ingestion
│   ├── query.py                   # Multi-strategy search
│   ├── manage.py                  # Document management
│   └── stats.py                   # System statistics
├── models/                        # Pydantic data models
│   ├── schemas.py                 # Base schemas
│   ├── contextual_schemas.py      # Contextual RAG models
│   └── graph_schemas.py           # Graph RAG models (entities, relationships)
└── utils/                         # Utilities
    ├── async_tasks.py             # Background task manager
    └── logger.py                  # Logging configuration
```

### RAG Strategy Comparison

| Feature | Traditional | Contextual | Graph |
|---------|-------------|------------|-------|
| **Speed** | ⚡️ Fastest | 🔄 Background | 🔄 Background |
| **Accuracy** | Good | Better | Best for relationships |
| **Use Case** | Direct facts | Complex queries | Multi-hop reasoning |
| **Dependencies** | ChromaDB only | + Ollama | + Ollama + Neo4j |
| **Storage** | ChromaDB | ChromaDB (2 collections) | ChromaDB + Neo4j |

## License

MIT License - See LICENSE file

---

**OpenRAG** - Privacy-first RAG for personal documents
