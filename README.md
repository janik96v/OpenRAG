# OpenRAG - Traditional RAG MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An open-source MCP (Model Context Protocol) server for Retrieval Augmented Generation (RAG) over personal documents. Built with ChromaDB, sentence-transformers, and designed for local-first, privacy-preserving document search.

## Features

- **🔒 Privacy-First**: All data stored locally using ChromaDB
- **📄 Document Ingestion**: Process .txt files OR ingest text directly (no file required)
- **🔍 Semantic Search**: Natural language queries with similarity scoring
- **⚡ Fast Embeddings**: Configurable models (all-mpnet-base-v2 default)
- **🎯 Token-Aware Chunking**: Uses tiktoken for accurate token counting
- **🔧 Configurable**: Chunk size, overlap, embedding model all customizable
- **🧪 Well-Tested**: Comprehensive test suite

## Quick Start

### Automated Setup (Recommended)

Following CLAUDE.md conventions:

```bash
# Clone repository
cd /path/to/OpenRAG

# Run automated setup (creates conda environment "OpenRAG")
./setup_environment.sh

# Activate environment
conda activate OpenRAG

# Configure (optional)
cp .env.example .env

# Run server
python -m openrag.server
```

### Manual Setup

```bash
# Create conda environment
conda create -n OpenRAG python=3.13 -y
conda activate OpenRAG

# Install dependencies
pip install -r requirements.txt

# Or install from pyproject.toml
pip install -e ".[dev]"

# Configure
cp .env.example .env

# Test
python quick_test.py
```

## Documentation

Comprehensive documentation available in the [docs/](docs/) directory:

- **[Quick Start Guide](docs/quick-start.md)** - Get started in 10 minutes
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[User Guide](docs/user-guide.md)** - Complete usage guide
- **[API Reference](docs/api-reference.md)** - MCP tools documentation
- **[Configuration Reference](docs/configuration.md)** - All configuration options
- **[Architecture Overview](docs/architecture.md)** - System design and components
- **[Developer Guide](docs/developer-guide.md)** - Contributing to OpenRAG
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[FAQ](docs/faq.md)** - Frequently asked questions

## MCP Tools

OpenRAG exposes 6 MCP tools:

1. **ingest_document**: Ingest .txt files into vector database
2. **ingest_text**: Ingest raw text directly (bypasses file I/O)
3. **query_documents**: Semantic search over documents
4. **list_documents**: List all ingested documents
5. **delete_document**: Remove documents and chunks
6. **get_stats**: System statistics

## Configuration

Create `.env` file or set environment variables:

```bash
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=60
LOG_LEVEL=INFO
```

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
├── server.py           # MCP server
├── config.py           # Settings
├── core/               # Core components
│   ├── chunker.py      # Text chunking
│   ├── embedder.py     # Embeddings
│   └── vector_store.py # ChromaDB
├── tools/              # MCP tools
└── models/             # Data models
```

## License

MIT License - See LICENSE file

---

**OpenRAG** - Privacy-first RAG for personal documents
