# Architecture Overview

Concise overview of OpenRAG's architecture and design decisions.

## System Overview

OpenRAG is an MCP (Model Context Protocol) server implementing three parallel RAG strategies:

1. **Traditional RAG**: Vector-based semantic search (ChromaDB + embeddings)
2. **Contextual RAG**: Document-level context enhancement (+ Ollama)
3. **Graph RAG**: Entity extraction and relationship mapping (+ Ollama + Neo4j)

All strategies run in parallel, allowing users to choose the best approach for each query.

## High-Level Architecture

```
Claude Code/Desktop (MCP Client)
         ↓ (stdio/JSON-RPC)
    MCP Server (server.py)
         ↓
    Tools Layer (ingest, query, manage, stats)
         ↓
┌────────┴────────┬─────────────────┬──────────────┐
│   Traditional   │   Contextual    │   Graph      │
│   RAG           │   RAG           │   RAG        │
├─────────────────┼─────────────────┼──────────────┤
│ Chunker         │ + Context       │ + Entity     │
│ Embedder        │   Processor     │   Extractor  │
│ VectorStore     │   (Ollama)      │   (Ollama)   │
└────────┬────────┴────────┬────────┴──────┬───────┘
         ↓                 ↓               ↓
    ChromaDB          ChromaDB        ChromaDB
    (base)         (contextual)    (graph) + Neo4j
```

## Core Components

### MCP Server (`server.py`)

- Exposes 5 MCP tools: `ingest_text`, `query_documents`, `list_documents`, `delete_document`, `get_stats`
- Initializes all RAG strategy components on startup
- Routes tool calls to appropriate handlers
- Manages component lifecycle (esp. Neo4j connections)

### Tools Layer (`tools/`)

**ingest.py**: Ingests text content into all enabled RAG strategies
- Traditional RAG: immediate processing
- Contextual RAG: background context generation
- Graph RAG: background entity extraction (3-5 min)

**query.py**: Searches across RAG strategies
- Supports `rag_type` parameter: "traditional", "contextual", "graph"
- Graph RAG supports `max_hops` for graph traversal depth

**manage.py**: Document lifecycle management (list, delete)

**stats.py**: System statistics and configuration info

### Core Layer (`core/`)

**chunker.py**: Token-aware text splitting
- Uses tiktoken for accurate token counting
- Default: 400 tokens/chunk, 60 token overlap
- Recursive splitting on semantic boundaries

**embedder.py**: Vector embedding generation
- Uses sentence-transformers (local execution)
- Default: `all-mpnet-base-v2` (768 dimensions)
- Alternative: `all-MiniLM-L6-v2` (faster, smaller)

**vector_store.py**: Traditional RAG storage (ChromaDB base collection)

**contextual_processor.py**: Context generation using Ollama
- Generates document-level context for each chunk
- Improves accuracy for complex queries

**contextual_vector_store.py**: Manages base + contextual collections

**graph_processor.py**: Entity and relationship extraction
- Uses Ollama LLM for entity extraction
- Stores entities/relationships in Neo4j
- Supports fallback mode (no Ollama)

**graph_vector_store.py**: Manages all three collections + graph traversal
- Base collection (traditional RAG)
- Contextual collection (contextual RAG)
- Graph collection (graph RAG) + Neo4j queries

**ollama_client.py**: HTTP client for Ollama API
- Handles retries and timeouts
- Used by contextual and graph processors

**schemas.py**: Base document and chunk models
- `DocumentChunk`: Individual text chunks with metadata
- `DocumentMetadata`: Document-level information
- `RAGType` enum: "traditional", "contextual", "graph"

**contextual_schemas.py**: Contextual RAG models
- `ContextualChunk`: Chunks with document context

**graph_schemas.py**: Graph RAG models
- `Entity`: Extracted entities (name, type, description)
- `Relationship`: Entity relationships (source, target, type)
- `GraphChunk`: Chunks with entities and relationships

### Configuration (`config.py`)

Centralized settings using Pydantic with `.env` support:

```python
# Core settings
chroma_db_path: str = "./chroma_db"
embedding_model: str = "all-mpnet-base-v2"
chunk_size: int = 400
chunk_overlap: int = 60

# Contextual RAG
ollama_base_url: str = "http://localhost:11434"
ollama_context_model: str = "llama3.2:3b"

# Graph RAG
graph_enabled: bool = True
neo4j_uri: str = "neo4j://localhost:7687"
neo4j_username: str = "neo4j"
neo4j_password: str  # from env
graph_entity_model: str = "llama3.2:3b"
graph_max_hops: int = 2
```

### Utilities (`utils/`)

**logger.py**: Logging to stderr (MCP requirement)
**validation.py**: Input validation and sanitization
**async_tasks.py**: Background task manager for Contextual/Graph RAG

## Data Flow

### Ingestion Flow

When `ingest_text` is called, all three RAG types are triggered in a single pass:

```
ingest_text tool
  → Chunker (400 tokens, 60 overlap)
  │
  ├── [BLOCKING] Traditional RAG
  │     → Embedder (vector generation)
  │     → ChromaDB base collection
  │     → Return document_id + stats  ← MCP responds here
  │
  ├── [BACKGROUND, concurrent] Contextual RAG       (~30-60 sec)
  │     → Generate document context per chunk (Ollama)
  │     → Store in contextual collection
  │
  └── [BACKGROUND, concurrent] Graph RAG            (~3-5 min)
        → Extract entities per chunk (Ollama)
        → Extract relationships per chunk (Ollama)
        → Store in Neo4j + graph collection
```

**Key details:**
- Traditional ingestion is **synchronous** (`add_document` blocks until complete)
- Contextual and Graph tasks are launched via `asyncio.create_task()` — they start **concurrently** and do **not** depend on each other
- The MCP tool returns immediately after traditional ingestion; background tasks continue running
- `BackgroundTaskManager` holds strong references to prevent garbage collection

### Query Flow

```
query_documents tool
  → Choose RAG type (traditional/contextual/graph)
  → Embed query
  → Search appropriate collection(s)
  → (Graph only) Traverse Neo4j graph (max_hops)
  → Return ranked results with scores
```

## Multi-Collection Storage Strategy

OpenRAG uses separate ChromaDB collections for each RAG type:

1. **Base collection** (`documents`): Traditional RAG chunks + embeddings
2. **Contextual collection** (`documents_contextual`): Chunks with document context
3. **Graph collection** (`documents_graph`): Chunks with entity/relationship metadata

Plus **Neo4j graph database** for entity-relationship storage (Graph RAG only).

This allows:
- Parallel ingestion to all strategies
- Independent querying of each strategy
- Gradual adoption (start with Traditional, add Contextual/Graph later)

## Key Design Decisions

### Why Three RAG Strategies?

Different queries benefit from different approaches:
- **Traditional**: Fast, direct fact retrieval
- **Contextual**: Better accuracy when document context matters
- **Graph**: Multi-hop reasoning, finding relationships

### Why MCP?

- Standardized protocol for AI-tool integration
- Claude Desktop/Code support out of the box
- Clean separation between AI and tooling

### Why Local-First?

- **Privacy**: All data stays on your machine
- **No API costs**: Local embeddings and LLMs (optional)
- **Offline capable**: Works without internet (after setup)

### Why Background Processing?

Contextual and Graph RAG use LLMs which take time:
- Contextual: ~30-60 seconds per document
- Graph: 3-5 minutes per document (entity extraction)

Background processing allows immediate ingestion response while processing continues.

## Performance Notes

### Traditional RAG
- Ingestion: Immediate (~1-2 sec for typical document)
- Query: Fast (~100-500ms)

### Contextual RAG
- Ingestion: ~30-60 seconds (background)
- Query: Fast (~100-500ms)
- Requires: Ollama running

### Graph RAG
- Ingestion: 3-5 minutes (background, LLM-intensive)
- Query: Moderate (~500ms-2sec with graph traversal)
- Requires: Ollama + Neo4j running

## Testing

Tests are organized by RAG type:
- `tests/test_*.py`: Unit tests for components
- `tests/test_*_integration.py`: Integration tests
- `quick_test_normalRAG.py`: Traditional RAG end-to-end
- `quick_test_contextualRAG.py`: Contextual RAG end-to-end
- `quick_test_graphRAG.py`: Graph RAG end-to-end

Run: `pytest tests/ -v`

## Related Documentation

- [Installation Guide](installation.md) - Setup and configuration
- [Quick Start Guide](quick-start.md) - Get started in 15 minutes
- [CLAUDE.md](../CLAUDE.md) - Development conventions
- [Lab Journal](lab_journal.md) - Research notes and decisions

---

**Last Updated**: 2026-03-06
