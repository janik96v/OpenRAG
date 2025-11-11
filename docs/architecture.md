# Architecture Overview

This document describes the high-level architecture of OpenRAG, explaining how components work together to provide RAG capabilities over personal documents.

## System Overview

OpenRAG is an MCP (Model Context Protocol) server that enables AI assistants like Claude to perform semantic search over personal documents using vector embeddings and ChromaDB.

### Core Design Principles

Following the principles defined in [CLAUDE.md](../CLAUDE.md):

- **KISS**: Simple, straightforward implementation
- **YAGNI**: Features implemented only when needed
- **Single Responsibility**: Each component has one clear purpose
- **Dependency Inversion**: High-level modules depend on abstractions

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Client (Claude)                      │
│                     (stdio/JSON-RPC protocol)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server (server.py)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Tool Router                           │   │
│  │  - ingest_document    - query_documents                  │   │
│  │  - list_documents     - delete_document                  │   │
│  │  - get_stats                                             │   │
│  └────┬──────────────────────┬──────────────────────────────┘   │
└───────┼──────────────────────┼──────────────────────────────────┘
        │                      │
        ▼                      ▼
┌──────────────────┐    ┌──────────────────┐
│  Tools Layer     │    │  Configuration   │
│  (tools/)        │    │  (config.py)     │
│  - ingest.py     │    │  - Settings      │
│  - query.py      │    │  - Validation    │
│  - manage.py     │    └──────────────────┘
│  - stats.py      │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer (core/)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐   │
│  │  Chunker    │  │  Embedder   │  │   Vector Store        │   │
│  │ (chunker.py)│  │(embedder.py)│  │  (vector_store.py)    │   │
│  │             │  │             │  │                       │   │
│  │ - Tiktoken  │  │ - Sentence  │  │ - ChromaDB Client     │   │
│  │   counting  │  │   Transformers│ │ - Collection Mgmt    │   │
│  │ - Recursive │  │ - all-mpnet │  │ - Query/Search        │   │
│  │   splitting │  │   -base-v2  │  │                       │   │
│  └─────────────┘  └─────────────┘  └───────────┬───────────┘   │
└────────────────────────────────────────────────┼───────────────┘
                                                  │
                                                  ▼
                                    ┌──────────────────────────┐
                                    │   ChromaDB Storage       │
                                    │   (Persistent SQLite)    │
                                    │                          │
                                    │  - Documents metadata    │
                                    │  - Embeddings            │
                                    │  - Chunks                │
                                    └──────────────────────────┘
```

## Component Details

### 1. MCP Server Layer (`server.py`)

**Purpose**: Entry point that handles MCP protocol communication.

**Responsibilities**:
- Initialize core components (chunker, embedder, vector store)
- Register and expose MCP tools
- Route tool calls to appropriate handlers
- Handle errors and return structured responses
- Manage server lifecycle

**Key Design Decisions**:
- Uses stdio transport for Claude Desktop integration
- Async-first design for non-blocking operations
- Global component initialization for efficiency
- JSON-RPC 2.0 protocol compliance

**Code Structure**:
```python
# Server initialization
def create_server() -> Server:
    - Register tool definitions
    - Set up tool call handlers

async def main() -> None:
    - Initialize components
    - Start server with stdio transport
```

### 2. Tools Layer (`tools/`)

**Purpose**: Implement the five MCP tools exposed to clients.

#### Tool: `ingest_document` (`tools/ingest.py`)
- **Input**: File path to .txt document
- **Process**: Read → Chunk → Embed → Store
- **Output**: Document ID, chunk count, status
- **Error Handling**: File validation, encoding errors

#### Tool: `query_documents` (`tools/query.py`)
- **Input**: Natural language query, max results, similarity threshold
- **Process**: Embed query → Vector search → Rank results
- **Output**: Ranked chunks with similarity scores
- **Error Handling**: Empty query, no results

#### Tool: `list_documents` (`tools/manage.py`)
- **Input**: None
- **Process**: Query metadata from vector store
- **Output**: List of documents with metadata
- **Error Handling**: Empty collection

#### Tool: `delete_document` (`tools/manage.py`)
- **Input**: Document ID
- **Process**: Remove document and all chunks
- **Output**: Success/failure status
- **Error Handling**: Invalid ID, deletion errors

#### Tool: `get_stats` (`tools/stats.py`)
- **Input**: None
- **Process**: Gather system statistics
- **Output**: Document count, chunk count, configuration
- **Error Handling**: Collection access errors

### 3. Core Layer (`core/`)

#### TextChunker (`core/chunker.py`)

**Purpose**: Split documents into optimal-sized chunks for embedding.

**Strategy**: RecursiveCharacterTextSplitter
- Default: 400 tokens per chunk
- 15% overlap (60 tokens)
- Hierarchical splitting on semantic boundaries

**Key Features**:
- Token counting using tiktoken
- Configurable separators
- Metadata preservation
- Character vs. token awareness

**Implementation**:
```python
class TextChunker:
    def chunk_text(text: str) -> list[str]:
        - Split on paragraph boundaries
        - Respect token limits
        - Maintain overlap
```

**Design Rationale**:
- 400 tokens balances context vs. precision
- 15% overlap prevents information loss
- Recursive splitting respects document structure

#### EmbeddingModel (`core/embedder.py`)

**Purpose**: Generate vector embeddings for text chunks.

**Default Model**: `all-mpnet-base-v2`
- 768-dimensional embeddings
- Best quality among base models
- ~420 MB model size

**Alternative**: `all-MiniLM-L6-v2`
- 384-dimensional embeddings
- 5x faster, smaller footprint
- ~80 MB model size

**Key Features**:
- Lazy loading (on first use)
- Batch embedding support
- Dimension reporting
- Model caching

**Implementation**:
```python
class EmbeddingModel:
    def embed(texts: list[str]) -> list[list[float]]:
        - Load model (cached)
        - Generate embeddings
        - Normalize vectors
```

**Design Rationale**:
- sentence-transformers for quality and compatibility
- Configurable model selection
- Local execution (privacy-preserving)

#### VectorStore (`core/vector_store.py`)

**Purpose**: Manage persistent vector storage using ChromaDB.

**Storage Strategy**:
- PersistentClient for local storage
- SQLite backend for metadata
- Columnar storage for embeddings

**Key Features**:
- Collection management
- Metadata filtering
- Similarity search (cosine similarity)
- Document-level operations
- Statistics and monitoring

**Implementation**:
```python
class VectorStore:
    def add_document(document: Document) -> None:
        - Store chunks with embeddings
        - Save metadata
        - Update indices

    def search(query: str, n_results: int) -> list[tuple]:
        - Embed query
        - Vector similarity search
        - Return ranked results
```

**Design Rationale**:
- ChromaDB for simplicity and local-first approach
- Persistent storage ensures data survives restarts
- Metadata enables filtering and management

### 4. Data Models (`models/schemas.py`)

**Purpose**: Define data structures using Pydantic for validation.

**Key Models**:

```python
class DocumentChunk:
    - chunk_id: str (UUID)
    - document_id: str
    - content: str
    - chunk_index: int
    - metadata: dict

class Document:
    - document_id: str (UUID)
    - metadata: DocumentMetadata
    - chunks: list[DocumentChunk]

class QueryResult:
    - chunk: DocumentChunk
    - similarity_score: float
    - document_name: str
```

**Design Rationale**:
- Pydantic v2 for fast validation
- Type safety throughout codebase
- Serialization support for MCP responses

### 5. Configuration (`config.py`)

**Purpose**: Centralized settings management with validation.

**Settings**:
```python
class Settings(BaseSettings):
    chroma_db_path: str = "./chroma_db"
    embedding_model: str = "all-mpnet-base-v2"
    chunk_size: int = 400
    chunk_overlap: int = 60
    log_level: str = "INFO"
```

**Features**:
- Environment variable support (.env)
- Validation with constraints
- Singleton pattern for consistency
- Path creation and validation

### 6. Utilities (`utils/`)

#### Logger (`utils/logger.py`)
- Structured logging to stderr (MCP requirement)
- Configurable log levels
- Context-aware messages

#### Validation (`utils/validation.py`)
- File path validation
- Input sanitization
- Error message formatting

## Data Flow

### Document Ingestion Flow

```
1. User calls ingest_document tool
   ↓
2. Validate file path and read content
   ↓
3. Chunker splits text into chunks (400 tokens)
   ↓
4. Embedder generates vectors for each chunk
   ↓
5. VectorStore saves chunks + embeddings + metadata
   ↓
6. Return document ID and statistics
```

### Query Flow

```
1. User calls query_documents tool
   ↓
2. Embedder generates vector for query
   ↓
3. VectorStore performs similarity search
   ↓
4. Rank results by cosine similarity
   ↓
5. Filter by minimum similarity threshold
   ↓
6. Return top N results with scores
```

## Design Decisions

### Why MCP?
- **Standardized protocol** for AI-tool integration
- **Claude Desktop support** out of the box
- **Extensibility** for future features
- **Clean separation** between AI and tooling

### Why ChromaDB?
- **Local-first** (privacy-preserving)
- **Simple API** (easy to use)
- **Persistent storage** (SQLite backend)
- **No external services** required

### Why sentence-transformers?
- **State-of-the-art** embedding quality
- **Local execution** (no API costs)
- **Flexible model selection** (speed vs. quality)
- **Active community** and updates

### Why Recursive Chunking?
- **Industry standard** (proven approach)
- **Respects structure** (paragraphs, sentences)
- **Configurable** (adapts to document types)
- **Good baseline** (easy to optimize from)

## Performance Characteristics

### Throughput
- **Ingestion**: ~1-2 pages/second (depends on embedding model)
- **Query**: ~100-500ms per query (depends on collection size)
- **Scaling**: Linear with document count (SQLite limitations)

### Memory Usage
- **Base**: ~500 MB (Python + dependencies)
- **Embedding Model**: 80 MB (MiniLM) or 420 MB (mpnet)
- **ChromaDB**: Scales with collection size

### Storage
- **Embeddings**: ~3 KB per chunk (768 dimensions)
- **Metadata**: ~1 KB per chunk (depends on content)
- **Growth Rate**: ~4 KB per chunk on average

## Scalability Considerations

### Current Limitations
- **Single writer**: SQLite backend limits concurrent writes
- **In-memory search**: Collection loaded into memory
- **No sharding**: Single collection per instance

### Future Enhancements
- **Client-server mode**: For multi-user scenarios
- **Batch processing**: Parallel ingestion
- **Index optimization**: Faster search for large collections
- **Caching**: Query result caching

## Security Considerations

### Current Implementation
- **File path validation**: Prevent directory traversal
- **Input sanitization**: Prevent injection attacks
- **Local execution**: No external API calls
- **Metadata isolation**: Documents stored separately

### Best Practices
- **Validate inputs**: All tool parameters validated
- **Fail fast**: Early error detection
- **Log to stderr**: Separation from protocol output
- **Limited permissions**: Read-only file access

## Extension Points

### Adding New Tools
1. Define tool in `tools/`
2. Register in `server.py`
3. Add input schema
4. Implement handler
5. Add tests

### Supporting New File Types
1. Add parser in `utils/`
2. Update `ingest.py` validation
3. Extend DocumentMetadata
4. Add type-specific chunking

### Alternative Embedding Models
1. Update `embedder.py`
2. Add model configuration
3. Handle dimension differences
4. Update documentation

## Testing Strategy

### Unit Tests
- Individual components tested in isolation
- Mock external dependencies
- Focus on edge cases

### Integration Tests
- Test component interactions
- Use temporary ChromaDB instance
- Validate full workflows

### End-to-End Tests
- Test MCP protocol integration
- Verify tool functionality
- Check error handling

See [Testing Guide](TESTING.md) for details.

## Related Documentation

- [Developer Guide](developer-guide.md) - Contributing to the codebase
- [API Reference](api-reference.md) - Tool specifications
- [Configuration Reference](configuration.md) - All settings explained
- [Lab Journal](lab_journal.md) - Research and design decisions

---

Last Updated: 2025-11-09
