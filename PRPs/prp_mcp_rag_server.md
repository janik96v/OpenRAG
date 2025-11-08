name: "MCP RAG Server - Traditional RAG Implementation v1"
description: |
  Build an open-source MCP server for Retrieval Augmented Generation (RAG) over personal documents.
  This PRP focuses on implementing traditional RAG with local ChromaDB storage, document chunking,
  embedding, and semantic search capabilities. The system will enable users to ingest .txt files,
  store embedded chunks locally, and query their personal knowledge base through an LLM.

---

## Goal

Create a fully functional MCP server that:
1. Accepts .txt documents from LLMs and processes them into searchable chunks
2. Stores embedded chunks in a local ChromaDB vector database with rich metadata
3. Provides semantic search capabilities to retrieve relevant chunks based on natural language queries
4. Returns retrieved context to the LLM for answer generation
5. Maintains user privacy by storing all data locally

The system must be modular, extensible, and follow Python best practices (PEP8, type hints, comprehensive testing).

## Why

- **User Privacy**: All document processing and storage happens locally on the user's machine
- **Knowledge Retrieval**: Enable users to efficiently search through their personal document collections
- **LLM Enhancement**: Provide relevant context to LLMs for more accurate and grounded responses
- **Extensibility**: Build foundation for future RAG types (contextual RAG, graph RAG, etc.)
- **Open Source**: Create a community-driven alternative to proprietary RAG solutions

## What

A Python-based MCP server with the following user-visible behavior:

### MCP Tools Exposed:
1. **ingest_document**: Accept .txt file path, chunk, embed, and store in vector database
2. **query_documents**: Search for relevant chunks using natural language query
3. **list_documents**: Show all ingested documents with metadata
4. **delete_document**: Remove a document and its chunks from the database
5. **get_stats**: Return system statistics (document count, chunk count, storage info)

### Technical Requirements:
- Local ChromaDB persistent storage
- Configurable embedding models (default: all-mpnet-base-v2)
- Metadata storage: chunk_id, source_document, chunk_text, created_at, chunk_index
- Chunk size: 400 tokens with 15% overlap (configurable)
- RecursiveCharacterTextSplitter for chunking
- Error handling with informative messages
- Comprehensive logging to stderr (MCP requirement)

### Success Criteria
- [x] MCP server runs and connects to Claude Desktop via stdio
- [x] Can ingest .txt files and store chunks in ChromaDB
- [x] Semantic search returns relevant chunks (top-5 by default)
- [x] All metadata fields correctly stored and retrievable
- [x] User can configure vector database storage location
- [x] Database persists across server restarts
- [x] All unit tests pass (80%+ coverage)
- [x] Ruff and mypy pass with no errors
- [x] Documentation includes setup and usage examples

## All Needed Context

### Documentation & References

```yaml
# CRITICAL READING - Must review before implementation

- url: https://modelcontextprotocol.io/docs
  section: Python SDK, Tools, Error Handling
  why: Official MCP protocol specification and Python implementation patterns
  critical: Stdout MUST only contain JSON-RPC messages, all logs go to stderr

- url: https://github.com/modelcontextprotocol/python-sdk
  why: Reference implementations and code examples
  critical: Shows proper async patterns and tool decoration

- url: https://docs.trychroma.com/docs/overview/introduction
  section: Getting Started, PersistentClient, Collections
  why: ChromaDB setup and usage patterns
  critical: Must explicitly specify embedding function, path configuration

- url: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
  why: Embedding model selection and usage
  critical: all-mpnet-base-v2 recommended for production (420MB, balanced quality/speed)

- file: /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG/docs/lab_journal.md
  why: Comprehensive research findings on MCP, ChromaDB, embeddings, and chunking
  critical: Contains deployment strategies, security considerations, performance benchmarks

- file: /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG/examples/vector_store.py
  why: Pattern for ChromaDB integration, batch operations, error handling
  critical: Shows proper PersistentClient setup, collection management

- file: /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG/examples/document_processor.py
  why: Document processing patterns, chunking logic, validation
  critical: Shows chunk creation with overlap, metadata structure

- file: /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG/examples/document_manager.py
  why: JSON-based metadata storage pattern
  critical: Persistent document tracking, status management

- file: /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG/CLAUDE.md
  why: Project-specific conventions, naming standards, testing requirements
  critical: Line length 100 chars, Google-style docstrings, TDD approach, conda environment
```

### Current Codebase Structure
```bash
OpenRAG/
├── .claude/              # Claude Code configuration
├── .git/                 # Git repository
├── .serena/             # Serena MCP server config
├── CLAUDE.md            # Development guidelines
├── README.md            # Project readme
├── PRPs/                # Project Research Plans
│   ├── issues_user/     # User feature requests
│   └── templates/       # PRP templates
├── docs/                # Documentation
│   └── lab_journal.md   # Research findings
└── examples/            # Reference implementations
    ├── document_processor.py
    ├── document_manager.py
    ├── vector_store.py
    ├── rag_engine.py
    └── document_watcher.py
```

### Desired Codebase Structure
```bash
OpenRAG/
├── src/
│   └── openrag/
│       ├── __init__.py
│       ├── server.py              # MCP server entry point
│       ├── config.py              # Configuration management (Pydantic)
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py         # Pydantic models (Document, Chunk, etc.)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── vector_store.py    # ChromaDB wrapper
│       │   ├── embedder.py        # Embedding model wrapper
│       │   └── chunker.py         # Text chunking logic
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── ingest.py          # Document ingestion tool
│       │   ├── query.py           # Search/query tool
│       │   ├── manage.py          # Document management tools
│       │   └── stats.py           # Statistics tool
│       └── utils/
│           ├── __init__.py
│           ├── logger.py          # Logging setup (stderr)
│           └── validation.py      # Input validation
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_vector_store.py
│   ├── test_embedder.py
│   ├── test_chunker.py
│   ├── test_ingest.py
│   ├── test_query.py
│   └── test_integration.py       # End-to-end tests
├── pyproject.toml                # Project config, dependencies, ruff/mypy
├── README.md                     # Setup and usage docs
└── .env.example                  # Example configuration
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: MCP Protocol Requirements
# 1. ALL logs must go to stderr, stdout is ONLY for JSON-RPC messages
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # CRITICAL: stderr only!
)

# 2. MCP tools must be async functions
@mcp.tool()
async def my_tool(param: str) -> str:
    # Async required even if operations are sync
    return result

# CRITICAL: ChromaDB Collection Management
# 1. ALWAYS specify embedding function explicitly
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function  # Don't rely on default!
)

# 2. PersistentClient path must exist or be creatable
persist_directory = Path(config.chroma_path)
persist_directory.mkdir(parents=True, exist_ok=True)

# 3. ChromaDB limits batch operations to 10,000 items
# Split large documents into batches
if len(chunks) > 10000:
    for i in range(0, len(chunks), 10000):
        batch = chunks[i:i+10000]
        collection.add(...)

# CRITICAL: sentence-transformers
# 1. Model downloads on first use (~420MB for all-mpnet-base-v2)
# 2. Cache in ~/.cache/torch/sentence_transformers/
# 3. Use encode() method, returns numpy arrays
embeddings = model.encode(texts)  # Returns np.ndarray

# CRITICAL: Token Counting vs Character Counting
# 1. Use tiktoken for accurate token counts, not len(text)
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(text)
token_count = len(tokens)

# CRITICAL: Pydantic v2 Patterns (not v1!)
from pydantic import BaseModel, Field
from datetime import datetime

class Document(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/openrag/models/schemas.py

from pydantic import BaseModel, Field
from datetime import datetime, UTC
from uuid import uuid4
from enum import Enum

class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentChunk(BaseModel):
    """Represents a chunk of a document with metadata"""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    chunk_index: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = Field(default_factory=dict)

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }

class DocumentMetadata(BaseModel):
    """Document metadata"""
    filename: str
    file_size: int  # bytes
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class Document(BaseModel):
    """Complete document with chunks"""
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: DocumentMetadata
    chunks: list[DocumentChunk] = Field(default_factory=list)

class QueryRequest(BaseModel):
    """Search query request"""
    query: str
    max_results: int = 5
    min_similarity: float = 0.1

class QueryResult(BaseModel):
    """Single query result"""
    chunk: DocumentChunk
    similarity_score: float
    document_name: str

class QueryResponse(BaseModel):
    """Query response with results"""
    results: list[QueryResult]
    query: str
    total_results: int
```

### Task List (Sequential Implementation Order)

```yaml
Task 1: Project Setup and Configuration
  CREATE pyproject.toml:
    - Define project metadata
    - Add dependencies: mcp, chromadb, sentence-transformers, pydantic, pytest, ruff, mypy
    - Configure ruff (line-length=100, target-version=py39)
    - Configure mypy (strict=true)

  CREATE src/openrag/__init__.py:
    - Package initialization
    - Version string

  CREATE src/openrag/config.py:
    - Pydantic Settings model
    - Environment variable loading (.env support)
    - Defaults: chroma_path="./chroma_db", embedding_model="all-mpnet-base-v2", chunk_size=400, chunk_overlap=60
    - Validation for paths, model names

  CREATE .env.example:
    - CHROMA_DB_PATH=./chroma_db
    - EMBEDDING_MODEL=all-mpnet-base-v2
    - CHUNK_SIZE=400
    - CHUNK_OVERLAP=60
    - LOG_LEVEL=INFO

  CREATE tests/conftest.py:
    - Pytest fixtures for temp directories
    - Mock embedding model fixture
    - Sample document fixtures

Task 2: Logging and Utilities
  CREATE src/openrag/utils/logger.py:
    - Configure logging to stderr only
    - Format: timestamp, name, level, message
    - Support LOG_LEVEL environment variable
    - CRITICAL: sys.stderr stream target

  CREATE src/openrag/utils/validation.py:
    - File path validation (exists, readable, .txt extension)
    - Query string validation (not empty, max length)
    - Parameter validation helpers
    - Security: prevent path traversal attacks

  CREATE tests/test_logger.py:
    - Test stderr output
    - Test log levels

  CREATE tests/test_validation.py:
    - Test file validation (valid, invalid paths, extensions)
    - Test query validation
    - Test security (path traversal attempts)

Task 3: Data Models
  CREATE src/openrag/models/__init__.py:
    - Export all models

  CREATE src/openrag/models/schemas.py:
    - Implement all Pydantic models (see Data Models section)
    - Type hints for all fields
    - Validation for string lengths, ranges
    - JSON serialization configuration

  CREATE tests/test_models.py:
    - Test model instantiation
    - Test validation rules
    - Test JSON serialization
    - Test datetime handling

Task 4: Text Chunking
  CREATE src/openrag/core/chunker.py:
    - TextChunker class
    - __init__(chunk_size, chunk_overlap, separators)
    - chunk_text(text: str) -> list[str]
    - RecursiveCharacterTextSplitter logic
    - Separators: ["\n\n", "\n", ".", "?", "!", " ", ""]
    - Token counting with tiktoken
    - PATTERN: Mirror examples/document_processor.py _create_chunks

  CREATE tests/test_chunker.py:
    - Test basic chunking
    - Test overlap behavior
    - Test separator hierarchy
    - Test edge cases (empty text, single word, very long text)
    - Test token counting accuracy

Task 5: Embedding Model Wrapper
  CREATE src/openrag/core/embedder.py:
    - EmbeddingModel class
    - __init__(model_name: str)
    - embed_texts(texts: list[str]) -> list[list[float]]
    - Model loading with sentence-transformers
    - Error handling for model download failures
    - Batch processing support
    - PATTERN: Mirror examples/vector_store.py _embedder usage

  CREATE tests/test_embedder.py:
    - Test model loading (use lightweight model for tests: all-MiniLM-L6-v2)
    - Test embedding generation
    - Test batch processing
    - Test error handling (invalid model name)
    - Mock sentence-transformers in tests for speed

Task 6: Vector Store (ChromaDB)
  CREATE src/openrag/core/vector_store.py:
    - VectorStore class
    - __init__(persist_directory: Path, collection_name: str, embedding_model: EmbeddingModel)
    - add_document(document: Document) -> None
    - search(query: str, n_results: int, min_similarity: float) -> list[tuple[DocumentChunk, float]]
    - delete_document(document_id: str) -> bool
    - get_stats() -> dict
    - list_documents() -> list[dict]
    - PersistentClient setup with explicit embedding function
    - Batch operations for large documents (max 10,000 per batch)
    - PATTERN: Mirror examples/vector_store.py structure
    - CRITICAL: Store all metadata fields (chunk_id, document_id, filename, created_at, chunk_index)

  CREATE tests/test_vector_store.py:
    - Test PersistentClient initialization
    - Test collection creation
    - Test add_document with single chunk
    - Test add_document with multiple chunks
    - Test add_document with >10k chunks (batch handling)
    - Test search functionality
    - Test delete_document
    - Test get_stats
    - Test list_documents
    - Use temporary directory for test database

Task 7: MCP Tools - Document Ingestion
  CREATE src/openrag/tools/ingest.py:
    - ingest_document_tool(file_path: str) async function
    - @mcp.tool() decorator
    - Validate file path (exists, .txt, readable)
    - Read file content
    - Chunk text using TextChunker
    - Create Document and DocumentChunk objects
    - Add to VectorStore
    - Return success message with stats
    - Error handling with helpful messages
    - PATTERN: Combine document_processor.py + vector_store.py patterns

  CREATE tests/test_ingest.py:
    - Test successful ingestion
    - Test file validation failures
    - Test empty file handling
    - Test very large file handling
    - Test chunking and embedding
    - Mock VectorStore for isolation

Task 8: MCP Tools - Query/Search
  CREATE src/openrag/tools/query.py:
    - query_documents_tool(query: str, max_results: int = 5) async function
    - @mcp.tool() decorator
    - Validate query (not empty)
    - Search VectorStore
    - Format results as QueryResponse
    - Return JSON-serializable results
    - Error handling

  CREATE tests/test_query.py:
    - Test successful query
    - Test empty results
    - Test max_results parameter
    - Test invalid query handling
    - Mock VectorStore with sample results

Task 9: MCP Tools - Document Management
  CREATE src/openrag/tools/manage.py:
    - list_documents_tool() async function
    - delete_document_tool(document_id: str) async function
    - @mcp.tool() decorators
    - Call VectorStore methods
    - Format responses
    - Error handling

  CREATE tests/test_manage.py:
    - Test list_documents
    - Test delete_document success
    - Test delete_document not found
    - Mock VectorStore

Task 10: MCP Tools - Statistics
  CREATE src/openrag/tools/stats.py:
    - get_stats_tool() async function
    - @mcp.tool() decorator
    - Collect system statistics
    - Format as JSON
    - Include: document count, chunk count, storage size, embedding model

  CREATE tests/test_stats.py:
    - Test stats collection
    - Test JSON formatting

Task 11: MCP Server Entry Point
  CREATE src/openrag/server.py:
    - Main server setup
    - Initialize MCP server instance
    - Load configuration
    - Initialize VectorStore, EmbeddingModel, TextChunker
    - Register all tools
    - Setup async main() function
    - Graceful shutdown handling
    - PATTERN: Follow official MCP Python SDK examples
    - CRITICAL: stdio transport for Claude Desktop integration

  CREATE tests/test_server.py:
    - Test server initialization
    - Test tool registration
    - Test configuration loading

Task 12: Integration Testing
  CREATE tests/test_integration.py:
    - End-to-end workflow tests
    - Test: ingest -> query -> delete cycle
    - Test: multiple documents
    - Test: search across documents
    - Test: persistence (restart server)
    - Use real ChromaDB (temp directory)
    - Use lightweight embedding model

Task 13: Documentation
  UPDATE README.md:
    - Project description
    - Features list
    - Installation instructions (conda environment, dependencies)
    - Configuration guide (environment variables)
    - Usage examples (Claude Desktop integration)
    - MCP tool documentation
    - Development setup
    - Testing instructions

  CREATE docs/DEPLOYMENT.md:
    - Deployment strategies (based on lab_journal.md findings)
    - Claude Desktop configuration
    - Security considerations
    - Performance tuning
    - Troubleshooting
```

### Pseudocode for Critical Components

```python
# Task 4: Text Chunking Core Logic
class TextChunker:
    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks using recursive strategy.

        PATTERN: Try separators in order until chunk_size met
        - First try "\n\n" (paragraphs)
        - Then "\n" (lines)
        - Then "." (sentences)
        - Then " " (words)
        - Finally "" (characters)

        CRITICAL: Use tiktoken for token counting, not len(text)
        CRITICAL: Overlap creates continuity at boundaries
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        chunks = []
        start = 0

        while start < len(text):
            # Calculate end based on token count
            end = self._find_chunk_end(text, start, encoding)

            # Try to end at semantic boundary
            if end < len(text):
                end = self._adjust_to_boundary(text, end)

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)

        return chunks

# Task 6: Vector Store - Search with Similarity Filtering
class VectorStore:
    async def search(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.1
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Semantic search with similarity threshold filtering.

        PATTERN:
        1. Embed query
        2. ChromaDB vector search
        3. Convert distance to similarity (1 - distance)
        4. Filter by min_similarity
        5. Reconstruct DocumentChunk objects from metadata

        GOTCHA: ChromaDB returns distances, we need similarity scores
        """
        # Embed query
        query_embedding = self.embedding_model.embed_texts([query])[0]

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Process results
        chunks_with_scores = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Convert distance to similarity (0-1, higher is better)
            similarity = max(0.0, 1.0 - distance)

            # Filter by threshold
            if similarity >= min_similarity:
                # Reconstruct DocumentChunk from metadata
                chunk = DocumentChunk(
                    chunk_id=metadata["chunk_id"],
                    document_id=metadata["document_id"],
                    content=doc,
                    chunk_index=metadata["chunk_index"],
                    created_at=datetime.fromisoformat(metadata["created_at"]),
                    metadata=metadata
                )
                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

# Task 7: Ingest Tool - Complete Workflow
@mcp.tool()
async def ingest_document(file_path: str) -> dict:
    """
    Ingest a document into the RAG system.

    WORKFLOW:
    1. Validate file (exists, .txt, readable, not too large)
    2. Read content
    3. Chunk text
    4. Create Document object with chunks
    5. Add to vector store (handles embedding)
    6. Return success with metadata

    ERROR HANDLING:
    - File not found -> helpful message with path
    - Not .txt -> explain supported formats
    - Too large -> suggest splitting
    - Embedding failure -> check model availability
    """
    # Validate
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.endswith('.txt'):
        raise ValueError(f"Only .txt files supported, got: {Path(file_path).suffix}")

    # Read
    content = Path(file_path).read_text(encoding='utf-8')

    # Chunk
    text_chunks = chunker.chunk_text(content)

    # Create Document
    document = Document(
        metadata=DocumentMetadata(
            filename=Path(file_path).name,
            file_size=Path(file_path).stat().st_size,
            chunk_count=len(text_chunks),
            status=DocumentStatus.PROCESSING
        )
    )

    # Create chunks
    for i, chunk_text in enumerate(text_chunks):
        chunk = DocumentChunk(
            document_id=document.document_id,
            content=chunk_text,
            chunk_index=i
        )
        document.chunks.append(chunk)

    # Add to vector store (embeddings happen here)
    vector_store.add_document(document)

    # Update status
    document.metadata.status = DocumentStatus.COMPLETED

    return {
        "status": "success",
        "document_id": document.document_id,
        "filename": document.metadata.filename,
        "chunk_count": len(document.chunks),
        "message": f"Successfully ingested {document.metadata.filename} with {len(document.chunks)} chunks"
    }
```

### Integration Points

```yaml
CONFIGURATION:
  - add to: pyproject.toml
  - dependencies:
      - mcp >= 0.1.0
      - chromadb >= 0.4.0
      - sentence-transformers >= 2.2.0
      - pydantic >= 2.0.0
      - pydantic-settings >= 2.0.0
      - tiktoken >= 0.5.0
      - pytest >= 7.0.0
      - pytest-asyncio >= 0.21.0
      - ruff >= 0.1.0
      - mypy >= 1.0.0
  - ruff config:
      - line-length = 100
      - target-version = "py39"
      - select = ["E", "F", "I", "N", "UP", "W"]
  - mypy config:
      - python_version = "3.9"
      - strict = true
      - warn_return_any = true

ENVIRONMENT:
  - create: .env file
  - variables:
      - CHROMA_DB_PATH: path to vector database
      - EMBEDDING_MODEL: model name
      - CHUNK_SIZE: chunk size in tokens
      - CHUNK_OVERLAP: overlap size in tokens
      - LOG_LEVEL: logging level

CLAUDE_DESKTOP:
  - add to: Claude Desktop MCP settings
  - config:
      command: python
      args: ["-m", "openrag.server"]
      env:
        CHROMA_DB_PATH: "/path/to/chroma_db"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Activate conda environment first
conda activate OpenRAG

# Run these FIRST - fix any errors before proceeding
ruff check src/ --fix
ruff format src/
mypy src/

# Expected: No errors. If errors, READ the error message and fix.
# Common issues:
# - Missing type hints
# - Line too long (>100 chars)
# - Import order wrong
# - Type mismatch
```

### Level 2: Unit Tests
```bash
# Run all unit tests with coverage
pytest tests/ -v --cov=src/openrag --cov-report=term-missing

# Expected: All tests pass, 80%+ coverage
# If failing:
# 1. Read the error message completely
# 2. Identify the failing assertion
# 3. Add print statements or use debugger
# 4. Fix the root cause, not the symptom
# 5. Re-run tests

# Run specific test file
pytest tests/test_chunker.py -v

# Run specific test
pytest tests/test_chunker.py::test_basic_chunking -v
```

### Level 3: Integration Test
```bash
# Test the complete workflow
pytest tests/test_integration.py -v

# Expected: Full ingest -> query -> delete cycle works
# Checks:
# - Document ingestion completes
# - Chunks stored in ChromaDB
# - Search returns relevant results
# - Document deletion removes all chunks
```

### Level 4: Manual MCP Server Test
```bash
# Start the server in development mode
python -m openrag.server

# In another terminal, test with MCP client or Claude Desktop
# OR use the MCP Inspector tool

# Test checklist:
# 1. Server starts without errors
# 2. Can list available tools
# 3. Can call ingest_document with sample .txt file
# 4. Can call query_documents and get results
# 5. Can call list_documents and see ingested doc
# 6. Can call delete_document
# 7. Can call get_stats
# 8. Check stderr logs for any warnings

# Expected output example:
# {"status": "success", "document_id": "...", "chunk_count": 42}
```

### Level 5: Claude Desktop Integration
```bash
# 1. Add server to Claude Desktop config
# 2. Restart Claude Desktop
# 3. Check that OpenRAG tools appear in Claude
# 4. Test workflow in conversation:
#    User: "Please ingest the file /path/to/document.txt"
#    Claude: [uses ingest_document tool]
#    User: "What does the document say about X?"
#    Claude: [uses query_documents tool]
# 5. Verify results make sense

# Debugging:
# - Check stderr logs in Claude Desktop console
# - Verify ChromaDB directory created
# - Check chroma.sqlite3 file exists
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/ -v --cov=src/openrag`
- [ ] Coverage >= 80%: Check coverage report
- [ ] No linting errors: `ruff check src/`
- [ ] No type errors: `mypy src/`
- [ ] Server starts: `python -m openrag.server`
- [ ] Can ingest document: Test with sample .txt file
- [ ] Can query document: Search returns relevant chunks
- [ ] Can list documents: Shows ingested documents
- [ ] Can delete document: Removes all chunks
- [ ] Can get stats: Returns system info
- [ ] Database persists: Restart server, data still there
- [ ] Error messages helpful: Test with invalid inputs
- [ ] Logs go to stderr: Verify no stdout pollution
- [ ] Documentation complete: README has setup/usage
- [ ] .env.example provided: Users know config options

---

## Anti-Patterns to Avoid

- ❌ Don't write to stdout (MCP protocol violation)
- ❌ Don't skip input validation (security risk)
- ❌ Don't hardcode paths (use configuration)
- ❌ Don't use default ChromaDB embedding (specify explicitly)
- ❌ Don't count tokens with len(text) (use tiktoken)
- ❌ Don't forget to create conda environment first
- ❌ Don't batch >10k items to ChromaDB (will fail)
- ❌ Don't expose file system paths without validation
- ❌ Don't assume embedding model is downloaded (handle first-use)
- ❌ Don't write sync code when MCP requires async

## Success Metrics

After implementation, the system should achieve:

1. **Functional**: All 5 MCP tools work correctly
2. **Quality**: 80%+ test coverage, no linting/type errors
3. **Performance**: Can ingest 10-page document in <30 seconds
4. **Usability**: Clear error messages, good documentation
5. **Reliability**: Database persists across restarts
6. **Security**: Input validation prevents path traversal
7. **Maintainability**: Modular code, <500 lines per file

## Implementation Confidence Score

**Score: 9/10**

**Rationale**:
- ✅ Clear technical specification with all requirements defined
- ✅ Comprehensive research in lab_journal.md provides best practices
- ✅ Example code from similar project shows patterns to follow
- ✅ All dependencies well-documented with version requirements
- ✅ Step-by-step task list with clear dependencies
- ✅ Validation gates are executable and comprehensive
- ✅ Error handling and edge cases documented
- ✅ Integration with Claude Desktop well-specified

**Potential Challenges**:
- ⚠️ First-time MCP server implementation (follow docs closely)
- ⚠️ Async patterns required by MCP (be careful with sync/async mixing)
- ⚠️ ChromaDB embedding model download on first use (may take time)

**Mitigation**:
- Follow MCP Python SDK examples exactly
- Use pytest-asyncio for testing async code
- Pre-download embedding model during setup
- Start with minimal implementation, then iterate

This PRP provides all necessary context for one-pass implementation success. The research, examples, and detailed task breakdown should enable smooth development with minimal rework.
