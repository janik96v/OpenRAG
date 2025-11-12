# Developer Guide

Guide for developers contributing to OpenRAG or extending its functionality.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Adding Features](#adding-features)
- [Debugging](#debugging)
- [Contributing](#contributing)

## Getting Started

### Prerequisites

- Python 3.10+ (3.12 recommended) 
- Anaconda or Miniconda
- Git
- Code editor (VS Code, PyCharm, or similar)
- Familiarity with async Python, Pydantic, and vector databases

### Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/OpenRAG.git
cd OpenRAG

# Create development environment
conda create -n OpenRAG python=3.12 -y
conda activate OpenRAG

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .

# Verify installation
python quick_test.py
pytest tests/ -v
```

## Development Environment

### Conda Environment

OpenRAG uses conda for environment management:

```bash
# Activate environment
conda activate OpenRAG

# List installed packages
conda list

# Add new dependency
pip install package_name

# Update requirements.txt
pip freeze > requirements-freeze.txt
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "/path/to/anaconda3/envs/OpenRAG/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add Conda Environment → Select "OpenRAG"
3. Enable Ruff for linting
4. Set line length to 100 characters

### Development Tools

#### Ruff (Linting and Formatting)

```bash
# Format code
ruff format src/ tests/

# Check for issues
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

#### MyPy (Type Checking)

```bash
# Run type checker
mypy src/openrag/

# Strict mode
mypy --strict src/openrag/
```

#### Pytest (Testing)

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/openrag --cov-report=html

# Run specific test file
pytest tests/test_chunker.py -v

# Run specific test
pytest tests/test_chunker.py::test_chunk_text -v
```

## Project Structure

Following vertical slice architecture:

```
OpenRAG/
├── src/openrag/
│   ├── __init__.py
│   ├── server.py              # MCP server entry point
│   ├── config.py              # Configuration management
│   │
│   ├── core/                  # Core components
│   │   ├── __init__.py
│   │   ├── chunker.py         # Text chunking logic
│   │   ├── embedder.py        # Embedding generation
│   │   └── vector_store.py    # ChromaDB interface
│   │
│   ├── tools/                 # MCP tools
│   │   ├── __init__.py
│   │   ├── ingest.py          # Document ingestion
│   │   ├── query.py           # Semantic search
│   │   ├── manage.py          # Document management
│   │   └── stats.py           # System statistics
│   │
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models
│   │
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── logger.py          # Logging setup
│       └── validation.py      # Input validation
│
├── tests/                     # Test suite
│   ├── conftest.py           # Pytest fixtures
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_tools.py
│   └── test_integration.py
│
├── docs/                      # Documentation
├── requirements.txt           # Core dependencies
├── requirements-dev.txt       # Dev dependencies
├── pyproject.toml            # Project configuration
├── .env.example              # Environment template
└── README.md                 # Project overview
```

### File Organization Principles

- **Max 500 lines** per file
- **Max 50 lines** per function
- **Max 100 lines** per class
- **Single Responsibility**: Each module has one purpose
- **Tests next to code**: Vertical slice architecture

## Code Standards

### Style Guide

#### Python Style

- **PEP 8 compliant**
- **Line length**: 100 characters (enforced by Ruff)
- **Quotes**: Double quotes for strings
- **Trailing commas**: In multi-line structures
- **Type hints**: Required for all functions

#### Naming Conventions

```python
# Variables and functions: snake_case
user_count = 10
def process_document(file_path: str) -> Document:
    pass

# Classes: PascalCase
class DocumentChunk:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CHUNK_SIZE = 2000
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

# Private: _leading_underscore
def _internal_helper() -> None:
    pass

# Type aliases: PascalCase
EmbeddingVector = list[float]
```

### Docstring Standards

Use Google-style docstrings:

```python
def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60
) -> list[str]:
    """
    Split text into overlapping chunks of specified size.

    Args:
        text: Input text to chunk
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks

    Raises:
        ValueError: If chunk_size <= chunk_overlap
        ValueError: If chunk_size > MAX_CHUNK_SIZE

    Example:
        >>> chunks = chunk_text("Long text here", chunk_size=100)
        >>> len(chunks)
        3
    """
```

Required sections:
- Brief description
- `Args:` - All parameters
- `Returns:` - Return value description
- `Raises:` - Exceptions that may be raised
- `Example:` - Usage example (optional but encouraged)

### Type Hints

Required for all public functions:

```python
from typing import Optional, Union
from pathlib import Path

# Good
def load_document(path: Path) -> Optional[str]:
    """Load document from path."""
    pass

def embed_text(
    texts: list[str],
    batch_size: int = 32
) -> list[list[float]]:
    """Generate embeddings for texts."""
    pass

# Bad - no type hints
def load_document(path):
    pass
```

### Error Handling

Use specific exceptions:

```python
class OpenRAGError(Exception):
    """Base exception for OpenRAG."""
    pass

class DocumentNotFoundError(OpenRAGError):
    """Document not found in vector store."""
    pass

class InvalidChunkSizeError(OpenRAGError):
    """Invalid chunk size configuration."""
    pass

# Usage
def validate_chunk_size(size: int) -> None:
    """Validate chunk size is within acceptable range."""
    if size < 100:
        raise InvalidChunkSizeError(
            f"Chunk size must be >= 100, got {size}"
        )
    if size > 2000:
        raise InvalidChunkSizeError(
            f"Chunk size must be <= 2000, got {size}"
        )
```

### Logging

Log to stderr (MCP requirement):

```python
from .utils.logger import setup_logger

logger = setup_logger(__name__)

# Different log levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)

# With context
logger.info(
    "Document ingested",
    extra={
        "document_id": doc_id,
        "chunk_count": len(chunks),
        "file_size": file_size
    }
)
```

## Testing

### Test Organization

Following TDD approach:

```
tests/
├── conftest.py              # Shared fixtures
├── test_chunker.py         # Unit tests for chunker
├── test_embedder.py        # Unit tests for embedder
├── test_vector_store.py    # Unit tests for vector store
├── test_tools.py           # Unit tests for tools
└── test_integration.py     # Integration tests
```

### Writing Tests

#### Unit Test Template

```python
import pytest
from openrag.core.chunker import TextChunker

class TestTextChunker:
    """Tests for TextChunker class."""

    @pytest.fixture
    def chunker(self):
        """Provide a TextChunker instance."""
        return TextChunker(chunk_size=400, chunk_overlap=60)

    @pytest.fixture
    def sample_text(self):
        """Provide sample text for testing."""
        return "This is a test. " * 100

    def test_chunk_text_basic(self, chunker, sample_text):
        """Test basic text chunking."""
        chunks = chunker.chunk_text(sample_text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_chunk_text_respects_size(self, chunker, sample_text):
        """Test chunks respect maximum size."""
        chunks = chunker.chunk_text(sample_text)

        for chunk in chunks:
            token_count = chunker.count_tokens(chunk)
            assert token_count <= chunker.chunk_size

    def test_chunk_text_empty_input(self, chunker):
        """Test chunking empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            chunker.chunk_text("")

    def test_chunk_text_overlap(self, chunker):
        """Test chunks have expected overlap."""
        text = "A B C D E F G H I J " * 50
        chunks = chunker.chunk_text(text)

        # Verify consecutive chunks have overlap
        for i in range(len(chunks) - 1):
            # Check some content appears in both chunks
            assert any(
                word in chunks[i+1]
                for word in chunks[i].split()[-10:]
            )
```

#### Integration Test Template

```python
import pytest
from pathlib import Path
from openrag.core.chunker import TextChunker
from openrag.core.embedder import EmbeddingModel
from openrag.core.vector_store import VectorStore
from openrag.tools.ingest import ingest_document_tool
from openrag.tools.query import query_documents_tool

@pytest.mark.asyncio
class TestFullWorkflow:
    """Integration tests for complete RAG workflow."""

    @pytest.fixture
    async def setup_system(self, tmp_path):
        """Set up complete RAG system."""
        # Create temporary database
        db_path = tmp_path / "chroma_db"

        # Initialize components
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        embedding_model = EmbeddingModel("all-MiniLM-L6-v2")
        vector_store = VectorStore(db_path, embedding_model)

        yield {
            "chunker": chunker,
            "vector_store": vector_store,
            "db_path": db_path
        }

        # Cleanup handled by tmp_path fixture

    @pytest.fixture
    def sample_document(self, tmp_path):
        """Create temporary test document."""
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text(
            "Machine learning is a field of AI.\n"
            "It involves training algorithms on data.\n"
            "Neural networks are a popular approach."
        )
        return str(doc_path)

    async def test_ingest_and_query(
        self, setup_system, sample_document
    ):
        """Test complete ingest and query workflow."""
        system = setup_system

        # Ingest document
        ingest_result = await ingest_document_tool(
            file_path=sample_document,
            vector_store=system["vector_store"],
            chunker=system["chunker"]
        )

        assert ingest_result["status"] == "success"
        assert ingest_result["chunk_count"] > 0

        # Query document
        query_result = await query_documents_tool(
            query="What is machine learning?",
            vector_store=system["vector_store"],
            max_results=3
        )

        assert query_result["status"] == "success"
        assert query_result["total_results"] > 0
        assert "machine learning" in \
            query_result["results"][0]["content"].lower()
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/openrag --cov-report=html --cov-report=term

# Specific test file
pytest tests/test_chunker.py -v

# Specific test class
pytest tests/test_chunker.py::TestTextChunker -v

# Specific test method
pytest tests/test_chunker.py::TestTextChunker::test_chunk_text_basic -v

# Tests matching pattern
pytest tests/ -k "chunk" -v

# Stop on first failure
pytest tests/ -x

# Verbose output with print statements
pytest tests/ -v -s
```

### Test Coverage Goals

- **Target**: 80%+ code coverage
- **Focus**: Critical paths and edge cases
- **Priority**: Core components > Tools > Utilities

Check coverage:
```bash
pytest tests/ --cov=src/openrag --cov-report=term-missing
```

## Adding Features

### Adding a New Tool

1. **Create tool module** in `src/openrag/tools/`

```python
# src/openrag/tools/my_new_tool.py
"""My new MCP tool."""

from ..core.vector_store import VectorStore

async def my_new_tool(
    param: str,
    vector_store: VectorStore
) -> dict:
    """
    Perform new functionality.

    Args:
        param: Description of parameter
        vector_store: VectorStore instance

    Returns:
        Result dictionary with status and data
    """
    try:
        # Implement logic here
        result = do_something(param, vector_store)

        return {
            "status": "success",
            "data": result,
            "message": "Operation completed"
        }
    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": "operation_failed",
            "message": str(e)
        }
```

2. **Register in server** (`src/openrag/server.py`)

```python
# In list_tools()
Tool(
    name="my_new_tool",
    description="Description of what the tool does",
    inputSchema={
        "type": "object",
        "properties": {
            "param": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param"]
    }
)

# In call_tool()
elif name == "my_new_tool":
    result = await my_new_tool(
        param=arguments["param"],
        vector_store=vector_store
    )
```

3. **Write tests**

```python
# tests/test_my_new_tool.py
import pytest
from openrag.tools.my_new_tool import my_new_tool

@pytest.mark.asyncio
async def test_my_new_tool(vector_store):
    """Test new tool functionality."""
    result = await my_new_tool(
        param="test",
        vector_store=vector_store
    )

    assert result["status"] == "success"
    assert "data" in result
```

4. **Update documentation**
   - Add to [API Reference](api-reference.md)
   - Update [User Guide](user-guide.md)
   - Update README.md

### Adding Core Functionality

1. Create module in `src/openrag/core/`
2. Write comprehensive unit tests
3. Update affected tools
4. Document in architecture.md

### Adding Configuration Options

1. **Add to Settings** (`src/openrag/config.py`)

```python
class Settings(BaseSettings):
    # Existing settings...

    new_setting: Annotated[
        int,
        Field(
            default=100,
            ge=1,
            le=1000,
            description="Description of new setting"
        )
    ]

    @field_validator("new_setting")
    @classmethod
    def validate_new_setting(cls, v: int) -> int:
        """Validate new setting."""
        if v < 10:
            raise ValueError("New setting must be >= 10")
        return v
```

2. **Add to .env.example**

```bash
# New Feature
NEW_SETTING=100
```

3. **Document** in [Configuration Reference](configuration.md)

## Debugging

### Using ipdb

```python
# Add breakpoint
import ipdb; ipdb.set_trace()

# Common commands
# n - next line
# s - step into
# c - continue
# p variable - print variable
# q - quit
```

### Logging for Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via .env
LOG_LEVEL=DEBUG
```

### Testing MCP Server

```bash
# Start server manually
python -m openrag.server

# In another terminal, send MCP messages
# (Use MCP inspector or client)
```

### Common Issues

**Import errors**:
```bash
# Reinstall in editable mode
pip install -e .
```

**Test failures**:
```bash
# Run with verbose output
pytest tests/ -v -s

# Run specific failing test
pytest tests/test_file.py::test_name -v -s
```

## Contributing

### Git Workflow

Following branching strategy:

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add .
git commit -m "feat(component): add new feature

- Implement feature logic
- Add tests
- Update documentation"

# Push and create PR
git push origin feature/my-new-feature
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(tools): add batch ingestion tool

- Implement batch_ingest_tool for multiple documents
- Add progress reporting
- Add tests for batch operations

Closes #42
```

### Pull Request Process

1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Run full test suite
5. Format and lint code
6. Push and create PR
7. Address review feedback
8. Merge when approved

### Code Review Checklist

- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Code formatted with Ruff
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] No linting errors

## Related Documentation

- [Architecture](architecture.md) - System design
- [API Reference](api-reference.md) - Tool specifications
- [Testing Guide](TESTING.md) - Testing strategies

---

Last Updated: 2025-11-09
