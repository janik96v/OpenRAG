# User Guide

Complete guide for using OpenRAG to build a personal RAG system over your documents.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Document Ingestion](#document-ingestion)
- [Searching Documents](#searching-documents)
- [Managing Documents](#managing-documents)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

OpenRAG enables you to:
- **Ingest** personal documents into a vector database
- **Search** using natural language queries
- **Retrieve** semantically similar content
- **Manage** your document collection

All data stays local on your machine - no external APIs or cloud services required.

## Core Concepts

### What is RAG?

Retrieval Augmented Generation (RAG) combines:
1. **Retrieval**: Finding relevant information from your documents
2. **Generation**: Using that information to answer questions (via LLM)

OpenRAG handles the retrieval part, providing context to AI assistants like Claude.

### How It Works

```
Your Document → Chunking → Embedding → Vector Storage
                    ↓
              Semantic Search ← Your Query
                    ↓
              Retrieved Context → AI Assistant → Answer
```

1. **Chunking**: Documents split into ~400-token chunks
2. **Embedding**: Each chunk converted to a vector (768 dimensions)
3. **Storage**: Vectors stored in ChromaDB (local SQLite database)
4. **Search**: Query embedded and matched against stored vectors
5. **Retrieval**: Most similar chunks returned to AI assistant

### Key Terms

- **Chunk**: A segment of text (typically 400 tokens)
- **Embedding**: Vector representation of text
- **Similarity**: Cosine similarity score (0.0-1.0)
- **Vector Store**: ChromaDB database containing embeddings
- **Document ID**: Unique identifier (UUID) for each document

## Document Ingestion

### Supported Formats

Currently supported:
- `.txt` files (UTF-8 encoding)

Coming soon:
- PDF files
- Markdown files
- HTML files

### Ingestion Workflow

#### 1. Prepare Your Document

Ensure your document:
- Is in `.txt` format
- Uses UTF-8 encoding
- Is under 10 MB (recommended)
- Has meaningful content (not just metadata)

#### 2. Ingest Using Python

```python
import asyncio
from pathlib import Path
from openrag.core.chunker import TextChunker
from openrag.core.embedder import EmbeddingModel
from openrag.core.vector_store import VectorStore
from openrag.tools.ingest import ingest_document_tool
from openrag.config import get_settings

async def ingest_document(file_path: str):
    """Ingest a document into OpenRAG."""
    settings = get_settings()

    # Initialize components
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    embedding_model = EmbeddingModel(model_name=settings.embedding_model)
    vector_store = VectorStore(
        persist_directory=Path(settings.chroma_db_path),
        embedding_model=embedding_model
    )

    # Ingest
    result = await ingest_document_tool(
        file_path=file_path,
        vector_store=vector_store,
        chunker=chunker
    )

    if result['status'] == 'success':
        print(f"✅ Success!")
        print(f"   Document ID: {result['document_id']}")
        print(f"   Filename: {result['filename']}")
        print(f"   Chunks: {result['chunk_count']}")
        print(f"   Size: {result['file_size']} bytes")
    else:
        print(f"❌ Error: {result['message']}")

    return result

# Run
result = asyncio.run(ingest_document("/path/to/your/document.txt"))
```

#### 3. Ingest Using Claude Desktop

If you have OpenRAG configured with Claude Desktop:

```
Please ingest this document: /Users/name/documents/research_paper.txt
```

Claude will use the `ingest_document` tool automatically.

### Understanding Chunk Count

The number of chunks depends on:
- **Document length**: Longer documents = more chunks
- **Chunk size**: Default 400 tokens ≈ 300 words ≈ 1 page
- **Overlap**: 15% overlap means some content appears in multiple chunks

Example:
- 10-page document ≈ 10-15 chunks
- 100-page book ≈ 100-150 chunks
- 1,000-page technical manual ≈ 1,000-1,500 chunks

### Storage Requirements

Approximate storage per chunk:
- Embeddings: ~3 KB (768 dimensions)
- Metadata: ~1 KB
- **Total**: ~4 KB per chunk

Examples:
- 10-page document (15 chunks) ≈ 60 KB
- 100-page book (150 chunks) ≈ 600 KB
- 1,000-page manual (1,500 chunks) ≈ 6 MB

### Ingestion Performance

Typical speeds:
- **all-MiniLM-L6-v2**: ~2-3 pages/second
- **all-mpnet-base-v2**: ~0.5-1 pages/second

For a 100-page document:
- MiniLM: ~30-60 seconds
- mpnet: ~2-3 minutes

## Searching Documents

### Basic Search

```python
from openrag.tools.query import query_documents_tool

async def search(query: str):
    """Search ingested documents."""
    # Initialize vector_store (same as ingestion)
    # ...

    result = await query_documents_tool(
        query=query,
        vector_store=vector_store,
        max_results=5,
        min_similarity=0.1
    )

    print(f"Query: {query}")
    print(f"Found {result['total_results']} results\n")

    for i, res in enumerate(result['results'], 1):
        print(f"{i}. Score: {res['similarity_score']:.3f}")
        print(f"   Source: {res['document_name']}")
        print(f"   {res['content'][:200]}...")
        print()

# Run
asyncio.run(search("What is machine learning?"))
```

### Advanced Search Parameters

#### max_results

Controls how many results to return.

```python
# Return top 3 results
result = await query_documents_tool(
    query="neural networks",
    vector_store=vector_store,
    max_results=3
)

# Return top 10 results
result = await query_documents_tool(
    query="neural networks",
    vector_store=vector_store,
    max_results=10
)
```

**Guidelines**:
- Use 3-5 for focused answers
- Use 10-20 for comprehensive exploration
- Use 1-2 for quick fact-checking

#### min_similarity

Filters results below a similarity threshold.

```python
# High precision (fewer, more relevant results)
result = await query_documents_tool(
    query="quantum computing",
    vector_store=vector_store,
    min_similarity=0.6  # Only very relevant results
)

# High recall (more results, some less relevant)
result = await query_documents_tool(
    query="quantum computing",
    vector_store=vector_store,
    min_similarity=0.2  # Include marginally relevant
)
```

**Similarity Score Interpretation**:

| Range | Meaning | Use Case |
|-------|---------|----------|
| 0.8-1.0 | Very high match | Exact topic/question |
| 0.6-0.8 | High match | Related information |
| 0.4-0.6 | Moderate match | Tangentially related |
| 0.2-0.4 | Low match | Broad topic search |
| 0.0-0.2 | Very low match | Usually not useful |

**Recommended Thresholds**:
- **Precision-focused**: 0.5-0.7 (best answers only)
- **Balanced**: 0.3-0.5 (default range)
- **Recall-focused**: 0.1-0.3 (explore broadly)

### Query Optimization

#### Good Queries

```
✅ "What are the main types of machine learning?"
✅ "Explain how transformers work in NLP"
✅ "What is the difference between supervised and unsupervised learning?"
✅ "How does gradient descent optimize neural networks?"
```

Characteristics:
- Natural language
- Specific and focused
- Question-like format
- Domain terminology included

#### Poor Queries

```
❌ "ML" (too short, ambiguous)
❌ "Tell me everything" (too broad)
❌ "asdfasdf" (nonsense)
❌ "123" (not semantic content)
```

### Using Results with Claude

When using Claude Desktop with OpenRAG:

```
Based on my documents, explain the concept of attention mechanisms in transformers.
```

Claude will:
1. Use `query_documents` tool to find relevant chunks
2. Read the retrieved content
3. Synthesize an answer based on your documents

## Managing Documents

### List All Documents

```python
from openrag.tools.manage import list_documents_tool

async def list_docs():
    """List all ingested documents."""
    # Initialize vector_store
    # ...

    result = await list_documents_tool(vector_store=vector_store)

    print(f"Total documents: {result['total_documents']}\n")

    for doc in result['documents']:
        print(f"📄 {doc['filename']}")
        print(f"   ID: {doc['document_id']}")
        print(f"   Chunks: {doc['chunk_count']}")
        print(f"   Size: {doc['file_size']:,} bytes")
        print(f"   Created: {doc['created_at']}")
        print()

asyncio.run(list_docs())
```

### Delete a Document

```python
from openrag.tools.manage import delete_document_tool

async def delete_doc(document_id: str):
    """Delete a document."""
    # Initialize vector_store
    # ...

    result = await delete_document_tool(
        document_id=document_id,
        vector_store=vector_store
    )

    if result['status'] == 'success':
        print(f"✅ Deleted document {document_id}")
        print(f"   Removed {result['chunks_deleted']} chunks")
    else:
        print(f"❌ Error: {result['message']}")

asyncio.run(delete_doc("550e8400-e29b-41d4-a716-446655440000"))
```

**Important**: Deletion is permanent and cannot be undone!

### Get System Statistics

```python
from openrag.tools.stats import get_stats_tool
from openrag.config import get_settings

async def get_stats():
    """Get system statistics."""
    settings = get_settings()
    # Initialize vector_store
    # ...

    result = await get_stats_tool(
        vector_store=vector_store,
        settings=settings
    )

    stats = result['statistics']
    print(f"📊 OpenRAG Statistics")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Chunks: {stats['total_chunks']}")
    print(f"   Storage: {stats['storage_path']}")
    print(f"   Model: {stats['embedding_model']}")
    print(f"   Dimensions: {stats['embedding_dimension']}")
    print(f"   Chunk size: {stats['chunk_size']} tokens")
    print(f"   Chunk overlap: {stats['chunk_overlap']} tokens")

asyncio.run(get_stats())
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# ChromaDB Storage
CHROMA_DB_PATH=./chroma_db

# Embedding Model
EMBEDDING_MODEL=all-mpnet-base-v2

# Chunking Strategy
CHUNK_SIZE=400
CHUNK_OVERLAP=60

# Logging
LOG_LEVEL=INFO
```

### Choosing an Embedding Model

| Model | Speed | Quality | Size | Use Case |
|-------|-------|---------|------|----------|
| all-MiniLM-L6-v2 | Fast | Good | 80 MB | Development, testing |
| all-mpnet-base-v2 | Moderate | Best | 420 MB | Production (default) |

Change model by editing `.env`:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2  # For speed
EMBEDDING_MODEL=all-mpnet-base-v2  # For quality
```

### Adjusting Chunk Size

Default: 400 tokens (≈300 words, ≈1 page)

**Smaller chunks (200-300 tokens)**:
- More precise retrieval
- Better for fact-finding
- More chunks = more storage

**Larger chunks (500-800 tokens)**:
- More context per result
- Better for comprehensive answers
- Fewer chunks = less storage

Change in `.env`:
```bash
CHUNK_SIZE=200  # Smaller, more precise
CHUNK_SIZE=600  # Larger, more context
```

### Adjusting Overlap

Default: 60 tokens (15% of 400)

**More overlap (20-25%)**:
- Less information loss at boundaries
- Better for capturing complete thoughts
- More storage required

**Less overlap (5-10%)**:
- More storage efficient
- Risk of losing context at boundaries

Change in `.env`:
```bash
CHUNK_OVERLAP=100  # 25% overlap
CHUNK_OVERLAP=20   # 5% overlap
```

See [Configuration Reference](configuration.md) for all options.

## Best Practices

### Document Organization

1. **Group related documents**: Ingest documents on similar topics together
2. **Use descriptive filenames**: Makes management easier
3. **Keep documents focused**: One topic per document works best
4. **Regular cleanup**: Delete outdated or irrelevant documents

### Query Strategies

1. **Be specific**: Use detailed, focused queries
2. **Use domain terminology**: Match language in your documents
3. **Iterate**: Refine queries based on results
4. **Adjust parameters**: Tune `max_results` and `min_similarity`

### Performance Optimization

1. **Choose right model**: Balance speed vs. quality for your use case
2. **Batch ingestion**: Ingest multiple documents in one session
3. **Monitor storage**: Check disk space as collection grows
4. **Clean old data**: Remove outdated documents regularly

### Quality Improvement

1. **Well-formatted documents**: Clear structure improves chunking
2. **Remove noise**: Clean documents before ingestion (headers, footers)
3. **Appropriate chunk size**: Adjust for your document type
4. **Test queries**: Verify retrieval quality with known questions

## Troubleshooting

### No Results for Query

**Causes**:
- No documents ingested yet
- Query too specific or uses different terminology
- Similarity threshold too high

**Solutions**:
```python
# Lower similarity threshold
result = await query_documents_tool(
    query="your query",
    vector_store=vector_store,
    min_similarity=0.1  # Lower threshold
)

# Increase max results
result = await query_documents_tool(
    query="your query",
    vector_store=vector_store,
    max_results=20  # More results
)
```

### Poor Quality Results

**Causes**:
- Wrong embedding model for content type
- Chunk size too large or too small
- Document quality issues

**Solutions**:
1. Try different embedding model
2. Adjust chunk size in `.env`
3. Clean and reformat documents
4. Re-ingest with new settings

### Slow Performance

**Causes**:
- Large collection (1000+ documents)
- Using slow embedding model (mpnet)
- Insufficient RAM

**Solutions**:
1. Switch to faster model (MiniLM)
2. Reduce chunk size to decrease total chunks
3. Increase system RAM
4. Delete unused documents

### Storage Issues

**Causes**:
- Very large collection
- Disk space running out

**Solutions**:
```bash
# Check storage size
du -sh ./chroma_db

# Clean up old documents
# (Use delete_document tool)

# Move to larger disk
mv ./chroma_db /path/to/larger/disk/chroma_db
# Update .env with new path
```

## Advanced Usage

### Programmatic Workflows

Batch ingest multiple documents:

```python
import asyncio
from pathlib import Path

async def batch_ingest(file_paths: list[str]):
    """Ingest multiple documents."""
    # Initialize components once
    settings = get_settings()
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    embedding_model = EmbeddingModel(settings.embedding_model)
    vector_store = VectorStore(
        Path(settings.chroma_db_path),
        embedding_model
    )

    results = []
    for file_path in file_paths:
        print(f"Ingesting {file_path}...")
        result = await ingest_document_tool(
            file_path, vector_store, chunker
        )
        results.append(result)
        print(f"  ✅ {result['chunk_count']} chunks")

    return results

# Run
files = [
    "/path/to/doc1.txt",
    "/path/to/doc2.txt",
    "/path/to/doc3.txt"
]
asyncio.run(batch_ingest(files))
```

### Export/Backup

Backup your vector database:

```bash
# Backup ChromaDB directory
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./chroma_db

# Restore from backup
tar -xzf chroma_backup_20251109.tar.gz
```

## Next Steps

- **[API Reference](api-reference.md)** - Detailed tool specifications
- **[Configuration](configuration.md)** - All configuration options
- **[Architecture](architecture.md)** - How OpenRAG works internally
- **[Developer Guide](developer-guide.md)** - Contributing to OpenRAG

---

Last Updated: 2025-11-09
