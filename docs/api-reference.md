# API Reference

Complete reference for all MCP tools exposed by OpenRAG.

## Overview

OpenRAG exposes 5 MCP tools that enable document ingestion, semantic search, and management of a personal RAG system.

### Tool Summary

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `ingest_document` | Add document to RAG system | File path | Document ID, stats |
| `query_documents` | Semantic search | Query text | Ranked results |
| `list_documents` | List all documents | None | Document list |
| `delete_document` | Remove document | Document ID | Success status |
| `get_stats` | System statistics | None | System stats |

## Common Patterns

### Response Format

All tools return JSON with a consistent structure:

```json
{
  "status": "success" | "error",
  "message": "Optional human-readable message",
  ... // Tool-specific fields
}
```

### Error Handling

Errors include:
- `status`: Always "error"
- `error`: Error type code
- `message`: Human-readable description

Example error response:
```json
{
  "status": "error",
  "error": "file_not_found",
  "message": "File does not exist: /path/to/file.txt"
}
```

## Tool Specifications

### 1. ingest_document

Ingest a text document into the RAG system by chunking, embedding, and storing in the vector database.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Absolute path to the .txt file to ingest"
    }
  },
  "required": ["file_path"]
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | Absolute path to .txt file |

#### Success Response

```json
{
  "status": "success",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.txt",
  "file_size": 15234,
  "chunk_count": 12,
  "message": "Document ingested successfully"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `document_id` | string | UUID of ingested document |
| `filename` | string | Name of the file |
| `file_size` | integer | File size in bytes |
| `chunk_count` | integer | Number of chunks created |
| `message` | string | Success message |

#### Error Responses

**File Not Found**:
```json
{
  "status": "error",
  "error": "file_not_found",
  "message": "File does not exist: /path/to/file.txt"
}
```

**Invalid File Type**:
```json
{
  "status": "error",
  "error": "invalid_file_type",
  "message": "Only .txt files are supported"
}
```

**File Too Large**:
```json
{
  "status": "error",
  "error": "file_too_large",
  "message": "File size exceeds maximum allowed size"
}
```

**Encoding Error**:
```json
{
  "status": "error",
  "error": "encoding_error",
  "message": "Failed to read file: invalid encoding"
}
```

#### Example Usage

```python
# Using MCP client
result = await client.call_tool(
    "ingest_document",
    {
        "file_path": "/Users/name/documents/research_paper.txt"
    }
)

print(f"Document ID: {result['document_id']}")
print(f"Created {result['chunk_count']} chunks")
```

#### Process Flow

1. Validate file path exists and is .txt
2. Read file content (UTF-8 encoding)
3. Split into chunks using TextChunker (400 tokens, 15% overlap)
4. Generate embeddings for each chunk
5. Store in ChromaDB with metadata
6. Return document ID and statistics

#### Performance

- **Speed**: ~1-2 seconds per page (depends on embedding model)
- **Memory**: Chunks processed in batches
- **Disk**: ~4 KB per chunk (embeddings + metadata)

---

### 2. query_documents

Perform semantic search over all ingested documents using natural language queries.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query"
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum number of results to return (default: 5)",
      "default": 5
    },
    "min_similarity": {
      "type": "number",
      "description": "Minimum similarity score threshold 0-1 (default: 0.1)",
      "default": 0.1
    }
  },
  "required": ["query"]
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language query |
| `max_results` | integer | No | 5 | Max results to return (1-100) |
| `min_similarity` | float | No | 0.1 | Min similarity threshold (0.0-1.0) |

#### Success Response

```json
{
  "status": "success",
  "query": "What is machine learning?",
  "total_results": 3,
  "results": [
    {
      "chunk_id": "chunk_uuid_1",
      "document_id": "doc_uuid_1",
      "document_name": "ml_intro.txt",
      "content": "Machine learning is a subset of artificial intelligence...",
      "similarity_score": 0.89,
      "chunk_index": 0
    },
    {
      "chunk_id": "chunk_uuid_2",
      "document_id": "doc_uuid_1",
      "document_name": "ml_intro.txt",
      "content": "Supervised learning algorithms train on labeled data...",
      "similarity_score": 0.76,
      "chunk_index": 2
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `query` | string | Original query text |
| `total_results` | integer | Number of results returned |
| `results` | array | Array of result objects (see below) |

**Result Object Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | UUID of the chunk |
| `document_id` | string | UUID of source document |
| `document_name` | string | Name of source document |
| `content` | string | Text content of chunk |
| `similarity_score` | float | Cosine similarity (0.0-1.0) |
| `chunk_index` | integer | Position in original document |

#### Error Responses

**Empty Query**:
```json
{
  "status": "error",
  "error": "empty_query",
  "message": "Query cannot be empty"
}
```

**No Results**:
```json
{
  "status": "success",
  "query": "obscure topic",
  "total_results": 0,
  "results": [],
  "message": "No results found matching the query"
}
```

**Empty Collection**:
```json
{
  "status": "error",
  "error": "empty_collection",
  "message": "No documents have been ingested yet"
}
```

#### Example Usage

```python
# Basic query
result = await client.call_tool(
    "query_documents",
    {
        "query": "What is deep learning?"
    }
)

# Advanced query with filters
result = await client.call_tool(
    "query_documents",
    {
        "query": "neural network architectures",
        "max_results": 10,
        "min_similarity": 0.5
    }
)

for item in result['results']:
    print(f"Score: {item['similarity_score']:.2f}")
    print(f"Source: {item['document_name']}")
    print(f"Content: {item['content'][:100]}...")
    print()
```

#### Similarity Score Interpretation

| Range | Interpretation |
|-------|----------------|
| 0.8 - 1.0 | Very high relevance |
| 0.6 - 0.8 | High relevance |
| 0.4 - 0.6 | Moderate relevance |
| 0.2 - 0.4 | Low relevance |
| 0.0 - 0.2 | Very low relevance |

#### Process Flow

1. Validate query is not empty
2. Generate embedding for query
3. Perform vector similarity search in ChromaDB
4. Rank results by cosine similarity
5. Filter by minimum similarity threshold
6. Return top N results

#### Performance

- **Speed**: 100-500ms per query (depends on collection size)
- **Accuracy**: Depends on embedding model quality
- **Scaling**: Linear with collection size

---

### 3. list_documents

List all ingested documents with their metadata.

#### Input Schema

```json
{
  "type": "object",
  "properties": {}
}
```

#### Parameters

None.

#### Success Response

```json
{
  "status": "success",
  "total_documents": 2,
  "documents": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "research_paper.txt",
      "file_size": 15234,
      "chunk_count": 12,
      "created_at": "2025-11-09T10:30:00Z",
      "status": "completed"
    },
    {
      "document_id": "660e8400-e29b-41d4-a716-446655440001",
      "filename": "notes.txt",
      "file_size": 8421,
      "chunk_count": 7,
      "created_at": "2025-11-09T11:15:00Z",
      "status": "completed"
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `total_documents` | integer | Total number of documents |
| `documents` | array | Array of document objects |

**Document Object Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | string | UUID of document |
| `filename` | string | Original filename |
| `file_size` | integer | File size in bytes |
| `chunk_count` | integer | Number of chunks |
| `created_at` | string | ISO 8601 timestamp |
| `status` | string | Processing status |

#### Error Responses

**Empty Collection**:
```json
{
  "status": "success",
  "total_documents": 0,
  "documents": [],
  "message": "No documents have been ingested yet"
}
```

#### Example Usage

```python
result = await client.call_tool("list_documents", {})

print(f"Total documents: {result['total_documents']}")
for doc in result['documents']:
    print(f"- {doc['filename']} ({doc['chunk_count']} chunks)")
```

---

### 4. delete_document

Delete a document and all its chunks from the vector database.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "document_id": {
      "type": "string",
      "description": "ID of the document to delete"
    }
  },
  "required": ["document_id"]
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `document_id` | string | Yes | UUID of document to delete |

#### Success Response

```json
{
  "status": "success",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_deleted": 12,
  "message": "Document and 12 chunks deleted successfully"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `document_id` | string | UUID of deleted document |
| `chunks_deleted` | integer | Number of chunks removed |
| `message` | string | Success message |

#### Error Responses

**Document Not Found**:
```json
{
  "status": "error",
  "error": "document_not_found",
  "message": "Document not found: 550e8400-e29b-41d4-a716-446655440000"
}
```

**Invalid Document ID**:
```json
{
  "status": "error",
  "error": "invalid_document_id",
  "message": "Invalid document ID format"
}
```

#### Example Usage

```python
result = await client.call_tool(
    "delete_document",
    {
        "document_id": "550e8400-e29b-41d4-a716-446655440000"
    }
)

print(f"Deleted {result['chunks_deleted']} chunks")
```

#### Important Notes

- This operation is **permanent** and cannot be undone
- All chunks associated with the document are deleted
- Embeddings are removed from the vector store
- Metadata is permanently deleted

---

### 5. get_stats

Get system statistics including document count, chunk count, and configuration.

#### Input Schema

```json
{
  "type": "object",
  "properties": {}
}
```

#### Parameters

None.

#### Success Response

```json
{
  "status": "success",
  "statistics": {
    "total_documents": 5,
    "total_chunks": 47,
    "storage_path": "/Users/name/openrag/chroma_db",
    "embedding_model": "all-mpnet-base-v2",
    "embedding_dimension": 768,
    "chunk_size": 400,
    "chunk_overlap": 60
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `statistics` | object | Statistics object (see below) |

**Statistics Object Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `total_documents` | integer | Number of ingested documents |
| `total_chunks` | integer | Total number of chunks |
| `storage_path` | string | ChromaDB storage location |
| `embedding_model` | string | Model name being used |
| `embedding_dimension` | integer | Embedding vector size |
| `chunk_size` | integer | Max tokens per chunk |
| `chunk_overlap` | integer | Overlap tokens |

#### Example Usage

```python
result = await client.call_tool("get_stats", {})

stats = result['statistics']
print(f"Documents: {stats['total_documents']}")
print(f"Chunks: {stats['total_chunks']}")
print(f"Model: {stats['embedding_model']}")
print(f"Storage: {stats['storage_path']}")
```

---

## Error Codes Reference

| Error Code | Description | Applicable Tools |
|------------|-------------|------------------|
| `file_not_found` | File does not exist | ingest_document |
| `invalid_file_type` | Wrong file format | ingest_document |
| `file_too_large` | File exceeds size limit | ingest_document |
| `encoding_error` | Failed to read file | ingest_document |
| `empty_query` | Query string is empty | query_documents |
| `empty_collection` | No documents ingested | query_documents, list_documents |
| `document_not_found` | Document ID not found | delete_document |
| `invalid_document_id` | Malformed document ID | delete_document |
| `tool_execution_failed` | Unexpected error | All tools |

## Best Practices

### File Ingestion
- Use absolute paths for reliability
- Verify file exists before calling tool
- Handle encoding errors gracefully
- Monitor chunk count for very large files

### Querying
- Start with default parameters
- Adjust `min_similarity` based on results
- Use higher thresholds (0.5+) for precision
- Use lower thresholds (0.1-0.3) for recall

### Document Management
- List documents before deletion
- Store document IDs for future reference
- Implement confirmation for deletions
- Monitor storage usage with get_stats

### Error Handling
- Always check `status` field
- Log error codes for debugging
- Provide user-friendly error messages
- Retry on transient failures

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider:
- Throttling ingestion for large batches
- Queuing concurrent requests
- Monitoring resource usage

## Future API Enhancements

Planned additions:
- Batch document ingestion
- Metadata filtering in queries
- Document update (re-ingestion)
- Collection management
- Export/import functionality

## Related Documentation

- [User Guide](user-guide.md) - Practical usage examples
- [Architecture](architecture.md) - Implementation details
- [Testing Guide](TESTING.md) - Testing the API

---

Last Updated: 2025-11-09
