# Configuration Reference

Complete reference for all OpenRAG configuration options.

## Overview

OpenRAG uses environment variables for configuration, managed through a `.env` file and Pydantic Settings.

### Configuration File

Create `.env` in the project root:

```bash
cp .env.example .env
```

Edit `.env` to customize settings.

## Configuration Options

### ChromaDB Settings

#### CHROMA_DB_PATH

Path to ChromaDB persistent storage directory.

**Type**: String (file path)
**Default**: `./chroma_db`
**Environment Variable**: `CHROMA_DB_PATH`

**Description**:
- Location where vector database files are stored
- Directory created automatically if doesn't exist
- Can be relative or absolute path
- Contains SQLite database and vector data

**Examples**:
```bash
# Relative path (default)
CHROMA_DB_PATH=./chroma_db

# Absolute path
CHROMA_DB_PATH=/Users/name/data/openrag_db

# Different location
CHROMA_DB_PATH=/mnt/storage/chroma_db
```

**Considerations**:
- Ensure sufficient disk space (plan for ~4 KB per chunk)
- Use SSD for better performance
- Include in backup strategy
- Don't use network drives (performance issues)

---

### Embedding Model Settings

#### EMBEDDING_MODEL

Name of the sentence-transformer model to use for embeddings.

**Type**: String (model name)
**Default**: `all-mpnet-base-v2`
**Environment Variable**: `EMBEDDING_MODEL`

**Description**:
- Determines quality and speed of embeddings
- Model downloaded on first use
- Affects retrieval accuracy and performance

**Available Models**:

| Model | Dimensions | Size | Speed | Quality | Use Case |
|-------|-----------|------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | 80 MB | Fast | Good | Development, testing |
| all-mpnet-base-v2 | 768 | 420 MB | Moderate | Best | Production (default) |
| instructor-xl | 768 | Large | Slow | Best | Research, multi-domain |

**Examples**:
```bash
# Default - best quality (recommended)
EMBEDDING_MODEL=all-mpnet-base-v2

# Fast model - good for development
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Advanced model - for research
EMBEDDING_MODEL=hkunlp/instructor-xl
```

**Performance Impact**:

| Model | Ingestion Speed | Query Speed | Memory |
|-------|----------------|-------------|--------|
| MiniLM | ~2-3 pages/sec | 50-100ms | ~500 MB |
| mpnet | ~0.5-1 page/sec | 100-200ms | ~800 MB |

**Changing Models**:

**Important**: Changing the embedding model requires re-ingesting all documents, as embeddings from different models are incompatible.

Process:
1. Backup existing data: `tar -czf chroma_backup.tar.gz ./chroma_db`
2. Change `EMBEDDING_MODEL` in `.env`
3. Delete or rename old database: `mv chroma_db chroma_db_old`
4. Re-ingest all documents with new model

---

### Chunking Settings

#### CHUNK_SIZE

Maximum number of tokens per chunk.

**Type**: Integer
**Default**: `400`
**Range**: 100-2000
**Environment Variable**: `CHUNK_SIZE`

**Description**:
- Controls size of text segments for embedding
- Larger chunks = more context per result
- Smaller chunks = more precise retrieval
- Affects storage and performance

**Recommendations by Use Case**:

| Use Case | Chunk Size | Rationale |
|----------|-----------|-----------|
| Fact extraction | 200-300 | Precise, focused results |
| General purpose | 400-512 | Balanced (default) |
| Long-form context | 600-800 | More comprehensive |
| Academic papers | 300-400 | Preserves paragraph structure |
| Code documentation | 200-300 | Function-level granularity |

**Examples**:
```bash
# Small chunks - precise retrieval
CHUNK_SIZE=200

# Default - balanced
CHUNK_SIZE=400

# Large chunks - more context
CHUNK_SIZE=600
```

**Storage Impact**:

With 1000 chunks:
- 200 tokens: Smaller but more chunks (~500 more) = ~2 MB extra
- 400 tokens: Baseline = ~4 MB
- 600 tokens: Larger but fewer chunks (~300 fewer) = ~3 MB

**Performance Impact**:
- Larger chunks = Slower embedding (more tokens to process)
- Smaller chunks = More chunks to search (slightly slower queries)
- Optimal range: 300-600 tokens

**Validation**:
- Must be >= 100
- Must be <= 2000
- Must be > `CHUNK_OVERLAP`

---

#### CHUNK_OVERLAP

Number of tokens that overlap between consecutive chunks.

**Type**: Integer
**Default**: `60` (15% of default chunk size)
**Range**: 0-500
**Environment Variable**: `CHUNK_OVERLAP`

**Description**:
- Prevents information loss at chunk boundaries
- Ensures complete thoughts captured
- Creates redundancy in storage

**Industry Standard**:
- **10-20% of chunk size** (recommended)
- For 400-token chunks: 40-80 token overlap

**Examples**:
```bash
# Minimal overlap (5%)
CHUNK_OVERLAP=20

# Default overlap (15%)
CHUNK_OVERLAP=60

# High overlap (25%)
CHUNK_OVERLAP=100
```

**Overlap Percentage Table**:

| Chunk Size | 10% Overlap | 15% Overlap | 20% Overlap |
|-----------|-------------|-------------|-------------|
| 200 | 20 | 30 | 40 |
| 400 | 40 | 60 | 80 |
| 600 | 60 | 90 | 120 |

**Trade-offs**:

**More Overlap**:
- ✅ Better information preservation
- ✅ Less risk of split concepts
- ❌ More storage required
- ❌ Potential duplicate results

**Less Overlap**:
- ✅ Less storage required
- ✅ Fewer duplicate results
- ❌ Risk of losing context at boundaries
- ❌ May split important information

**Validation**:
- Must be >= 0
- Must be < `CHUNK_SIZE`
- Recommended: 10-20% of `CHUNK_SIZE`

---

### Logging Settings

#### LOG_LEVEL

Logging verbosity level.

**Type**: String (enum)
**Default**: `INFO`
**Valid Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
**Environment Variable**: `LOG_LEVEL`

**Description**:
- Controls how much information is logged to stderr
- All logs go to stderr (MCP requirement)
- Higher levels = less output

**Log Levels**:

| Level | When to Use | What You See |
|-------|-------------|--------------|
| DEBUG | Development, troubleshooting | All operations, variable values |
| INFO | Normal operation | Key events, document operations |
| WARNING | Production (default) | Warnings and errors only |
| ERROR | Minimal logging | Errors only |
| CRITICAL | Critical issues only | Fatal errors only |

**Examples**:
```bash
# Development - see everything
LOG_LEVEL=DEBUG

# Production - normal operation (recommended)
LOG_LEVEL=INFO

# Production - minimal output
LOG_LEVEL=WARNING
```

**Example Output by Level**:

**DEBUG**:
```
DEBUG - Initializing TextChunker with size=400, overlap=60
DEBUG - Reading file: /path/to/doc.txt
DEBUG - File size: 15234 bytes
DEBUG - Creating 12 chunks
DEBUG - Embedding chunk 1/12
...
```

**INFO**:
```
INFO - Document ingested: doc.txt (12 chunks)
INFO - Query executed: "machine learning" (3 results)
```

**WARNING**:
```
WARNING - Low similarity scores for query
ERROR - Failed to read file: permission denied
```

**Performance Impact**:
- DEBUG: Slight performance overhead
- INFO: Negligible impact
- WARNING+: No impact

---

## Configuration Profiles

### Development Profile

Fast iteration, verbose logging:

```bash
CHROMA_DB_PATH=./chroma_db_dev
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=300
CHUNK_OVERLAP=45
LOG_LEVEL=DEBUG
```

### Production Profile

Best quality, standard settings:

```bash
CHROMA_DB_PATH=/var/data/openrag/chroma_db
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=60
LOG_LEVEL=INFO
```

### Research Profile

High quality, larger context:

```bash
CHROMA_DB_PATH=/research/data/chroma_db
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=100
LOG_LEVEL=INFO
```

### Resource-Constrained Profile

Minimal resources:

```bash
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=250
CHUNK_OVERLAP=25
LOG_LEVEL=WARNING
```

---

## Configuration Validation

Settings are validated on startup. Invalid configurations raise errors:

```python
# Invalid chunk overlap
CHUNK_SIZE=400
CHUNK_OVERLAP=500  # ERROR: overlap must be < chunk_size

# Invalid log level
LOG_LEVEL=VERBOSE  # ERROR: must be DEBUG, INFO, WARNING, ERROR, or CRITICAL

# Invalid chunk size
CHUNK_SIZE=50  # ERROR: chunk_size must be >= 100
CHUNK_SIZE=3000  # ERROR: chunk_size must be <= 2000
```

## Runtime Configuration

### Reading Configuration

```python
from openrag.config import get_settings

settings = get_settings()
print(f"DB Path: {settings.chroma_db_path}")
print(f"Model: {settings.embedding_model}")
print(f"Chunk size: {settings.chunk_size}")
```

### Configuration Caching

Settings are cached (singleton pattern):
- Loaded once on first access
- Changes to `.env` require restart
- Efficient - no repeated file reads

---

## Environment-Specific Configuration

### Using Multiple Environments

```bash
# Development
cp .env.development .env

# Production
cp .env.production .env

# Testing
cp .env.test .env
```

### Docker Configuration

```dockerfile
# In Dockerfile
ENV CHROMA_DB_PATH=/data/chroma_db
ENV EMBEDDING_MODEL=all-mpnet-base-v2
ENV CHUNK_SIZE=400
ENV CHUNK_OVERLAP=60
ENV LOG_LEVEL=INFO
```

### Cloud Deployment

```yaml
# Example: Kubernetes ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: openrag-config
data:
  CHROMA_DB_PATH: "/mnt/data/chroma_db"
  EMBEDDING_MODEL: "all-mpnet-base-v2"
  CHUNK_SIZE: "400"
  CHUNK_OVERLAP: "60"
  LOG_LEVEL: "INFO"
```

---

## Performance Tuning

### For Speed

```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fastest model
CHUNK_SIZE=300                     # Smaller chunks
CHUNK_OVERLAP=30                   # Less overlap
```

### For Quality

```bash
EMBEDDING_MODEL=all-mpnet-base-v2  # Best quality
CHUNK_SIZE=400                      # Optimal size
CHUNK_OVERLAP=60                    # Standard overlap
```

### For Storage Efficiency

```bash
CHUNK_SIZE=500                      # Fewer chunks
CHUNK_OVERLAP=50                    # Minimal overlap (10%)
```

### For Precision

```bash
CHUNK_SIZE=200                      # Small, focused chunks
CHUNK_OVERLAP=40                    # Good overlap (20%)
```

---

## Troubleshooting Configuration

### Verify Current Configuration

```bash
python -c "from openrag.config import get_settings; s = get_settings(); print(f'Model: {s.embedding_model}'); print(f'Chunk size: {s.chunk_size}'); print(f'DB path: {s.chroma_db_path}')"
```

### Common Issues

**Settings not loading**:
```bash
# Ensure .env exists in project root
ls -la .env

# Check for syntax errors
cat .env
```

**Invalid values**:
```bash
# Check validation errors in logs
python -m openrag.server
```

**Changes not taking effect**:
- Restart server after changing `.env`
- Clear cached settings (restart Python)

---

## Related Documentation

- [User Guide](user-guide.md) - Using configured settings
- [Architecture](architecture.md) - How configuration is used
- [Performance Guide](troubleshooting.md) - Optimization tips

---

Last Updated: 2025-11-09
