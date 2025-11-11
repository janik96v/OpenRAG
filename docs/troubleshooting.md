# Troubleshooting Guide

Solutions to common issues when using OpenRAG.

## Quick Diagnostics

Run these commands to identify issues:

```bash
# Check Python version
python --version  # Should be 3.10+

# Check environment
conda activate OpenRAG
which python

# Check installation
python -c "import openrag; print('OpenRAG installed')"

# Run quick test
python quick_test.py

# Check configuration
python -c "from openrag.config import get_settings; print(get_settings())"
```

## Installation Issues

### conda: command not found

**Problem**: Conda not in PATH

**Solution**:
```bash
# Find conda installation
ls /opt/anaconda3/bin/conda
ls ~/anaconda3/bin/conda
ls ~/miniconda3/bin/conda

# Add to PATH (for zsh)
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Add to PATH (for bash)
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
conda --version
```

### pip install fails

**Problem**: Network issues, missing dependencies

**Solutions**:

**Network timeout**:
```bash
# Increase timeout
pip install --timeout=300 -r requirements.txt

# Use different index
pip install --index-url https://pypi.org/simple -r requirements.txt
```

**Permission denied**:
```bash
# Ensure environment is activated
conda activate OpenRAG

# Don't use sudo with conda
pip install -r requirements.txt
```

**Dependency conflicts**:
```bash
# Start fresh
conda env remove -n OpenRAG
conda create -n OpenRAG python=3.13 -y
conda activate OpenRAG
pip install -r requirements.txt
```

### Module not found errors

**Problem**: Package not installed or wrong environment

**Solutions**:

**Verify environment**:
```bash
# Check active environment
conda info --envs
# Look for * next to OpenRAG

# Activate if needed
conda activate OpenRAG
```

**Reinstall package**:
```bash
pip install -e .
```

**Check PYTHONPATH**:
```bash
export PYTHONPATH=/path/to/OpenRAG:$PYTHONPATH
python -c "import openrag; print(openrag.__file__)"
```

## Runtime Issues

### MCP server won't start

**Problem**: Configuration or dependency issues

**Diagnostic**:
```bash
# Check configuration
python -c "from openrag.config import get_settings; print(get_settings())"

# Check imports
python -c "from openrag.server import create_server; print('OK')"

# Run with debug logging
LOG_LEVEL=DEBUG python -m openrag.server
```

**Solutions**:

**Missing .env**:
```bash
cp .env.example .env
```

**Invalid configuration**:
```bash
# Check .env syntax
cat .env

# Validate each setting
python -c "from openrag.config import Settings; s = Settings(); print('Valid')"
```

**Port already in use** (if using HTTP transport):
```bash
# Find process using port
lsof -i :PORT_NUMBER

# Kill process
kill -9 PID
```

### Embedding model download fails

**Problem**: Network issues, disk space, or permissions

**Solutions**:

**Check disk space**:
```bash
df -h
# Ensure 2+ GB free
```

**Manual download**:
```bash
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Downloaded successfully')
"
```

**Check model cache**:
```bash
# Default cache location
ls ~/.cache/torch/sentence_transformers/

# Set custom cache
export SENTENCE_TRANSFORMERS_HOME=/path/to/cache
```

**Network issues**:
```bash
# Use proxy if needed
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### ChromaDB errors

**Problem**: Database corruption, permissions, or locking

**Solutions**:

**Permission denied**:
```bash
# Fix permissions
chmod -R 755 ./chroma_db

# Or use different location
mkdir ~/openrag_data
export CHROMA_DB_PATH=~/openrag_data/chroma_db
```

**Database locked**:
```bash
# Kill any running processes
pkill -f openrag

# Remove lock file
rm -f ./chroma_db/chroma.sqlite3-wal
rm -f ./chroma_db/chroma.sqlite3-shm
```

**Corrupted database**:
```bash
# Restore from backup
tar -xzf chroma_backup_YYYYMMDD.tar.gz

# Or start fresh
mv chroma_db chroma_db_backup
mkdir chroma_db
```

## Document Ingestion Issues

### File not found

**Problem**: Incorrect path or permissions

**Solutions**:

**Use absolute path**:
```python
import os
file_path = os.path.abspath("my_document.txt")
print(f"Using path: {file_path}")
```

**Check file exists**:
```python
from pathlib import Path
path = Path("my_document.txt")
print(f"Exists: {path.exists()}")
print(f"Is file: {path.is_file()}")
```

**Check permissions**:
```bash
ls -l my_document.txt
# Should show read permissions
```

### Encoding errors

**Problem**: File not UTF-8 encoded

**Solutions**:

**Convert to UTF-8**:
```bash
# On macOS/Linux
iconv -f LATIN1 -t UTF-8 input.txt > output.txt

# Or using Python
python -c "
with open('input.txt', 'r', encoding='latin1') as f:
    content = f.read()
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(content)
"
```

**Force encoding**:
```python
# Modify ingestion code to handle different encodings
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
```

### File too large

**Problem**: Large files consume too much memory

**Solutions**:

**Split file**:
```bash
# Split into 10MB chunks
split -b 10M large_file.txt chunk_

# Ingest each chunk separately
for chunk in chunk_*; do
    # Ingest $chunk
done
```

**Streaming ingestion** (future feature):
```python
# Process file in chunks (not yet implemented)
# Will be added in future version
```

### Too many chunks

**Problem**: Document creates excessive chunks

**Solutions**:

**Increase chunk size**:
```bash
# In .env
CHUNK_SIZE=600  # Larger chunks = fewer total
```

**Split document**:
- Break large document into logical sections
- Ingest sections separately
- Better for organization and management

## Query Issues

### No results returned

**Problem**: Empty collection or query mismatch

**Diagnostic**:
```python
# Check if documents exist
from openrag.tools.manage import list_documents_tool
result = await list_documents_tool(vector_store)
print(f"Documents: {result['total_documents']}")

# Check stats
from openrag.tools.stats import get_stats_tool
stats = await get_stats_tool(vector_store, settings)
print(f"Chunks: {stats['statistics']['total_chunks']}")
```

**Solutions**:

**Lower similarity threshold**:
```python
result = await query_documents_tool(
    query="your query",
    vector_store=vector_store,
    min_similarity=0.0  # Show all results
)
```

**Increase max results**:
```python
result = await query_documents_tool(
    query="your query",
    vector_store=vector_store,
    max_results=50  # More results
)
```

**Rephrase query**:
```python
# Instead of:
"ML algorithms"

# Try:
"What are machine learning algorithms?"
```

### Poor quality results

**Problem**: Wrong model, chunk size, or query formulation

**Solutions**:

**Try different embedding model**:
```bash
# Switch to higher quality model
EMBEDDING_MODEL=all-mpnet-base-v2
```

**Adjust chunk size**:
```bash
# For more focused results
CHUNK_SIZE=200

# For more context
CHUNK_SIZE=600
```

**Improve query**:
```python
# Bad: "AI"
# Good: "What is artificial intelligence and how does it work?"

# Bad: "data"
# Good: "Explain the difference between structured and unstructured data"
```

**Re-ingest with new settings**:
```bash
# Backup current database
tar -czf chroma_backup.tar.gz ./chroma_db

# Change settings in .env
# Delete old database
rm -rf ./chroma_db

# Re-ingest documents
```

### Slow queries

**Problem**: Large collection or slow model

**Diagnostic**:
```python
import time

start = time.time()
result = await query_documents_tool(query="test", vector_store=vector_store)
elapsed = time.time() - start
print(f"Query took {elapsed:.2f} seconds")

# Check collection size
stats = await get_stats_tool(vector_store, settings)
print(f"Total chunks: {stats['statistics']['total_chunks']}")
```

**Solutions**:

**Use faster model**:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**Reduce collection size**:
```python
# Delete unused documents
await delete_document_tool(document_id, vector_store)
```

**Limit results**:
```python
result = await query_documents_tool(
    query="test",
    vector_store=vector_store,
    max_results=3  # Fewer results
)
```

## Performance Issues

### High memory usage

**Problem**: Large model or collection in memory

**Solutions**:

**Use smaller model**:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Uses ~500 MB vs 800 MB
```

**Reduce chunk size**:
```bash
CHUNK_SIZE=300  # Fewer, smaller chunks
```

**Monitor memory**:
```bash
# Check memory usage
ps aux | grep python

# On macOS
top -o mem
```

### Slow ingestion

**Problem**: Large files or slow model

**Solutions**:

**Use faster model**:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2  # 5x faster than mpnet
```

**Batch processing**:
```python
# Process multiple small documents instead of one large one
for doc in small_docs:
    await ingest_document_tool(doc, vector_store, chunker)
```

**Expected speeds**:
- MiniLM: ~2-3 pages/second
- mpnet: ~0.5-1 pages/second

### Disk space issues

**Problem**: Database growing too large

**Diagnostic**:
```bash
# Check database size
du -sh ./chroma_db

# Check available space
df -h
```

**Solutions**:

**Delete unused documents**:
```python
# List all documents
docs = await list_documents_tool(vector_store)

# Delete old ones
for doc in docs['documents']:
    if should_delete(doc):
        await delete_document_tool(doc['document_id'], vector_store)
```

**Increase overlap**:
```bash
# Less overlap = less storage
CHUNK_OVERLAP=20  # 5% instead of 15%
```

**Larger chunks**:
```bash
# Fewer chunks = less storage
CHUNK_SIZE=600
```

## Claude Desktop Integration Issues

### MCP server not appearing

**Problem**: Configuration error or path issues

**Solutions**:

**Verify config location**:
```bash
# macOS
ls ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
dir %APPDATA%\Claude\claude_desktop_config.json
```

**Check Python path**:
```bash
conda activate OpenRAG
which python
# Use this exact path in config
```

**Validate JSON**:
```bash
# Check for syntax errors
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | python -m json.tool
```

**Example correct config**:
```json
{
  "mcpServers": {
    "openrag": {
      "command": "/Users/name/anaconda3/envs/OpenRAG/bin/python",
      "args": ["-m", "openrag.server"],
      "env": {
        "CHROMA_DB_PATH": "/Users/name/openrag/chroma_db",
        "EMBEDDING_MODEL": "all-mpnet-base-v2"
      }
    }
  }
}
```

### Server crashes in Claude

**Problem**: Runtime error or configuration issue

**Diagnostic**:
```bash
# Check Claude logs
# macOS: Claude Desktop → Help → Show Logs
# Look for openrag errors
```

**Solutions**:

**Test server standalone**:
```bash
conda activate OpenRAG
python -m openrag.server
# Should start without errors
# Press Ctrl+C to stop
```

**Check environment variables**:
```json
{
  "env": {
    "CHROMA_DB_PATH": "/absolute/path/to/chroma_db",
    "EMBEDDING_MODEL": "all-mpnet-base-v2",
    "LOG_LEVEL": "DEBUG"
  }
}
```

**Restart Claude Desktop**:
- Quit Claude completely
- Restart
- Wait for MCP server to initialize

## Debugging Tips

### Enable debug logging

```bash
# In .env
LOG_LEVEL=DEBUG

# Or in Claude Desktop config
{
  "env": {
    "LOG_LEVEL": "DEBUG"
  }
}
```

### Check server logs

```bash
# Run server manually to see all output
python -m openrag.server
```

### Test components individually

```python
# Test chunker
from openrag.core.chunker import TextChunker
chunker = TextChunker()
chunks = chunker.chunk_text("Test text here")
print(f"Chunks: {len(chunks)}")

# Test embedder
from openrag.core.embedder import EmbeddingModel
model = EmbeddingModel("all-MiniLM-L6-v2")
embeddings = model.embed(["Test"])
print(f"Dimensions: {len(embeddings[0])}")

# Test vector store
from openrag.core.vector_store import VectorStore
from pathlib import Path
store = VectorStore(Path("./test_db"), model)
print("Vector store initialized")
```

## Getting Help

If you're still stuck:

1. **Check existing documentation**:
   - [User Guide](user-guide.md)
   - [API Reference](api-reference.md)
   - [Configuration Reference](configuration.md)

2. **Review logs**:
   - Enable DEBUG logging
   - Check for error messages
   - Note any stack traces

3. **Minimal reproducible example**:
   - Isolate the issue
   - Create minimal test case
   - Document steps to reproduce

4. **Open an issue**:
   - Describe the problem
   - Include configuration
   - Attach relevant logs
   - Share reproducible example

---

Last Updated: 2025-11-09
