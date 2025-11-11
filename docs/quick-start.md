# Quick Start Guide

Get OpenRAG up and running in under 10 minutes.

## Prerequisites

- Anaconda or Miniconda installed
- 10+ GB free disk space
- Internet connection (for initial setup)

## Step 1: Installation (5 minutes)

Run the automated setup script:

```bash
cd /path/to/OpenRAG
./setup_environment.sh
```

This will:
- Create conda environment "OpenRAG" with Python 3.12
- Install all dependencies
- Set up the project

**Alternative - Manual Setup**:
```bash
conda create -n OpenRAG python=3.12 -y
conda activate OpenRAG
pip install -r requirements.txt
```

## Step 2: Configuration (1 minute)

Create your configuration file:

```bash
cp .env.example .env
```

Default configuration works out of the box. Optionally edit `.env`:

```bash
# .env file
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-mpnet-base-v2    # Best quality
CHUNK_SIZE=400
CHUNK_OVERLAP=60
LOG_LEVEL=INFO
```

## Step 3: Quick Test (2 minutes)

Verify everything works:

```bash
conda activate OpenRAG
python quick_test.py
```

Expected output:
```
================================================================================
OPENRAG QUICK TEST
================================================================================

📦 Test 1: Testing imports...
✅ All imports successful

⚙️  Test 2: Testing configuration...
✅ Settings loaded

🔪 Test 3: Testing text chunker...
✅ Chunker working

🧮 Test 4: Testing embedding model...
✅ Embedding model loaded

💾 Test 5: Testing vector store...
✅ Vector store initialized

🔄 Test 6: Testing async tools...
✅ Async tools working

================================================================================
✅ ALL TESTS PASSED!
================================================================================
```

## Step 4: Your First Document (3 minutes)

### Create a test document

```bash
cat > my_first_doc.txt << 'EOF'
# Introduction to Artificial Intelligence

Artificial Intelligence (AI) is the simulation of human intelligence
processes by machines, especially computer systems. These processes
include learning, reasoning, and self-correction.

## Machine Learning

Machine learning is a subset of AI that provides systems the ability
to automatically learn and improve from experience without being
explicitly programmed.

## Applications

AI is being used in various fields including:
- Healthcare for diagnosis and treatment
- Finance for fraud detection
- Autonomous vehicles
- Natural language processing
EOF
```

### Ingest the document

Start Python and run:

```python
import asyncio
from pathlib import Path
from src.openrag.core.chunker import TextChunker
from src.openrag.core.embedder import EmbeddingModel
from src.openrag.core.vector_store import VectorStore
from src.openrag.tools.ingest import ingest_document_tool

async def ingest():
    # Initialize components
    chunker = TextChunker(chunk_size=400, chunk_overlap=60)
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(
        persist_directory=Path("./chroma_db"),
        embedding_model=embedding_model
    )

    # Ingest document
    result = await ingest_document_tool(
        file_path="my_first_doc.txt",
        vector_store=vector_store,
        chunker=chunker
    )

    print(f"✅ Document ingested!")
    print(f"   Document ID: {result['document_id']}")
    print(f"   Chunks: {result['chunk_count']}")
    return result

# Run
result = asyncio.run(ingest())
```

### Query your document

```python
from src.openrag.tools.query import query_documents_tool

async def query():
    # Initialize components (same as above)
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(
        persist_directory=Path("./chroma_db"),
        embedding_model=embedding_model
    )

    # Search
    result = await query_documents_tool(
        query="What is machine learning?",
        vector_store=vector_store,
        max_results=3
    )

    print(f"\n🔍 Query: What is machine learning?")
    print(f"   Found {result['total_results']} results:\n")

    for i, res in enumerate(result['results'], 1):
        print(f"{i}. Score: {res['similarity_score']:.3f}")
        print(f"   {res['content'][:150]}...\n")

# Run
asyncio.run(query())
```

Expected output:
```
🔍 Query: What is machine learning?
   Found 2 results:

1. Score: 0.856
   Machine learning is a subset of AI that provides systems the ability
   to automatically learn and improve from experience without being
   explicitly programmed...

2. Score: 0.734
   Artificial Intelligence (AI) is the simulation of human intelligence
   processes by machines, especially computer systems...
```

## Step 5: Use with Claude Desktop (Optional)

### Add to Claude Desktop Configuration

Edit your Claude Desktop config:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add OpenRAG server:

```json
{
  "mcpServers": {
    "openrag": {
      "command": "/path/to/anaconda3/envs/OpenRAG/bin/python",
      "args": [
        "-m",
        "openrag.server"
      ],
      "env": {
        "CHROMA_DB_PATH": "/path/to/OpenRAG/chroma_db",
        "EMBEDDING_MODEL": "all-mpnet-base-v2",
        "CHUNK_SIZE": "400",
        "CHUNK_OVERLAP": "60",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Get your Python path**:
```bash
conda activate OpenRAG
which python
```

### Restart Claude Desktop

Restart Claude Desktop to load the MCP server.

### Use in Claude

In Claude, you can now:

```
Can you ingest this document: /Users/name/documents/notes.txt

Search my documents for information about neural networks

What documents do I have ingested?

Delete the document with ID xyz-123

Show me the system statistics
```

Claude will automatically use the OpenRAG tools!

## Common Commands

### Activate environment
```bash
conda activate OpenRAG
```

### Run tests
```bash
pytest tests/ -v
```

### Start MCP server manually
```bash
python -m openrag.server
```

### Check configuration
```bash
python -c "from openrag.config import get_settings; s = get_settings(); print(f'Model: {s.embedding_model}'); print(f'DB: {s.chroma_db_path}')"
```

### View stats
```python
from src.openrag.tools.stats import get_stats_tool
from src.openrag.config import get_settings
# ... initialize vector_store ...
result = await get_stats_tool(vector_store=vector_store, settings=get_settings())
print(result)
```

## Troubleshooting

### Import errors
```bash
# Ensure environment is activated
conda activate OpenRAG

# Reinstall in editable mode
pip install -e .
```

### Model download fails
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### ChromaDB permission error
```bash
# Fix permissions
chmod -R 755 ./chroma_db
```

### Claude Desktop not seeing server
1. Check Python path in config is correct
2. Check environment variables are set
3. Restart Claude Desktop
4. Check logs in Claude Desktop → Help → Show Logs

## Next Steps

Now that you're up and running:

1. **Read the [User Guide](user-guide.md)** - Learn all features
2. **Check the [API Reference](api-reference.md)** - Understand the tools
3. **Explore [Configuration](configuration.md)** - Optimize for your needs
4. **Review [Testing Guide](TESTING.md)** - Advanced testing

## Tips for Success

### Performance
- Use `all-MiniLM-L6-v2` for faster processing (development)
- Use `all-mpnet-base-v2` for better quality (production)
- Adjust chunk size based on your document type

### Best Practices
- Always use absolute file paths
- Ingest related documents together
- Use descriptive queries for better results
- Monitor disk space as collection grows

### Optimization
- Batch ingest multiple documents
- Use higher similarity thresholds for precision
- Use lower thresholds for broad exploration
- Periodically clean up unused documents

## Getting Help

- **Installation issues**: See [INSTALLATION.md](INSTALLATION.md)
- **Usage questions**: See [User Guide](user-guide.md)
- **Errors**: See [Troubleshooting](troubleshooting.md)
- **API details**: See [API Reference](api-reference.md)

---

**Congratulations!** You now have a working RAG system for your personal documents.

Last Updated: 2025-11-09
