# Installation Guide

Complete guide to installing and configuring OpenRAG as an MCP server in Claude Code.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [External Dependencies](#external-dependencies)
- [Installing OpenRAG](#installing-openrag)
- [MCP Server Configuration](#mcp-server-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required

- **Python 3.12** (NOT 3.13 - PyTorch compatibility requirement)
- **Conda** (Anaconda or Miniconda) - [Download here](https://docs.conda.io/en/latest/miniconda.html)
- **10+ GB free disk space** (for dependencies and embedding models)
- **Internet connection** (for initial setup and model downloads)

### Optional (for enhanced RAG features)

- **Ollama** - Required for Contextual RAG and Graph RAG
- **Neo4j** - Required for Graph RAG only

## Environment Setup

### Option 1: Automated Setup (Recommended)

\`\`\`bash
# Clone or navigate to OpenRAG directory
cd /path/to/OpenRAG

# Run automated setup script
./setup_environment.sh

# Follow prompts - this will:
# - Create conda environment "OpenRAG" with Python 3.12
# - Install all dependencies
# - Install development tools (optional)
# - Set up the package in editable mode
# - Run verification tests (optional)
\`\`\`

**Time required**: ~10-15 minutes (depending on internet speed)

### Option 2: Manual Setup

\`\`\`bash
# 1. Create conda environment with Python 3.12
conda create -n OpenRAG python=3.12 -y

# 2. Activate environment
conda activate OpenRAG

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install development dependencies (optional, for testing/development)
pip install -r requirements-dev.txt

# 5. Install OpenRAG in editable mode
pip install -e .

# 6. Create configuration file
cp .env.example .env
\`\`\`

### Verify Python Installation

\`\`\`bash
# Activate environment
conda activate OpenRAG

# Check Python version (must be 3.12)
python --version

# Verify core imports
python -c "import chromadb, sentence_transformers, tiktoken, mcp; print('✅ All core dependencies installed')"
\`\`\`

## External Dependencies

OpenRAG supports three RAG strategies, each with different dependencies:

### Traditional RAG (No External Dependencies)

Works out of the box with just the Python environment. Uses:
- ChromaDB for vector storage (embedded SQLite)
- sentence-transformers for embeddings (local models)

**No additional setup required!**

### Contextual RAG (Requires Ollama)

Adds document-level context to each chunk using Ollama LLM.

**Setup**:

\`\`\`bash
# Install Ollama
brew install ollama  # macOS
# For Linux/Windows: https://ollama.ai/download

# Start Ollama service
ollama serve  # Run in separate terminal, or use system service

# Pull the required model
ollama pull llama3.2:3b

# Verify Ollama is running
curl http://localhost:11434/api/tags
\`\`\`

**Configure** in \`.env\`:
\`\`\`bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CONTEXT_MODEL=llama3.2:3b
\`\`\`

### Graph RAG (Requires Ollama + Neo4j)

Extracts entities and relationships for graph-based reasoning.

**Setup Ollama** (see above), then install Neo4j:

\`\`\`bash
# Install Neo4j
brew install neo4j  # macOS
# For Linux/Windows: https://neo4j.com/download/

# Start Neo4j service
brew services start neo4j  # macOS (runs in background)
# OR
neo4j start  # Manual start

# Set initial password
# Navigate to http://localhost:7474 in browser
# Login with username: neo4j, password: neo4j
# You'll be prompted to change the password
# Remember this password for configuration!
\`\`\`

**Configure** in \`.env\`:
\`\`\`bash
GRAPH_ENABLED=true
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
GRAPH_ENTITY_MODEL=llama3.2:3b
GRAPH_MAX_HOPS=2
\`\`\`

**Verify Neo4j**:
\`\`\`bash
# Check Neo4j is running
brew services list | grep neo4j  # macOS
# OR
curl http://localhost:7474

# Test connection (in Neo4j browser at http://localhost:7474)
# Run Cypher query: MATCH (n) RETURN count(n)
\`\`\`

## Installing OpenRAG

After environment setup, configure OpenRAG settings:

### Configuration

Edit \`.env\` file (created from \`.env.example\`):

\`\`\`bash
# Core Settings (Required)
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=60
LOG_LEVEL=INFO

# Contextual RAG (Optional - requires Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CONTEXT_MODEL=llama3.2:3b

# Graph RAG (Optional - requires Ollama + Neo4j)
GRAPH_ENABLED=true
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GRAPH_ENTITY_MODEL=llama3.2:3b
GRAPH_MAX_HOPS=2
\`\`\`

### Embedding Model Options

Choose based on your needs:

| Model | Size | Speed | Quality | Dimensions | Best For |
|-------|------|-------|---------|------------|----------|
| \`all-MiniLM-L6-v2\` | 80 MB | Fast | Good | 384 | Development, testing |
| \`all-mpnet-base-v2\` | 420 MB | Moderate | Best | 768 | Production (default) |

Set in \`.env\`:
\`\`\`bash
EMBEDDING_MODEL=all-mpnet-base-v2  # Recommended
# OR
EMBEDDING_MODEL=all-MiniLM-L6-v2   # Faster, smaller
\`\`\`

## MCP Server Configuration

### For Claude Code (Recommended)

Claude Code uses MCP servers configured in \`~/.claude/config.json\`.

#### Step 1: Find Your Python Path

\`\`\`bash
# Activate OpenRAG environment
conda activate OpenRAG

# Get Python path
which python

# Example output: /opt/anaconda3/envs/OpenRAG/bin/python
# Copy this path for the next step
\`\`\`

#### Step 2: Configure MCP Server

Edit or create \`~/.claude/config.json\`:

\`\`\`json
{
  "mcpServers": {
    "openrag": {
      "command": "/opt/anaconda3/envs/OpenRAG/bin/python",
      "args": ["-m", "openrag.server"],
      "env": {
        "CHROMA_DB_PATH": "/absolute/path/to/your/chroma_db",
        "EMBEDDING_MODEL": "all-mpnet-base-v2",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
\`\`\`

**Important**:
- Use the **absolute path** to Python from Step 1
- Use **absolute paths** for \`CHROMA_DB_PATH\` (not relative paths like \`./chroma_db\`)
- Do NOT use \`conda run\` - it doesn't work in non-interactive shells

#### Step 3: Advanced Configuration Options

**Project-Specific Database**:

To use different databases for different projects:

\`\`\`json
{
  "mcpServers": {
    "openrag": {
      "command": "/opt/anaconda3/envs/OpenRAG/bin/python",
      "args": ["-m", "openrag.server"],
      "cwd": "/path/to/your/project",
      "env": {
        "CHROMA_DB_PATH": "./data/chroma_db",
        "EMBEDDING_MODEL": "all-mpnet-base-v2"
      }
    }
  }
}
\`\`\`

With \`cwd\` set, relative paths in \`env\` are resolved from that directory.

**Enable Contextual RAG**:

\`\`\`json
{
  "mcpServers": {
    "openrag": {
      "command": "/opt/anaconda3/envs/OpenRAG/bin/python",
      "args": ["-m", "openrag.server"],
      "env": {
        "CHROMA_DB_PATH": "/absolute/path/to/chroma_db",
        "EMBEDDING_MODEL": "all-mpnet-base-v2",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_CONTEXT_MODEL": "llama3.2:3b"
      }
    }
  }
}
\`\`\`

**Enable Graph RAG**:

\`\`\`json
{
  "mcpServers": {
    "openrag": {
      "command": "/opt/anaconda3/envs/OpenRAG/bin/python",
      "args": ["-m", "openrag.server"],
      "env": {
        "CHROMA_DB_PATH": "/absolute/path/to/chroma_db",
        "EMBEDDING_MODEL": "all-mpnet-base-v2",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_CONTEXT_MODEL": "llama3.2:3b",
        "GRAPH_ENABLED": "true",
        "NEO4J_URI": "neo4j://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your_password",
        "GRAPH_ENTITY_MODEL": "llama3.2:3b",
        "GRAPH_MAX_HOPS": "2"
      }
    }
  }
}
\`\`\`

### For Claude Desktop App

Edit Claude Desktop configuration:

**macOS**: \`~/Library/Application Support/Claude/claude_desktop_config.json\`

**Windows**: \`%APPDATA%\Claude\claude_desktop_config.json\`

**Linux**: \`~/.config/Claude/claude_desktop_config.json\`

Add OpenRAG server (same format as Claude Code above):

\`\`\`json
{
  "mcpServers": {
    "openrag": {
      "command": "/opt/anaconda3/envs/OpenRAG/bin/python",
      "args": ["-m", "openrag.server"],
      "env": {
        "CHROMA_DB_PATH": "/absolute/path/to/chroma_db",
        "EMBEDDING_MODEL": "all-mpnet-base-v2"
      }
    }
  }
}
\`\`\`

**Restart Claude Desktop** after making changes.

## Verification

### Test 1: Quick Test Script

\`\`\`bash
# Activate environment
conda activate OpenRAG

# Run quick test for Traditional RAG
python quick_test_normalRAG.py

# Expected: All tests pass, document ingested and queried successfully
\`\`\`

### Test 2: Contextual RAG Test (if Ollama installed)

\`\`\`bash
# Ensure Ollama is running
ollama list  # Should show llama3.2:3b

# Run Contextual RAG test
python quick_test_contextualRAG.py

# Expected: Document ingested with contextual processing
\`\`\`

### Test 3: Graph RAG Test (if Ollama + Neo4j installed)

\`\`\`bash
# Ensure both Ollama and Neo4j are running
ollama list
brew services list | grep neo4j  # macOS

# Run Graph RAG test (takes 3-5 minutes for entity extraction)
python quick_test_graphRAG.py

# Expected: Entities and relationships extracted and stored in Neo4j
\`\`\`

### Test 4: MCP Server Test

\`\`\`bash
# Start server manually to check for errors
conda activate OpenRAG
python -m openrag.server

# Expected output:
# ================================================================================
# OpenRAG MCP Server Starting
# ================================================================================
# Initializing components...
# Loading embedding model: all-mpnet-base-v2
# ... (initialization logs)
# MCP server ready and listening for requests
\`\`\`

Press \`Ctrl+C\` to stop the server.

### Test 5: Claude Code Integration

In Claude Code, try:

\`\`\`
List available MCP tools
\`\`\`

Expected response should include OpenRAG tools:
- \`ingest_text\`
- \`query_documents\`
- \`list_documents\`
- \`delete_document\`
- \`get_stats\`

Test ingestion:

\`\`\`
Ingest this text into OpenRAG with name "test.txt":

OpenRAG is a multi-strategy RAG system supporting traditional,
contextual, and graph-based retrieval over personal documents.
\`\`\`

Test query:

\`\`\`
Query OpenRAG for "multi-strategy RAG"
\`\`\`

## Troubleshooting

### Issue: MCP server not appearing in Claude Code

**Check**:
1. Verify \`~/.claude/config.json\` syntax is valid JSON
2. Check Python path is absolute and correct: \`which python\` (in OpenRAG env)
3. Restart Claude Code completely
4. Check Claude Code logs for errors

**Debug**:
\`\`\`bash
# Test server startup manually
conda activate OpenRAG
python -m openrag.server

# Look for initialization errors
\`\`\`

### Issue: ImportError or module not found

**Solution**:
\`\`\`bash
# Ensure environment is activated
conda activate OpenRAG

# Reinstall package in editable mode
pip install -e .

# Verify installation
python -c "import openrag; print(openrag.__file__)"
\`\`\`

### Issue: Ollama connection failed

**Check**:
\`\`\`bash
# Verify Ollama is running
ollama list

# Test API
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
\`\`\`

**If using different Ollama URL**:
\`\`\`bash
# Update .env or MCP config
OLLAMA_BASE_URL=http://your-ollama-host:11434
\`\`\`

### Issue: Neo4j connection failed

**Check**:
\`\`\`bash
# Verify Neo4j is running
brew services list | grep neo4j  # macOS
neo4j status  # Linux/Windows

# Start Neo4j if stopped
brew services start neo4j  # macOS
neo4j start  # Linux/Windows

# Test connection in browser
open http://localhost:7474
\`\`\`

**Verify credentials**:
\`\`\`bash
# Ensure password in .env or MCP config matches Neo4j password
NEO4J_PASSWORD=your_actual_password
\`\`\`

### Issue: Embedding model download fails

**Solution**:
\`\`\`bash
# Pre-download model manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Check disk space
df -h

# Check internet connection
ping google.com
\`\`\`

### Issue: ChromaDB permission errors

**Solution**:
\`\`\`bash
# Fix permissions
chmod -R 755 ./chroma_db

# Or use different directory with write permissions
export CHROMA_DB_PATH=/tmp/chroma_db
\`\`\`

### Issue: Python 3.13 instead of 3.12

**Solution**:
\`\`\`bash
# Remove existing environment
conda env remove -n OpenRAG

# Recreate with Python 3.12 explicitly
conda create -n OpenRAG python=3.12 -y
conda activate OpenRAG

# Reinstall dependencies
pip install -r requirements.txt
pip install -e .
\`\`\`

### Issue: Background tasks not processing (Contextual/Graph RAG)

**Check**:
\`\`\`bash
# Verify Ollama is accessible
curl http://localhost:11434/api/tags

# Check server logs for background task errors
# Look for "Contextual processing started" and "completed" messages
\`\`\`

**Note**: Background processing for Contextual and Graph RAG can take time:
- Contextual RAG: ~30-60 seconds per document
- Graph RAG: 3-5 minutes per document (entity extraction is LLM-intensive)

### Issue: Graph traversal queries slow

**Optimize**:
\`\`\`bash
# Reduce max_hops in query (default: 2)
# In .env or MCP config:
GRAPH_MAX_HOPS=1  # Faster, less context

# Or specify in query:
# query_documents(query="...", rag_type="graph", max_hops=1)
\`\`\`

### Issue: Running out of disk space

**Check storage**:
\`\`\`bash
# Check ChromaDB size
du -sh ./chroma_db

# Check embedding model cache
du -sh ~/.cache/torch/sentence_transformers

# Check Neo4j database size
du -sh /opt/homebrew/var/neo4j  # macOS
\`\`\`

**Clean up**:
\`\`\`bash
# Delete specific documents using delete_document tool
# Or delete entire database to start fresh:
rm -rf ./chroma_db

# Clear Neo4j data (WARNING: deletes all graph data)
# In Neo4j browser: MATCH (n) DETACH DELETE n
\`\`\`

## Getting Help

- Review [Quick Start Guide](quick-start.md) for usage examples
- Check [Architecture Overview](architecture.md) for system design
- See [CLAUDE.md](../CLAUDE.md) for development conventions
- Review [Lab Journal](lab_journal.md) for research notes

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quick-start.md) to learn basic usage
2. Try ingesting your first document
3. Explore different RAG strategies (traditional, contextual, graph)
4. Experiment with configuration options

---

**Last Updated**: 2026-03-06
