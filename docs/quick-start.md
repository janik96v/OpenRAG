# Quick Start Guide

Get OpenRAG up and running in Claude Code in under 15 minutes.

## Overview

This guide will help you:
1. Install OpenRAG and dependencies
2. Configure it as an MCP server in Claude Code
3. Test basic functionality with all three RAG strategies

## Prerequisites

- Conda installed
- Claude Code installed
- 10+ GB free disk space

## Step 1: Install OpenRAG (5 minutes)

\`\`\`bash
# Navigate to OpenRAG directory
cd /path/to/OpenRAG

# Run automated setup
./setup_environment.sh

# When prompted:
# - Install dev dependencies: y (recommended)
# - Run verification: y (recommended)

# This creates conda environment "OpenRAG" with Python 3.12
# and installs all required packages
\`\`\`

**Verify**:
\`\`\`bash
conda activate OpenRAG
python --version  # Should show Python 3.12.x
\`\`\`

## Step 2: Configure MCP Server (3 minutes)

### Find Python Path

\`\`\`bash
conda activate OpenRAG
which python
# Copy the output (e.g., /opt/anaconda3/envs/OpenRAG/bin/python)
\`\`\`

### Configure Claude Code

Claude Code stores MCP servers **per-project** in `~/.claude.json`. The easiest way to add OpenRAG is via the CLI:

\`\`\`bash
claude mcp add openrag \
  --scope project \
  -- /opt/anaconda3/envs/OpenRAG/bin/python -m openrag.server
\`\`\`

Replace `/opt/anaconda3/envs/OpenRAG/bin/python` with YOUR Python path from above.

**Alternatively**, you can edit `~/.claude.json` directly. Find your project entry under the `projects` key and add the `openrag` entry to its `mcpServers` object:

\`\`\`json
"mcpServers": {
  "openrag": {
    "type": "stdio",
    "command": "/opt/anaconda3/envs/OpenRAG/bin/python",
    "args": ["-m", "openrag.server"],
    "env": {
      "CHROMA_DB_PATH": "/path/to/OpenRAG/chroma_db",
      "EMBEDDING_MODEL": "all-mpnet-base-v2",
      "LOG_LEVEL": "INFO"
    }
  }
}
\`\`\`

**Important**:
- Replace `command` with YOUR Python path from above
- Set `CHROMA_DB_PATH` to an absolute path where you want to store data
- `~/.claude/config.json` does **not** exist — `~/.claude.json` is the correct file

### Restart Claude Code

Completely quit and restart Claude Code for changes to take effect.

## Step 3: Test Traditional RAG (2 minutes)

In Claude Code, try:

\`\`\`
List available MCP tools
\`\`\`

You should see 5 OpenRAG tools. Now test ingestion and query:

\`\`\`
Please ingest the following text with name "ai_intro.txt":

Artificial Intelligence (AI) is the simulation of human intelligence by machines.
Machine learning is a subset of AI that enables systems to learn from data.
Deep learning uses neural networks with multiple layers to process information.
Natural language processing allows machines to understand and generate human language.
\`\`\`

After successful ingestion, query:

\`\`\`
Query OpenRAG for "what is machine learning"
\`\`\`

You should get relevant results with similarity scores!

## Step 4: Enable Contextual RAG (Optional, 5 minutes)

Contextual RAG adds document-level context to improve accuracy.

### Install Ollama

\`\`\`bash
# Install Ollama
brew install ollama  # macOS

# Start Ollama
ollama serve  # Keep this running

# In a new terminal, pull model
ollama pull llama3.2:3b
\`\`\`

### Update MCP Config

Edit the `openrag` entry in `~/.claude.json` to add Ollama settings:

\`\`\`json
"openrag": {
  "type": "stdio",
  "command": "/opt/anaconda3/envs/OpenRAG/bin/python",
  "args": ["-m", "openrag.server"],
  "env": {
    "CHROMA_DB_PATH": "/path/to/OpenRAG/chroma_db",
    "EMBEDDING_MODEL": "all-mpnet-base-v2",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_CONTEXT_MODEL": "llama3.2:3b",
    "LOG_LEVEL": "INFO"
  }
}
\`\`\`

### Restart and Test

Restart Claude Code, then:

\`\`\`
Query OpenRAG using contextual RAG for "machine learning"
\`\`\`

Contextual RAG provides better accuracy for complex queries by including document-level context.

## Step 5: Enable Graph RAG (Optional, 10 minutes)

Graph RAG extracts entities and relationships for advanced reasoning.

### Install Neo4j

\`\`\`bash
# Install Neo4j
brew install neo4j  # macOS

# Start Neo4j
brew services start neo4j

# Open browser to set password
open http://localhost:7474
# Login: neo4j / neo4j
# Set new password (remember it!)
\`\`\`

### Update MCP Config

Edit the `openrag` entry in `~/.claude.json` to add Graph RAG settings:

\`\`\`json
"openrag": {
  "type": "stdio",
  "command": "/opt/anaconda3/envs/OpenRAG/bin/python",
  "args": ["-m", "openrag.server"],
  "env": {
    "CHROMA_DB_PATH": "/path/to/OpenRAG/chroma_db",
    "EMBEDDING_MODEL": "all-mpnet-base-v2",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_CONTEXT_MODEL": "llama3.2:3b",
    "GRAPH_ENABLED": "true",
    "NEO4J_URI": "neo4j://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "your_password_here",
    "GRAPH_ENTITY_MODEL": "llama3.2:3b",
    "GRAPH_MAX_HOPS": "2",
    "LOG_LEVEL": "INFO"
  }
}
\`\`\`

### Restart and Test

Restart Claude Code, then ingest a document with entities:

\`\`\`
Ingest this text with name "research_team.txt":

Dr. Sarah Chen leads the AI research team at MIT.
She collaborates with Prof. James Wilson from Stanford University.
Their recent paper on neural networks was published in Nature.
The research was funded by the National Science Foundation.
\`\`\`

**Note**: Graph processing takes 3-5 minutes as it extracts entities and relationships using the LLM.

After processing completes, query with Graph RAG:

\`\`\`
Query OpenRAG using graph RAG for "who works at MIT"
\`\`\`

Graph RAG will traverse the knowledge graph to find entities and their relationships!

## Common Tasks

### Check System Status

\`\`\`
Get OpenRAG statistics
\`\`\`

Shows document count, chunk count, and configuration.

### List All Documents

\`\`\`
List all documents in OpenRAG
\`\`\`

### Delete a Document

\`\`\`
Delete document with ID [document_id] from OpenRAG
\`\`\`

### Query with Different RAG Types

\`\`\`
Query OpenRAG using traditional RAG for "query text"
Query OpenRAG using contextual RAG for "query text"
Query OpenRAG using graph RAG for "query text"
\`\`\`

### Control Graph Traversal Depth

\`\`\`
Query OpenRAG using graph RAG with max_hops 1 for "query text"
\`\`\`

Lower max_hops = faster, less context. Higher = slower, more context (range: 1-5).

## RAG Strategy Comparison

| Feature | Traditional | Contextual | Graph |
|---------|------------|------------|-------|
| **Speed** | Fastest | Fast | Moderate |
| **Setup** | None | Ollama | Ollama + Neo4j |
| **Best For** | Direct facts | Complex queries | Relationships, entities |
| **Processing Time** | Immediate | ~30-60 sec | 3-5 min |

**Recommendation**: Start with Traditional RAG, add Contextual for better accuracy, add Graph for relationship-based queries.

## Troubleshooting

### MCP Tools Not Appearing

1. Check `~/.claude.json` syntax is valid JSON (find your project entry, verify the `openrag` key exists)
2. Verify Python path with \`which python\` in OpenRAG environment
3. Restart Claude Code completely
4. Check Claude Code logs

### Test Manually

\`\`\`bash
conda activate OpenRAG
python -m openrag.server
# Should start without errors
# Press Ctrl+C to stop
\`\`\`

### Ollama Not Connecting

\`\`\`bash
# Check Ollama is running
ollama list

# Start if needed
ollama serve
\`\`\`

### Neo4j Not Connecting

\`\`\`bash
# Check Neo4j status
brew services list | grep neo4j

# Start if needed
brew services start neo4j

# Verify password in MCP config matches Neo4j password
\`\`\`

### Background Processing Not Working

Check server logs. Contextual and Graph RAG process in background:
- Contextual: ~30-60 seconds per document
- Graph: 3-5 minutes per document

You can continue using the system while processing happens.

## Next Steps

- Read [Installation Guide](installation.md) for detailed setup options
- Review [Architecture Overview](architecture.md) to understand the system
- Check [CLAUDE.md](../CLAUDE.md) for development information
- Explore [Lab Journal](lab_journal.md) for research notes

## Tips

1. **Start simple**: Use Traditional RAG first, add Contextual/Graph as needed
2. **Use absolute paths**: Always use absolute paths in MCP config
3. **Keep Ollama running**: For Contextual/Graph RAG, ensure \`ollama serve\` is running
4. **Monitor disk space**: ChromaDB and Neo4j grow with more documents
5. **Test incrementally**: Test each RAG type separately to isolate issues

---

**Last Updated**: 2026-03-06
