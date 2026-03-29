# Installation Guide

Install and configure OpenRAG as an MCP server using Docker — no Python, conda, or manual dependency management required.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [MCP Server Configuration](#mcp-server-configuration)
- [RAG Strategy Setup](#rag-strategy-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Docker Desktop** — [Download here](https://www.docker.com/products/docker-desktop/)
- **10+ GB free disk space** (image ~3–4 GB + models + data)
- **Ollama** on your host machine — required for Contextual RAG and Graph RAG

### Install Ollama (for Contextual/Graph RAG)

```bash
brew install ollama        # macOS
# Linux/Windows: https://ollama.ai/download

ollama serve               # Start the service
ollama pull llama3.2:3b    # Pull the required model
```

> **Traditional RAG only?** Ollama is not required. Skip this step and set `CONTEXTUAL_ENABLED=false` and `GRAPH_ENABLED=false` in your `.env`.

---

## Quick Start

### Option A: Docker Compose (Recommended — includes Neo4j for Graph RAG)

```bash
# 1. Clone the repository
git clone https://github.com/janik96v/OpenRAG.git
cd OpenRAG

# 2. Create your configuration file
cp .env.docker.example .env
# Edit .env if needed (change NEO4J_PASSWORD etc.)

# 3. Build and start
docker compose up -d
```

The MCP server and Neo4j start automatically. Neo4j data and ChromaDB data are persisted in Docker volumes and a local `./chroma_db` folder.

### Option B: Standalone Container (Traditional RAG only — no Neo4j)

```bash
docker run --rm -i \
  -v ./chroma_db:/app/chroma_db \
  -e GRAPH_ENABLED=false \
  -e CONTEXTUAL_ENABLED=false \
  janik96v/openrag
```

> Pull from Docker Hub once the image is published — see [Publishing to Docker Hub](#publishing-to-docker-hub).

---

## Configuration

Copy `.env.docker.example` to `.env` and adjust:

```env
# RAG Types — disable what you don't need
TRADITIONAL_ENABLED=true
CONTEXTUAL_ENABLED=true    # Requires Ollama on host
GRAPH_ENABLED=true         # Requires Ollama on host + Neo4j (included in compose)

# Ollama — runs on your host machine
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_CONTEXT_MODEL=llama3.2:3b

# Neo4j — managed by docker-compose (must match NEO4J_AUTH in compose)
NEO4J_PASSWORD=openrag

# Chunking (optional tuning)
CHUNK_SIZE=400
CHUNK_OVERLAP=60

LOG_LEVEL=INFO
```

### Embedding Model Options

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `all-mpnet-base-v2` | 420 MB | Moderate | Best | Production (default) |
| `all-MiniLM-L6-v2` | 80 MB | Fast | Good | Development, testing |

```env
EMBEDDING_MODEL=all-mpnet-base-v2
```

The model is downloaded on first startup and cached in a Docker volume — no re-download on restart.

---

## MCP Server Configuration

### Claude Code

Add to `~/.claude/config.json`:

```json
{
  "mcpServers": {
    "openrag": {
      "command": "docker",
      "args": [
        "compose",
        "-f", "/absolute/path/to/OpenRAG/docker-compose.yml",
        "run", "--rm", "-i", "openrag"
      ]
    }
  }
}
```

### Claude Desktop

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "openrag": {
      "command": "docker",
      "args": [
        "compose",
        "-f", "/absolute/path/to/OpenRAG/docker-compose.yml",
        "run", "--rm", "-i", "openrag"
      ]
    }
  }
}
```

Restart Claude Desktop after saving.

> **Note**: The `-i` flag keeps stdin open, which is required for the MCP stdio protocol. `--rm` cleans up the container after each session.

---

## RAG Strategy Setup

### Traditional RAG

No extra services needed. Works immediately after `docker compose up`.

### Contextual RAG

Requires Ollama running on your host machine:

```bash
ollama serve
ollama pull llama3.2:3b
```

Ensure `CONTEXTUAL_ENABLED=true` and `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`.

### Graph RAG

Requires Ollama (above) + Neo4j (included in `docker-compose.yml` — no extra setup needed).

Set `GRAPH_ENABLED=true` in `.env`. The compose file handles the Neo4j connection automatically.

**Access the Neo4j browser** (for debugging/inspection):

```
http://localhost:7474
Username: neo4j
Password: <your NEO4J_PASSWORD from .env>
```

---

## Verification

### Check Services Are Running

```bash
docker compose ps
# Expected: openrag and neo4j both "running" or "healthy"
```

### Check Server Logs

```bash
docker compose logs openrag
# Expected: "MCP server ready and listening for requests"
```

### Test in Claude

Once configured, ask Claude:

```
List available MCP tools
```

Expected tools: `ingest_text`, `query_documents`, `list_documents`, `delete_document`, `get_stats`

Then test ingestion:

```
Ingest this text into OpenRAG with name "test.txt":
OpenRAG is a multi-strategy RAG system supporting traditional, contextual, and graph-based retrieval.
```

And query:

```
Query OpenRAG for "multi-strategy RAG"
```

---

## Troubleshooting

### Neo4j not ready / server crashes on startup

Neo4j takes ~30 seconds to initialize. The compose healthcheck handles this — the openrag service waits until Neo4j is healthy before starting. If it still fails:

```bash
docker compose restart openrag
```

### Ollama connection refused

The container reaches Ollama via `host.docker.internal`. Ensure Ollama is running on your host:

```bash
ollama serve
curl http://localhost:11434/api/tags
```

On Linux, `host.docker.internal` may not resolve automatically. Add to `docker-compose.yml` under the `openrag` service:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

### Embedding model not downloading

The model downloads on first startup (420 MB). Check logs:

```bash
docker compose logs -f openrag
```

If the download fails, ensure outbound internet access from Docker is allowed.

### ChromaDB permission errors

```bash
chmod -R 755 ./chroma_db
```

### Delete all data and start fresh

```bash
docker compose down -v          # Removes volumes (Neo4j data + HuggingFace cache)
rm -rf ./chroma_db              # Removes ChromaDB data
docker compose up -d            # Start fresh
```

> **Warning**: This permanently deletes all ingested documents.

---

## Publishing to Docker Hub

To push the image so others can `docker pull` it without cloning the repo:

### Manual Push

```bash
# 1. Build the image with your Docker Hub username as the tag
docker build -t janik96v/openrag:latest .

# 2. (Optional) also tag with a version
docker tag janik96v/openrag:latest janik96v/openrag:0.1.0

# 3. Login to Docker Hub
docker login

# 4. Push
docker push janik96v/openrag:latest
docker push janik96v/openrag:0.1.0  # if versioned
```

### Automated Push via GitHub Actions

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Publish Docker Image

on:
  push:
    tags:
      - "v*"          # Trigger on version tags like v0.1.0
  workflow_dispatch:  # Allow manual trigger

jobs:
  push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract version tag
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: janik96v/openrag
          tags: |
            type=semver,pattern={{version}}
            type=raw,value=latest

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          platforms: linux/amd64,linux/arm64   # supports Apple Silicon + x86
          tags: ${{ steps.meta.outputs.tags }}
```

**Setup steps:**
1. Go to Docker Hub → Account Settings → Security → New Access Token
2. In your GitHub repo → Settings → Secrets → New repository secret:
   - `DOCKERHUB_USERNAME` = your Docker Hub username
   - `DOCKERHUB_TOKEN` = the token from step 1
3. Push a version tag to trigger the workflow:

```bash
git tag v0.1.0
git push origin v0.1.0
```

After the workflow completes, users can run:

```bash
docker pull janik96v/openrag:latest
```

---

## Getting Help

- [Quick Start Guide](quick-start.md) — usage examples
- [Architecture Overview](architecture.md) — system design
- [CLAUDE.md](../CLAUDE.md) — development conventions

---

**Last Updated**: 2026-03-28
