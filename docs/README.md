# OpenRAG Documentation

Concise, practical documentation for OpenRAG - a multi-strategy RAG MCP server for personal documents.

## Essential Documentation

### For New Users

**[Installation Guide](installation.md)** - Complete setup and MCP configuration (15 min)
- Environment setup (Python 3.12 + Conda)
- Installing OpenRAG
- **MCP server configuration for Claude Code** (critical)
- External dependencies (Ollama, Neo4j)
- Verification and troubleshooting

**[Quick Start Guide](quick-start.md)** - Get started in 15 minutes
- Quick installation steps
- MCP configuration for Claude Code
- Testing Traditional, Contextual, and Graph RAG
- Common tasks and usage patterns

### Technical Reference

**[Architecture Overview](architecture.md)** - System design and components
- Three RAG strategies (Traditional, Contextual, Graph)
- Component architecture
- Data flow and storage strategy
- Design decisions and performance notes

### Development

**[CLAUDE.md](../CLAUDE.md)** - Development conventions and standards
- Project structure and organization
- Development workflow
- Testing strategy
- Code style and standards

**[Lab Journal](lab_journal.md)** - Research notes and design decisions
- Experimental findings
- Technology evaluations
- Architecture evolution

## Quick Links

**New to OpenRAG?**
1. [Installation Guide](installation.md) - Set up environment and MCP server
2. [Quick Start Guide](quick-start.md) - Try it out in Claude Code
3. [Architecture Overview](architecture.md) - Understand how it works

**Troubleshooting?**
- Check [Installation Guide - Troubleshooting](installation.md#troubleshooting)
- Review [Quick Start - Troubleshooting](quick-start.md#troubleshooting)
- See [CLAUDE.md](../CLAUDE.md) for development issues

## Project Information

- **Python**: 3.12 (NOT 3.13)
- **License**: MIT
- **Main Dependencies**: ChromaDB, sentence-transformers, Ollama (optional), Neo4j (optional)

## Three RAG Strategies

| Strategy | Setup | Speed | Best For |
|----------|-------|-------|----------|
| **Traditional** | None | Fastest | Direct facts, quick setup |
| **Contextual** | + Ollama | Fast | Complex queries, better accuracy |
| **Graph** | + Ollama + Neo4j | Moderate | Relationships, multi-hop reasoning |

All strategies run in parallel - choose the best one for each query.

## MCP Tools

OpenRAG exposes 5 MCP tools for Claude Code:

1. `ingest_text` - Ingest text content into RAG system
2. `query_documents` - Search with chosen RAG strategy
3. `list_documents` - List all documents
4. `delete_document` - Remove document
5. `get_stats` - System statistics

See [Quick Start Guide](quick-start.md) for usage examples.

## Contributing

Documentation follows these principles:
- **Concise**: Get to the point quickly
- **Practical**: Focus on what users need to do
- **Accurate**: Keep in sync with code changes
- **Current**: Update dates when modified

Found an issue? Update the relevant file and verify cross-references.

---

**Last Updated**: 2026-03-06
