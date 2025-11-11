# Frequently Asked Questions

Answers to common questions about OpenRAG.

## General Questions

### What is OpenRAG?

OpenRAG is an open-source MCP (Model Context Protocol) server that enables AI assistants like Claude to perform semantic search over your personal documents. It combines vector embeddings, ChromaDB storage, and an MCP interface to provide Retrieval Augmented Generation capabilities without requiring external APIs or cloud services.

### What does "RAG" mean?

RAG stands for Retrieval Augmented Generation. It's a technique where an AI system:
1. **Retrieves** relevant information from a knowledge base (your documents)
2. **Augments** its knowledge with that retrieved information
3. **Generates** answers based on both its training and your documents

OpenRAG handles the retrieval part, providing context to AI assistants.

### Is my data kept private?

Yes, completely. OpenRAG:
- Runs entirely on your local machine
- Stores all data in a local ChromaDB database
- Uses local embedding models (no API calls)
- Never sends your documents to external services
- Keeps all processing offline (except initial model download)

### What file formats are supported?

Currently supported:
- `.txt` files (UTF-8 encoding)

Planned for future releases:
- PDF files
- Markdown (.md)
- HTML files
- Microsoft Word (.docx)
- Code files (.py, .js, .java, etc.)

## Installation & Setup

### What are the system requirements?

**Minimum**:
- macOS, Linux, or Windows
- 8 GB RAM
- 10 GB free disk space
- Python 3.10 or higher
- Internet connection (for initial setup)

**Recommended**:
- 16 GB RAM
- 20+ GB free disk space
- SSD for better performance
- Python 3.13

### Do I need a GPU?

No, OpenRAG works fine on CPU. However:
- GPU can speed up embedding generation
- sentence-transformers automatically uses GPU if available
- Most users won't notice significant difference for personal use

### How long does installation take?

- **Automated**: ~10-15 minutes (including downloads)
- **Manual**: ~5-10 minutes
- First run: Additional 1-2 minutes to download embedding model

### Can I use this without Claude Desktop?

Yes! OpenRAG can be used:
1. **Python API**: Call tools directly from Python code
2. **MCP Client**: Any MCP-compatible client
3. **Claude Desktop**: Integrated experience (recommended)

See [User Guide](user-guide.md) for Python API examples.

## Usage Questions

### How many documents can I ingest?

**Practical limits**:
- Small documents (1-10 pages): Thousands
- Medium documents (10-100 pages): Hundreds
- Large documents (100+ pages): Dozens to hundreds

**Limiting factors**:
- Disk space (~4 KB per chunk)
- Query speed (grows with collection size)
- RAM (collection loaded during queries)

**Example capacity**:
- 100 MB storage ≈ 25,000 chunks ≈ 100-200 medium documents
- 1 GB storage ≈ 250,000 chunks ≈ 1,000-2,000 medium documents

### How do I choose between embedding models?

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| Development/Testing | all-MiniLM-L6-v2 | Faster iteration |
| Production | all-mpnet-base-v2 | Best quality |
| Limited resources | all-MiniLM-L6-v2 | Smaller, faster |
| Research/Academic | all-mpnet-base-v2 | Highest accuracy |

See [Configuration Reference](configuration.md) for details.

### What's a good chunk size?

**Default**: 400 tokens (recommended for most cases)

**Adjust based on use case**:
- **200-300 tokens**: Precise fact-finding, code documentation
- **400-500 tokens**: General purpose, balanced (default)
- **600-800 tokens**: Long-form content, comprehensive answers

See [User Guide](user-guide.md) for detailed guidance.

### How do I improve search quality?

1. **Use better queries**: Specific, natural language questions
2. **Adjust similarity threshold**: Higher for precision, lower for recall
3. **Choose right model**: mpnet for quality, MiniLM for speed
4. **Optimize chunk size**: Match to your document type
5. **Clean documents**: Remove noise before ingestion
6. **Re-ingest**: Try different settings if results are poor

### Can I search across multiple languages?

The default models (all-MiniLM-L6-v2 and all-mpnet-base-v2) work best with English.

For multilingual support:
- Use `paraphrase-multilingual-MiniLM-L12-v2`
- Or `paraphrase-multilingual-mpnet-base-v2`

Change in `.env`:
```bash
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

**Note**: Requires re-ingesting all documents.

## Performance Questions

### How fast is document ingestion?

**Typical speeds**:
- all-MiniLM-L6-v2: ~2-3 pages/second
- all-mpnet-base-v2: ~0.5-1 pages/second

**Examples**:
- 10-page document: 5-20 seconds
- 100-page document: 1-3 minutes
- 1,000-page book: 10-30 minutes

**Factors affecting speed**:
- Embedding model (MiniLM is 5x faster)
- Document complexity
- System resources
- Chunk size

### How fast are queries?

**Typical speeds**:
- Small collection (100 chunks): 50-100ms
- Medium collection (1,000 chunks): 100-200ms
- Large collection (10,000 chunks): 200-500ms

**Factors**:
- Collection size (larger = slower)
- max_results parameter
- System RAM and CPU

### Why is the first query slow?

The first query loads the embedding model and collection into memory. Subsequent queries are much faster.

**First query**: 2-5 seconds
**Following queries**: 100-500ms

This is normal behavior.

### How much disk space do I need?

**Storage calculation**:
- ~4 KB per chunk (embeddings + metadata)
- ~10-15 chunks per 10-page document

**Examples**:
- 10-page document: ~60 KB
- 100-page document: ~600 KB
- 1,000-page book: ~6 MB

**Plan for**:
- 100 MB: ~200 medium documents
- 1 GB: ~2,000 medium documents
- 10 GB: ~20,000 medium documents

## Technical Questions

### What is ChromaDB?

ChromaDB is an open-source vector database designed for AI applications. It:
- Stores embeddings and metadata
- Provides fast similarity search
- Runs locally (no external service)
- Uses SQLite for persistence

### What are embeddings?

Embeddings are numerical representations (vectors) of text that capture semantic meaning. Similar concepts have similar vectors, enabling semantic search.

Example:
- "dog" and "puppy" have similar embeddings
- "dog" and "car" have very different embeddings

### How does semantic search work?

1. Document chunks are converted to embeddings (vectors)
2. Your query is converted to an embedding
3. ChromaDB finds chunks with similar embeddings
4. Results are ranked by similarity (cosine distance)
5. Top N results are returned

### Can I backup my data?

Yes! Backup the ChromaDB directory:

```bash
# Backup
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./chroma_db

# Restore
tar -xzf chroma_backup_20251109.tar.gz
```

**Important**: Backup includes all documents and embeddings.

### Can I migrate to a different embedding model?

Yes, but requires re-ingesting all documents:

1. Backup current database
2. Change `EMBEDDING_MODEL` in `.env`
3. Delete old database (or rename it)
4. Re-ingest all documents

**Why?** Embeddings from different models are incompatible.

### How do I update OpenRAG?

```bash
# Pull latest code
git pull origin main

# Update dependencies
conda activate OpenRAG
pip install -r requirements.txt

# Reinstall in editable mode
pip install -e .

# Run tests
pytest tests/ -v
```

### Can I use this in production?

Yes, but consider:
- **Backup strategy**: Regular database backups
- **Monitoring**: Log analysis and error tracking
- **Resource limits**: Plan for concurrent access
- **Updates**: Keep dependencies current
- **Security**: Validate all file paths

For high-throughput scenarios, consider ChromaDB client-server mode.

## MCP & Integration Questions

### What is MCP?

MCP (Model Context Protocol) is an open standard by Anthropic for integrating external tools and data with AI assistants. It provides a standardized way for AI systems to access tools, databases, and APIs.

### Can I use OpenRAG with other AI assistants?

Currently optimized for Claude Desktop, but technically works with any MCP-compatible client.

### Can I add custom tools?

Yes! See [Developer Guide](developer-guide.md) for adding new tools.

### Does this work offline?

**After initial setup**: Yes, completely offline.

**Initial setup requires internet** for:
- Installing dependencies
- Downloading embedding models

Once set up, OpenRAG runs fully offline.

## Troubleshooting Questions

### Why are my query results poor?

Common causes:
1. **Wrong embedding model**: Try mpnet instead of MiniLM
2. **Poor query**: Be more specific
3. **Chunk size mismatch**: Adjust for your content
4. **Low similarity threshold**: Increase min_similarity
5. **Document quality**: Clean and reformat documents

See [Troubleshooting Guide](troubleshooting.md) for solutions.

### Why is ingestion failing?

Common causes:
1. **File not found**: Use absolute paths
2. **Encoding error**: Convert to UTF-8
3. **File too large**: Split into smaller files
4. **Permission denied**: Check file permissions
5. **Disk space**: Ensure sufficient free space

### Why can't Claude Desktop see my server?

Common causes:
1. **Wrong Python path**: Use `which python` in activated environment
2. **Invalid JSON**: Validate config file syntax
3. **Missing environment variables**: Check config.env section
4. **Claude not restarted**: Quit and restart Claude Desktop

See [Troubleshooting Guide](troubleshooting.md) for detailed solutions.

### How do I get help?

1. **Check documentation**:
   - [User Guide](user-guide.md)
   - [Troubleshooting Guide](troubleshooting.md)
   - This FAQ

2. **Enable debug logging**:
   ```bash
   LOG_LEVEL=DEBUG
   ```

3. **Review error messages**: Check stderr output

4. **Create minimal example**: Isolate the issue

5. **Open GitHub issue**: With details and logs

## Future Plans

### What features are planned?

**Near term**:
- PDF support
- Markdown support
- Batch ingestion tool
- Document update capability

**Long term**:
- Multiple RAG strategies (contextual, graph)
- Reranking models
- Multi-modal search (text + images)
- Web interface

See [Lab Journal](lab_journal.md) for research notes.

### Can I contribute?

Yes! Contributions welcome:
- Bug fixes
- New features
- Documentation improvements
- Test coverage
- Performance optimizations

See [Developer Guide](developer-guide.md) for contribution guidelines.

### Is there a roadmap?

Check the GitHub repository for:
- Issues tagged "enhancement"
- Project boards
- Milestones
- [Lab Journal](lab_journal.md) for research directions

## Still Have Questions?

- **Installation**: See [Installation Guide](INSTALLATION.md)
- **Usage**: See [User Guide](user-guide.md)
- **API**: See [API Reference](api-reference.md)
- **Configuration**: See [Configuration Reference](configuration.md)
- **Troubleshooting**: See [Troubleshooting Guide](troubleshooting.md)
- **Development**: See [Developer Guide](developer-guide.md)

---

**Have a question not answered here?** Open a GitHub issue or discussion.

Last Updated: 2025-11-09
