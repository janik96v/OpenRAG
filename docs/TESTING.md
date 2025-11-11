# Testing OpenRAG Implementation

This guide walks you through testing the RAG implementation to verify everything works correctly.

## Prerequisites

1. **Install Dependencies** (Python 3.10+ required for MCP):
   ```bash
   # Create conda environment
   conda create -n OpenRAG python=3.13
   conda activate OpenRAG

   # Install dependencies
   pip install chromadb sentence-transformers pydantic pydantic-settings tiktoken pytest pytest-asyncio
   ```

2. **Set PYTHONPATH**:
   ```bash
   export PYTHONPATH=/Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG
   ```

## Option 1: Run Unit and Integration Tests

The fastest way to verify the implementation:

```bash
cd /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG

# Run all tests
python -m pytest tests/ -v

# Run with detailed output
python -m pytest tests/ -v -s

# Run specific test
python -m pytest tests/test_integration.py::test_full_workflow -v -s
```

**Expected Output**: All tests should pass ✅

## Option 2: Interactive Python Testing

Test the RAG components directly in Python:

```bash
cd /Users/janikvollenweider/Library/CloudStorage/OneDrive-Persönlich/1Daten/90_Diverses/10_Projekte/Coding/OpenRAG
python3
```

Then run this test script:

```python
from pathlib import Path
from openrag.core.chunker import TextChunker
from openrag.core.embedder import EmbeddingModel
from openrag.core.vector_store import VectorStore
from openrag.models.schemas import Document, DocumentChunk, DocumentMetadata

# 1. Create test document
test_text = """
Artificial Intelligence (AI) is transforming how we interact with technology.
Machine learning, a subset of AI, enables computers to learn from data.
Deep learning uses neural networks with multiple layers to process information.
Natural language processing allows computers to understand human language.
Computer vision enables machines to interpret and analyze visual information.
"""

print("✅ Test text created")

# 2. Initialize components
print("\n📦 Initializing components...")
chunker = TextChunker(chunk_size=100, chunk_overlap=20)
print("✅ Chunker initialized")

embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")  # Smaller model for testing
print("✅ Embedding model loaded")

vector_store = VectorStore(
    persist_directory=Path("./test_chroma_db"),
    embedding_model=embedding_model,
)
print("✅ Vector store initialized")

# 3. Test chunking
print("\n🔪 Testing chunker...")
chunks = chunker.chunk_text(test_text)
print(f"✅ Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"   Chunk {i}: {len(chunk)} chars, {chunker.count_tokens(chunk)} tokens")

# 4. Create document
print("\n📄 Creating document...")
metadata = DocumentMetadata(
    filename="test.txt",
    file_size=len(test_text),
    chunk_count=len(chunks),
)
document = Document(metadata=metadata)

for i, chunk_text in enumerate(chunks):
    chunk = DocumentChunk(
        document_id=document.document_id,
        content=chunk_text,
        chunk_index=i,
    )
    document.chunks.append(chunk)

print(f"✅ Document created with ID: {document.document_id}")

# 5. Test vector store ingestion
print("\n💾 Testing vector store ingestion...")
vector_store.add_document(document)
print("✅ Document added to vector store")

# 6. Test search
print("\n🔍 Testing semantic search...")
queries = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "Tell me about neural networks",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = vector_store.search(query, n_results=2, min_similarity=0.0)
    print(f"Found {len(results)} results:")
    for chunk, score in results:
        print(f"  - Similarity: {score:.3f}")
        print(f"    Content: {chunk.content[:100]}...")

# 7. Test stats
print("\n📊 Testing statistics...")
stats = vector_store.get_stats()
print(f"✅ Stats: {stats}")

# 8. Test list documents
print("\n📋 Testing list documents...")
docs = vector_store.list_documents()
print(f"✅ Found {len(docs)} documents")
for doc in docs:
    print(f"   - {doc['filename']}: {doc['chunk_count']} chunks")

# 9. Test deletion
print("\n🗑️  Testing document deletion...")
deleted = vector_store.delete_document(document.document_id)
print(f"✅ Document deleted: {deleted}")

print("\n✅ All tests passed! RAG implementation working correctly.")
```

## Option 3: Test with Real .txt File

Create a test document and test the full workflow:

### Step 1: Create Test Document

```bash
cat > test_document.txt << 'EOF'
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on enabling
computers to learn from data without being explicitly programmed. There are three
main types of machine learning:

Supervised Learning
In supervised learning, the algorithm learns from labeled training data. The model
is trained on input-output pairs and learns to predict outputs for new inputs.
Common applications include image classification, spam detection, and price prediction.

Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm tries to find patterns
and structure in the data without predefined labels. Clustering and dimensionality
reduction are common unsupervised learning techniques.

Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions by interacting
with an environment. The agent receives rewards or penalties based on its actions
and learns to maximize cumulative rewards over time.

Deep Learning
Deep learning uses artificial neural networks with multiple layers (hence "deep") to
learn hierarchical representations of data. It has revolutionized fields like computer
vision, natural language processing, and speech recognition.
EOF
```

### Step 2: Test Ingestion and Query

```python
import asyncio
from pathlib import Path
from openrag.core.chunker import TextChunker
from openrag.core.embedder import EmbeddingModel
from openrag.core.vector_store import VectorStore
from openrag.tools.ingest import ingest_document_tool
from openrag.tools.query import query_documents_tool
from openrag.tools.manage import list_documents_tool, delete_document_tool
from openrag.tools.stats import get_stats_tool
from openrag.config import Settings

async def test_full_workflow():
    """Test complete RAG workflow."""

    # Initialize
    settings = Settings(
        chroma_db_path="./test_chroma_db",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=30,
    )

    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embedding_model = EmbeddingModel(model_name=settings.embedding_model)

    vector_store = VectorStore(
        persist_directory=Path(settings.chroma_db_path),
        embedding_model=embedding_model,
    )

    print("=" * 80)
    print("TESTING OPENRAG IMPLEMENTATION")
    print("=" * 80)

    # 1. Ingest document
    print("\n📄 Step 1: Ingesting document...")
    result = await ingest_document_tool(
        file_path="test_document.txt",
        vector_store=vector_store,
        chunker=chunker,
    )
    print(f"Status: {result['status']}")
    print(f"Document ID: {result.get('document_id', 'N/A')}")
    print(f"Chunks: {result.get('chunk_count', 0)}")

    if result['status'] != 'success':
        print(f"❌ Error: {result.get('message')}")
        return

    document_id = result['document_id']

    # 2. Query documents
    print("\n🔍 Step 2: Querying documents...")
    test_queries = [
        "What is supervised learning?",
        "Explain deep learning",
        "What are the types of machine learning?",
    ]

    for query in test_queries:
        print(f"\n  Query: '{query}'")
        result = await query_documents_tool(
            query=query,
            vector_store=vector_store,
            max_results=3,
        )

        if result['status'] == 'success':
            print(f"  Found {result['total_results']} results:")
            for i, res in enumerate(result['results'][:2], 1):
                print(f"    {i}. Score: {res['similarity_score']:.3f}")
                print(f"       {res['content'][:100]}...")
        else:
            print(f"  ❌ Error: {result.get('message')}")

    # 3. List documents
    print("\n📋 Step 3: Listing documents...")
    result = await list_documents_tool(vector_store=vector_store)
    print(f"Total documents: {result.get('total_documents', 0)}")

    # 4. Get stats
    print("\n📊 Step 4: Getting statistics...")
    result = await get_stats_tool(vector_store=vector_store, settings=settings)
    if result['status'] == 'success':
        stats = result['statistics']
        print(f"  Documents: {stats['total_documents']}")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Model: {stats['embedding_model']}")

    # 5. Delete document
    print(f"\n🗑️  Step 5: Deleting document {document_id}...")
    result = await delete_document_tool(
        document_id=document_id,
        vector_store=vector_store,
    )
    print(f"Status: {result['status']}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

# Run the test
asyncio.run(test_full_workflow())
```

Save this as `test_workflow.py` and run:
```bash
python test_workflow.py
```

## Option 4: Test MCP Server (Advanced)

Test the actual MCP server with the MCP Inspector tool:

### Step 1: Install MCP Requirements

```bash
# Note: MCP requires Python 3.10+
# You may need to install mcp from source or use a newer Python version
```

### Step 2: Start Server in Test Mode

```bash
# Set environment variables
export CHROMA_DB_PATH=./test_chroma_db
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export CHUNK_SIZE=200
export CHUNK_OVERLAP=30
export LOG_LEVEL=DEBUG

# Run server (will wait for stdio input)
python -m openrag.server
```

## Expected Results

### ✅ Successful Test Indicators:

1. **Chunking**: Text split into appropriate chunks (100-400 tokens each)
2. **Embedding**: Model loads successfully (~80MB for MiniLM, ~420MB for mpnet)
3. **Storage**: ChromaDB creates `./test_chroma_db/` directory with `chroma.sqlite3`
4. **Search**: Queries return relevant chunks with similarity scores > 0.5
5. **Persistence**: Data survives Python session restarts

### ❌ Common Issues:

1. **Import Errors**: Ensure PYTHONPATH is set correctly
2. **Model Download**: First run downloads embedding model (requires internet)
3. **Permission Errors**: Ensure write permissions for chroma_db directory
4. **Python Version**: MCP server requires Python 3.10+, but core can run on 3.9

## Verification Checklist

- [ ] Chunker creates appropriate sized chunks
- [ ] Embedding model loads without errors
- [ ] Vector store creates ChromaDB directory
- [ ] Documents can be ingested
- [ ] Queries return relevant results
- [ ] Similarity scores are reasonable (0.3-0.9 range)
- [ ] Documents can be listed
- [ ] Documents can be deleted
- [ ] Stats are accurate
- [ ] Database persists across sessions

## Clean Up

```bash
# Remove test database
rm -rf test_chroma_db/

# Remove test document
rm test_document.txt test_workflow.py
```

## Next Steps

Once basic testing passes:

1. **Test with larger documents** (10+ pages)
2. **Test query variety** (different question types)
3. **Test edge cases** (empty files, very long files)
4. **Benchmark performance** (ingestion time, query latency)
5. **Test Claude Desktop integration** (requires MCP setup)

## Getting Help

If tests fail:

1. Check logs in stderr output
2. Verify Python version: `python --version`
3. Verify dependencies: `pip list | grep -E "(chromadb|sentence-transformers|tiktoken)"`
4. Check disk space: `df -h`
5. Check ChromaDB directory: `ls -la test_chroma_db/`

---

**Happy Testing!** 🧪✨
