#!/usr/bin/env python3
"""Quick test script for OpenRAG implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("OPENRAG QUICK TEST")
print("=" * 80)

# Test 1: Imports
print("\n📦 Test 1: Testing imports...")
try:
    from src.openrag.core.chunker import TextChunker
    from src.openrag.core.embedder import EmbeddingModel
    from src.openrag.core.vector_store import VectorStore
    from src.openrag.models.schemas import Document, DocumentChunk, DocumentMetadata
    from src.openrag.config import Settings
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Configuration
print("\n⚙️  Test 2: Testing configuration...")
try:
    settings = Settings(
        chroma_db_path="./tests/test_chroma_db",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=20,
    )
    print(f"✅ Settings loaded:")
    print(f"   - ChromaDB: {settings.chroma_db_path}")
    print(f"   - Model: {settings.embedding_model}")
    print(f"   - Chunk size: {settings.chunk_size}")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Test 3: Chunker
print("\n🔪 Test 3: Testing text chunker...")
try:
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    test_text = """
    Artificial intelligence is the simulation of human intelligence by machines.
    Machine learning is a subset of AI that enables systems to learn from data.
    Deep learning uses neural networks to process complex patterns.
    Natural language processing helps computers understand human language.
    """

    chunks = chunker.chunk_text(test_text)
    print(f"✅ Chunker working:")
    print(f"   - Input: {len(test_text)} chars, {chunker.count_tokens(test_text)} tokens")
    print(f"   - Output: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        tokens = chunker.count_tokens(chunk)
        print(f"   - Chunk {i}: {tokens} tokens")

except Exception as e:
    print(f"❌ Chunker failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Embedding Model
print("\n🧮 Test 4: Testing embedding model...")
print("   (This may take a moment on first run - downloading model...)")
try:
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    print(f"✅ Embedding model loaded:")
    print(f"   - Dimension: {embedding_model.get_embedding_dimension()}")

    # Test embedding
    test_texts = ["Hello world", "Machine learning is fascinating"]
    embeddings = embedding_model.embed_texts(test_texts)
    print(f"   - Generated {len(embeddings)} embeddings")
    print(f"   - Embedding shape: {len(embeddings[0])} dimensions")

except Exception as e:
    print(f"❌ Embedding model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Vector Store
print("\n💾 Test 5: Testing vector store...")
try:
    vector_store = VectorStore(
        persist_directory=Path("./tests/test_chroma_db"),
        embedding_model=embedding_model,
    )
    print("✅ Vector store initialized")

    # Create test document
    metadata = DocumentMetadata(
        filename="./tests/test_chunk.txt",
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

    # Add to vector store
    vector_store.add_document(document)
    print(f"   - Document added: {document.document_id}")

    # Test search
    results = vector_store.search("What is artificial intelligence?", n_results=2)
    print(f"   - Search returned {len(results)} results")

    if results:
        chunk, score = results[0]
        print(f"   - Top result similarity: {score:.3f}")

    # Test stats
    stats = vector_store.get_stats()
    print(f"   - Stats: {stats}")

    # Clean up
    vector_store.delete_document(document.document_id)
    print(f"   - Document deleted")

except Exception as e:
    print(f"❌ Vector store failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Full Workflow (Async)
print("\n🔄 Test 6: Testing async tools...")
try:
    import asyncio
    from openrag.tools.ingest import ingest_text_tool
    from openrag.tools.query import query_documents_tool
    from openrag.tools.stats import get_stats_tool

    # Test text content
    test_text_content = """
    Python is a high-level programming language.
    It is widely used for web development, data science, and automation.
    Python's syntax is designed to be readable and straightforward.
    """

    async def test_async():
        # Reinitialize for clean state
        vs = VectorStore(
            persist_directory=Path("./tests/test_chroma_db"),
            embedding_model=embedding_model,
        )
        ch = TextChunker(chunk_size=100, chunk_overlap=20)

        # Ingest
        result = await ingest_text_tool(
            text=test_text_content,
            document_name="test_python.txt",
            vector_store=vs,
            chunker=ch,
        )
        assert result['status'] == 'success', f"Ingest failed: {result}"
        doc_id = result['document_id']
        print(f"   - Ingested: {result['document_name']} ({result['chunk_count']} chunks)")

        # Query
        result = await query_documents_tool(
            query="What is Python?",
            vector_store=vs,
            max_results=2,
        )
        assert result['status'] == 'success', f"Query failed: {result}"
        print(f"   - Query found {result['total_results']} results")

        # Stats
        result = await get_stats_tool(vector_store=vs, settings=settings)
        assert result['status'] == 'success', f"Stats failed: {result}"
        print(f"   - Stats: {result['statistics']['total_chunks']} chunks")

        # Clean up
        vs.delete_document(doc_id)
        test_file.unlink()

    asyncio.run(test_async())
    print("✅ Async tools working")

except Exception as e:
    print(f"❌ Async tools failed: {e}")
    import traceback
    traceback.print_exc()
    # Clean up
    if test_file.exists():
        test_file.unlink()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\n🎉 OpenRAG implementation is working correctly!")
print("\nNext steps:")
print("1. Run full test suite: pytest tests/ -v")
print("2. Test with your own documents")
print("3. Integrate with Claude Desktop (see README.md)")
print("\n📖 See TESTING.md for more detailed testing options")
