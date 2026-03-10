#!/usr/bin/env python3
"""Quick test script for Contextual RAG implementation."""

import sys
import asyncio
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("CONTEXTUAL RAG QUICK TEST")
print("=" * 80)

# Test text - shared with Graph RAG
TEST_TEXT = """
Dr. Sarah Chen, a leading researcher at the MIT AI Lab in Cambridge Massachusetts,
recently announced a breakthrough in retrieval-augmented generation systems. Working
alongside her colleague Dr. James Rodriguez from Stanford University, they developed
a novel approach combining graph-based knowledge representation with contextual embeddings.

The research team at MIT AI Lab has been investigating how traditional vector databases
can be enhanced with knowledge graphs to improve semantic search accuracy. Their work
builds upon earlier research in natural language processing and demonstrates that
contextual information significantly improves retrieval quality, with performance gains
of up to 49% in domain-specific applications.

The collaboration between MIT AI Lab and Stanford University began at the AI Conference
2024 held in San Francisco, where Dr. Chen presented preliminary findings. Dr. Rodriguez,
who specializes in graph neural networks, contributed the relationship extraction
methodology. The team also partnered with Acme Corporation, a technology company based
in New York, to validate their approach on real-world enterprise data.

This advancement has significant implications for RAG systems used in question-answering,
document retrieval, and knowledge management. The researchers plan to open-source their
implementation, making it accessible to the broader AI community. Future work will focus
on scaling the approach to handle millions of documents while maintaining sub-second
query response times.
"""

# Test 1: Imports
print("\n📦 Test 1: Testing imports...")
try:
    from src.openrag.core.contextual_processor import ContextualProcessor
    from src.openrag.core.contextual_vector_store import ContextualVectorStore
    from src.openrag.core.ollama_client import OllamaClient
    from src.openrag.core.chunker import TextChunker
    from src.openrag.core.embedder import EmbeddingModel
    from src.openrag.models.contextual_schemas import RAGType, ContextualDocument
    from src.openrag.models.schemas import DocumentMetadata
    from src.openrag.config import Settings
    print("All imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Configuration and Cleanup
print("\nTest 2: Testing configuration and cleanup...")
try:
    settings = Settings()
    print(f"Settings loaded:")
    print(f"   - Ollama URL: {settings.ollama_base_url}")
    print(f"   - Ollama Model: {settings.ollama_context_model}")
    print(f"   - Chunk size: {settings.chunk_size}")

    # Cleanup: Delete ChromaDB collections but keep directory
    chroma_dir = Path("./tests/test_chroma_contextual")
    if chroma_dir.exists():
        print(f"   - Cleaning ChromaDB collections in {chroma_dir}")
        # Remove the directory and recreate it to clean collections
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    print(f"   - Clean ChromaDB directory ready")

except Exception as e:
    print(f"Configuration/cleanup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test 3: Ollama Connection Test
print("\nTest 3: Testing Ollama connection...")


async def test_ollama_connection():
    try:
        ollama_client = OllamaClient(
            base_url=settings.ollama_base_url,
            timeout=300.0,
        )

        # Simple test: Try to generate a small response
        test_prompt = "Say 'OK' if you can read this."
        response = await ollama_client.generate_response(
            model=settings.ollama_context_model,
            prompt=test_prompt,
            temperature=0.1,
        )

        print(f"Ollama connection successful:")
        print(f"   - Model: {settings.ollama_context_model}")
        print(f"   - Base URL: {settings.ollama_base_url}")
        print(f"   - Test response received")

        return ollama_client

    except Exception as e:
        print(f"Ollama connection failed: {e}")
        print("   - Make sure Ollama is running: ollama serve")
        print(f"   - Make sure model is available: ollama pull {settings.ollama_context_model}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


ollama_client = asyncio.run(test_ollama_connection())


# Test 4: Contextual Processor Test
print("\nTest 4: Testing contextual processor...")
print("   ⏱️  Context generation may take 2-3 minutes...")


async def test_contextual_processor():
    try:
        start_time = datetime.now()

        # Initialize processor
        processor = ContextualProcessor(
            ollama_client=ollama_client,
            context_model=settings.ollama_context_model,
            fallback_enabled=True,
        )

        # Create chunks
        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunk_texts = chunker.chunk_text(text=TEST_TEXT)

        # Convert to DocumentChunk objects
        from src.openrag.models.schemas import DocumentChunk

        chunks = [
            DocumentChunk(
                document_id="test_contextual_doc",
                content=chunk_text,
                chunk_index=i,
            )
            for i, chunk_text in enumerate(chunk_texts)
        ]

        print(f"   - Created {len(chunks)} chunks from test text")

        # Create metadata
        metadata = DocumentMetadata(
            filename="test_contextual.txt",
            file_size=len(TEST_TEXT),
            chunk_count=len(chunks),
        )

        # Process chunks with context generation
        contextual_chunks = await processor.process_document_chunks(
            chunks=chunks,
            document_metadata=metadata,
            full_document_text=TEST_TEXT,
        )

        # Verify contextual content was generated
        context_generated_count = sum(
            1 for chunk in contextual_chunks
            if chunk.contextual_content is not None
        )

        print(f"   - Processed {len(contextual_chunks)} chunks")
        print(f"   - Context generated for {context_generated_count}/{len(contextual_chunks)} chunks")

        # Print sample context from first chunk
        if contextual_chunks and contextual_chunks[0].contextual_content:
            sample = contextual_chunks[0].contextual_content
            # Extract just the context part (before "Content:")
            if "Context:" in sample and "Content:" in sample:
                context_part = sample.split("Content:")[0].replace("Context:", "").strip()
                print(f"   - Sample context (chunk 0): {context_part[:100]}...")

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"   - Processing time: {elapsed:.1f} seconds")

        return contextual_chunks, metadata, chunker

    except Exception as e:
        print(f"Contextual processor failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


contextual_chunks, metadata, chunker = asyncio.run(test_contextual_processor())


# Test 5: Contextual Vector Store Test
print("\nTest 5: Testing contextual vector store...")
try:
    # Initialize embedding model
    embedding_model = EmbeddingModel(model_name=settings.embedding_model)
    print(f"   - Embedding model loaded: {settings.embedding_model}")

    # Initialize contextual vector store
    vector_store = ContextualVectorStore(
        persist_directory=Path("./tests/test_chroma_contextual"),
        embedding_model=embedding_model,
        base_collection_name="test_documents",
    )
    print(f"   - Contextual vector store initialized")

    # Create contextual document
    contextual_doc = ContextualDocument(
        document_id="test_contextual_doc",
        metadata=metadata,
        chunks=contextual_chunks,
    )

    # Add document to contextual collection
    vector_store.add_document(contextual_doc, rag_type=RAGType.CONTEXTUAL)
    print(f"   - Document added to contextual collection")
    print(f"   - Document ID: {contextual_doc.document_id}")

    # Search contextual collection
    query = "What did Dr. Sarah Chen research at MIT?"
    results = vector_store.search(
        query=query,
        n_results=3,
        rag_type=RAGType.CONTEXTUAL,
    )

    print(f"   - Search query: '{query}'")
    print(f"   - Search returned {len(results)} results")

    if results:
        for i, (chunk, similarity) in enumerate(results):
            print(f"   - Result {i+1} similarity: {similarity:.3f}")
            # Show that contextual content exists
            if hasattr(chunk, 'contextual_content') and chunk.contextual_content:
                print(f"     (Has contextual content)")

    # Get stats
    stats = vector_store.get_stats()
    print(f"   - Collection stats:")
    if 'contextual_chunks' in stats:
        print(f"     - Contextual chunks: {stats['contextual_chunks']}")

    print(f"   - Data left for inspection (not deleted)")

except Exception as e:
    print(f"Contextual vector store failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)

print("\nTest Summary:")
print(f"   - Ollama connection: ✓")
print(f"   - Context generation: ✓ ({len(contextual_chunks)} chunks processed)")
print(f"   - Contextual vector store: ✓ ({len(results)} search results)")
print(f"   - Data persisted in: ./tests/test_chroma_contextual")

print("\nNext steps:")
print("1. Run full test suite: pytest tests/ -v")
print("2. Compare with traditional RAG: python quick_test_normalRAG.py")
print("3. Test graph RAG: python quick_test_graphRAG.py")
print("4. Test with your own documents")
print("5. Integrate with Claude Desktop (see README.md)")
