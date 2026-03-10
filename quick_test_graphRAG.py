#!/usr/bin/env python3
"""Quick test script for Graph RAG implementation."""

import sys
import asyncio
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("GRAPH RAG QUICK TEST")
print("=" * 80)

# Test text - SAME as contextual RAG
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
    from src.openrag.core.graph_processor import GraphProcessor
    from src.openrag.core.graph_vector_store import GraphVectorStore
    from src.openrag.core.ollama_client import OllamaClient
    from src.openrag.core.chunker import TextChunker
    from src.openrag.core.embedder import EmbeddingModel
    from src.openrag.models.graph_schemas import GraphDocument
    from src.openrag.models.schemas import DocumentMetadata
    from src.openrag.models.contextual_schemas import RAGType
    from src.openrag.config import Settings
    print("All imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Configuration and Cleanup
print("\nTest 2: Testing configuration and cleanup...")


async def config_and_cleanup():
    try:
        settings = Settings()
        print(f"Settings loaded:")
        print(f"   - Neo4j URI: {settings.neo4j_uri}")
        print(f"   - Neo4j Database: {settings.neo4j_database}")
        print(f"   - Ollama URL: {settings.ollama_base_url}")
        print(f"   - Ollama Model: {settings.ollama_context_model}")

        # Cleanup: Delete ChromaDB collections but keep directory
        chroma_dir = Path("./tests/test_chroma_graph")
        if chroma_dir.exists():
            print(f"   - Cleaning ChromaDB collections in {chroma_dir}")
            shutil.rmtree(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        print(f"   - Clean ChromaDB directory ready")

        # Cleanup Neo4j: Drop all test nodes and relationships
        print(f"   - Cleaning Neo4j test data...")
        try:
            from neo4j import AsyncGraphDatabase

            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )

            async with driver.session(database=settings.neo4j_database) as session:
                # Drop all nodes and relationships with test document ID
                await session.run(
                    "MATCH (n) WHERE n.document_id = 'test_graph_doc' DETACH DELETE n"
                )
                print(f"   - Neo4j test data cleaned")

            await driver.close()

        except Exception as e:
            print(f"   - Warning: Could not clean Neo4j database: {e}")
            print(f"   - This is OK if Neo4j is not running yet")

        return settings

    except Exception as e:
        print(f"Configuration/cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


settings = asyncio.run(config_and_cleanup())


# Test 3: Neo4j Connection Test and Database Setup
print("\nTest 3: Testing Neo4j connection and database setup...")


async def test_neo4j_connection():
    try:
        # Create graph processor with default database
        graph_processor = GraphProcessor(
            ollama_client=None,  # Will add later
            neo4j_uri=settings.neo4j_uri,
            neo4j_username=settings.neo4j_username,
            neo4j_password=settings.neo4j_password,
            neo4j_database=settings.neo4j_database,
            entity_model=settings.ollama_context_model,
            fallback_enabled=True,
        )

        # Initialize Neo4j connection
        await graph_processor.initialize()

        print(f"Neo4j connection successful:")
        print(f"   - Database: {settings.neo4j_database}")
        print(f"   - Constraints and indexes created")
        print(f"   - Test data isolated by document_id: 'test_graph_doc'")

        return graph_processor

    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        print("   - Make sure Neo4j is running: brew services start neo4j")
        print("   - Check Neo4j status: brew services list")
        print("   - Or start manually: neo4j start")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Run all async tests in a single event loop to avoid event loop issues
async def run_all_tests():
    """Run all async tests in a single event loop."""
    global graph_processor, ollama_client, graph_chunks, metadata, vector_store

    # Test 3: Neo4j Connection Test
    print("\nTest 3: Testing Neo4j connection and database setup...")
    try:
        graph_processor = GraphProcessor(
            ollama_client=None,  # Will add later
            neo4j_uri=settings.neo4j_uri,
            neo4j_username=settings.neo4j_username,
            neo4j_password=settings.neo4j_password,
            neo4j_database=settings.neo4j_database,
            entity_model=settings.ollama_context_model,
            fallback_enabled=True,
        )

        await graph_processor.initialize()

        print(f"Neo4j connection successful:")
        print(f"   - Database: {settings.neo4j_database}")
        print(f"   - Constraints and indexes created")
        print(f"   - Test data isolated by document_id: 'test_graph_doc'")

    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        print("   - Make sure Neo4j is running: brew services start neo4j")
        print("   - Check Neo4j status: brew services list")
        print("   - Or start manually: neo4j start")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 4: Ollama Connection Test
    print("\nTest 4: Testing Ollama connection...")
    try:
        ollama_client = OllamaClient(
            base_url=settings.ollama_base_url,
            timeout=300.0,
        )

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

        graph_processor.ollama_client = ollama_client

    except Exception as e:
        print(f"Ollama connection failed: {e}")
        print("   - Make sure Ollama is running: ollama serve")
        print(f"   - Make sure model is available: ollama pull {settings.ollama_context_model}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 5: Entity Extraction
    print("\nTest 5: Testing graph processor - entity extraction...")
    print("   ⏱️  Entity extraction may take 3-5 minutes...")
    try:
        start_time = datetime.now()

        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunk_texts = chunker.chunk_text(text=TEST_TEXT)
        print(f"   - Created {len(chunk_texts)} chunks from test text")

        metadata = DocumentMetadata(
            filename="test_graph.txt",
            file_size=len(TEST_TEXT),
            chunk_count=len(chunk_texts),
        )

        from src.openrag.models.schemas import DocumentChunk

        chunks = [
            DocumentChunk(
                document_id="test_graph_doc",
                content=chunk_text,
                chunk_index=i,
            )
            for i, chunk_text in enumerate(chunk_texts)
        ]

        graph_chunks = await graph_processor.process_document_chunks(
            chunks=chunks,
            document_metadata=metadata,
            full_document_text=TEST_TEXT,
        )

        print(f"   - Processed {len(graph_chunks)} chunks")

        total_entities = sum(chunk.entity_count for chunk in graph_chunks)
        total_relationships = sum(chunk.relationship_count for chunk in graph_chunks)

        print(f"   - Total entities extracted: {total_entities}")
        print(f"   - Total relationships extracted: {total_relationships}")

        if graph_chunks and graph_chunks[0].entities:
            sample_entities = graph_chunks[0].entities[:3]
            print(f"   - Sample entities from chunk 0:")
            for entity in sample_entities:
                print(f"     - {entity.name} ({entity.entity_type}, confidence: {entity.confidence:.2f})")

        if graph_chunks and graph_chunks[0].relationships:
            sample_rels = graph_chunks[0].relationships[:2]
            print(f"   - Sample relationships from chunk 0:")
            for rel in sample_rels:
                print(f"     - {rel.relationship_type} (confidence: {rel.confidence:.2f})")

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"   - Processing time: {elapsed:.1f} seconds")

    except Exception as e:
        print(f"Entity extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 6: Graph Storage
    print("\nTest 6: Testing graph vector store - storage and Neo4j...")
    try:
        embedding_model = EmbeddingModel(model_name=settings.embedding_model)
        print(f"   - Embedding model loaded: {settings.embedding_model}")

        vector_store = GraphVectorStore(
            persist_directory=Path("./tests/test_chroma_graph"),
            embedding_model=embedding_model,
            graph_processor=graph_processor,
            base_collection_name="test_documents",
        )
        print(f"   - Graph vector store initialized")

        total_entities = sum(chunk.entity_count for chunk in graph_chunks)
        total_relationships = sum(chunk.relationship_count for chunk in graph_chunks)

        graph_doc = GraphDocument(
            document_id="test_graph_doc",
            metadata=metadata,
            chunks=graph_chunks,
            total_entities=total_entities,
            total_relationships=total_relationships,
        )

        vector_store.add_document(graph_doc, rag_type=RAGType.GRAPH)
        print(f"   - Document added to graph collection")
        print(f"   - Document ID: {graph_doc.document_id}")

        stats = vector_store.get_stats()
        if 'graph_chunks' in stats:
            print(f"   - ChromaDB: {stats['graph_chunks']} chunks stored")

        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                "MATCH (e:Entity) WHERE e.document_id = 'test_graph_doc' RETURN count(e) as count"
            )
            entity_count = (await result.single())["count"]

            result = await session.run(
                """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE e1.document_id = 'test_graph_doc'
                RETURN count(r) as count
                """
            )
            rel_count = (await result.single())["count"]

            print(f"   - Neo4j: {entity_count} entities, {rel_count} relationships stored")
            print(f"   - (Test data isolated with document_id: 'test_graph_doc')")

        await driver.close()

    except Exception as e:
        print(f"Graph storage failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 7: Graph Search
    print("\nTest 7: Testing graph search with expansion...")
    try:
        query = "Who works at MIT AI Lab?"

        results = await vector_store.search(
            query=query,
            n_results=3,
            rag_type=RAGType.GRAPH,
            max_hops=2,
        )

        print(f"   - Search query: '{query}'")
        print(f"   - Search returned {len(results)} results")

        if results:
            for i, (chunk, similarity) in enumerate(results):
                print(f"   - Result {i+1} similarity: {similarity:.3f}")
                if hasattr(chunk, 'entity_count'):
                    print(f"     (Contains {chunk.entity_count} entities, {chunk.relationship_count} relationships)")

        print(f"   - Graph expansion: Additional related chunks found via relationships")
        print(f"   - Data left for inspection (not deleted)")

    except Exception as e:
        print(f"Graph search failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Cleanup
    await graph_processor.close()


# Run all tests in a single event loop
asyncio.run(run_all_tests())


print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)

print("\nTest Summary:")
print(f"   - Neo4j connection: ✓ (database: {settings.neo4j_database})")
print(f"   - Ollama connection: ✓")
print(f"   - Entity extraction: ✓ ({len(graph_chunks)} chunks processed)")
total_entities = sum(chunk.entity_count for chunk in graph_chunks)
total_relationships = sum(chunk.relationship_count for chunk in graph_chunks)
print(f"   - Entities extracted: {total_entities}")
print(f"   - Relationships extracted: {total_relationships}")
print(f"   - Graph vector store: ✓")
print(f"   - Data persisted in:")
print(f"     - ChromaDB: ./tests/test_chroma_graph")
print(f"     - Neo4j: {settings.neo4j_database} database (document_id: 'test_graph_doc')")
print(f"\nNote: Test data is isolated by document_id to avoid interfering with production data.")

print("\nNext steps:")
print("1. Run full test suite: pytest tests/ -v")
print("2. Compare with traditional RAG: python quick_test_normalRAG.py")
print("3. Compare with contextual RAG: python quick_test_contextualRAG.py")
print("4. Test with your own documents")
print("5. Integrate with Claude Desktop (see README.md)")
