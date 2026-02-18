"""Integration tests for Graph RAG end-to-end functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openrag.core.chunker import TextChunker
from openrag.core.embedder import EmbeddingModel
from openrag.core.graph_processor import GraphProcessor
from openrag.core.graph_vector_store import GraphVectorStore
from openrag.models.contextual_schemas import RAGType
from openrag.models.schemas import Document, DocumentMetadata, DocumentStatus


@pytest.fixture
def sample_document_text():
    """Sample text with clear entities and relationships."""
    return """
    John Doe is the CEO of Acme Corporation, which is headquartered in New York City.
    The company was founded in 2010 and specializes in artificial intelligence solutions.

    Sarah Smith, the CTO of Acme Corporation, leads the engineering team. She previously
    worked at Tech Innovations Inc. in San Francisco.

    Acme Corporation recently partnered with Global Tech Alliance to expand their services
    to Europe. The partnership was announced at the 2024 AI Summit in London.
    """.strip()


@pytest.fixture
def mock_graph_processor_with_ollama():
    """Create a mock GraphProcessor that simulates Ollama responses."""

    async def mock_extract(chunk, full_document_text):
        """Mock entity extraction that returns realistic entities."""
        # Parse the content to return relevant entities
        content = chunk.content.lower()

        from openrag.models.graph_schemas import Entity, Relationship

        entities = []
        relationships = []

        # Define entity patterns
        if "john doe" in content:
            entities.append(
                Entity(
                    name="John Doe",
                    entity_type="PERSON",
                    source_chunk_id=chunk.chunk_id,
                    confidence=0.95,
                )
            )

        if "acme corporation" in content:
            entities.append(
                Entity(
                    name="Acme Corporation",
                    entity_type="ORGANIZATION",
                    source_chunk_id=chunk.chunk_id,
                    confidence=0.95,
                )
            )

        if "new york" in content:
            entities.append(
                Entity(
                    name="New York City",
                    entity_type="LOCATION",
                    source_chunk_id=chunk.chunk_id,
                    confidence=0.90,
                )
            )

        if "sarah smith" in content:
            entities.append(
                Entity(
                    name="Sarah Smith",
                    entity_type="PERSON",
                    source_chunk_id=chunk.chunk_id,
                    confidence=0.95,
                )
            )

        # Create relationships based on content
        if len(entities) >= 2:
            if "ceo" in content or "works" in content:
                relationships.append(
                    Relationship(
                        source_entity_id=entities[0].entity_id,
                        target_entity_id=entities[1].entity_id,
                        relationship_type="WORKS_AT",
                        source_chunk_id=chunk.chunk_id,
                        confidence=0.90,
                    )
                )

        return entities, relationships

    async def mock_store_graph(entities, relationships, chunk):
        """Mock Neo4j storage."""
        return {
            "entity_ids": {e.entity_id: i for i, e in enumerate(entities)},
            "relationship_ids": list(range(len(relationships))),
        }

    processor = MagicMock(spec=GraphProcessor)
    processor.extract_entities_and_relationships = mock_extract
    processor.store_graph_in_neo4j = mock_store_graph
    processor.process_document_chunks = AsyncMock()
    processor.driver = MagicMock()
    processor.neo4j_database = "neo4j"

    # Mock process_document_chunks to use our mock extraction
    async def mock_process_chunks(chunks, document_metadata, full_document_text):
        from openrag.models.graph_schemas import GraphChunk

        graph_chunks = []
        for chunk in chunks:
            entities, relationships = await mock_extract(chunk, full_document_text)
            graph_data = await mock_store_graph(entities, relationships, chunk)

            graph_chunk = GraphChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                rag_type=RAGType.GRAPH,
                entities=entities,
                relationships=relationships,
                graph_data=graph_data,
                entity_count=len(entities),
                relationship_count=len(relationships),
            )
            graph_chunks.append(graph_chunk)

        return graph_chunks

    processor.process_document_chunks = mock_process_chunks

    return processor


@pytest.mark.asyncio
class TestGraphRAGIntegration:
    """Integration tests for complete Graph RAG workflow."""

    async def test_complete_graph_rag_workflow(
        self,
        chroma_dir: Path,
        sample_document_text: str,
        mock_graph_processor_with_ollama,
    ):
        """Test complete workflow: ingest -> process -> query with Graph RAG."""

        # Setup components
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

        vector_store = GraphVectorStore(
            persist_directory=chroma_dir,
            embedding_model=embedding_model,
            graph_processor=mock_graph_processor_with_ollama,
            base_collection_name="test_docs",
        )

        # Step 1: Chunk the document
        chunks = chunker.chunk_text(
            text=sample_document_text,
            document_id="test_doc_1",
        )

        assert len(chunks) > 0

        # Step 2: Create document
        metadata = DocumentMetadata(
            filename="test_company.txt",
            file_size=len(sample_document_text),
            chunk_count=len(chunks),
            status=DocumentStatus.COMPLETED,
        )

        document = Document(
            document_id="test_doc_1",
            metadata=metadata,
            chunks=chunks,
        )

        # Step 3: Process chunks for Graph RAG
        graph_chunks = await mock_graph_processor_with_ollama.process_document_chunks(
            chunks=document.chunks,
            document_metadata=metadata,
            full_document_text=sample_document_text,
        )

        assert len(graph_chunks) > 0

        # Verify entities were extracted
        total_entities = sum(chunk.entity_count for chunk in graph_chunks)
        assert total_entities > 0

        # Step 4: Add to graph vector store
        from openrag.models.graph_schemas import GraphDocument

        graph_doc = GraphDocument(
            document_id=document.document_id,
            metadata=metadata,
            chunks=graph_chunks,
            total_entities=total_entities,
            total_relationships=sum(chunk.relationship_count for chunk in graph_chunks),
        )

        vector_store.add_document(graph_doc, rag_type=RAGType.GRAPH)

        # Step 5: Query the graph
        results = await vector_store.search(
            query="Who is the CEO of Acme Corporation?",
            n_results=5,
            rag_type=RAGType.GRAPH,
            min_similarity=0.1,
        )

        assert len(results) > 0

        # Verify we got relevant results
        result_texts = [chunk.content for chunk, _ in results]
        assert any("John Doe" in text for text in result_texts)
        assert any("Acme Corporation" in text for text in result_texts)

    async def test_parallel_rag_types(
        self,
        chroma_dir: Path,
        sample_document_text: str,
        mock_graph_processor_with_ollama,
    ):
        """Test that all three RAG types work in parallel."""

        # Setup
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

        vector_store = GraphVectorStore(
            persist_directory=chroma_dir,
            embedding_model=embedding_model,
            graph_processor=mock_graph_processor_with_ollama,
            base_collection_name="test_docs",
        )

        # Create document
        chunks = chunker.chunk_text(
            text=sample_document_text,
            document_id="test_doc_1",
        )

        metadata = DocumentMetadata(
            filename="test.txt",
            file_size=len(sample_document_text),
            chunk_count=len(chunks),
            status=DocumentStatus.COMPLETED,
        )

        document = Document(
            document_id="test_doc_1",
            metadata=metadata,
            chunks=chunks,
        )

        # Add to traditional collection
        vector_store.add_document(document, rag_type=RAGType.TRADITIONAL)

        # Add to contextual collection
        from openrag.models.contextual_schemas import ContextualDocument, ContextualDocumentChunk

        contextual_doc = ContextualDocument(
            document_id=document.document_id,
            metadata=metadata,
            chunks=[
                ContextualDocumentChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    chunk_index=c.chunk_index,
                    contextual_content=f"This is from a document about companies. {c.content}",
                )
                for c in document.chunks
            ],
        )

        vector_store.add_document(contextual_doc, rag_type=RAGType.CONTEXTUAL)

        # Add to graph collection
        graph_chunks = await mock_graph_processor_with_ollama.process_document_chunks(
            chunks=document.chunks,
            document_metadata=metadata,
            full_document_text=sample_document_text,
        )

        from openrag.models.graph_schemas import GraphDocument

        graph_doc = GraphDocument(
            document_id=document.document_id,
            metadata=metadata,
            chunks=graph_chunks,
        )

        vector_store.add_document(graph_doc, rag_type=RAGType.GRAPH)

        # Query all three types
        query = "Tell me about Acme Corporation"

        traditional_results = await vector_store.search(
            query=query,
            n_results=3,
            rag_type=RAGType.TRADITIONAL,
        )

        contextual_results = await vector_store.search(
            query=query,
            n_results=3,
            rag_type=RAGType.CONTEXTUAL,
        )

        graph_results = await vector_store.search(
            query=query,
            n_results=3,
            rag_type=RAGType.GRAPH,
        )

        # All should return results
        assert len(traditional_results) > 0
        assert len(contextual_results) > 0
        assert len(graph_results) > 0

        # Verify stats show all three collections
        stats = vector_store.get_stats()
        assert stats["traditional_chunks"] > 0
        assert stats["contextual_chunks"] > 0
        assert stats["graph_chunks"] > 0

    async def test_graph_expansion_improves_results(
        self,
        chroma_dir: Path,
        sample_document_text: str,
        mock_graph_processor_with_ollama,
    ):
        """Test that graph expansion finds related chunks via relationships."""

        # Setup
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)  # Smaller chunks
        embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

        vector_store = GraphVectorStore(
            persist_directory=chroma_dir,
            embedding_model=embedding_model,
            graph_processor=mock_graph_processor_with_ollama,
            base_collection_name="test_docs",
        )

        # Process and add document
        chunks = chunker.chunk_text(
            text=sample_document_text,
            document_id="test_doc_1",
        )

        metadata = DocumentMetadata(
            filename="test.txt",
            file_size=len(sample_document_text),
            chunk_count=len(chunks),
            status=DocumentStatus.COMPLETED,
        )

        document = Document(
            document_id="test_doc_1",
            metadata=metadata,
            chunks=chunks,
        )

        graph_chunks = await mock_graph_processor_with_ollama.process_document_chunks(
            chunks=document.chunks,
            document_metadata=metadata,
            full_document_text=sample_document_text,
        )

        from openrag.models.graph_schemas import GraphDocument

        graph_doc = GraphDocument(
            document_id=document.document_id,
            metadata=metadata,
            chunks=graph_chunks,
        )

        vector_store.add_document(graph_doc, rag_type=RAGType.GRAPH)

        # Mock graph expansion to return additional chunks
        async def mock_expand(chunk_ids, max_hops):
            # Return some additional chunk IDs
            all_chunk_ids = [c.chunk_id for c in graph_chunks]
            # Return chunks not in the initial set
            return [cid for cid in all_chunk_ids if cid not in chunk_ids][:2]

        with patch.object(
            vector_store,
            "_expand_via_graph",
            side_effect=mock_expand,
        ):
            results = await vector_store.search(
                query="CEO",
                n_results=5,
                rag_type=RAGType.GRAPH,
                min_similarity=0.1,
                max_hops=2,
            )

            # Should have results from both vector search and graph expansion
            assert len(results) > 0
