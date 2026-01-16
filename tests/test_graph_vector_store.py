"""Unit tests for Graph Vector Store."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openrag.core.embedder import EmbeddingModel
from openrag.core.graph_processor import GraphProcessor
from openrag.core.graph_vector_store import GraphVectorStore
from openrag.models.contextual_schemas import RAGType
from openrag.models.graph_schemas import Entity, GraphChunk, GraphDocument, Relationship
from openrag.models.schemas import DocumentChunk, DocumentMetadata, DocumentStatus


@pytest.fixture
def mock_graph_processor():
    """Create a mock GraphProcessor."""
    processor = MagicMock(spec=GraphProcessor)
    processor.driver = MagicMock()
    return processor


@pytest.fixture
def graph_vector_store(chroma_dir: Path, embedding_model: EmbeddingModel, mock_graph_processor):
    """Create a GraphVectorStore instance for testing."""
    return GraphVectorStore(
        persist_directory=chroma_dir,
        embedding_model=embedding_model,
        graph_processor=mock_graph_processor,
        base_collection_name="test_documents",
        max_batch_size=100,
    )


@pytest.fixture
def sample_graph_document():
    """Create a sample GraphDocument with entities and relationships."""
    metadata = DocumentMetadata(
        filename="test.txt",
        file_size=1000,
        chunk_count=2,
        status=DocumentStatus.COMPLETED,
    )

    # Create entities
    entities_chunk1 = [
        Entity(
            entity_id="entity_1",
            name="John Doe",
            entity_type="PERSON",
            source_chunk_id="chunk_1",
            confidence=0.95,
        ),
        Entity(
            entity_id="entity_2",
            name="Acme Corp",
            entity_type="ORGANIZATION",
            source_chunk_id="chunk_1",
            confidence=0.90,
        ),
    ]

    relationships_chunk1 = [
        Relationship(
            relationship_id="rel_1",
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            relationship_type="WORKS_AT",
            source_chunk_id="chunk_1",
            confidence=0.90,
        ),
    ]

    # Create graph chunks
    chunk1 = GraphChunk(
        chunk_id="chunk_1",
        document_id="doc_123",
        content="John Doe works at Acme Corp.",
        chunk_index=0,
        rag_type=RAGType.GRAPH,
        entities=entities_chunk1,
        relationships=relationships_chunk1,
        entity_count=2,
        relationship_count=1,
        graph_data={"entity_ids": {"entity_1": 1, "entity_2": 2}, "relationship_ids": [1]},
    )

    chunk2 = GraphChunk(
        chunk_id="chunk_2",
        document_id="doc_123",
        content="Acme Corp is based in New York.",
        chunk_index=1,
        rag_type=RAGType.GRAPH,
        entities=[],
        relationships=[],
        entity_count=0,
        relationship_count=0,
    )

    document = GraphDocument(
        document_id="doc_123",
        metadata=metadata,
        chunks=[chunk1, chunk2],
        total_entities=2,
        total_relationships=1,
        entity_types={"PERSON": 1, "ORGANIZATION": 1},
    )

    return document


@pytest.mark.asyncio
class TestGraphVectorStore:
    """Test suite for GraphVectorStore."""

    def test_initialization(self, graph_vector_store):
        """Test GraphVectorStore initialization."""
        assert graph_vector_store is not None
        assert RAGType.TRADITIONAL in graph_vector_store._collections
        assert RAGType.CONTEXTUAL in graph_vector_store._collections
        assert RAGType.GRAPH in graph_vector_store._collections

    def test_add_graph_document(self, graph_vector_store, sample_graph_document):
        """Test adding a graph document to the store."""
        graph_vector_store.add_document(sample_graph_document, rag_type=RAGType.GRAPH)

        # Verify collection has chunks
        collection = graph_vector_store._collections[RAGType.GRAPH]
        assert collection.count() == 2

    def test_add_document_with_graph_metadata(
        self, graph_vector_store, sample_graph_document
    ):
        """Test that graph metadata is properly stored."""
        graph_vector_store.add_document(sample_graph_document, rag_type=RAGType.GRAPH)

        # Get the stored data
        collection = graph_vector_store._collections[RAGType.GRAPH]
        result = collection.get(ids=["chunk_1"], include=["metadatas"])

        metadata = result["metadatas"][0]
        assert metadata["rag_type"] == "graph"
        assert metadata["entity_count"] == 2
        assert metadata["relationship_count"] == 1
        assert "graph_data_json" in metadata

    async def test_search_graph_without_processor(
        self, chroma_dir, embedding_model, sample_graph_document
    ):
        """Test graph search falls back to vector search when no processor."""
        # Create store without graph processor
        store = GraphVectorStore(
            persist_directory=chroma_dir,
            embedding_model=embedding_model,
            graph_processor=None,
            base_collection_name="test_documents",
        )

        # Add document
        store.add_document(sample_graph_document, rag_type=RAGType.GRAPH)

        # Search should work (fallback to vector search)
        results = await store.search(
            query="John Doe works",
            n_results=5,
            rag_type=RAGType.GRAPH,
            min_similarity=0.1,
        )

        assert len(results) > 0

    async def test_graph_search_with_expansion(
        self, graph_vector_store, sample_graph_document
    ):
        """Test graph search with graph expansion."""
        # Add document
        graph_vector_store.add_document(sample_graph_document, rag_type=RAGType.GRAPH)

        # Mock graph expansion
        with patch.object(
            graph_vector_store,
            "_expand_via_graph",
            new_callable=AsyncMock,
            return_value=["chunk_2"],
        ):
            results = await graph_vector_store.search(
                query="Acme Corp",
                n_results=5,
                rag_type=RAGType.GRAPH,
                min_similarity=0.1,
                max_hops=2,
            )

            # Should have results from both vector search and graph expansion
            assert len(results) > 0

    async def test_expand_via_graph(self, graph_vector_store, mock_graph_processor):
        """Test graph expansion via Neo4j."""
        # Mock Neo4j session and query results
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"chunk_id": "chunk_2"},
            {"chunk_id": "chunk_3"},
        ]
        mock_session.run.return_value = mock_result

        mock_graph_processor.driver.session.return_value.__aenter__.return_value = (
            mock_session
        )
        mock_graph_processor.neo4j_database = "neo4j"

        expanded_ids = await graph_vector_store._expand_via_graph(
            chunk_ids=["chunk_1"],
            max_hops=2,
        )

        assert len(expanded_ids) == 2
        assert "chunk_2" in expanded_ids
        assert "chunk_3" in expanded_ids

    async def test_expand_via_graph_no_processor(self, chroma_dir, embedding_model):
        """Test graph expansion returns empty when no processor."""
        store = GraphVectorStore(
            persist_directory=chroma_dir,
            embedding_model=embedding_model,
            graph_processor=None,
        )

        expanded_ids = await store._expand_via_graph(
            chunk_ids=["chunk_1"],
            max_hops=2,
        )

        assert len(expanded_ids) == 0

    async def test_retrieve_chunks_by_ids(
        self, graph_vector_store, sample_graph_document
    ):
        """Test retrieving specific chunks by their IDs."""
        # Add document first
        graph_vector_store.add_document(sample_graph_document, rag_type=RAGType.GRAPH)

        # Retrieve chunks
        chunks = await graph_vector_store._retrieve_chunks_by_ids(
            chunk_ids=["chunk_1", "chunk_2"]
        )

        assert len(chunks) == 2
        chunk_ids = [c.chunk_id for c in chunks]
        assert "chunk_1" in chunk_ids
        assert "chunk_2" in chunk_ids

    async def test_retrieve_chunks_empty_list(self, graph_vector_store):
        """Test retrieving chunks with empty ID list."""
        chunks = await graph_vector_store._retrieve_chunks_by_ids(chunk_ids=[])
        assert len(chunks) == 0

    def test_get_stats_with_graph(self, graph_vector_store, sample_graph_document):
        """Test getting statistics including graph collection."""
        # Add document
        graph_vector_store.add_document(sample_graph_document, rag_type=RAGType.GRAPH)

        stats = graph_vector_store.get_stats()

        assert "graph_chunks" in stats
        assert stats["graph_chunks"] == 2
        assert "total_chunks" in stats
        assert "traditional_chunks" in stats
        assert "contextual_chunks" in stats

    async def test_search_traditional_still_works(
        self, graph_vector_store, sample_graph_document
    ):
        """Test that traditional search still works in GraphVectorStore."""
        # Convert to regular document and add to traditional collection
        from openrag.models.schemas import Document

        doc = Document(
            document_id=sample_graph_document.document_id,
            metadata=sample_graph_document.metadata,
            chunks=[
                DocumentChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    chunk_index=c.chunk_index,
                )
                for c in sample_graph_document.chunks
            ],
        )

        graph_vector_store.add_document(doc, rag_type=RAGType.TRADITIONAL)

        results = await graph_vector_store.search(
            query="John Doe",
            n_results=5,
            rag_type=RAGType.TRADITIONAL,
            min_similarity=0.1,
        )

        assert len(results) > 0

    async def test_search_contextual_still_works(
        self, graph_vector_store, sample_graph_document
    ):
        """Test that contextual search still works in GraphVectorStore."""
        # Convert to contextual document
        from openrag.models.contextual_schemas import ContextualDocument, ContextualDocumentChunk

        doc = ContextualDocument(
            document_id=sample_graph_document.document_id,
            metadata=sample_graph_document.metadata,
            chunks=[
                ContextualDocumentChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    chunk_index=c.chunk_index,
                    contextual_content=f"Context: {c.content}",
                )
                for c in sample_graph_document.chunks
            ],
        )

        graph_vector_store.add_document(doc, rag_type=RAGType.CONTEXTUAL)

        results = await graph_vector_store.search(
            query="Acme Corp",
            n_results=5,
            rag_type=RAGType.CONTEXTUAL,
            min_similarity=0.1,
        )

        assert len(results) > 0
