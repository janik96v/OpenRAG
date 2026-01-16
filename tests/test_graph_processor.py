"""Unit tests for Graph RAG processor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openrag.core.graph_processor import GraphProcessor, GraphProcessorError
from openrag.core.ollama_client import OllamaClient, OllamaError
from openrag.models.graph_schemas import Entity, Relationship
from openrag.models.schemas import DocumentChunk


@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client."""
    client = MagicMock(spec=OllamaClient)
    client.generate_response = AsyncMock()
    return client


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session = MagicMock(return_value=session)
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
async def graph_processor(mock_ollama_client):
    """Create a GraphProcessor instance for testing."""
    processor = GraphProcessor(
        ollama_client=mock_ollama_client,
        neo4j_uri="neo4j://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        entity_model="llama3.2:3b",
        fallback_enabled=True,
    )
    return processor


@pytest.fixture
def sample_chunk():
    """Create a sample DocumentChunk."""
    return DocumentChunk(
        chunk_id="chunk_123",
        document_id="doc_456",
        content="John Doe works at Acme Corporation in New York.",
        chunk_index=0,
    )


@pytest.mark.asyncio
class TestGraphProcessor:
    """Test suite for GraphProcessor."""

    async def test_initialization(self, graph_processor, mock_neo4j_driver):
        """Test GraphProcessor initialization."""
        with patch(
            "openrag.core.graph_processor.AsyncGraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            await graph_processor.initialize()
            assert graph_processor.driver is not None
            mock_neo4j_driver.verify_connectivity.assert_called_once()

    async def test_initialization_failure(self, graph_processor):
        """Test GraphProcessor initialization failure."""
        with patch(
            "openrag.core.graph_processor.AsyncGraphDatabase.driver",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(GraphProcessorError, match="Neo4j initialization failed"):
                await graph_processor.initialize()

    async def test_parse_entities(self, graph_processor):
        """Test entity parsing from LLM response."""
        response = """
        <entities>
        - [PERSON] John Doe
        - [ORGANIZATION] Acme Corporation
        - [LOCATION] New York
        </entities>
        """

        entities = graph_processor._parse_entities(response, "chunk_123")

        assert len(entities) == 3
        assert entities[0].name == "John Doe"
        assert entities[0].entity_type == "PERSON"
        assert entities[0].source_chunk_id == "chunk_123"
        assert entities[1].name == "Acme Corporation"
        assert entities[1].entity_type == "ORGANIZATION"
        assert entities[2].name == "New York"
        assert entities[2].entity_type == "LOCATION"

    async def test_parse_entities_empty_response(self, graph_processor):
        """Test entity parsing with empty response."""
        response = "No entities found"
        entities = graph_processor._parse_entities(response, "chunk_123")
        assert len(entities) == 0

    async def test_parse_relationships(self, graph_processor):
        """Test relationship parsing from LLM response."""
        # First create entities
        entities = [
            Entity(
                entity_id="entity_1",
                name="John Doe",
                entity_type="PERSON",
                source_chunk_id="chunk_123",
            ),
            Entity(
                entity_id="entity_2",
                name="Acme Corporation",
                entity_type="ORGANIZATION",
                source_chunk_id="chunk_123",
            ),
        ]

        response = """
        <relationships>
        - John Doe [WORKS_AT] Acme Corporation
        </relationships>
        """

        relationships = graph_processor._parse_relationships(
            response, entities, "chunk_123"
        )

        assert len(relationships) == 1
        assert relationships[0].source_entity_id == "entity_1"
        assert relationships[0].target_entity_id == "entity_2"
        assert relationships[0].relationship_type == "WORKS_AT"

    async def test_parse_relationships_no_match(self, graph_processor):
        """Test relationship parsing when entities don't match."""
        entities = [
            Entity(
                entity_id="entity_1",
                name="John Doe",
                entity_type="PERSON",
                source_chunk_id="chunk_123",
            ),
        ]

        response = """
        <relationships>
        - Unknown Person [WORKS_AT] Unknown Company
        </relationships>
        """

        relationships = graph_processor._parse_relationships(
            response, entities, "chunk_123"
        )

        # Should be empty because entities don't exist
        assert len(relationships) == 0

    async def test_extract_entities_success(
        self, graph_processor, mock_ollama_client, sample_chunk
    ):
        """Test successful entity extraction."""
        mock_ollama_client.generate_response.return_value = """
        <entities>
        - [PERSON] John Doe
        - [ORGANIZATION] Acme Corporation
        </entities>
        <relationships>
        - John Doe [WORKS_AT] Acme Corporation
        </relationships>
        """

        entities, relationships = await graph_processor.extract_entities_and_relationships(
            chunk=sample_chunk,
            full_document_text="Sample document text",
        )

        assert len(entities) == 2
        assert len(relationships) == 1
        mock_ollama_client.generate_response.assert_called_once()

    async def test_extract_entities_ollama_failure_with_fallback(
        self, graph_processor, mock_ollama_client, sample_chunk
    ):
        """Test entity extraction fallback when Ollama fails."""
        mock_ollama_client.generate_response.side_effect = OllamaError(
            "Connection failed"
        )

        entities, relationships = await graph_processor.extract_entities_and_relationships(
            chunk=sample_chunk,
            full_document_text="Sample document text",
        )

        # Fallback should extract capitalized words
        assert len(entities) > 0
        # Check that we got some entities from fallback
        entity_names = [e.name for e in entities]
        assert "John Doe" in entity_names or "Acme Corporation" in entity_names

    async def test_fallback_entity_extraction(self, graph_processor, sample_chunk):
        """Test fallback entity extraction with pattern matching."""
        entities, relationships = await graph_processor._fallback_entity_extraction(
            sample_chunk
        )

        # Should extract capitalized words
        assert len(entities) > 0

        # Check for entities with confidence < 1.0 (fallback indicator)
        assert all(e.confidence <= 0.7 for e in entities)

        # No relationships in fallback mode
        assert len(relationships) == 0

    async def test_store_graph_in_neo4j(
        self, graph_processor, mock_neo4j_driver, sample_chunk
    ):
        """Test storing graph data in Neo4j."""
        entities = [
            Entity(
                entity_id="entity_1",
                name="John Doe",
                entity_type="PERSON",
                source_chunk_id=sample_chunk.chunk_id,
            ),
            Entity(
                entity_id="entity_2",
                name="Acme Corp",
                entity_type="ORGANIZATION",
                source_chunk_id=sample_chunk.chunk_id,
            ),
        ]

        relationships = [
            Relationship(
                relationship_id="rel_1",
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                relationship_type="WORKS_AT",
                source_chunk_id=sample_chunk.chunk_id,
            ),
        ]

        # Mock Neo4j session and results
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single.return_value = {"entity_id": 123, "rel_id": 456}
        mock_session.run.return_value = mock_result

        graph_processor.driver = mock_neo4j_driver
        mock_neo4j_driver.session.return_value.__aenter__.return_value = mock_session

        graph_data = await graph_processor.store_graph_in_neo4j(
            entities=entities,
            relationships=relationships,
            chunk=sample_chunk,
        )

        # Should have entity and relationship IDs
        assert "entity_ids" in graph_data
        assert "relationship_ids" in graph_data

    async def test_close_connection(self, graph_processor, mock_neo4j_driver):
        """Test closing Neo4j connection."""
        graph_processor.driver = mock_neo4j_driver
        await graph_processor.close()
        mock_neo4j_driver.close.assert_called_once()
