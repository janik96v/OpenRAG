"""Graph RAG specific data models.

This module defines Pydantic models for Graph RAG functionality, including:
- Entity: Named entities extracted from documents
- Relationship: Connections between entities
- GraphChunk: Document chunks enhanced with graph information
- GraphDocument: Complete document with graph-enhanced chunks
- GraphQueryResult: Query results with graph context
"""

from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .contextual_schemas import ContextualChunk
from .schemas import DocumentChunk


class Entity(BaseModel):
    """Represents an extracted entity from a document chunk.

    Entities are named elements (people, organizations, locations, concepts, etc.)
    identified in the text that form nodes in the knowledge graph.

    Attributes:
        entity_id: Unique identifier for the entity
        name: Entity name or mention as it appears in text
        entity_type: Classification (PERSON, ORGANIZATION, LOCATION, CONCEPT, DATE, EVENT)
        attributes: Additional metadata about the entity
        confidence: Extraction confidence score (0.0 to 1.0)
        source_chunk_id: ID of the chunk this entity was extracted from
    """

    entity_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Entity name/mention")
    entity_type: str = Field(
        ...,
        description="Entity type: PERSON, ORGANIZATION, LOCATION, CONCEPT, DATE, EVENT",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Additional attributes"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    source_chunk_id: str = Field(
        ..., description="Chunk this entity was extracted from"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "John Doe",
                "entity_type": "PERSON",
                "attributes": {"role": "researcher"},
                "confidence": 0.95,
                "source_chunk_id": "chunk_123",
            }
        }
    )


class Relationship(BaseModel):
    """Represents a relationship between two entities.

    Relationships form edges in the knowledge graph, connecting entities
    and enabling multi-hop reasoning.

    Attributes:
        relationship_id: Unique identifier for the relationship
        source_entity_id: ID of the source entity
        target_entity_id: ID of the target entity
        relationship_type: Type of relationship (WORKS_AT, LOCATED_IN, COLLABORATES_WITH, etc.)
        attributes: Additional metadata about the relationship
        confidence: Extraction confidence score (0.0 to 1.0)
        source_chunk_id: ID of the chunk this relationship was extracted from
    """

    relationship_id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(
        ..., description="Relationship type: WORKS_AT, LOCATED_IN, etc."
    )
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: str = Field(
        ..., description="Chunk this relationship was extracted from"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "relationship_id": "550e8400-e29b-41d4-a716-446655440001",
                "source_entity_id": "entity_1",
                "target_entity_id": "entity_2",
                "relationship_type": "WORKS_AT",
                "attributes": {"since": "2023"},
                "confidence": 0.90,
                "source_chunk_id": "chunk_123",
            }
        }
    )


class GraphChunk(ContextualChunk):
    """Document chunk enhanced with graph information.

    Extends ContextualChunk to add entity and relationship data extracted
    from the chunk. Stores a serialized subgraph for efficient querying.

    Attributes:
        entities: List of entities found in this chunk
        relationships: List of relationships found in this chunk
        graph_data: Serialized NetworkX graph data (node-link format)
        entity_count: Number of entities in this chunk
        relationship_count: Number of relationships in this chunk
    """

    entities: list[Entity] = Field(
        default_factory=list, description="Entities in this chunk"
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="Relationships in this chunk"
    )
    graph_data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Serialized graph data for storage (Neo4j node/relationship IDs)",
    )

    # Graph-specific metadata
    entity_count: int = Field(default=0, description="Number of entities")
    relationship_count: int = Field(
        default=0, description="Number of relationships"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "chunk_123",
                "content": "John Doe works at Acme Corporation in New York.",
                "entities": [
                    {"name": "John Doe", "entity_type": "PERSON"},
                    {"name": "Acme Corporation", "entity_type": "ORGANIZATION"},
                    {"name": "New York", "entity_type": "LOCATION"},
                ],
                "entity_count": 3,
                "relationship_count": 2,
            }
        }
    )


class GraphDocument(BaseModel):
    """Document with graph-enhanced chunks.

    Represents a complete document that has been processed for Graph RAG,
    including all extracted entities, relationships, and graph statistics.

    Attributes:
        document_id: Unique document identifier
        metadata: Document metadata (filename, source, etc.)
        chunks: List of graph-enhanced chunks
        total_entities: Total number of unique entities across all chunks
        total_relationships: Total number of relationships across all chunks
        entity_types: Count of entities by type (e.g., {"PERSON": 5, "ORG": 3})
    """

    document_id: str
    metadata: Any  # DocumentMetadata from schemas.py
    chunks: list[GraphChunk] = Field(default_factory=list)

    # Document-level graph summary
    total_entities: int = Field(default=0)
    total_relationships: int = Field(default=0)
    entity_types: dict[str, int] = Field(
        default_factory=dict, description="Entity type counts"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_456",
                "total_entities": 15,
                "total_relationships": 12,
                "entity_types": {
                    "PERSON": 5,
                    "ORGANIZATION": 4,
                    "LOCATION": 3,
                    "CONCEPT": 3,
                },
            }
        }
    )


class GraphQueryResult(BaseModel):
    """Result from graph-enhanced query including graph context.

    Combines traditional chunk retrieval with graph-based context,
    providing entity relationships and multi-hop reasoning paths.

    Attributes:
        chunk: The retrieved document chunk
        similarity_score: Vector similarity score
        related_entities: Key entities found in this chunk
        graph_context: Human-readable description of entity relationships
        multi_hop_path: Sequence of entities traversed to reach this result
    """

    chunk: DocumentChunk
    similarity_score: float

    # Graph context
    related_entities: list[Entity] = Field(
        default_factory=list, description="Key entities in chunk"
    )
    graph_context: Optional[str] = Field(
        default=None,
        description="Human-readable graph context (e.g., 'John Doe works at Acme Corp in NYC')",
    )
    multi_hop_path: Optional[list[str]] = Field(
        default=None, description="Entity path for multi-hop queries"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "similarity_score": 0.87,
                "related_entities": [
                    {"name": "John Doe", "entity_type": "PERSON"},
                    {"name": "Acme Corp", "entity_type": "ORGANIZATION"},
                ],
                "graph_context": "John Doe works at Acme Corp, which is located in New York",
                "multi_hop_path": ["John Doe", "WORKS_AT", "Acme Corp", "LOCATED_IN", "New York"],
            }
        }
    )
