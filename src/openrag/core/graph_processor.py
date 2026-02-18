"""Graph RAG processor for entity extraction and knowledge graph construction using Neo4j."""

import logging
import re
import time
from typing import Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

from ..models.contextual_schemas import RAGType
from ..models.graph_schemas import Entity, GraphChunk, Relationship
from ..models.schemas import DocumentChunk, DocumentMetadata
from .ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)


class GraphProcessorError(Exception):
    """Exception raised during graph processing operations."""

    pass


class GraphProcessor:
    """Processor for extracting entities, relationships, and building knowledge graphs."""

    # Entity extraction prompt template (XML format for structured output)
    ENTITY_EXTRACTION_PROMPT = """<document>
{full_document}
</document>

<chunk>
{chunk_content}
</chunk>

Extract all entities and their relationships from the chunk above.

Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, DATE, EVENT

Provide output in this exact format:
<entities>
- [PERSON] Full Name
- [ORGANIZATION] Company Name
- [LOCATION] City, Country
- [CONCEPT] Technical Term
- [DATE] Time Reference
- [EVENT] Event Name
</entities>

<relationships>
- Entity1 [WORKS_AT] Entity2
- Entity1 [LOCATED_IN] Entity2
- Entity1 [PARTICIPATES_IN] Entity2
</relationships>

Answer only with entities and relationships, nothing else."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        entity_model: str = "llama3.2:3b",
        fallback_enabled: bool = True,
    ):
        """
        Initialize the graph processor with Neo4j connection.

        Args:
            ollama_client: Ollama client for LLM-based entity extraction
            neo4j_uri: Neo4j database URI (e.g., 'neo4j://localhost:7687')
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            entity_model: Ollama model for entity extraction
            fallback_enabled: Use simple pattern matching if Ollama fails
        """
        self.ollama_client = ollama_client
        self.entity_model = entity_model
        self.fallback_enabled = fallback_enabled

        # Neo4j connection
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.driver: Optional[AsyncDriver] = None

        logger.info(
            f"Initialized GraphProcessor with model: {entity_model}, "
            f"Neo4j: {neo4j_uri}"
        )

    async def initialize(self):
        """Initialize Neo4j connection and create indexes."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password)
            )

            # Verify connection
            await self.driver.verify_connectivity()

            # Create indexes and constraints
            async with self.driver.session(database=self.neo4j_database) as session:
                # Entity uniqueness constraint
                await session.run(
                    """
                    CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.name IS UNIQUE
                    """
                )

                # Chunk ID index
                await session.run(
                    """
                    CREATE INDEX chunk_id_index IF NOT EXISTS
                    FOR (c:Chunk) ON (c.chunk_id)
                    """
                )

            logger.info("Neo4j connection initialized and indexes created")

        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            raise GraphProcessorError(f"Neo4j initialization failed: {e}")

    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def extract_entities_and_relationships(
        self,
        chunk: DocumentChunk,
        full_document_text: str,
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract entities and relationships from a chunk using Ollama LLM.

        Args:
            chunk: Document chunk to process
            full_document_text: Full document for context

        Returns:
            Tuple of (entities, relationships)
        """
        try:
            logger.debug(f"Extracting entities from chunk {chunk.chunk_id}")

            # Create prompt
            prompt = self.ENTITY_EXTRACTION_PROMPT.format(
                full_document=full_document_text, chunk_content=chunk.content
            )

            # Generate with Ollama
            response = await self.ollama_client.generate_response(
                prompt=prompt,
                model=self.entity_model,
                temperature=0.2,  # Low temp for consistent extraction
                max_tokens=500,
            )

            # Parse entities and relationships from response
            entities = self._parse_entities(response, chunk.chunk_id)
            relationships = self._parse_relationships(
                response, entities, chunk.chunk_id
            )

            logger.debug(
                f"Extracted {len(entities)} entities and "
                f"{len(relationships)} relationships from chunk {chunk.chunk_id}"
            )

            return entities, relationships

        except OllamaError as e:
            logger.warning(
                f"Ollama extraction failed for chunk {chunk.chunk_id}: {e}"
            )
            if self.fallback_enabled:
                return await self._fallback_entity_extraction(chunk)
            raise GraphProcessorError(f"Entity extraction failed: {e}")

    def _parse_entities(self, response: str, chunk_id: str) -> list[Entity]:
        """Parse entities from LLM response."""
        entities = []

        # Extract entities block
        entities_match = re.search(
            r"<entities>(.*?)</entities>", response, re.DOTALL
        )
        if not entities_match:
            logger.warning("No entities block found in response")
            return entities

        entities_text = entities_match.group(1)

        # Parse each entity line: "- [TYPE] Name"
        for line in entities_text.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("-"):
                continue

            match = re.match(r"-\s*\[(\w+)\]\s+(.+)", line)
            if match:
                entity_type, name = match.groups()
                entities.append(
                    Entity(
                        name=name.strip(),
                        entity_type=entity_type.upper(),
                        source_chunk_id=chunk_id,
                        confidence=0.9,  # LLM extraction
                    )
                )

        return entities

    def _parse_relationships(
        self, response: str, entities: list[Entity], chunk_id: str
    ) -> list[Relationship]:
        """Parse relationships from LLM response."""
        relationships = []

        # Extract relationships block
        rel_match = re.search(
            r"<relationships>(.*?)</relationships>", response, re.DOTALL
        )
        if not rel_match:
            logger.warning("No relationships block found in response")
            return relationships

        rel_text = rel_match.group(1)

        # Create entity name to ID mapping
        entity_map = {e.name: e.entity_id for e in entities}

        # Parse each relationship line: "- Entity1 [REL_TYPE] Entity2"
        for line in rel_text.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("-"):
                continue

            match = re.match(r"-\s*(.+?)\s*\[(\w+)\]\s*(.+)", line)
            if match:
                source_name, rel_type, target_name = match.groups()
                source_name = source_name.strip()
                target_name = target_name.strip()

                # Find entity IDs
                source_id = entity_map.get(source_name)
                target_id = entity_map.get(target_name)

                if source_id and target_id:
                    relationships.append(
                        Relationship(
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            relationship_type=rel_type.upper(),
                            source_chunk_id=chunk_id,
                            confidence=0.9,
                        )
                    )

        return relationships

    async def _fallback_entity_extraction(
        self, chunk: DocumentChunk
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Simple pattern-based entity extraction fallback.

        Extracts:
        - Capitalized words as potential entities
        - Organizations (words ending with Inc., Corp., Ltd.)
        - Dates (basic patterns)
        """
        entities = []
        text = chunk.content

        # Extract capitalized words (potential PERSON or ORGANIZATION)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for name in set(capitalized):
            # Skip common words
            if name.lower() in [
                "the",
                "a",
                "an",
                "this",
                "that",
                "these",
                "those",
            ]:
                continue

            # Check if organization
            entity_type = (
                "ORGANIZATION"
                if any(
                    suffix in name
                    for suffix in ["Inc", "Corp", "Ltd", "LLC", "Company"]
                )
                else "PERSON"
            )

            entities.append(
                Entity(
                    name=name,
                    entity_type=entity_type,
                    source_chunk_id=chunk.chunk_id,
                    confidence=0.5,  # Lower confidence for fallback
                )
            )

        # Extract dates
        dates = re.findall(r"\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)
        for date in set(dates):
            entities.append(
                Entity(
                    name=date,
                    entity_type="DATE",
                    source_chunk_id=chunk.chunk_id,
                    confidence=0.7,
                )
            )

        logger.info(
            f"Fallback extraction: {len(entities)} entities from chunk {chunk.chunk_id}"
        )

        # No relationships in fallback mode
        return entities, []

    async def store_graph_in_neo4j(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        chunk: DocumentChunk,
    ) -> dict:
        """
        Store entities and relationships in Neo4j graph database.

        Args:
            entities: Entities to store
            relationships: Relationships to store
            chunk: Source chunk

        Returns:
            Dict with Neo4j node/relationship IDs for later retrieval
        """
        if not self.driver:
            raise GraphProcessorError("Neo4j driver not initialized")

        graph_data = {"entity_ids": {}, "relationship_ids": []}

        async with self.driver.session(database=self.neo4j_database) as session:
            # Create chunk node
            await session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.content = $content,
                    c.document_id = $document_id
                """,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                document_id=chunk.document_id,
            )

            # Create entity nodes
            for entity in entities:
                result = await session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.confidence = $confidence
                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    RETURN id(e) as entity_id
                    """,
                    name=entity.name,
                    type=entity.entity_type,
                    confidence=entity.confidence,
                    chunk_id=chunk.chunk_id,
                )

                record = await result.single()
                if record:
                    graph_data["entity_ids"][entity.entity_id] = record["entity_id"]

            # Create relationships
            for rel in relationships:
                source_neo4j_id = graph_data["entity_ids"].get(
                    rel.source_entity_id
                )
                target_neo4j_id = graph_data["entity_ids"].get(
                    rel.target_entity_id
                )

                if source_neo4j_id and target_neo4j_id:
                    result = await session.run(
                        f"""
                        MATCH (source:Entity) WHERE id(source) = $source_id
                        MATCH (target:Entity) WHERE id(target) = $target_id
                        MERGE (source)-[r:{rel.relationship_type}]->(target)
                        SET r.confidence = $confidence
                        RETURN id(r) as rel_id
                        """,
                        source_id=source_neo4j_id,
                        target_id=target_neo4j_id,
                        confidence=rel.confidence,
                    )

                    record = await result.single()
                    if record:
                        graph_data["relationship_ids"].append(record["rel_id"])

        logger.info(
            f"Stored {len(entities)} entities and {len(relationships)} relationships in Neo4j"
        )

        return graph_data

    async def process_document_chunks(
        self,
        chunks: list[DocumentChunk],
        document_metadata: DocumentMetadata,
        full_document_text: str,
    ) -> list[GraphChunk]:
        """
        Process all chunks of a document: extract entities, build graph, store in Neo4j.

        Args:
            chunks: Document chunks to process
            document_metadata: Document metadata
            full_document_text: Full document text for context

        Returns:
            List of GraphChunk objects with entities and relationships
        """
        start_time = time.time()
        graph_chunks = []

        logger.info(
            f"Processing {len(chunks)} chunks for graph extraction: "
            f"{document_metadata.filename}"
        )

        for chunk in chunks:
            try:
                # Extract entities and relationships
                entities, relationships = await self.extract_entities_and_relationships(
                    chunk=chunk, full_document_text=full_document_text
                )

                # Store in Neo4j
                graph_data = await self.store_graph_in_neo4j(
                    entities=entities, relationships=relationships, chunk=chunk
                )

                # Create GraphChunk
                graph_chunk = GraphChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                    rag_type=RAGType.GRAPH,
                    entities=entities,
                    relationships=relationships,
                    graph_data=graph_data,
                    entity_count=len(entities),
                    relationship_count=len(relationships),
                )

                graph_chunks.append(graph_chunk)

            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
                # Continue with other chunks
                continue

        processing_time = time.time() - start_time
        logger.info(
            f"Graph processing completed in {processing_time:.2f}s: "
            f"{len(graph_chunks)}/{len(chunks)} chunks processed"
        )

        return graph_chunks

    def _generate_graph_context(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> str:
        """
        Generate human-readable context from entities and relationships.

        Example: "John Doe works at Acme Corp, which is located in New York"
        """
        if not entities:
            return ""

        # Build relationship map
        rel_map = {}
        for rel in relationships:
            if rel.source_entity_id not in rel_map:
                rel_map[rel.source_entity_id] = []
            rel_map[rel.source_entity_id].append(rel)

        # Generate context sentences
        context_parts = []
        entity_map = {e.entity_id: e.name for e in entities}

        for entity in entities[:3]:  # Top 3 entities
            rels = rel_map.get(entity.entity_id, [])
            if rels:
                rel = rels[0]  # First relationship
                target_name = entity_map.get(rel.target_entity_id, "unknown")
                rel_type = rel.relationship_type.replace("_", " ").lower()
                context_parts.append(
                    f"{entity.name} {rel_type} {target_name}"
                )

        return "; ".join(context_parts) if context_parts else ""
