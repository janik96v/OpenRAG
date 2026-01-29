"""Graph-enhanced vector store managing traditional, contextual, and graph RAG collections."""

import logging
from pathlib import Path
from typing import Any, Optional

from ..models.contextual_schemas import RAGType
from ..models.graph_schemas import GraphChunk, GraphDocument
from ..models.schemas import Document, DocumentChunk
from .contextual_vector_store import ContextualVectorStore
from .embedder import EmbeddingModel
from .graph_processor import GraphProcessor
from .vector_store import VectorStoreError

logger = logging.getLogger(__name__)


class GraphVectorStore(ContextualVectorStore):
    """
    Extended vector store managing traditional, contextual, and graph RAG collections.

    Architecture:
    - Three collections: {base}_traditional_v1, {base}_contextual_v1, {base}_graph_v1
    - Graph data stored in Neo4j
    - Vector embeddings for graph chunks stored in ChromaDB
    - Hybrid search combining vector similarity and graph traversal
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: EmbeddingModel,
        graph_processor: Optional[GraphProcessor] = None,
        base_collection_name: str = "documents",
        max_batch_size: int = 10000,
    ):
        """
        Initialize with traditional, contextual, and graph collections.

        Args:
            persist_directory: Directory for persistent ChromaDB storage
            embedding_model: Embedding model instance
            graph_processor: GraphProcessor for entity extraction (optional)
            base_collection_name: Base name for collections
            max_batch_size: Maximum batch size for ChromaDB operations
        """
        # Initialize parent (ContextualVectorStore)
        super().__init__(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            base_collection_name=base_collection_name,
            max_batch_size=max_batch_size,
        )

        self.graph_processor = graph_processor

        # Initialize graph collection
        self._ensure_graph_collection()

        logger.info(
            "Initialized GraphVectorStore with traditional, contextual, and graph collections"
        )

    def _ensure_graph_collection(self) -> None:
        """Create or get graph RAG collection."""
        try:
            graph_name = f"{self.base_collection_name}_graph_v1"

            self._collections[RAGType.GRAPH] = self.client.get_or_create_collection(
                name=graph_name,
                metadata={
                    "description": "Graph RAG document chunks with entity relationships",
                    "rag_type": "graph",
                },
            )

            logger.info(
                f"Initialized graph collection '{graph_name}' "
                f"with {self._collections[RAGType.GRAPH].count()} existing chunks"
            )

        except Exception as e:
            logger.error(f"Failed to initialize graph collection: {e}")
            raise VectorStoreError(f"Failed to initialize graph collection: {e}") from e

    def add_document(
        self,
        document: Document | GraphDocument,
        rag_type: RAGType = RAGType.TRADITIONAL,
    ) -> None:
        """
        Add document to specified collection.

        For Graph RAG:
        - Embeds original content
        - Stores graph metadata (entity counts, Neo4j IDs)
        - Graph structure itself is in Neo4j (not ChromaDB)

        Args:
            document: Document or GraphDocument to add
            rag_type: Which collection to add to

        Raises:
            VectorStoreError: If adding document fails
        """
        try:
            if not document.chunks:
                logger.warning(f"Document {document.document_id} has no chunks to add")
                return

            collection = self._collections[rag_type]

            # Determine content to embed
            if rag_type == RAGType.GRAPH:
                # For graph RAG, embed original content (not contextual)
                # Graph structure is in Neo4j, not in embeddings
                chunk_contents = [chunk.content for chunk in document.chunks]
                logger.info(
                    f"Generating graph RAG embeddings for {len(document.chunks)} chunks "
                    f"from document {document.metadata.filename}"
                )
            elif rag_type == RAGType.CONTEXTUAL:
                chunk_contents = [
                    getattr(chunk, "contextual_content", None) or chunk.content
                    for chunk in document.chunks
                ]
                logger.info(
                    f"Generating contextual embeddings for {len(document.chunks)} chunks "
                    f"from document {document.metadata.filename}"
                )
            else:
                chunk_contents = [chunk.content for chunk in document.chunks]
                logger.info(
                    f"Generating traditional embeddings for {len(document.chunks)} chunks "
                    f"from document {document.metadata.filename}"
                )

            # Generate embeddings
            embeddings = self.embedding_model.embed_texts(chunk_contents)

            # Prepare data for ChromaDB
            ids: list[str] = []
            metadatas: list[dict[str, Any]] = []
            documents: list[str] = []

            for chunk, embedding in zip(document.chunks, embeddings):
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)

                # Build metadata
                metadata = {
                    "document_id": document.document_id,
                    "filename": document.metadata.filename,
                    "chunk_index": chunk.chunk_index,
                    "file_size": document.metadata.file_size,
                    "created_at": chunk.created_at.isoformat(),
                    "rag_type": rag_type.value,
                }

                # Add graph-specific metadata
                if rag_type == RAGType.GRAPH and isinstance(chunk, GraphChunk):
                    metadata["entity_count"] = chunk.entity_count
                    metadata["relationship_count"] = chunk.relationship_count

                    # Store Neo4j node/relationship IDs for later retrieval
                    if chunk.graph_data:
                        metadata["has_graph_data"] = True
                        # Store entity/relationship IDs as JSON string
                        import json

                        metadata["graph_data_json"] = json.dumps(chunk.graph_data)

                # Add any additional chunk metadata
                for key, value in chunk.metadata.items():
                    if value is not None:
                        metadata[key] = self._validate_metadata_value(value)

                # Validate metadata before adding
                metadata = self._validate_metadata_for_chromadb(metadata)
                metadatas.append(metadata)

            # Add to collection in batches
            batch_size = min(self.max_batch_size, len(ids))
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                batch_documents = documents[i : i + batch_size]

                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents,
                )

            logger.info(
                f"Successfully added {len(document.chunks)} chunks to {rag_type.value} collection"
            )

        except Exception as e:
            logger.error(
                f"Failed to add document to {rag_type.value} collection: {e}"
            )
            raise VectorStoreError(f"Failed to add document: {e}") from e

    async def search(
        self,
        query: str,
        n_results: int = 5,
        rag_type: RAGType = RAGType.TRADITIONAL,
        min_similarity: float = 0.1,
        max_hops: int = 2,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search for relevant chunks with optional graph traversal.

        For Graph RAG:
        1. Vector search for initial chunks
        2. Extract entities from query (if GraphProcessor available)
        3. Traverse graph to find related entities/chunks
        4. Combine and re-rank results

        Args:
            query: Search query text
            n_results: Number of results to return
            rag_type: Which collection to search
            min_similarity: Minimum similarity threshold
            max_hops: Maximum graph traversal hops (Graph RAG only)

        Returns:
            List of (chunk, similarity_score) tuples
        """
        try:
            # For Graph RAG with processor, use hybrid search
            if rag_type == RAGType.GRAPH and self.graph_processor:
                return await self._graph_search(
                    query=query,
                    n_results=n_results,
                    min_similarity=min_similarity,
                    max_hops=max_hops,
                )

            # Otherwise, standard vector search
            collection = self._collections[rag_type]

            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)

            # Search collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            # Convert to chunks
            chunks_with_scores = []

            if results["ids"] and results["ids"][0]:
                for idx in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][idx]
                    content = results["documents"][0][idx]
                    metadata = results["metadatas"][0][idx]
                    distance = results["distances"][0][idx]

                    # Convert distance to similarity (cosine distance to similarity)
                    similarity = 1 - distance

                    if similarity >= min_similarity:
                        chunk = DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=metadata.get("document_id", ""),
                            content=content,
                            chunk_index=metadata.get("chunk_index", 0),
                            metadata=metadata,
                        )
                        chunks_with_scores.append((chunk, float(similarity)))

            logger.info(
                f"Found {len(chunks_with_scores)} results for query in {rag_type.value} collection"
            )

            return chunks_with_scores

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e

    async def _graph_search(
        self,
        query: str,
        n_results: int,
        min_similarity: float,
        max_hops: int,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Hybrid graph search: vector similarity + graph traversal.

        Strategy:
        1. Vector search for initial relevant chunks
        2. Extract entities from those chunks
        3. Traverse Neo4j graph to find related entities
        4. Retrieve chunks connected to those entities
        5. Combine and re-rank
        """
        if not self.graph_processor or not self.graph_processor.driver:
            logger.warning("GraphProcessor not available, falling back to vector search")
            return await self.search(
                query=query,
                n_results=n_results,
                rag_type=RAGType.GRAPH,
                min_similarity=min_similarity,
            )

        # Step 1: Vector search for initial chunks
        initial_results = await self.search(
            query=query,
            n_results=n_results * 2,  # Get more for expansion
            rag_type=RAGType.GRAPH,
            min_similarity=min_similarity,
        )

        if not initial_results:
            return []

        # Step 2: Extract chunk IDs
        chunk_ids = [chunk.chunk_id for chunk, _ in initial_results]

        # Step 3: Query Neo4j for graph expansion
        expanded_chunk_ids = await self._expand_via_graph(
            chunk_ids=chunk_ids, max_hops=max_hops
        )

        # Step 4: Retrieve expanded chunks from ChromaDB
        all_chunks = await self._retrieve_chunks_by_ids(
            chunk_ids=list(set(chunk_ids + expanded_chunk_ids))
        )

        # Step 5: Re-rank (keep original vector scores for initial results)
        # New chunks get lower scores
        result_map = {chunk.chunk_id: score for chunk, score in initial_results}
        final_results = []

        for chunk in all_chunks:
            if chunk.chunk_id in result_map:
                final_results.append((chunk, result_map[chunk.chunk_id]))
            else:
                # Graph-expanded chunks get lower score
                final_results.append((chunk, min_similarity * 0.8))

        # Sort by score and limit
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:n_results]

    async def _expand_via_graph(
        self, chunk_ids: list[str], max_hops: int
    ) -> list[str]:
        """
        Traverse Neo4j graph to find related chunks via entity relationships.

        Args:
            chunk_ids: Starting chunk IDs
            max_hops: Maximum traversal depth

        Returns:
            List of additional chunk IDs found via graph traversal
        """
        if not self.graph_processor or not self.graph_processor.driver:
            return []

        try:
            async with self.graph_processor.driver.session(
                database=self.graph_processor.neo4j_database
            ) as session:
                # Cypher query: find chunks connected via entity relationships
                query = f"""
                MATCH (start:Chunk)
                WHERE start.chunk_id IN $chunk_ids
                MATCH (start)<-[:MENTIONED_IN]-(e:Entity)-[*1..{max_hops}]-(related:Entity)
                MATCH (related)-[:MENTIONED_IN]->(related_chunk:Chunk)
                WHERE related_chunk.chunk_id <> start.chunk_id
                RETURN DISTINCT related_chunk.chunk_id as chunk_id
                LIMIT 20
                """

                result = await session.run(query, chunk_ids=chunk_ids)
                records = await result.data()

                expanded_ids = [record["chunk_id"] for record in records]

                logger.info(
                    f"Graph expansion: {len(chunk_ids)} initial -> "
                    f"{len(expanded_ids)} related chunks"
                )

                return expanded_ids

        except Exception as e:
            logger.warning(f"Graph expansion failed: {e}")
            return []

    async def _retrieve_chunks_by_ids(
        self, chunk_ids: list[str]
    ) -> list[DocumentChunk]:
        """Retrieve chunks from ChromaDB by their IDs."""
        if not chunk_ids:
            return []

        try:
            collection = self._collections[RAGType.GRAPH]

            result = collection.get(
                ids=chunk_ids, include=["documents", "metadatas"]
            )

            chunks = []
            for idx in range(len(result["ids"])):
                chunk = DocumentChunk(
                    chunk_id=result["ids"][idx],
                    document_id=result["metadatas"][idx].get("document_id", ""),
                    content=result["documents"][idx],
                    chunk_index=result["metadatas"][idx].get("chunk_index", 0),
                    metadata=result["metadatas"][idx],
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Failed to retrieve chunks by IDs: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics for all collections including graph collection.

        Returns:
            Dictionary with stats for traditional, contextual, and graph collections
        """
        stats = super().get_stats()

        # Add graph collection stats
        try:
            graph_collection = self._collections.get(RAGType.GRAPH)
            if graph_collection:
                stats["graph_chunks"] = graph_collection.count()
            else:
                stats["graph_chunks"] = 0
        except Exception as e:
            logger.error(f"Failed to get graph collection stats: {e}")
            stats["graph_chunks"] = 0

        return stats
