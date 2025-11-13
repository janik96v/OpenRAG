"""Extended vector store managing both traditional and contextual RAG collections."""

import logging
from pathlib import Path
from typing import Any, Optional

from ..models.contextual_schemas import ContextualDocument, RAGType
from ..models.schemas import Document
from .embedder import EmbeddingModel
from .vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class ContextualVectorStore(VectorStore):
    """
    Extended vector store managing both traditional and contextual collections.

    Architecture:
    - Two collections: {base}_traditional_v1 and {base}_contextual_v1
    - Separate but parallel storage
    - Each collection has its own embeddings
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: EmbeddingModel,
        base_collection_name: str = "documents",
        max_batch_size: int = 10000,
    ):
        """
        Initialize with both traditional and contextual collections.

        Args:
            persist_directory: Directory for persistent ChromaDB storage
            embedding_model: Embedding model instance
            base_collection_name: Base name for collections (will add _traditional_v1 suffix)
            max_batch_size: Maximum batch size for ChromaDB operations
        """
        self.base_collection_name = base_collection_name

        # Initialize parent class with traditional collection
        super().__init__(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            collection_name=f"{base_collection_name}_traditional_v1",
            max_batch_size=max_batch_size,
        )

        # Store both collection references
        self._collections = {
            RAGType.TRADITIONAL: self.collection,  # From parent
        }

        # Initialize contextual collection
        self._ensure_contextual_collection()

        logger.info("Initialized ContextualVectorStore with both collections")

    def _ensure_contextual_collection(self) -> None:
        """Create or get contextual collection."""
        try:
            contextual_name = f"{self.base_collection_name}_contextual_v1"

            self._collections[RAGType.CONTEXTUAL] = self.client.get_or_create_collection(
                name=contextual_name,
                metadata={
                    "description": "Contextual RAG document chunks",
                    "rag_type": "contextual",
                },
            )

            logger.info(
                f"Initialized contextual collection '{contextual_name}' "
                f"with {self._collections[RAGType.CONTEXTUAL].count()} existing chunks"
            )

        except Exception as e:
            logger.error(f"Failed to initialize contextual collection: {e}")
            raise VectorStoreError(f"Failed to initialize contextual collection: {e}") from e

    def add_document(
        self,
        document: Document | ContextualDocument,
        rag_type: RAGType = RAGType.TRADITIONAL,
    ) -> None:
        """
        Add document to specified collection.

        CRITICAL: For contextual RAG, embeds contextual_content (not original content)

        Args:
            document: Document or ContextualDocument to add
            rag_type: Which collection to add to

        Raises:
            VectorStoreError: If adding document fails
        """
        try:
            if not document.chunks:
                logger.warning(f"Document {document.document_id} has no chunks to add")
                return

            collection = self._collections[rag_type]

            # Determine what content to embed based on rag_type
            if rag_type == RAGType.CONTEXTUAL:
                # For contextual RAG, embed contextual_content (or fallback to original)
                chunk_contents = [
                    getattr(chunk, "contextual_content", None) or chunk.content
                    for chunk in document.chunks
                ]
                logger.info(
                    f"Generating contextual embeddings for {len(document.chunks)} chunks "
                    f"from document {document.metadata.filename}"
                )
            else:
                # For traditional RAG, embed original content
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
                documents.append(chunk.content)  # Always store original content

                # Build metadata (must be JSON-serializable, no None values!)
                metadata = {
                    "document_id": document.document_id,
                    "filename": document.metadata.filename,
                    "chunk_index": chunk.chunk_index,
                    "file_size": document.metadata.file_size,
                    "created_at": chunk.created_at.isoformat(),
                    "rag_type": rag_type.value,  # Important: mark the RAG type
                }

                # Add any additional chunk metadata
                for key, value in chunk.metadata.items():
                    if value is not None:  # CRITICAL: ChromaDB rejects None values
                        metadata[key] = value

                # Validate metadata for ChromaDB
                metadata = self._validate_metadata_for_chromadb(metadata, chunk.chunk_id)

                metadatas.append(metadata)

            # Add to collection in batches
            total_chunks = len(document.chunks)

            if total_chunks <= self.max_batch_size:
                # Small document - add all at once
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info(f"Added {total_chunks} chunks to {rag_type.value} collection")
            else:
                # Large document - split into batches
                logger.info(
                    f"Document has {total_chunks} chunks, "
                    f"splitting into batches of {self.max_batch_size}"
                )

                for i in range(0, total_chunks, self.max_batch_size):
                    end_idx = min(i + self.max_batch_size, total_chunks)

                    collection.add(
                        ids=ids[i:end_idx],
                        embeddings=embeddings[i:end_idx],
                        documents=documents[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                    )

                    logger.info(
                        f"Added batch {i // self.max_batch_size + 1}: "
                        f"chunks {i + 1}-{end_idx} of {total_chunks}"
                    )

            logger.info(
                f"Successfully added document {document.metadata.filename} "
                f"with {total_chunks} chunks to {rag_type.value} collection"
            )

        except Exception as e:
            error_msg = (
                f"Failed to add document {document.document_id} to {rag_type.value}: {str(e)}"
            )
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def _validate_metadata_for_chromadb(
        self, metadata: dict[str, Any], chunk_id: str
    ) -> dict[str, Any]:
        """
        CRITICAL: ChromaDB rejects None values and has strict type requirements.

        Args:
            metadata: Metadata dictionary to validate
            chunk_id: Chunk ID for logging

        Returns:
            Validated metadata dictionary
        """
        validated = {}

        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue

            # Skip empty strings
            if isinstance(value, str) and not value.strip():
                continue

            # Ensure valid types (ChromaDB accepts: str, int, float, bool)
            if isinstance(value, (str, int, float, bool)):
                validated[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert sequences to comma-separated strings
                validated[key] = ",".join(str(v) for v in value)
            else:
                # Convert other types to string
                validated[key] = str(value)

        return validated

    def search(
        self,
        query: str,
        n_results: int = 5,
        rag_type: RAGType = RAGType.TRADITIONAL,
        min_similarity: float = 0.4,
    ) -> list[tuple[Any, float]]:
        """
        Search in specified collection.

        Args:
            query: Search query text
            n_results: Maximum number of results to return
            rag_type: Which collection to search
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of (DocumentChunk, similarity_score) tuples

        Raises:
            VectorStoreError: If search fails
        """
        try:
            collection = self._collections[rag_type]

            # Generate query embedding
            logger.debug(f"Searching {rag_type.value} collection for: '{query[:100]}...'")
            query_embedding = self.embedding_model.embed_text(query)

            # Search collection with rag_type filter
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"rag_type": rag_type.value},  # Filter by RAG type
                include=["documents", "metadatas", "distances"],
            )

            # Convert results to DocumentChunk objects
            from ..models.schemas import DocumentChunk

            chunks_with_scores: list[tuple[DocumentChunk, float]] = []

            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    # Convert distance to similarity score
                    similarity = max(0.0, 1.0 - (distance / 2.0))

                    # Filter by minimum similarity
                    if similarity >= min_similarity:
                        # Reconstruct DocumentChunk from metadata
                        chunk = DocumentChunk(
                            chunk_id=results["ids"][0][i],
                            document_id=metadata["document_id"],
                            content=doc,
                            chunk_index=metadata["chunk_index"],
                            metadata={
                                k: v
                                for k, v in metadata.items()
                                if k not in ["document_id", "chunk_index", "rag_type"]
                            },
                        )

                        chunks_with_scores.append((chunk, similarity))

            logger.info(
                f"Found {len(chunks_with_scores)} relevant chunks in {rag_type.value} collection"
            )
            return chunks_with_scores

        except Exception as e:
            error_msg = f"Search failed in {rag_type.value} collection: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def delete_document(self, document_id: str, rag_type: Optional[RAGType] = None) -> bool:
        """
        Delete document from one or both collections.

        Args:
            document_id: ID of the document to delete
            rag_type: Specific collection to delete from, or None for both

        Returns:
            True if document was found and deleted from at least one collection

        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            deleted_from_any = False

            # Determine which collections to delete from
            collections_to_check = (
                [rag_type] if rag_type is not None else [RAGType.TRADITIONAL, RAGType.CONTEXTUAL]
            )

            for coll_type in collections_to_check:
                collection = self._collections[coll_type]

                # Find all chunks for the document in this collection
                results = collection.get(
                    where={"document_id": document_id, "rag_type": coll_type.value}
                )

                if results["ids"]:
                    # Delete all chunks
                    chunk_ids = results["ids"]
                    collection.delete(ids=chunk_ids)

                    logger.info(
                        f"Deleted {len(chunk_ids)} chunks for document {document_id} "
                        f"from {coll_type.value} collection"
                    )
                    deleted_from_any = True

            if not deleted_from_any:
                logger.warning(f"No chunks found for document {document_id} in any collection")

            return deleted_from_any

        except Exception as e:
            error_msg = f"Failed to delete document {document_id}: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def list_documents(self, rag_type: Optional[RAGType] = None) -> list[dict[str, Any]]:
        """
        List documents from one or both collections.

        Args:
            rag_type: Specific collection to list from, or None for both

        Returns:
            List of document information dictionaries
        """
        try:
            all_documents: dict[str, dict[str, Any]] = {}

            # Determine which collections to list from
            collections_to_check = (
                [rag_type] if rag_type is not None else [RAGType.TRADITIONAL, RAGType.CONTEXTUAL]
            )

            for coll_type in collections_to_check:
                collection = self._collections[coll_type]
                count = collection.count()

                if count == 0:
                    continue

                # Get all documents metadata
                results = collection.get(include=["metadatas"])

                if not results["metadatas"]:
                    continue

                # Group by document_id
                for metadata in results["metadatas"]:
                    if metadata.get("rag_type") != coll_type.value:
                        continue

                    doc_id = metadata["document_id"]
                    key = f"{doc_id}_{coll_type.value}"

                    if key not in all_documents:
                        all_documents[key] = {
                            "document_id": doc_id,
                            "filename": metadata.get("filename", "Unknown"),
                            "file_size": metadata.get("file_size", 0),
                            "created_at": metadata.get("created_at", "Unknown"),
                            "chunk_count": 0,
                            "rag_type": coll_type.value,
                        }

                    all_documents[key]["chunk_count"] += 1

            return list(all_documents.values())

        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return []

    def get_stats(self, rag_type: Optional[RAGType] = None) -> dict[str, Any]:
        """
        Get statistics for one or both collections.

        Args:
            rag_type: Specific collection to get stats for, or None for both

        Returns:
            Dictionary with statistics
        """
        try:
            stats = {}

            # Determine which collections to get stats for
            collections_to_check = (
                [rag_type] if rag_type is not None else [RAGType.TRADITIONAL, RAGType.CONTEXTUAL]
            )

            for coll_type in collections_to_check:
                collection = self._collections[coll_type]
                total_chunks = collection.count()

                prefix = f"{coll_type.value}_"

                if total_chunks == 0:
                    stats[f"{prefix}chunks"] = 0
                    stats[f"{prefix}documents"] = 0
                    continue

                # Get unique documents count
                results = collection.get(include=["metadatas"])

                unique_documents = set()
                if results["metadatas"]:
                    unique_documents = {
                        metadata["document_id"]
                        for metadata in results["metadatas"]
                        if metadata.get("rag_type") == coll_type.value
                    }

                stats[f"{prefix}chunks"] = total_chunks
                stats[f"{prefix}documents"] = len(unique_documents)

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                "traditional_chunks": 0,
                "traditional_documents": 0,
                "contextual_chunks": 0,
                "contextual_documents": 0,
            }
