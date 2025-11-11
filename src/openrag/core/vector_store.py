"""ChromaDB vector store for document embeddings and similarity search."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from ..models.schemas import Document, DocumentChunk
from ..utils.logger import setup_logger
from .embedder import EmbeddingModel

logger = setup_logger(__name__)


class VectorStoreError(Exception):
    """Exception raised during vector store operations."""

    pass


class VectorStore:
    """
    Manages document embeddings and similarity search using ChromaDB.

    Provides persistent storage of document chunks with their embeddings,
    and efficient semantic search capabilities.
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: EmbeddingModel,
        collection_name: str = "documents",
        max_batch_size: int = 10000,
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory for persistent ChromaDB storage
            embedding_model: Embedding model instance
            collection_name: Name of the ChromaDB collection
            max_batch_size: Maximum batch size for ChromaDB operations

        Raises:
            VectorStoreError: If initialization fails
        """
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.max_batch_size = max_batch_size

        try:
            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB persistent client
            logger.info(f"Initializing ChromaDB at {self.persist_directory}")
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Get or create collection
            # CRITICAL: Don't specify embedding_function, we'll provide embeddings directly
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "RAG document chunks"},
            )

            logger.info(
                f"Initialized ChromaDB collection '{collection_name}' "
                f"with {self.collection.count()} existing chunks"
            )

        except Exception as e:
            error_msg = f"Failed to initialize vector store: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def add_document(self, document: Document) -> None:
        """
        Add a document and its chunks to the vector store.

        Args:
            document: Document with chunks to add

        Raises:
            VectorStoreError: If adding document fails
        """
        try:
            if not document.chunks:
                logger.warning(f"Document {document.document_id} has no chunks to add")
                return

            # Generate embeddings for all chunks
            logger.info(
                f"Generating embeddings for {len(document.chunks)} chunks "
                f"from document {document.metadata.filename}"
            )
            chunk_contents = [chunk.content for chunk in document.chunks]
            embeddings = self.embedding_model.embed_texts(chunk_contents)

            # Prepare data for ChromaDB
            ids: list[str] = []
            metadatas: list[dict[str, Any]] = []
            documents: list[str] = []

            for chunk, embedding in zip(document.chunks, embeddings):
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)

                # Build metadata (must be JSON-serializable)
                metadata = {
                    "document_id": document.document_id,
                    "filename": document.metadata.filename,
                    "chunk_index": chunk.chunk_index,
                    "file_size": document.metadata.file_size,
                    "created_at": chunk.created_at.isoformat(),
                    **chunk.metadata,  # Include any additional metadata
                }
                metadatas.append(metadata)

            # Add to collection in batches
            total_chunks = len(document.chunks)

            if total_chunks <= self.max_batch_size:
                # Small document - add all at once
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info(f"Added {total_chunks} chunks to vector store")
            else:
                # Large document - split into batches
                logger.info(
                    f"Document has {total_chunks} chunks, "
                    f"splitting into batches of {self.max_batch_size}"
                )

                for i in range(0, total_chunks, self.max_batch_size):
                    end_idx = min(i + self.max_batch_size, total_chunks)

                    self.collection.add(
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
                f"with {total_chunks} chunks"
            )

        except Exception as e:
            error_msg = f"Failed to add document {document.document_id}: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def search(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.1,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search for similar document chunks.

        Args:
            query: Search query text
            n_results: Maximum number of results to return
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of (DocumentChunk, similarity_score) tuples, sorted by relevance

        Raises:
            VectorStoreError: If search fails
        """
        try:
            # Generate query embedding
            logger.debug(f"Searching for: '{query[:100]}...'")
            query_embedding = self.embedding_model.embed_text(query)

            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            # Convert results to DocumentChunk objects
            chunks_with_scores: list[tuple[DocumentChunk, float]] = []

            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    # Convert distance to similarity score (0-1, higher is better)
                    # ChromaDB uses L2 distance, convert to similarity
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
                                if k not in ["document_id", "chunk_index"]
                            },
                        )

                        chunks_with_scores.append((chunk, similarity))

            logger.info(f"Found {len(chunks_with_scores)} relevant chunks")
            return chunks_with_scores

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks of a document from the vector store.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if document was found and deleted, False if not found

        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            # Find all chunks for the document
            results = self.collection.get(where={"document_id": document_id})

            if not results["ids"]:
                logger.warning(f"No chunks found for document {document_id}")
                return False

            # Delete all chunks
            chunk_ids = results["ids"]
            self.collection.delete(ids=chunk_ids)

            logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
            return True

        except Exception as e:
            error_msg = f"Failed to delete document {document_id}: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents in the collection.

        Returns:
            List of document information dictionaries
        """
        try:
            count = self.collection.count()

            if count == 0:
                return []

            # Get all documents metadata
            results = self.collection.get(include=["metadatas"])

            if not results["metadatas"]:
                return []

            # Group by document_id and collect info
            documents: dict[str, dict[str, Any]] = {}

            for metadata in results["metadatas"]:
                doc_id = metadata["document_id"]

                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "file_size": metadata.get("file_size", 0),
                        "created_at": metadata.get("created_at", "Unknown"),
                        "chunk_count": 0,
                    }

                documents[doc_id]["chunk_count"] += 1

            return list(documents.values())

        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        try:
            total_chunks = self.collection.count()

            if total_chunks == 0:
                return {
                    "total_chunks": 0,
                    "unique_documents": 0,
                }

            # Get unique documents count
            results = self.collection.get(include=["metadatas"])

            unique_documents = set()
            if results["metadatas"]:
                unique_documents = {metadata["document_id"] for metadata in results["metadatas"]}

            return {
                "total_chunks": total_chunks,
                "unique_documents": len(unique_documents),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                "total_chunks": 0,
                "unique_documents": 0,
            }
