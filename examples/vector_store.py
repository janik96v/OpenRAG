"""Vector store management using ChromaDB."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..models.schemas import Document, DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Exception raised during vector store operations."""

    pass


class VectorStore:
    """Manages document embeddings and similarity search using ChromaDB."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str = "research_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_batch_size: int = 10000,
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the sentence transformer model
            max_batch_size: Maximum batch size for ChromaDB operations
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.max_batch_size = max_batch_size

        # Initialize embedding model
        try:
            self._embedder = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            raise VectorStoreError(f"Failed to load embedding model: {str(e)}") from e

        # Initialize ChromaDB client
        try:
            if persist_directory:
                persist_directory = Path(persist_directory)
                persist_directory.mkdir(parents=True, exist_ok=True)

                self._client = chromadb.PersistentClient(
                    path=str(persist_directory),
                    settings=Settings(allow_reset=True, anonymized_telemetry=False),
                )
            else:
                self._client = chromadb.Client(
                    settings=Settings(allow_reset=True, anonymized_telemetry=False)
                )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=collection_name, metadata={"description": "Research documents and chunks"}
            )

            logger.info(f"Initialized ChromaDB collection: {collection_name}")

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}") from e

    def add_document(self, document: Document) -> None:
        """
        Add a document and its chunks to the vector store.

        Args:
            document: Document to add with chunks

        Raises:
            VectorStoreError: If adding document fails
        """
        try:
            if not document.chunks:
                logger.warning(f"Document {document.document_id} has no chunks to add")
                return

            # Generate embeddings for all chunks
            chunk_contents = [chunk.content for chunk in document.chunks]
            embeddings = self._embedder.encode(chunk_contents).tolist()

            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in document.chunks]
            metadatas = []

            for chunk in document.chunks:
                metadata = {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "filename": document.metadata.filename,
                    "file_format": document.metadata.format,
                    "upload_date": document.metadata.upload_date.isoformat(),
                    "chunk_length": len(chunk.content),
                    **chunk.metadata,
                }
                metadatas.append(metadata)

            # Add to collection in batches to handle large documents
            total_chunks = len(document.chunks)

            if total_chunks <= self.max_batch_size:
                # Small document - add all at once
                self._collection.add(
                    ids=ids, embeddings=embeddings, documents=chunk_contents, metadatas=metadatas
                )
            else:
                # Large document - split into batches
                logger.info(f"Document has {total_chunks} chunks, splitting into batches of {self.max_batch_size}")

                for i in range(0, total_chunks, self.max_batch_size):
                    end_idx = min(i + self.max_batch_size, total_chunks)
                    batch_ids = ids[i:end_idx]
                    batch_embeddings = embeddings[i:end_idx]
                    batch_contents = chunk_contents[i:end_idx]
                    batch_metadatas = metadatas[i:end_idx]

                    logger.info(f"Adding batch {i // self.max_batch_size + 1}: chunks {i+1}-{end_idx} of {total_chunks}")
                    self._collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_contents,
                        metadatas=batch_metadatas
                    )

            logger.info(
                f"Added {len(document.chunks)} chunks from document "
                f"{document.metadata.filename} to vector store"
            )

        except Exception as e:
            raise VectorStoreError(
                f"Failed to add document {document.document_id}: {str(e)}"
            ) from e

    def search(
        self, query: str, n_results: int = 5, document_filter: dict[str, str] | None = None
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search for similar document chunks.

        Args:
            query: Search query
            n_results: Maximum number of results to return
            document_filter: Optional metadata filter

        Returns:
            List of (DocumentChunk, similarity_score) tuples

        Raises:
            VectorStoreError: If search fails
        """
        try:
            # Generate query embedding
            query_embedding = self._embedder.encode([query]).tolist()[0]

            # Prepare filter for ChromaDB
            where_filter = None
            if document_filter:
                where_filter = document_filter

            # Search collection
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            # Convert results to DocumentChunk objects
            chunks = []

            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                        strict=False,
                    )
                ):
                    # Create DocumentChunk from metadata
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

                    # Convert distance to similarity score (0-1, higher is more similar)
                    similarity_score = max(0.0, 1.0 - distance)

                    chunks.append((chunk, similarity_score))

            logger.info(f"Found {len(chunks)} similar chunks for query")
            return chunks

        except Exception as e:
            raise VectorStoreError(f"Search failed: {str(e)}") from e

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks of a document from the vector store.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if document was found and deleted

        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            # Find all chunks for the document
            results = self._collection.get(
                where={"document_id": document_id},
            )

            if not results["ids"]:
                logger.warning(f"No chunks found for document {document_id}")
                return False

            # Delete all chunks
            chunk_ids = results["ids"]
            self._collection.delete(ids=chunk_ids)

            logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
            return True

        except Exception as e:
            raise VectorStoreError(f"Failed to delete document {document_id}: {str(e)}") from e

    def get_collection_stats(self) -> dict[str, int]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self._collection.count()

            if count == 0:
                return {"total_chunks": 0, "unique_documents": 0}

            # Get unique documents count
            all_results = self._collection.get(include=["metadatas"])

            unique_documents = set()
            if all_results["metadatas"]:
                unique_documents = {
                    metadata["document_id"] for metadata in all_results["metadatas"]
                }

            return {"total_chunks": count, "unique_documents": len(unique_documents)}

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"total_chunks": 0, "unique_documents": 0}

    def list_documents(self) -> list[dict[str, str]]:
        """
        List all documents in the collection.

        Returns:
            List of document information dictionaries
        """
        try:
            count = self._collection.count()

            if count == 0:
                return []

            # Get all documents metadata
            results = self._collection.get(include=["metadatas"])

            if not results["metadatas"]:
                return []

            # Group by document_id and collect info
            documents = {}
            for metadata in results["metadatas"]:
                doc_id = metadata["document_id"]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "file_format": metadata.get("file_format", "Unknown"),
                        "upload_date": metadata.get("upload_date", "Unknown"),
                        "chunk_count": 0,
                    }
                documents[doc_id]["chunk_count"] += 1

            return list(documents.values())

        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return []

    def reset_collection(self) -> None:
        """
        Delete all documents from the collection.

        Raises:
            VectorStoreError: If reset fails
        """
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"description": "Research documents and chunks"}
            )
            logger.info(f"Reset collection: {self.collection_name}")

        except Exception as e:
            raise VectorStoreError(f"Failed to reset collection: {str(e)}") from e
