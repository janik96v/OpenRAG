"""Document metadata management with persistent storage."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models.contextual_schemas import ContextualDocument, ContextualDocumentMetadata
from ..models.schemas import Document

logger = logging.getLogger(__name__)


class DocumentManagerError(Exception):
    """Exception raised during document manager operations."""

    pass


class DocumentManager:
    """Manages document metadata with persistent JSON storage."""

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize the document manager.

        Args:
            storage_path: Path to the documents.json file
        """
        if storage_path is None:
            storage_path = Path("data/documents.json")

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize storage file if it doesn't exist
        if not self.storage_path.exists():
            self._save_documents({})

        logger.info(f"Initialized document manager with storage: {self.storage_path}")

    def _load_documents(self) -> dict[str, dict[str, Any]]:
        """
        Load documents metadata from JSON file.

        Returns:
            Dictionary of document metadata keyed by document_id

        Raises:
            DocumentManagerError: If loading fails
        """
        try:
            with open(self.storage_path, encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except FileNotFoundError:
            logger.info(f"Documents file not found at {self.storage_path}, creating new one")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in documents file {self.storage_path}: {str(e)}")
            raise DocumentManagerError(f"Failed to load document metadata: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to load documents from {self.storage_path}: {str(e)}")
            raise DocumentManagerError(f"Failed to load document metadata: {str(e)}") from e

    def _save_documents(self, documents: dict[str, dict[str, Any]]) -> None:
        """
        Save documents metadata to JSON file.

        Args:
            documents: Dictionary of document metadata

        Raises:
            DocumentManagerError: If saving fails
        """
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(documents, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Failed to save documents to {self.storage_path}: {str(e)}")
            raise DocumentManagerError(f"Failed to save document metadata: {str(e)}") from e

    def add_document(self, document: Document | ContextualDocument) -> None:
        """
        Add a document to the metadata storage.

        Args:
            document: Document to add

        Raises:
            DocumentManagerError: If adding document fails
        """
        try:
            documents = self._load_documents()

            upload_time = datetime.now(UTC).isoformat()

            document_metadata = {
                "document_id": document.document_id,
                "filename": document.metadata.filename,
                "file_size": document.metadata.file_size,
                "format": document.metadata.format.value,
                "chunk_count": document.metadata.chunk_count,
                "upload_date": upload_time,
                "status": document.metadata.status.value,
                "processed_date": (
                    document.metadata.processed_date.isoformat()
                    if document.metadata.processed_date
                    else None
                ),
            }

            # Add RAG type and contextual metadata if available
            if isinstance(document.metadata, ContextualDocumentMetadata):
                document_metadata["rag_type"] = document.metadata.rag_type.value
                if document.metadata.context_generation_model:
                    document_metadata["context_generation_model"] = document.metadata.context_generation_model
            else:
                # Default to traditional RAG for regular documents
                document_metadata["rag_type"] = "traditional"

            documents[document.document_id] = document_metadata
            self._save_documents(documents)

            logger.info(f"Added document metadata: {document.metadata.filename}")

        except Exception as e:
            logger.error(f"Failed to add document {document.document_id}: {str(e)}")
            raise DocumentManagerError(f"Failed to add document metadata: {str(e)}") from e

    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from the metadata storage.

        Args:
            document_id: ID of the document to remove

        Returns:
            True if document was removed, False if not found

        Raises:
            DocumentManagerError: If removal fails
        """
        try:
            documents = self._load_documents()

            if document_id not in documents:
                return False

            filename = documents[document_id].get("filename", "Unknown")
            del documents[document_id]
            self._save_documents(documents)

            logger.info(f"Removed document metadata: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {str(e)}")
            raise DocumentManagerError(f"Failed to remove document metadata: {str(e)}") from e

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """
        Get document metadata by ID.

        Args:
            document_id: ID of the document

        Returns:
            Document metadata or None if not found

        Raises:
            DocumentManagerError: If retrieval fails
        """
        try:
            documents = self._load_documents()
            return documents.get(document_id)

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            raise DocumentManagerError(f"Failed to get document metadata: {str(e)}") from e

    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents with their metadata.

        Returns:
            List of document metadata sorted by upload date (newest first)

        Raises:
            DocumentManagerError: If listing fails
        """
        try:
            documents = self._load_documents()
            document_list = list(documents.values())

            # Sort by upload_date, newest first
            document_list.sort(
                key=lambda x: x.get("upload_date", "1970-01-01T00:00:00Z"), reverse=True
            )

            return document_list

        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise DocumentManagerError(f"Failed to list documents: {str(e)}") from e

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about stored documents.

        Returns:
            Dictionary with document statistics

        Raises:
            DocumentManagerError: If getting stats fails
        """
        try:
            documents = self._load_documents()

            total_documents = len(documents)
            total_chunks = sum(doc.get("chunk_count", 0) for doc in documents.values())
            total_size = sum(doc.get("file_size", 0) for doc in documents.values())

            formats = {}
            for doc in documents.values():
                fmt = doc.get("format", "unknown")
                formats[fmt] = formats.get(fmt, 0) + 1

            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_size": total_size,
                "formats": formats,
            }

        except Exception as e:
            logger.error(f"Failed to get document stats: {str(e)}")
            raise DocumentManagerError(f"Failed to get document stats: {str(e)}") from e

    def update_document_status(self, document_id: str, status: str) -> bool:
        """
        Update the status of a document.

        Args:
            document_id: ID of the document
            status: New status value

        Returns:
            True if updated successfully, False if document not found

        Raises:
            DocumentManagerError: If update fails
        """
        try:
            documents = self._load_documents()

            if document_id not in documents:
                return False

            documents[document_id]["status"] = status
            self._save_documents(documents)

            logger.info(f"Updated document status: {document_id} -> {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document status {document_id}: {str(e)}")
            raise DocumentManagerError(f"Failed to update document status: {str(e)}") from e
