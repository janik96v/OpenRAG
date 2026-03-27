"""Document management MCP tools (list, delete, cancel ingestion)."""

from typing import Any

from ..core.contextual_vector_store import ContextualVectorStore
from ..utils.async_tasks import BackgroundTaskManager
from ..utils.logger import setup_logger
from ..utils.validation import ValidationError, validate_document_id

logger = setup_logger(__name__)


async def list_documents_tool(vector_store: ContextualVectorStore) -> dict[str, Any]:
    """
    List all ingested documents with metadata.

    This tool retrieves information about all documents currently stored
    in the vector database.

    Args:
        vector_store: VectorStore instance

    Returns:
        Dictionary with list of documents

    Example Response:
        {
            "status": "success",
            "documents": [
                {
                    "document_id": "...",
                    "filename": "guide.txt",
                    "file_size": 12345,
                    "chunk_count": 42,
                    "created_at": "2025-11-08T10:30:00Z"
                }
            ],
            "total_documents": 3
        }
    """
    try:
        logger.info("Listing all documents")

        # Get documents from vector store
        documents = vector_store.list_documents()

        logger.info(f"Found {len(documents)} documents")

        return {
            "status": "success",
            "documents": documents,
            "total_documents": len(documents),
        }

    except Exception as e:
        logger.error(f"List documents error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": "list_failed",
            "message": f"Failed to list documents: {str(e)}",
        }


async def delete_document_tool(
    document_id: str,
    vector_store: ContextualVectorStore,
) -> dict[str, Any]:
    """
    Delete a document and all its chunks from the vector database.

    This tool permanently removes a document and all its associated chunks.

    Args:
        document_id: ID of the document to delete
        vector_store: VectorStore instance

    Returns:
        Dictionary with deletion result

    Example Response:
        {
            "status": "success",
            "document_id": "123e4567-e89b-12d3-a456-426614174000",
            "message": "Successfully deleted document and 42 chunks"
        }
    """
    try:
        # Validate document ID
        logger.info(f"Validating document ID: {document_id}")
        validated_id = validate_document_id(document_id)

        # Delete document
        logger.info(f"Deleting document: {validated_id}")
        deleted = vector_store.delete_document(validated_id)

        if not deleted:
            logger.warning(f"Document not found: {validated_id}")
            return {
                "status": "error",
                "error": "not_found",
                "message": f"Document not found: {validated_id}",
            }

        logger.info(f"Successfully deleted document: {validated_id}")

        return {
            "status": "success",
            "document_id": validated_id,
            "message": f"Successfully deleted document {validated_id}",
        }

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return {
            "status": "error",
            "error": "validation_error",
            "message": str(e),
        }

    except Exception as e:
        logger.error(f"Delete error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": "delete_failed",
            "message": f"Failed to delete document: {str(e)}",
        }


async def cancel_ingestion_tool(
    rag_type: str,
    task_manager: BackgroundTaskManager | None,
) -> dict[str, Any]:
    """
    Cancel all running background ingestion tasks for a given RAG type.

    Traditional RAG ingestion is synchronous and cannot be cancelled.
    Only 'contextual' and 'graph' background tasks can be stopped.

    Args:
        rag_type: RAG type whose background tasks should be cancelled
                  ('contextual' or 'graph')
        task_manager: BackgroundTaskManager instance (None if background tasks disabled)

    Returns:
        Dictionary with cancellation result

    Example Response:
        {
            "status": "success",
            "rag_type": "contextual",
            "tasks_cancelled": 2,
            "message": "Cancelled 2 contextual background ingestion task(s)."
        }
    """
    _VALID_TYPES = ("contextual", "graph")

    if rag_type not in _VALID_TYPES:
        return {
            "status": "error",
            "error": "validation_error",
            "message": (
                f"Invalid rag_type '{rag_type}'. "
                f"Only {_VALID_TYPES} background tasks can be cancelled. "
                "Traditional RAG is synchronous and cannot be cancelled."
            ),
        }

    if task_manager is None:
        return {
            "status": "error",
            "error": "not_available",
            "message": (
                "Background task manager is not available. "
                "Ollama is required for contextual and graph RAG processing."
            ),
        }

    cancelled = task_manager.cancel_by_prefix(f"{rag_type}_")

    logger.info(f"Cancelled {cancelled} {rag_type} background ingestion task(s)")

    return {
        "status": "success",
        "rag_type": rag_type,
        "tasks_cancelled": cancelled,
        "message": f"Cancelled {cancelled} {rag_type} background ingestion task(s).",
    }
