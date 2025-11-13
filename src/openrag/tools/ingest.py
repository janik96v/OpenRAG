"""Document ingestion MCP tool with dual traditional and contextual RAG support."""

from typing import Any

from ..core.chunker import TextChunker
from ..core.contextual_processor import ContextualProcessor
from ..core.contextual_vector_store import ContextualVectorStore
from ..models.contextual_schemas import ContextualDocument, RAGType
from ..models.schemas import Document, DocumentChunk, DocumentMetadata, DocumentStatus
from ..utils.async_tasks import BackgroundTaskManager
from ..utils.logger import setup_logger
from ..utils.validation import ValidationError

logger = setup_logger(__name__)


async def ingest_text_tool(
    text: str,
    document_name: str,
    vector_store: ContextualVectorStore,
    chunker: TextChunker,
    contextual_processor: ContextualProcessor,
    task_manager: BackgroundTaskManager,
) -> dict[str, Any]:
    """
    Ingest raw text content directly into both traditional and contextual RAG systems.

    This tool accepts raw text content, chunks it, and:
    1. Immediately stores it in the traditional RAG collection (blocking)
    2. Starts background processing for contextual RAG (non-blocking)

    The function returns immediately after traditional ingestion completes, while
    contextual processing continues in the background.

    Args:
        text: Raw text content to ingest
        document_name: Name/identifier for this document (e.g., "report.pdf")
        vector_store: ContextualVectorStore instance
        chunker: TextChunker instance
        contextual_processor: ContextualProcessor instance
        task_manager: BackgroundTaskManager instance

    Returns:
        Dictionary with ingestion results and metadata

    Raises:
        ValidationError: If text validation fails
        Exception: If ingestion fails

    Example Response:
        {
            "status": "success",
            "document_id": "123e4567-e89b-12d3-a456-426614174000",
            "document_name": "report.pdf",
            "chunk_count": 42,
            "traditional_ingestion": "completed",
            "contextual_ingestion": "processing_in_background",
            "message": "Successfully ingested report.pdf with 42 chunks..."
        }
    """
    try:
        # Validate inputs
        if not isinstance(text, str):
            raise ValidationError(f"Text must be a string, got {type(text).__name__}")

        if not text.strip():
            raise ValidationError("Text content cannot be empty or only whitespace")

        if not isinstance(document_name, str) or not document_name.strip():
            raise ValidationError("Document name must be a non-empty string")

        document_name = document_name.strip()
        text = text.strip()

        logger.info(f"Processing text content for document: {document_name}")

        # Chunk text
        logger.info(f"Chunking document: {document_name}")
        text_chunks = chunker.chunk_text(text)

        if not text_chunks:
            raise ValueError("No chunks generated from text content")

        # Create Document object
        metadata = DocumentMetadata(
            filename=document_name,
            file_size=len(text.encode("utf-8")),  # Size in bytes
            chunk_count=len(text_chunks),
            status=DocumentStatus.PROCESSING,
        )

        document = Document(metadata=metadata)

        # Create DocumentChunk objects
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                document_id=document.document_id,
                content=chunk_text,
                chunk_index=i,
                metadata={
                    "token_count": chunker.count_tokens(chunk_text),
                    "char_count": len(chunk_text),
                },
            )
            document.chunks.append(chunk)

        # Step 1: Add to TRADITIONAL collection immediately (blocking)
        logger.info(f"Adding document to traditional collection: {document_name}")
        vector_store.add_document(document, rag_type=RAGType.TRADITIONAL)

        # Step 2: Start CONTEXTUAL processing in background (non-blocking)
        logger.info(f"Starting contextual processing in background for: {document_name}")

        # CRITICAL: Use task_manager.create_task() to prevent garbage collection
        task_manager.create_task(
            _process_contextual_async(
                document=document,
                full_document_text=text,  # CRITICAL: Pass full document for context
                vector_store=vector_store,
                contextual_processor=contextual_processor,
            ),
            name=f"contextual_{document.document_id}",
        )

        # Update status
        document.metadata.status = DocumentStatus.COMPLETED

        logger.info(
            f"Successfully ingested {document_name} with {len(document.chunks)} chunks "
            f"(traditional completed, contextual processing in background)"
        )

        # Return immediately (don't await background task!)
        return {
            "status": "success",
            "document_id": document.document_id,
            "document_name": document.metadata.filename,
            "chunk_count": len(document.chunks),
            "text_size_bytes": document.metadata.file_size,
            "traditional_ingestion": "completed",
            "contextual_ingestion": "processing_in_background",
            "message": (
                f"Successfully ingested {document.metadata.filename} "
                f"with {len(document.chunks)} chunks to traditional store. "
                f"Contextual processing running in background."
            ),
        }

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return {
            "status": "error",
            "error": "validation_error",
            "message": str(e),
        }

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": "ingestion_failed",
            "message": f"Failed to ingest text content: {str(e)}",
        }


async def _process_contextual_async(
    document: Document,
    full_document_text: str,
    vector_store: ContextualVectorStore,
    contextual_processor: ContextualProcessor,
) -> None:
    """
    Background task for contextual processing.

    CRITICAL: This function runs in background - MUST have proper error handling.
    Any exception here will be logged by BackgroundTaskManager.

    Args:
        document: Original document with chunks
        full_document_text: Complete document text for context generation
        vector_store: ContextualVectorStore instance
        contextual_processor: ContextualProcessor instance
    """
    try:
        logger.info(f"Starting contextual processing for document {document.document_id}")

        # Generate contextual chunks
        contextual_chunks = await contextual_processor.process_document_chunks(
            chunks=document.chunks,
            document_metadata=document.metadata,
            full_document_text=full_document_text,
        )

        # Create contextual document
        contextual_doc = ContextualDocument(
            document_id=document.document_id,
            metadata=document.metadata,
            chunks=contextual_chunks,
        )

        # Add to contextual collection
        vector_store.add_document(contextual_doc, rag_type=RAGType.CONTEXTUAL)

        logger.info(
            f"Completed contextual processing for document {document.document_id} "
            f"({len(contextual_chunks)} chunks)"
        )

    except Exception as e:
        logger.error(
            f"Contextual processing failed for document {document.document_id}: {e}",
            exc_info=True,
        )
        # Don't raise - this is a background task, just log the error
        # Document still exists in traditional collection
