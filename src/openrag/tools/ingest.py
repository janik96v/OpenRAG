"""Document ingestion MCP tool."""

from typing import Any

from ..core.chunker import TextChunker
from ..core.vector_store import VectorStore
from ..models.schemas import Document, DocumentChunk, DocumentMetadata, DocumentStatus
from ..utils.logger import setup_logger
from ..utils.validation import ValidationError

logger = setup_logger(__name__)


async def ingest_text_tool(
    text: str,
    document_name: str,
    vector_store: VectorStore,
    chunker: TextChunker,
) -> dict[str, Any]:
    """
    Ingest raw text content directly into the RAG system.

    This tool accepts raw text content (e.g., extracted from PDF by an LLM client),
    chunks it, generates embeddings, and stores everything in the vector database.
    This bypasses file I/O and allows clients to handle document parsing.

    Args:
        text: Raw text content to ingest
        document_name: Name/identifier for this document (e.g., "report.pdf")
        vector_store: VectorStore instance
        chunker: TextChunker instance

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
            "message": "Successfully ingested report.pdf with 42 chunks"
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

        # Add to vector store (this generates embeddings and stores them)
        logger.info(f"Adding document to vector store: {document_name}")
        vector_store.add_document(document)

        # Update status
        document.metadata.status = DocumentStatus.COMPLETED

        logger.info(
            f"Successfully ingested {document_name} with {len(document.chunks)} chunks"
        )

        # Return success response
        return {
            "status": "success",
            "document_id": document.document_id,
            "document_name": document.metadata.filename,
            "chunk_count": len(document.chunks),
            "text_size_bytes": document.metadata.file_size,
            "message": (
                f"Successfully ingested {document.metadata.filename} "
                f"with {len(document.chunks)} chunks"
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
