"""Document ingestion MCP tool."""

from typing import Any

from ..core.chunker import TextChunker
from ..core.vector_store import VectorStore
from ..models.schemas import Document, DocumentChunk, DocumentMetadata, DocumentStatus
from ..utils.logger import setup_logger
from ..utils.validation import ValidationError, validate_txt_file

logger = setup_logger(__name__)


async def ingest_document_tool(
    file_path: str,
    vector_store: VectorStore,
    chunker: TextChunker,
) -> dict[str, Any]:
    """
    Ingest a document into the RAG system.

    This tool accepts a .txt file path, chunks the document, generates embeddings,
    and stores everything in the vector database.

    Args:
        file_path: Path to the .txt file to ingest
        vector_store: VectorStore instance
        chunker: TextChunker instance

    Returns:
        Dictionary with ingestion results and metadata

    Raises:
        ValidationError: If file validation fails
        Exception: If ingestion fails

    Example Response:
        {
            "status": "success",
            "document_id": "123e4567-e89b-12d3-a456-426614174000",
            "filename": "mydoc.txt",
            "chunk_count": 42,
            "message": "Successfully ingested mydoc.txt with 42 chunks"
        }
    """
    try:
        # Validate file path
        logger.info(f"Validating file: {file_path}")
        file_path_obj = validate_txt_file(file_path)

        # Read file content
        logger.info(f"Reading file: {file_path_obj.name}")
        try:
            content = file_path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            content = file_path_obj.read_text(encoding="latin-1")
        except Exception as e:
            raise ValueError(f"Failed to read file: {str(e)}") from e

        if not content.strip():
            raise ValueError("File is empty or contains only whitespace")

        # Chunk text
        logger.info(f"Chunking document: {file_path_obj.name}")
        text_chunks = chunker.chunk_text(content)

        if not text_chunks:
            raise ValueError("No chunks generated from document")

        # Create Document object
        metadata = DocumentMetadata(
            filename=file_path_obj.name,
            file_size=file_path_obj.stat().st_size,
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
        logger.info(f"Adding document to vector store: {file_path_obj.name}")
        vector_store.add_document(document)

        # Update status
        document.metadata.status = DocumentStatus.COMPLETED

        logger.info(
            f"Successfully ingested {file_path_obj.name} with {len(document.chunks)} chunks"
        )

        # Return success response
        return {
            "status": "success",
            "document_id": document.document_id,
            "filename": document.metadata.filename,
            "chunk_count": len(document.chunks),
            "file_size": document.metadata.file_size,
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
            "message": f"Failed to ingest document: {str(e)}",
        }
