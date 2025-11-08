"""Document processing functionality for extracting text from various formats."""

import logging
from pathlib import Path

from PyPDF2 import PdfReader

from ..models.contextual_schemas import (
    ContextualDocument,
    ContextualDocumentChunk,
    ContextualDocumentMetadata,
    RAGType,
)
from ..models.schemas import (
    Document,
    DocumentChunk,
    DocumentFormat,
    DocumentMetadata,
    DocumentStatus,
)

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Exception raised during document processing."""

    pass


class DocumentProcessor:
    """Handles extraction and chunking of documents."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_file_size_mb: int = 10,
        contextual_processor=None
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
            max_file_size_mb: Maximum file size in MB
            contextual_processor: Optional contextual processor for context generation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size_mb = max_file_size_mb
        self.contextual_processor = contextual_processor

    def process_document(
        self, file_path: Path, rag_type: RAGType = RAGType.TRADITIONAL
    ) -> Document | ContextualDocument:
        """
        Process a document and extract text chunks.

        Args:
            file_path: Path to the document file
            rag_type: Type of RAG processing (traditional or contextual)

        Returns:
            Processed document with chunks (Document or ContextualDocument)

        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            # Determine document format
            file_extension = file_path.suffix.lower().lstrip(".")
            document_format = self._get_document_format(file_extension)

            # Create metadata based on RAG type
            if rag_type == RAGType.CONTEXTUAL:
                metadata = ContextualDocumentMetadata(
                    filename=file_path.name,
                    file_size=file_path.stat().st_size,
                    format=document_format,
                    status=DocumentStatus.PROCESSING,
                    rag_type=rag_type,
                )
            else:
                metadata = DocumentMetadata(
                    filename=file_path.name,
                    file_size=file_path.stat().st_size,
                    format=document_format,
                    status=DocumentStatus.PROCESSING,
                )

            # Extract text content
            text_content = self._extract_text(file_path, document_format)

            # Create document based on RAG type
            if rag_type == RAGType.CONTEXTUAL:
                document = ContextualDocument(metadata=metadata)
            else:
                document = Document(metadata=metadata)

            # Generate chunks
            if rag_type == RAGType.CONTEXTUAL:
                chunks = self._create_contextual_chunks(document.document_id, text_content, metadata)
            else:
                chunks = self._create_chunks(document.document_id, text_content)

            document.chunks = chunks

            # Update metadata
            document.metadata.chunk_count = len(chunks)
            document.metadata.status = DocumentStatus.COMPLETED

            logger.info(f"Successfully processed document: {file_path.name} with {rag_type.value} RAG")
            return document

        except Exception as e:
            logger.error(f"Failed to process document {file_path.name}: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}") from e

    def _get_document_format(self, file_extension: str) -> DocumentFormat:
        """
        Determine document format from file extension.

        Args:
            file_extension: File extension without dot

        Returns:
            DocumentFormat enum value

        Raises:
            DocumentProcessingError: If format is not supported
        """
        format_mapping = {
            "pdf": DocumentFormat.PDF,
            "md": DocumentFormat.MARKDOWN,
            "markdown": DocumentFormat.MARKDOWN,
            "txt": DocumentFormat.TEXT,
        }

        if file_extension not in format_mapping:
            raise DocumentProcessingError(f"Unsupported file format: {file_extension}")

        return format_mapping[file_extension]

    def _extract_text(self, file_path: Path, document_format: DocumentFormat) -> str:
        """
        Extract text content from a document.

        Args:
            file_path: Path to the document file
            document_format: Format of the document

        Returns:
            Extracted text content

        Raises:
            DocumentProcessingError: If text extraction fails
        """
        try:
            if document_format == DocumentFormat.PDF:
                return self._extract_pdf_text(file_path)
            elif document_format in [DocumentFormat.MARKDOWN, DocumentFormat.TEXT]:
                return self._extract_text_file_content(file_path)
            else:
                raise DocumentProcessingError(f"Unsupported document format: {document_format}")

        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract text: {str(e)}") from e

    def _extract_pdf_text(self, file_path: Path) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        text_content = ""

        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\\n\\n--- Page {page_num} ---\\n\\n{page_text}"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")

        if not text_content.strip():
            raise DocumentProcessingError("No text content extracted from PDF")

        return text_content.strip()

    def _extract_text_file_content(self, file_path: Path) -> str:
        """
        Extract content from text-based files.

        Args:
            file_path: Path to the text file

        Returns:
            File content as string
        """
        try:
            with open(file_path, encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            with open(file_path, encoding="latin-1") as file:
                content = file.read()

        if not content.strip():
            raise DocumentProcessingError("File appears to be empty")

        return content.strip()

    def _create_chunks(self, document_id: str, text: str) -> list[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            document_id: ID of the parent document
            text: Text content to chunk

        Returns:
            List of document chunks
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))

            # If this is not the last chunk, try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the overlap buffer
                sentence_ends = [".", "!", "?", "\\n\\n"]
                best_end = end

                for i in range(end - self.chunk_overlap, end):
                    if i >= 0 and text[i] in sentence_ends:
                        best_end = i + 1
                        break

                end = best_end

            # Extract chunk content
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunk = DocumentChunk(
                    document_id=document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        "start_position": start,
                        "end_position": end,
                        "length": len(chunk_content),
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)

        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks

    def validate_file(self, file_path: Path, max_size_mb: int | None = None) -> bool:
        """
        Validate if a file can be processed.

        Args:
            file_path: Path to the file
            max_size_mb: Maximum file size in MB (defaults to instance config)

        Returns:
            True if file is valid for processing

        Raises:
            DocumentProcessingError: If validation fails
        """
        if not file_path.exists():
            raise DocumentProcessingError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise DocumentProcessingError(f"Path is not a file: {file_path}")

        # Use instance max_file_size_mb if not provided
        max_size = max_size_mb if max_size_mb is not None else self.max_file_size_mb

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size:
            raise DocumentProcessingError(f"File too large: {file_size_mb:.1f}MB > {max_size}MB")

        # Check file extension
        file_extension = file_path.suffix.lower().lstrip(".")
        supported_extensions = {"pdf", "md", "markdown", "txt"}

        if file_extension not in supported_extensions:
            raise DocumentProcessingError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(supported_extensions)}"
            )

        return True

    def _create_contextual_chunks(
        self,
        document_id: str,
        text: str,
        metadata: ContextualDocumentMetadata
    ) -> list[ContextualDocumentChunk]:
        """
        Create contextual chunks with generated context.

        Args:
            document_id: ID of the parent document
            text: Text content to chunk
            metadata: Document metadata for context generation

        Returns:
            List of contextual document chunks

        Raises:
            DocumentProcessingError: If contextual processing fails
        """
        if not self.contextual_processor:
            logger.warning("No contextual processor available, falling back to traditional chunks")
            # Create traditional chunks and convert to contextual format
            traditional_chunks = self._create_chunks(document_id, text)
            contextual_chunks = []

            for chunk in traditional_chunks:
                contextual_chunk = ContextualDocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    contextual_content=None,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                    rag_type=RAGType.TRADITIONAL,  # Fallback to traditional
                    created_at=chunk.created_at,
                )
                contextual_chunks.append(contextual_chunk)

            return contextual_chunks

        try:
            # First create traditional chunks
            traditional_chunks = self._create_chunks(document_id, text)

            logger.info(f"Generating context for {len(traditional_chunks)} chunks")

            # Process chunks with contextual processor
            contextual_chunks = self.contextual_processor.process_document_chunks(
                traditional_chunks, metadata, text
            )

            # Update metadata with context generation info
            if hasattr(metadata, 'context_generation_model'):
                metadata.context_generation_model = self.contextual_processor.context_model

                # Calculate total context generation time from successful generations
                # Note: We don't have individual timing data, so we'll set this to None
                # The timing is handled in the contextual processor response
                metadata.context_generation_time = None

            return contextual_chunks

        except Exception as e:
            logger.error(f"Failed to create contextual chunks: {str(e)}")
            raise DocumentProcessingError(f"Contextual chunk creation failed: {str(e)}") from e
