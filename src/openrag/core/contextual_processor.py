"""Context generation processor for Contextual RAG implementation."""

import asyncio
import logging
import time
from datetime import UTC, datetime

from ..models.contextual_schemas import (
    ContextGenerationRequest,
    ContextGenerationResponse,
    ContextualDocumentChunk,
    RAGType,
)
from ..models.schemas import DocumentChunk, DocumentMetadata
from .ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)


class ContextualProcessorError(Exception):
    """Exception raised during contextual processing operations."""

    pass


class ContextualProcessor:
    """Processor for generating contextual information for document chunks."""

    # Anthropic's exact prompt template for context generation
    CONTEXT_PROMPT_TEMPLATE = """<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context (2-3 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        context_model: str = "llama3.2:3b",
        fallback_enabled: bool = True,
    ):
        """
        Initialize the contextual processor.

        Args:
            ollama_client: Ollama client for LLM calls
            context_model: Model to use for context generation
            fallback_enabled: Whether to use fallback context on errors
        """
        self.ollama_client = ollama_client
        self.context_model = context_model
        self.fallback_enabled = fallback_enabled

        logger.info(f"Initialized ContextualProcessor with model: {context_model}")

    async def generate_context(
        self,
        chunk: DocumentChunk,
        document_metadata: DocumentMetadata,
        full_document_text: str,
    ) -> ContextGenerationResponse:
        """
        Generate contextual information for a document chunk using the full document.

        CRITICAL: This method requires the FULL document text to generate meaningful context.

        Args:
            chunk: Document chunk to generate context for
            document_metadata: Metadata about the source document
            full_document_text: Complete text content of the document for context generation

        Returns:
            Context generation response with generated context

        Raises:
            ContextualProcessorError: If context generation fails and fallback is disabled
        """
        start_time = time.time()

        try:
            logger.debug(f"Generating context for chunk {chunk.chunk_id}")

            # Create prompt using Anthropic's template
            prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
                full_document=full_document_text, chunk_content=chunk.content
            )

            # Generate context using Ollama
            generated_context = await self.ollama_client.generate_response(
                prompt=prompt,
                model=self.context_model,
                temperature=0.3,
                max_tokens=100,  # Keep context concise
            )

            # Clean up generated context (remove extra whitespace)
            generated_context = generated_context.strip()

            # Combine context with original content
            contextual_content = self._create_contextual_content(generated_context, chunk.content)

            generation_time = time.time() - start_time

            response = ContextGenerationResponse(
                generated_context=generated_context,
                contextual_content=contextual_content,
                generation_time=generation_time,
                model_used=self.context_model,
                success=True,
                error_message=None,
            )

            logger.debug(
                f"Successfully generated context for chunk {chunk.chunk_id} in {generation_time:.2f}s"
            )

            return response

        except Exception as e:
            generation_time = time.time() - start_time
            error_message = f"Context generation failed for chunk {chunk.chunk_id}: {str(e)}"

            logger.warning(error_message)

            if self.fallback_enabled:
                # Use fallback context
                fallback_context = self._create_fallback_context(document_metadata)
                contextual_content = self._create_contextual_content(
                    fallback_context, chunk.content
                )

                return ContextGenerationResponse(
                    generated_context=fallback_context,
                    contextual_content=contextual_content,
                    generation_time=generation_time,
                    model_used="fallback",
                    success=False,
                    error_message=error_message,
                )
            else:
                raise ContextualProcessorError(error_message) from e

    async def process_document_chunks(
        self,
        chunks: list[DocumentChunk],
        document_metadata: DocumentMetadata,
        full_document_text: str,
    ) -> list[ContextualDocumentChunk]:
        """
        Process all chunks in a document to generate contextual versions.

        Uses batched concurrent processing to handle multiple chunks efficiently
        while avoiding overwhelming the system.

        Args:
            chunks: List of document chunks to process
            document_metadata: Metadata about the source document
            full_document_text: Complete text content of the document for context generation

        Returns:
            List of contextual document chunks with generated context

        Raises:
            ContextualProcessorError: If processing fails
        """
        try:
            logger.info(f"Processing {len(chunks)} chunks for contextual RAG")

            contextual_chunks = []
            batch_size = 10  # Process 10 chunks concurrently at a time

            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                logger.debug(f"Processing batch {i // batch_size + 1} ({len(batch)} chunks)")

                # Generate contexts concurrently for this batch
                tasks = [
                    self.generate_context(chunk, document_metadata, full_document_text)
                    for chunk in batch
                ]

                # Gather results with exception handling
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Create contextual chunks from responses
                for chunk, response in zip(batch, responses):
                    if isinstance(response, Exception):
                        # Error occurred, create traditional chunk with fallback
                        logger.error(f"Failed to process chunk {chunk.chunk_id}: {response}")

                        contextual_chunk = ContextualDocumentChunk(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            content=chunk.content,
                            contextual_content=chunk.content,  # Fallback to original
                            chunk_index=chunk.chunk_index,
                            metadata=chunk.metadata,
                            rag_type=RAGType.TRADITIONAL,  # Mark as traditional on failure
                            context_generated_at=None,
                            created_at=chunk.created_at,
                        )
                    else:
                        # Success, create contextual chunk
                        contextual_chunk = ContextualDocumentChunk(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            content=chunk.content,
                            contextual_content=response.contextual_content,
                            chunk_index=chunk.chunk_index,
                            metadata={
                                **chunk.metadata,
                                "context_model": response.model_used,
                                "context_success": response.success,
                                "context_generation_time": response.generation_time,
                            },
                            rag_type=RAGType.CONTEXTUAL,
                            context_generated_at=datetime.now(UTC),
                            created_at=chunk.created_at,
                        )

                    contextual_chunks.append(contextual_chunk)

            logger.info(f"Successfully processed {len(contextual_chunks)} contextual chunks")

            return contextual_chunks

        except Exception as e:
            logger.exception(f"Failed to process document chunks: {e}")
            raise ContextualProcessorError(f"Document processing failed: {e}") from e

    def _create_contextual_content(self, context: str, original_content: str) -> str:
        """
        Combine context with original content in a structured format.

        Args:
            context: Generated context
            original_content: Original chunk content

        Returns:
            Combined contextual content
        """
        return f"Context: {context}\n\nContent: {original_content}"

    def _create_fallback_context(self, document_metadata: DocumentMetadata) -> str:
        """
        Create a simple fallback context when generation fails.

        Args:
            document_metadata: Document metadata

        Returns:
            Fallback context string
        """
        return f"This is a section from the document '{document_metadata.filename}'."
