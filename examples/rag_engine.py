"""RAG (Retrieval Augmented Generation) engine implementation."""

import logging
import time
from pathlib import Path

from ..models.contextual_schemas import (
    ContextualDocument,
    ContextualQueryRequest,
    RAGType,
)
from ..models.schemas import (
    Conversation,
    ConversationMessage,
    Document,
    QueryRequest,
    QueryResponse,
)
from .base_llm_client import BaseLLMClient
from .contextual_processor import ContextualProcessor
from .contextual_vector_store import ContextualVectorStore
from .document_manager import DocumentManager
from .document_processor import DocumentProcessor
from .system_prompt_loader import load_system_prompt
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEngineError(Exception):
    """Exception raised during RAG engine operations."""

    pass


class RAGEngine:
    """
    Retrieval Augmented Generation engine that combines document retrieval
    with LLM generation.
    """

    def __init__(
        self,
        vector_store: VectorStore | ContextualVectorStore,
        llm_client: BaseLLMClient,
        document_processor: DocumentProcessor | None = None,
        document_manager: DocumentManager | None = None,
        contextual_processor: ContextualProcessor | None = None,
        max_context_chunks: int = 5,
        similarity_threshold: float = 0.1,
    ):
        """
        Initialize the RAG engine.

        Args:
            vector_store: Vector store for document retrieval (can be ContextualVectorStore)
            llm_client: LLM client for generation (Ollama, HuggingFace, or other providers)
            document_processor: Document processor for ingesting new documents
            document_manager: Document manager for persistent metadata storage
            contextual_processor: Optional contextual processor for context generation
            max_context_chunks: Maximum chunks to use as context
            similarity_threshold: Minimum similarity score for relevance
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.contextual_processor = contextual_processor

        # Initialize document processor with contextual processor if available
        if self.contextual_processor and document_processor is None:
            self.document_processor = DocumentProcessor(contextual_processor=self.contextual_processor)
        else:
            self.document_processor = document_processor or DocumentProcessor()

        self.document_manager = document_manager or DocumentManager()
        self.max_context_chunks = max_context_chunks
        self.similarity_threshold = similarity_threshold

        # In-memory conversation storage (could be replaced with database)
        self._conversations: dict[str, Conversation] = {}

        logger.info("RAG Engine initialized successfully")

    def ingest_document(
        self, file_path: Path, rag_type: RAGType = RAGType.TRADITIONAL
    ) -> Document | ContextualDocument:
        """
        Process and ingest a document into the system.

        Args:
            file_path: Path to the document file
            rag_type: Type of RAG processing to use

        Returns:
            Processed document (Document or ContextualDocument)

        Raises:
            RAGEngineError: If document ingestion fails
        """
        try:
            # Validate file
            self.document_processor.validate_file(file_path)

            # Check if contextual RAG is requested but not available
            if rag_type == RAGType.CONTEXTUAL and not self.contextual_processor:
                logger.warning(
                    f"Contextual RAG requested but no contextual processor available. "
                    f"Falling back to traditional RAG for {file_path.name}"
                )
                rag_type = RAGType.TRADITIONAL

            # Process document with specified RAG type
            logger.info(f"Processing document: {file_path.name} with {rag_type.value} RAG")
            document = self.document_processor.process_document(file_path, rag_type)

            # Add to vector store with appropriate RAG type
            logger.info(f"Adding document to vector store: {document.document_id}")
            if isinstance(self.vector_store, ContextualVectorStore):
                self.vector_store.add_document(document, rag_type)
            else:
                # Traditional vector store - only supports traditional RAG
                if rag_type == RAGType.CONTEXTUAL:
                    logger.warning("Traditional vector store cannot handle contextual documents")
                self.vector_store.add_document(document)

            # Add to document manager
            logger.info(f"Adding document to metadata storage: {document.document_id}")
            self.document_manager.add_document(document)

            logger.info(f"Successfully ingested document: {file_path.name} with {rag_type.value} RAG")
            return document

        except Exception as e:
            raise RAGEngineError(f"Failed to ingest document: {str(e)}") from e

    def query(self, request: QueryRequest | ContextualQueryRequest) -> QueryResponse:
        """
        Process a query using RAG workflow.

        Args:
            request: Query request with user question (can include RAG type preference)

        Returns:
            Query response with generated answer

        Raises:
            RAGEngineError: If query processing fails
        """
        start_time = time.time()

        try:
            # Determine RAG type for query
            rag_type = getattr(request, 'rag_type', RAGType.TRADITIONAL)
            logger.info(f"Processing query with {rag_type.value} RAG: {request.query[:100]}...")

            # Retrieve relevant document chunks
            if isinstance(self.vector_store, ContextualVectorStore):
                chunks_with_scores = self.vector_store.search(
                    query=request.query,
                    n_results=request.max_results,
                    rag_type=rag_type
                )
            else:
                # Traditional vector store
                if rag_type == RAGType.CONTEXTUAL:
                    logger.warning("Contextual query requested but using traditional vector store")
                chunks_with_scores = self.vector_store.search(
                    query=request.query, n_results=request.max_results
                )

            # Filter by similarity threshold
            relevant_chunks = [
                chunk for chunk, score in chunks_with_scores if score >= self.similarity_threshold
            ]

            logger.info(f"Found {len(relevant_chunks)} relevant chunks")

            # Limit context to max_context_chunks
            context_chunks = relevant_chunks[: self.max_context_chunks]

            # Generate response using LLM
            system_prompt = self._get_system_prompt()

            response_text = self.llm_client.generate_response(
                prompt=request.query,
                context_chunks=context_chunks,
                system_prompt=system_prompt,
                temperature=0.7,
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create response
            response = QueryResponse(
                response=response_text,
                sources=context_chunks,
                conversation_id=request.conversation_id or "default",
                processing_time=processing_time,
            )

            # Update conversation if conversation_id is provided
            if request.conversation_id:
                self._update_conversation(request, response)

            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response

        except Exception as e:
            raise RAGEngineError(f"Failed to process query: {str(e)}") from e

    async def stream_query(self, request: QueryRequest):
        """
        Process a query with async streaming response.

        Args:
            request: Query request with user question

        Yields:
            Response chunks as they are generated

        Raises:
            RAGEngineError: If streaming fails
        """
        try:
            logger.info(f"Processing streaming query: {request.query[:100]}...")

            # Retrieve relevant document chunks
            chunks_with_scores = self.vector_store.search(
                query=request.query, n_results=request.max_results
            )

            # Filter by similarity threshold
            relevant_chunks = [
                chunk for chunk, score in chunks_with_scores if score >= self.similarity_threshold
            ]

            # Limit context to max_context_chunks
            context_chunks = relevant_chunks[: self.max_context_chunks]

            # Generate streaming response
            system_prompt = self._get_system_prompt()

            # Use sync generator but yield control to event loop after each chunk
            # Chunks are now typed: {"type": "thinking"|"content", "data": str}
            for typed_chunk in self.llm_client.generate_streaming_response(
                prompt=request.query,
                context_chunks=context_chunks,
                system_prompt=system_prompt,
                temperature=0.7,
            ):
                # Pass through typed chunks directly
                yield typed_chunk
                # Yield control to allow WebSocket to send
                import asyncio
                await asyncio.sleep(0)

            # Send sources at the end
            yield {
                "type": "sources",
                "data": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content[:200] + "..."
                        if len(chunk.content) > 200
                        else chunk.content,
                        "filename": chunk.metadata.get("filename", "Unknown"),
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in context_chunks
                ],
            }

        except Exception as e:
            yield {"type": "error", "data": f"Failed to process streaming query: {str(e)}"}

    def create_conversation(self, title: str | None = None) -> Conversation:
        """
        Create a new conversation.

        Args:
            title: Optional conversation title

        Returns:
            New conversation object
        """
        conversation = Conversation(title=title)
        self._conversations[conversation.conversation_id] = conversation

        logger.info(f"Created conversation: {conversation.conversation_id}")
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """
        Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Conversation object or None if not found
        """
        return self._conversations.get(conversation_id)

    def list_conversations(self) -> list[Conversation]:
        """
        List all conversations.

        Returns:
            List of conversation objects
        """
        return list(self._conversations.values())

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: ID of the conversation to delete

        Returns:
            True if conversation was deleted
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the system.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if document was deleted

        Raises:
            RAGEngineError: If deletion fails
        """
        try:
            # Delete from vector store
            vector_success = self.vector_store.delete_document(document_id)

            # Delete from document manager
            manager_success = self.document_manager.remove_document(document_id)

            # Both operations should succeed for complete deletion
            success = vector_success and manager_success

            if success:
                logger.info(f"Successfully deleted document: {document_id}")
            else:
                logger.warning(f"Document not found or partially deleted: {document_id}")
            return success

        except Exception as e:
            raise RAGEngineError(f"Failed to delete document: {str(e)}") from e

    def get_system_stats(self) -> dict:
        """
        Get system statistics.

        Returns:
            Dictionary with system statistics
        """
        try:
            document_stats = self.document_manager.get_stats()

            # Check LLM connection
            llm_connected = self.llm_client.check_connection()

            # Get available models
            available_models = []
            if llm_connected:
                try:
                    available_models = self.llm_client.list_models()
                except Exception:
                    pass

            return {
                "documents": {
                    "total_documents": document_stats.get("total_documents", 0),
                    "total_chunks": document_stats.get("total_chunks", 0),
                    "total_size": document_stats.get("total_size", 0),
                    "formats": document_stats.get("formats", {}),
                },
                "conversations": {"active_conversations": len(self._conversations)},
                "llm": {
                    "provider": self.llm_client.provider_name,
                    "connected": llm_connected,
                    "available_models": available_models,
                    "current_model": self.llm_client.model,
                },
                "vector_store": {
                    "collection_name": self.vector_store.collection_name,
                    "embedding_model": self.vector_store.embedding_model,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get system stats: {str(e)}")
            return {"error": str(e)}

    def _update_conversation(self, request: QueryRequest, response: QueryResponse) -> None:
        """
        Update conversation with new messages.

        Args:
            request: Original query request
            response: Generated response
        """
        conversation_id = request.conversation_id
        if not conversation_id:
            return

        # Get or create conversation
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            conversation = Conversation(conversation_id=conversation_id)
            self._conversations[conversation_id] = conversation

        # Add user message
        user_message = ConversationMessage(
            conversation_id=conversation_id, content=request.query, is_user=True
        )
        conversation.messages.append(user_message)

        # Add assistant message
        assistant_message = ConversationMessage(
            conversation_id=conversation_id,
            content=response.response,
            is_user=False,
            sources=response.sources,
        )
        conversation.messages.append(assistant_message)

        # Update conversation title if empty and this is the first exchange
        if not conversation.title and len(conversation.messages) == 2:
            # Use first few words of the query as title
            title_words = request.query.split()[:6]
            conversation.title = " ".join(title_words)
            if len(title_words) == 6:
                conversation.title += "..."

        conversation.updated_at = response.created_at

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM.

        Loads the system prompt from the custom system_prompt.md file if it exists,
        otherwise falls back to the default prompt.

        Returns:
            System prompt string
        """
        return load_system_prompt()
