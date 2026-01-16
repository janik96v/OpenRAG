"""MCP server entry point for OpenRAG."""

import asyncio
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from .config import get_settings
from .core.chunker import TextChunker
from .core.contextual_processor import ContextualProcessor
from .core.contextual_vector_store import ContextualVectorStore
from .core.embedder import EmbeddingModel
from .core.ollama_client import OllamaClient
from .tools.ingest import ingest_text_tool
from .tools.manage import delete_document_tool, list_documents_tool
from .tools.query import query_documents_tool
from .tools.stats import get_stats_tool
from .utils.async_tasks import BackgroundTaskManager
from .utils.logger import configure_root_logger, setup_logger

# Global instances (initialized in main)
vector_store: ContextualVectorStore | None = None
chunker: TextChunker | None = None
embedding_model: EmbeddingModel | None = None
ollama_client: OllamaClient | None = None
contextual_processor: ContextualProcessor | None = None
task_manager: BackgroundTaskManager | None = None
settings = get_settings()

logger = setup_logger(__name__)


def create_server() -> Server:
    """
    Create and configure the MCP server.

    Returns:
        Configured MCP Server instance
    """
    server = Server("openrag")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """
        List all available MCP tools.

        Returns:
            List of tool definitions
        """
        return [
            Tool(
                name="ingest_text",
                description=(
                    "Ingest raw text content directly into the RAG system. "
                    "Use this when the LLM client has already extracted/parsed content from "
                    "documents (e.g., PDF, DOCX). The text will be chunked, embedded, and "
                    "stored in the vector database."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Raw text content to ingest",
                        },
                        "document_name": {
                            "type": "string",
                            "description": (
                                "Name/identifier for this document (e.g., 'report.pdf', "
                                "'meeting_notes.docx')"
                            ),
                        },
                    },
                    "required": ["text", "document_name"],
                },
            ),
            Tool(
                name="query_documents",
                description=(
                    "Search for relevant document chunks using natural language query. "
                    "Returns the most semantically similar chunks with similarity scores. "
                    "Supports both traditional and contextual RAG."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5,
                        },
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity score threshold 0-1 (default: 0.4)",
                            "default": 0.4,
                        },
                        "rag_type": {
                            "type": "string",
                            "description": (
                                "Type of RAG to use: 'traditional' (default) or 'contextual'. "
                                "Contextual RAG provides better results by adding document context "
                                "to each chunk."
                            ),
                            "enum": ["traditional", "contextual"],
                            "default": "traditional",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="list_documents",
                description=(
                    "List all ingested documents with metadata including filename, "
                    "size, chunk count, and creation date."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="delete_document",
                description=(
                    "Delete a document and all its chunks from the vector database. "
                    "This operation is permanent and cannot be undone."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "ID of the document to delete",
                        }
                    },
                    "required": ["document_id"],
                },
            ),
            Tool(
                name="get_stats",
                description=(
                    "Get system statistics including document count, chunk count, "
                    "storage configuration, and embedding model information."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[dict]:
        """
        Handle tool calls from MCP clients.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            List of content items (text results)
        """
        global \
            vector_store, \
            chunker, \
            embedding_model, \
            ollama_client, \
            contextual_processor, \
            task_manager, \
            settings

        if vector_store is None or chunker is None or embedding_model is None:
            return [
                {
                    "type": "text",
                    "text": "Error: Server not properly initialized",
                }
            ]

        try:
            if name == "ingest_text":
                result = await ingest_text_tool(
                    text=arguments["text"],
                    document_name=arguments["document_name"],
                    vector_store=vector_store,
                    chunker=chunker,
                    contextual_processor=contextual_processor,
                    task_manager=task_manager,
                )
            elif name == "query_documents":
                result = await query_documents_tool(
                    query=arguments["query"],
                    vector_store=vector_store,
                    max_results=arguments.get("max_results", 5),
                    min_similarity=arguments.get("min_similarity", 0.1),
                    rag_type=arguments.get("rag_type", "traditional"),
                )
            elif name == "list_documents":
                result = await list_documents_tool(vector_store=vector_store)
            elif name == "delete_document":
                result = await delete_document_tool(
                    document_id=arguments["document_id"],
                    vector_store=vector_store,
                )
            elif name == "get_stats":
                result = await get_stats_tool(
                    vector_store=vector_store,
                    settings=settings,
                )
            else:
                result = {
                    "status": "error",
                    "error": "unknown_tool",
                    "message": f"Unknown tool: {name}",
                }

            # Format result as MCP response
            import json

            return [{"type": "text", "text": json.dumps(result, indent=2)}]

        except Exception as e:
            logger.error(f"Tool call error: {str(e)}", exc_info=True)
            return [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "status": "error",
                            "error": "tool_execution_failed",
                            "message": str(e),
                        }
                    ),
                }
            ]

    return server


async def main() -> None:
    """
    Main entry point for the MCP server.

    Initializes all components and starts the server with stdio transport.
    """
    global \
        vector_store, \
        chunker, \
        embedding_model, \
        ollama_client, \
        contextual_processor, \
        task_manager, \
        settings

    # Configure logging
    configure_root_logger(settings.log_level)

    logger.info("=" * 80)
    logger.info("OpenRAG MCP Server Starting")
    logger.info("=" * 80)

    try:
        # Initialize components
        logger.info("Initializing components...")

        # Create text chunker
        logger.info(
            f"Creating text chunker (size={settings.chunk_size}, overlap={settings.chunk_overlap})"
        )
        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # Load embedding model
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        embedding_model = EmbeddingModel(model_name=settings.embedding_model)

        # Initialize Ollama client for context generation (optional)
        try:
            logger.info(f"Initializing Ollama client at: {settings.OLLAMA_BASE_URL}")
            ollama_client = OllamaClient(
                base_url=settings.OLLAMA_BASE_URL,
                timeout=settings.OLLAMA_TIMEOUT,
                max_retries=settings.OLLAMA_MAX_RETRIES,
            )
            logger.info(f"Ollama model: {settings.OLLAMA_CONTEXT_MODEL}")

            # Initialize contextual processor
            logger.info("Initializing contextual processor...")
            contextual_processor = ContextualProcessor(
                ollama_client=ollama_client,
                context_model=settings.OLLAMA_CONTEXT_MODEL,
                fallback_enabled=settings.OLLAMA_FALLBACK_ENABLED,
            )

            # Initialize task manager for background processing
            logger.info("Initializing background task manager...")
            task_manager = BackgroundTaskManager()

            logger.info("Contextual RAG components initialized successfully")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Ollama client: {e}. "
                "Contextual RAG features will be disabled. Traditional RAG remains available."
            )
            ollama_client = None
            contextual_processor = None
            task_manager = None

        # Initialize contextual vector store (manages both traditional and contextual collections)
        logger.info(f"Initializing contextual vector store at: {settings.chroma_db_path}")
        vector_store = ContextualVectorStore(
            persist_directory=Path(settings.chroma_db_path),
            embedding_model=embedding_model,
            base_collection_name=settings.collection_base_name,
            max_batch_size=10000,
        )

        logger.info("Core components initialized successfully")
        logger.info(f"ChromaDB path: {settings.chroma_db_path}")
        logger.info(f"Embedding model: {settings.embedding_model}")
        logger.info(f"Chunk size: {settings.chunk_size} tokens")
        logger.info(f"Chunk overlap: {settings.chunk_overlap} tokens")

        if contextual_processor is not None and task_manager is not None:
            logger.info(f"Contextual RAG enabled with model: {settings.OLLAMA_CONTEXT_MODEL}")
            logger.info(f"Background tasks: {task_manager.task_count}")
        else:
            logger.info("Contextual RAG disabled (Ollama not available)")

        # Create and run server
        logger.info("Starting MCP server with stdio transport...")
        server = create_server()

        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server ready and listening for requests")
            await server.run(read_stream, write_stream, server.create_initialization_options())

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("OpenRAG MCP Server stopped")
        logger.info("=" * 80)


def run() -> None:
    """Run the server (synchronous entry point)."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
