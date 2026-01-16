"""Statistics MCP tool."""

from typing import Any

from ..config import Settings
from ..core.contextual_vector_store import ContextualVectorStore
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


async def get_stats_tool(
    vector_store: ContextualVectorStore,
    settings: Settings,
) -> dict[str, Any]:
    """
    Get system statistics and configuration information.

    This tool returns statistics about the RAG system, including document counts,
    chunk counts, and configuration details.

    Args:
        vector_store: VectorStore instance
        settings: Application settings

    Returns:
        Dictionary with system statistics

    Example Response:
        {
            "status": "success",
            "statistics": {
                "total_documents": 5,
                "total_chunks": 247,
                "storage_path": "/path/to/chroma_db",
                "embedding_model": "all-mpnet-base-v2",
                "chunk_size": 400,
                "chunk_overlap": 60
            }
        }
    """
    try:
        logger.info("Collecting system statistics")

        # Get vector store stats (both collections)
        store_stats = vector_store.get_stats()

        # Combine with configuration
        stats = {
            "traditional": {
                "documents": store_stats.get("traditional_documents", 0),
                "chunks": store_stats.get("traditional_chunks", 0),
            },
            "contextual": {
                "documents": store_stats.get("contextual_documents", 0),
                "chunks": store_stats.get("contextual_chunks", 0),
            },
            "storage_path": settings.chroma_db_path,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "ollama_model": settings.OLLAMA_CONTEXT_MODEL,
        }

        logger.info(
            f"Statistics: Traditional={stats['traditional']}, Contextual={stats['contextual']}"
        )

        return {
            "status": "success",
            "statistics": stats,
        }

    except Exception as e:
        logger.error(f"Stats error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": "stats_failed",
            "message": f"Failed to get statistics: {str(e)}",
        }
