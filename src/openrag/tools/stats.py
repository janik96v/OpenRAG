"""Statistics MCP tool."""

from typing import Any

from ..config import Settings
from ..core.vector_store import VectorStore
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


async def get_stats_tool(
    vector_store: VectorStore,
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

        # Get vector store stats
        store_stats = vector_store.get_stats()

        # Combine with configuration
        stats = {
            "total_documents": store_stats.get("total_documents", 0),
            "total_chunks": store_stats.get("total_chunks", 0),
            "storage_path": settings.chroma_db_path,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }

        logger.info(
            f"Statistics: {stats['total_documents']} documents, {stats['total_chunks']} chunks"
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
