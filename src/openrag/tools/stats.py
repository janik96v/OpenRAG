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

        # Get vector store stats (all collections)
        store_stats = vector_store.get_stats()

        # Combine with configuration
        # Handle both VectorStore (unique_documents/total_chunks) and
        # ContextualVectorStore (traditional_documents/traditional_chunks) key formats
        trad_docs = store_stats.get(
            "traditional_documents", store_stats.get("unique_documents", 0)
        )
        trad_chunks = store_stats.get(
            "traditional_chunks", store_stats.get("total_chunks", 0)
        )
        stats = {
            "total_documents": trad_docs,
            "total_chunks": trad_chunks,
            "traditional": {
                "enabled": settings.traditional_enabled,
                "documents": trad_docs,
                "chunks": trad_chunks,
            },
            "contextual": {
                "enabled": settings.contextual_enabled,
                "documents": store_stats.get("contextual_documents", 0),
                "chunks": store_stats.get("contextual_chunks", 0),
            },
            "graph": {
                "enabled": settings.graph_enabled,
                "documents": store_stats.get("graph_documents", 0),
                "chunks": store_stats.get("graph_chunks", 0),
            },
            "rag_types_enabled": {
                "traditional": settings.traditional_enabled,
                "contextual": settings.contextual_enabled,
                "graph": settings.graph_enabled,
            },
            "storage_path": settings.chroma_db_path,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "ollama_model": settings.OLLAMA_CONTEXT_MODEL,
        }

        logger.info(
            f"Statistics: Traditional={stats['traditional']}, "
            f"Contextual={stats['contextual']}, Graph={stats['graph']}"
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
