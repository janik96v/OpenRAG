"""Query/search MCP tool."""

from typing import Any

from ..core.vector_store import VectorStore
from ..utils.logger import setup_logger
from ..utils.validation import ValidationError, validate_max_results, validate_query

logger = setup_logger(__name__)


async def query_documents_tool(
    query: str,
    vector_store: VectorStore,
    max_results: int = 5,
    min_similarity: float = 0.1,
) -> dict[str, Any]:
    """
    Search for relevant document chunks using natural language query.

    This tool performs semantic search over ingested documents and returns
    the most relevant chunks with similarity scores.

    Args:
        query: Natural language search query
        vector_store: VectorStore instance
        max_results: Maximum number of results to return (default: 5)
        min_similarity: Minimum similarity score threshold 0-1 (default: 0.1)

    Returns:
        Dictionary with search results

    Example Response:
        {
            "status": "success",
            "query": "How to train a model?",
            "results": [
                {
                    "chunk_id": "...",
                    "document_id": "...",
                    "document_name": "ml_guide.txt",
                    "content": "Training a model involves...",
                    "similarity_score": 0.87,
                    "chunk_index": 5
                }
            ],
            "total_results": 3
        }
    """
    try:
        # Validate query
        logger.info(f"Validating query: '{query[:100]}...'")
        validated_query = validate_query(query)

        # Validate max_results
        validated_max_results = validate_max_results(max_results)

        # Perform search
        logger.info(f"Searching for: '{validated_query[:100]}...'")
        results = vector_store.search(
            query=validated_query,
            n_results=validated_max_results,
            min_similarity=min_similarity,
        )

        # Format results
        formatted_results = []

        for chunk, similarity_score in results:
            formatted_results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.metadata.get("filename", "Unknown"),
                    "content": chunk.content,
                    "similarity_score": round(similarity_score, 4),
                    "chunk_index": chunk.chunk_index,
                }
            )

        logger.info(f"Found {len(formatted_results)} relevant chunks")

        return {
            "status": "success",
            "query": validated_query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "max_results_requested": validated_max_results,
            "min_similarity": min_similarity,
        }

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return {
            "status": "error",
            "error": "validation_error",
            "message": str(e),
        }

    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": "query_failed",
            "message": f"Failed to execute query: {str(e)}",
        }
