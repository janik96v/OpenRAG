"""Integration tests for OpenRAG end-to-end workflows."""

import pytest

from openrag.tools.ingest import ingest_document_tool
from openrag.tools.manage import delete_document_tool, list_documents_tool
from openrag.tools.query import query_documents_tool
from openrag.tools.stats import get_stats_tool


@pytest.mark.asyncio
async def test_full_workflow(sample_txt_file, vector_store, chunker, test_settings):
    """Test complete workflow: ingest -> query -> list -> delete."""
    # 1. Ingest document
    ingest_result = await ingest_document_tool(
        file_path=str(sample_txt_file),
        vector_store=vector_store,
        chunker=chunker,
    )

    assert ingest_result["status"] == "success"
    assert "document_id" in ingest_result
    assert ingest_result["chunk_count"] > 0
    document_id = ingest_result["document_id"]

    # 2. Query documents
    query_result = await query_documents_tool(
        query="What is artificial intelligence?",
        vector_store=vector_store,
        max_results=5,
    )

    assert query_result["status"] == "success"
    assert len(query_result["results"]) > 0
    assert query_result["total_results"] > 0

    # Check result structure
    first_result = query_result["results"][0]
    assert "content" in first_result
    assert "similarity_score" in first_result
    assert "document_id" in first_result

    # 3. List documents
    list_result = await list_documents_tool(vector_store=vector_store)

    assert list_result["status"] == "success"
    assert list_result["total_documents"] >= 1
    assert len(list_result["documents"]) >= 1

    # 4. Get stats
    stats_result = await get_stats_tool(
        vector_store=vector_store,
        settings=test_settings,
    )

    assert stats_result["status"] == "success"
    assert stats_result["statistics"]["total_documents"] >= 1
    assert stats_result["statistics"]["total_chunks"] > 0

    # 5. Delete document
    delete_result = await delete_document_tool(
        document_id=document_id,
        vector_store=vector_store,
    )

    assert delete_result["status"] == "success"
    assert delete_result["document_id"] == document_id

    # 6. Verify deletion
    list_after_delete = await list_documents_tool(vector_store=vector_store)
    # Document count should be reduced (or 0 if it was the only one)
    assert (
        list_after_delete["total_documents"] < list_result["total_documents"]
        or list_after_delete["total_documents"] == 0
    )


@pytest.mark.asyncio
async def test_query_no_documents(vector_store):
    """Test querying when no documents are ingested."""
    result = await query_documents_tool(
        query="test query",
        vector_store=vector_store,
    )

    assert result["status"] == "success"
    assert result["total_results"] == 0
    assert len(result["results"]) == 0


@pytest.mark.asyncio
async def test_delete_nonexistent_document(vector_store):
    """Test deleting a document that doesn't exist."""
    result = await delete_document_tool(
        document_id="nonexistent-id-12345",
        vector_store=vector_store,
    )

    assert result["status"] == "error"
    assert result["error"] == "not_found"


@pytest.mark.asyncio
async def test_ingest_invalid_file(vector_store, chunker):
    """Test ingesting an invalid file path."""
    result = await ingest_document_tool(
        file_path="/nonexistent/file.txt",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "error"
    assert "error" in result


@pytest.mark.asyncio
async def test_query_validation(vector_store):
    """Test query with invalid parameters."""
    # Empty query
    result = await query_documents_tool(
        query="",
        vector_store=vector_store,
    )

    assert result["status"] == "error"
    assert result["error"] == "validation_error"
