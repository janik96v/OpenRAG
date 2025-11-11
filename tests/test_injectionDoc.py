#! /usr/bin/env python3
"""Test script for injecting a text in OpenRAG."""

import pytest
from pathlib import Path
from openrag.core.chunker import TextChunker
from openrag.core.embedder import EmbeddingModel
from openrag.core.vector_store import VectorStore
from openrag.tools.ingest import ingest_document_tool


@pytest.mark.asyncio
async def test_ingest_document():
    """Test document ingestion into vector store."""
    # Initialize components
    chunker = TextChunker(chunk_size=400, chunk_overlap=60)
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(
        persist_directory=Path("./tests/test_chroma_db"),
        embedding_model=embedding_model
    )

    # Ingest document
    result = await ingest_document_tool(
        file_path="./tests/test_doc.txt",
        vector_store=vector_store,
        chunker=chunker
    )

    # Assertions
    assert result is not None
    assert "document_id" in result
    assert "chunk_count" in result
    assert result["chunk_count"] > 0

    print(f"✅ Document ingested!")
    print(f"   Document ID: {result['document_id']}")
    print(f"   Chunks: {result['chunk_count']}")