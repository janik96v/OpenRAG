#! usr/bin/env python3
"""Test script for querying documents in OpenRAG."""

import asyncio
import pytest
from pathlib import Path
from openrag.tools.query import query_documents_tool
from openrag.core.embedder import EmbeddingModel
from openrag.core.vector_store import VectorStore

@pytest.mark.asyncio
async def test_query():
    # Initialize components (same as above)
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(
        persist_directory=Path("./tests/test_chroma_db"),
        embedding_model=embedding_model
    )

    # Search
    result = await query_documents_tool(
        query="What is Machine Learning?",
        vector_store=vector_store,
        max_results=3
    )

    print(f"\n🔍 Query: What is machine learning?")
    print(f"   Found {result['total_results']} results:\n")

    for i, res in enumerate(result['results'], 1):
        print(f"{i}. Score: {res['similarity_score']:.3f}")
        print(f"   {res['content'][:150]}...\n")
