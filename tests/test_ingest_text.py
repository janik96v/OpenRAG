"""Tests for text-based document ingestion."""

import pytest

from openrag.core.chunker import TextChunker
from openrag.core.embedder import EmbeddingModel
from openrag.core.vector_store import VectorStore
from openrag.tools.ingest import ingest_text_tool


@pytest.fixture
def sample_text_content() -> str:
    """Provide sample text content for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on
    the development of algorithms and statistical models. These models enable
    computers to improve their performance on tasks through experience.

    Deep learning is a specialized branch of machine learning that uses neural
    networks with multiple layers. This approach has revolutionized fields like
    computer vision and natural language processing.

    The future of AI involves combining various techniques to create more
    robust and general-purpose systems that can adapt to new situations.
    """


@pytest.mark.asyncio
async def test_ingest_text_success(
    vector_store: VectorStore,
    chunker: TextChunker,
    sample_text_content: str,
):
    """Test successful text ingestion."""
    result = await ingest_text_tool(
        text=sample_text_content,
        document_name="ml_guide.pdf",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "success"
    assert "document_id" in result
    assert result["document_name"] == "ml_guide.pdf"
    assert result["chunk_count"] > 0
    assert "text_size_bytes" in result
    assert "Successfully ingested" in result["message"]


@pytest.mark.asyncio
async def test_ingest_text_empty_string(
    vector_store: VectorStore,
    chunker: TextChunker,
):
    """Test that empty text is rejected."""
    result = await ingest_text_tool(
        text="",
        document_name="empty.txt",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "error"
    assert result["error"] == "validation_error"
    assert "empty" in result["message"].lower()


@pytest.mark.asyncio
async def test_ingest_text_whitespace_only(
    vector_store: VectorStore,
    chunker: TextChunker,
):
    """Test that whitespace-only text is rejected."""
    result = await ingest_text_tool(
        text="   \n\t  ",
        document_name="whitespace.txt",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "error"
    assert result["error"] == "validation_error"
    assert "whitespace" in result["message"].lower()


@pytest.mark.asyncio
async def test_ingest_text_empty_document_name(
    vector_store: VectorStore,
    chunker: TextChunker,
    sample_text_content: str,
):
    """Test that empty document name is rejected."""
    result = await ingest_text_tool(
        text=sample_text_content,
        document_name="",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "error"
    assert result["error"] == "validation_error"
    assert "document name" in result["message"].lower()


@pytest.mark.asyncio
async def test_ingest_text_invalid_text_type(
    vector_store: VectorStore,
    chunker: TextChunker,
):
    """Test that non-string text is rejected."""
    result = await ingest_text_tool(
        text=12345,  # type: ignore
        document_name="test.txt",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "error"
    assert result["error"] == "validation_error"
    assert "string" in result["message"].lower()


@pytest.mark.asyncio
async def test_ingest_text_various_document_names(
    vector_store: VectorStore,
    chunker: TextChunker,
    sample_text_content: str,
):
    """Test ingestion with various document name formats."""
    document_names = [
        "report.pdf",
        "meeting_notes.docx",
        "research_paper.txt",
        "presentation.pptx",
        "data_analysis.xlsx",
    ]

    for doc_name in document_names:
        result = await ingest_text_tool(
            text=sample_text_content,
            document_name=doc_name,
            vector_store=vector_store,
            chunker=chunker,
        )

        assert result["status"] == "success"
        assert result["document_name"] == doc_name


@pytest.mark.asyncio
async def test_ingest_text_large_content(
    vector_store: VectorStore,
    chunker: TextChunker,
):
    """Test ingestion of large text content."""
    # Create large text (approximately 10KB)
    large_text = "This is a test sentence. " * 400

    result = await ingest_text_tool(
        text=large_text,
        document_name="large_document.pdf",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "success"
    assert result["chunk_count"] > 1  # Should be chunked into multiple pieces


@pytest.mark.asyncio
async def test_ingest_text_unicode_content(
    vector_store: VectorStore,
    chunker: TextChunker,
):
    """Test ingestion of text with unicode characters."""
    unicode_text = """
    This document contains unicode: 你好世界 🌍
    Mathematical symbols: ∑∫∂∇
    Special characters: é ñ ü ö
    Emojis: 😀 🚀 💡
    """

    result = await ingest_text_tool(
        text=unicode_text,
        document_name="unicode_doc.pdf",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "success"
    assert result["chunk_count"] > 0


@pytest.mark.asyncio
async def test_ingest_text_retrieval(
    vector_store: VectorStore,
    chunker: TextChunker,
    sample_text_content: str,
):
    """Test that ingested text can be retrieved via search."""
    # Ingest text
    result = await ingest_text_tool(
        text=sample_text_content,
        document_name="ml_concepts.pdf",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "success"
    document_id = result["document_id"]

    # Search for content
    search_results = vector_store.search(
        query="What is deep learning?",
        n_results=5,
        min_similarity=0.1,
    )

    # Should find at least one relevant chunk
    assert len(search_results) > 0

    # Verify at least one result is from our document
    found_our_doc = any(chunk.document_id == document_id for chunk, _ in search_results)
    assert found_our_doc


@pytest.mark.asyncio
async def test_ingest_text_whitespace_trimming(
    vector_store: VectorStore,
    chunker: TextChunker,
):
    """Test that leading/trailing whitespace is trimmed."""
    text_with_whitespace = """

        This text has lots of whitespace around it.


    """

    result = await ingest_text_tool(
        text=text_with_whitespace,
        document_name="  whitespace_doc.txt  ",
        vector_store=vector_store,
        chunker=chunker,
    )

    assert result["status"] == "success"
    assert result["document_name"] == "whitespace_doc.txt"  # Trimmed
