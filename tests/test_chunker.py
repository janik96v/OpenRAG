"""Tests for text chunking functionality."""

import pytest

from openrag.core.chunker import TextChunker


def test_chunker_initialization():
    """Test TextChunker initialization."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 20
    assert len(chunker.separators) > 0


def test_count_tokens():
    """Test token counting."""
    chunker = TextChunker()
    text = "Hello world"
    token_count = chunker.count_tokens(text)
    assert isinstance(token_count, int)
    assert token_count > 0


def test_chunk_text_basic(sample_text: str):
    """Test basic text chunking."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_text)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunker = TextChunker()
    chunks = chunker.chunk_text("")
    assert chunks == []

    chunks = chunker.chunk_text("   ")
    assert chunks == []


def test_chunk_text_respects_token_limit():
    """Test that chunks respect token size limits."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "word " * 200  # Create long text
    chunks = chunker.chunk_text(text)

    for chunk in chunks:
        token_count = chunker.count_tokens(chunk)
        # Allow small overflow due to separator handling
        assert token_count <= chunker.chunk_size + 10


def test_chunk_metadata():
    """Test chunk metadata generation."""
    chunker = TextChunker()
    chunks = ["Hello world", "This is a test", "Final chunk"]
    metadata = chunker.get_chunk_metadata(chunks)

    assert len(metadata) == len(chunks)
    for i, meta in enumerate(metadata):
        assert meta["chunk_index"] == i
        assert "token_count" in meta
        assert "char_count" in meta
