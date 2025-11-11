"""Pytest configuration and fixtures for OpenRAG tests."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from openrag.config import Settings
from openrag.core.chunker import TextChunker
from openrag.core.embedder import EmbeddingModel
from openrag.core.vector_store import VectorStore
from openrag.models.schemas import Document, DocumentChunk, DocumentMetadata, DocumentStatus


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def chroma_dir(temp_dir: Path) -> Path:
    """Create a temporary ChromaDB directory."""
    chroma_path = temp_dir / "chroma_db"
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chroma_path


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to
    the natural intelligence displayed by humans and animals. Leading AI textbooks define
    the field as the study of "intelligent agents": any device that perceives its
    environment and takes actions that maximize its chance of successfully achieving its
    goals.

    Machine learning (ML) is the study of computer algorithms that can improve automatically
    through experience and by the use of data. It is seen as a part of artificial
    intelligence. Machine learning algorithms build a model based on sample data, known as
    training data, in order to make predictions or decisions without being explicitly
    programmed to do so.

    Deep learning is part of a broader family of machine learning methods based on
    artificial neural networks with representation learning. Learning can be supervised,
    semi-supervised or unsupervised.
    """.strip()


@pytest.fixture
def sample_txt_file(temp_dir: Path, sample_text: str) -> Path:
    """Create a sample .txt file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text(sample_text, encoding="utf-8")
    return file_path


@pytest.fixture
def test_settings(chroma_dir: Path) -> Settings:
    """Create test settings."""
    return Settings(
        chroma_db_path=str(chroma_dir),
        embedding_model="all-MiniLM-L6-v2",  # Smaller model for faster tests
        chunk_size=200,  # Smaller chunks for faster tests
        chunk_overlap=30,
        log_level="WARNING",  # Reduce log noise in tests
    )


@pytest.fixture
def chunker(test_settings: Settings) -> TextChunker:
    """Create a TextChunker instance for testing."""
    return TextChunker(
        chunk_size=test_settings.chunk_size,
        chunk_overlap=test_settings.chunk_overlap,
    )


@pytest.fixture
def embedding_model() -> EmbeddingModel:
    """Create an EmbeddingModel instance for testing."""
    # Use lightweight model for fast tests
    return EmbeddingModel(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(chroma_dir: Path, embedding_model: EmbeddingModel) -> VectorStore:
    """Create a VectorStore instance for testing."""
    return VectorStore(
        persist_directory=chroma_dir,
        embedding_model=embedding_model,
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    """Create a sample Document with chunks."""
    metadata = DocumentMetadata(
        filename="test.txt",
        file_size=len(sample_text),
        chunk_count=0,
        status=DocumentStatus.COMPLETED,
    )

    document = Document(metadata=metadata)

    # Create a few chunks
    chunks_text = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines.",
        "Machine learning (ML) is the study of computer algorithms.",
        "Deep learning is part of a broader family of machine learning methods.",
    ]

    for i, chunk_text in enumerate(chunks_text):
        chunk = DocumentChunk(
            document_id=document.document_id,
            content=chunk_text,
            chunk_index=i,
        )
        document.chunks.append(chunk)

    document.metadata.chunk_count = len(document.chunks)

    return document
