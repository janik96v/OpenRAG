"""Configuration management for OpenRAG using Pydantic Settings."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ChromaDB configuration
    chroma_db_path: Annotated[
        str, Field(default="./chroma_db", description="Path to ChromaDB persistent storage")
    ]

    # Embedding model configuration
    embedding_model: Annotated[
        str,
        Field(
            default="all-mpnet-base-v2",
            description="Name of the sentence transformer model to use",
        ),
    ]

    # Chunking configuration
    chunk_size: Annotated[
        int,
        Field(
            default=400,
            ge=100,
            le=2000,
            description="Maximum chunk size in tokens",
        ),
    ]

    chunk_overlap: Annotated[
        int,
        Field(
            default=60,
            ge=0,
            le=500,
            description="Overlap size in tokens between chunks",
        ),
    ]

    # Logging configuration
    log_level: Annotated[
        str,
        Field(
            default="INFO",
            description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ]

    # Ollama configuration for Contextual RAG
    ollama_base_url: Annotated[
        str,
        Field(
            default="http://localhost:11434",
            description="Base URL for Ollama API",
        ),
    ]

    ollama_context_model: Annotated[
        str,
        Field(
            default="llama3.2:3b",
            description="Ollama model to use for context generation",
        ),
    ]

    ollama_timeout: Annotated[
        float,
        Field(
            default=300.0,
            ge=10.0,
            le=600.0,
            description="Timeout in seconds for Ollama API calls (5 minutes default)",
        ),
    ]

    ollama_max_retries: Annotated[
        int,
        Field(
            default=3,
            ge=1,
            le=10,
            description="Maximum number of retry attempts for Ollama requests",
        ),
    ]

    ollama_fallback_enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Enable fallback context generation when Ollama fails",
        ),
    ]

    # Contextual RAG configuration
    contextual_batch_size: Annotated[
        int,
        Field(
            default=10,
            ge=1,
            le=50,
            description="Number of chunks to process concurrently for context generation",
        ),
    ]

    contextual_chunk_size: Annotated[
        int,
        Field(
            default=800,
            ge=100,
            le=2000,
            description="Chunk size for contextual RAG (larger than traditional due to context overhead)",
        ),
    ]

    # Collection naming
    collection_base_name: Annotated[
        str,
        Field(
            default="documents",
            description="Base name for ChromaDB collections (will add _traditional_v1 or _contextual_v1)",
        ),
    ]

    @field_validator("chroma_db_path")
    @classmethod
    def validate_chroma_path(cls, v: str) -> str:
        """Validate and create ChromaDB path if needed."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: dict) -> int:
        """Validate chunk overlap is less than chunk size."""
        # Note: info.data contains already-validated fields
        chunk_size = info.data.get("chunk_size", 400)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v

    @property
    def STORAGE_PATH(self) -> Path:
        """Get storage path as Path object for compatibility."""
        return Path(self.chroma_db_path)

    @property
    def EMBEDDING_MODEL(self) -> str:
        """Get embedding model name for compatibility."""
        return self.embedding_model

    @property
    def CHUNK_SIZE(self) -> int:
        """Get chunk size for compatibility."""
        return self.chunk_size

    @property
    def CHUNK_OVERLAP(self) -> int:
        """Get chunk overlap for compatibility."""
        return self.chunk_overlap

    @property
    def OLLAMA_BASE_URL(self) -> str:
        """Get Ollama base URL."""
        return self.ollama_base_url

    @property
    def OLLAMA_CONTEXT_MODEL(self) -> str:
        """Get Ollama context model."""
        return self.ollama_context_model

    @property
    def OLLAMA_TIMEOUT(self) -> float:
        """Get Ollama timeout."""
        return self.ollama_timeout

    @property
    def OLLAMA_MAX_RETRIES(self) -> int:
        """Get Ollama max retries."""
        return self.ollama_max_retries

    @property
    def OLLAMA_FALLBACK_ENABLED(self) -> bool:
        """Get Ollama fallback enabled."""
        return self.ollama_fallback_enabled


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
