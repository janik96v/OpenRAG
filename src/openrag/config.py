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


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
