"""Pydantic data models for OpenRAG."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentChunk(BaseModel):
    """Represents a chunk of a document with metadata."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    chunk_index: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Document metadata."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    filename: str
    file_size: int  # bytes
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Document(BaseModel):
    """Complete document with chunks."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: DocumentMetadata
    chunks: list[DocumentChunk] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Search query request."""

    query: str
    max_results: int = 5
    min_similarity: float = 0.1


class QueryResult(BaseModel):
    """Single query result."""

    chunk: DocumentChunk
    similarity_score: float
    document_name: str


class QueryResponse(BaseModel):
    """Query response with results."""

    results: list[QueryResult]
    query: str
    total_results: int


class DocumentInfo(BaseModel):
    """Document information for listing."""

    document_id: str
    filename: str
    chunk_count: int
    file_size: int
    created_at: datetime
    status: DocumentStatus


class StatsResponse(BaseModel):
    """System statistics response."""

    total_documents: int
    total_chunks: int
    storage_path: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
