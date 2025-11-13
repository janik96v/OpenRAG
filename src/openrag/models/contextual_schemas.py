"""Pydantic data models for Contextual RAG functionality."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .schemas import DocumentMetadata


class RAGType(str, Enum):
    """Type of RAG processing applied to chunks."""

    TRADITIONAL = "traditional"
    CONTEXTUAL = "contextual"


class ContextualDocumentChunk(BaseModel):
    """Document chunk with optional contextual information."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str  # Original chunk content
    contextual_content: Optional[str] = None  # "Context: <summary>\n\nContent: <original>"
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    rag_type: RAGType = RAGType.TRADITIONAL
    context_generated_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ContextGenerationRequest(BaseModel):
    """Request for generating contextual information for a chunk."""

    chunk_content: str
    document_filename: str
    document_format: str
    full_document_text: str  # CRITICAL: Full doc for context awareness
    model: str = "llama3.2:3b"
    temperature: float = 0.3


class ContextGenerationResponse(BaseModel):
    """Response from context generation."""

    generated_context: str
    contextual_content: str  # Combined: "Context: ...\n\nContent: ..."
    generation_time: float
    model_used: str
    success: bool
    error_message: Optional[str] = None


class ContextualDocument(BaseModel):
    """Document with contextual chunks."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: DocumentMetadata
    chunks: list[ContextualDocumentChunk] = Field(default_factory=list)
