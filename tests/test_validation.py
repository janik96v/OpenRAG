"""Tests for validation utilities."""

from pathlib import Path

import pytest

from openrag.utils.validation import (
    ValidationError,
    validate_document_id,
    validate_file_path,
    validate_max_results,
    validate_query,
    validate_txt_file,
)


def test_validate_file_path_valid(sample_txt_file: Path):
    """Test validation of valid file path."""
    result = validate_file_path(str(sample_txt_file))
    assert result.exists()
    assert result.is_file()


def test_validate_file_path_not_exists():
    """Test validation of non-existent file."""
    with pytest.raises(ValidationError, match="File not found"):
        validate_file_path("/nonexistent/file.txt")


def test_validate_file_path_traversal():
    """Test path traversal prevention."""
    with pytest.raises(ValidationError, match="Path traversal"):
        validate_file_path("../../../etc/passwd")


def test_validate_txt_file_valid(sample_txt_file: Path):
    """Test validation of valid .txt file."""
    result = validate_txt_file(str(sample_txt_file))
    assert result.suffix == ".txt"


def test_validate_txt_file_wrong_extension(temp_dir: Path):
    """Test validation fails for non-.txt files."""
    wrong_file = temp_dir / "test.pdf"
    wrong_file.touch()

    with pytest.raises(ValidationError, match="Only .txt files supported"):
        validate_txt_file(str(wrong_file))


def test_validate_query_valid():
    """Test validation of valid query."""
    query = "What is machine learning?"
    result = validate_query(query)
    assert result == query


def test_validate_query_empty():
    """Test validation fails for empty query."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_query("")


def test_validate_query_whitespace_only():
    """Test validation fails for whitespace-only query."""
    with pytest.raises(ValidationError, match="cannot be only whitespace"):
        validate_query("   ")


def test_validate_query_too_long():
    """Test validation fails for overly long query."""
    long_query = "word " * 500
    with pytest.raises(ValidationError, match="too long"):
        validate_query(long_query, max_length=100)


def test_validate_max_results_valid():
    """Test validation of valid max_results."""
    assert validate_max_results(5) == 5
    assert validate_max_results(100) == 100


def test_validate_max_results_too_small():
    """Test validation fails for max_results < 1."""
    with pytest.raises(ValidationError, match="must be at least 1"):
        validate_max_results(0)


def test_validate_max_results_too_large():
    """Test validation fails for max_results > limit."""
    with pytest.raises(ValidationError, match="too large"):
        validate_max_results(200, max_allowed=100)


def test_validate_document_id_valid():
    """Test validation of valid document ID."""
    doc_id = "123e4567-e89b-12d3-a456-426614174000"
    result = validate_document_id(doc_id)
    assert result == doc_id


def test_validate_document_id_empty():
    """Test validation fails for empty document ID."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_document_id("")


def test_validate_document_id_too_long():
    """Test validation fails for overly long document ID."""
    long_id = "x" * 100
    with pytest.raises(ValidationError, match="too long"):
        validate_document_id(long_id)
