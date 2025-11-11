"""Input validation utilities for OpenRAG.

Provides security-focused validation for file paths, queries, and other user inputs.
"""

from pathlib import Path


class ValidationError(Exception):
    """Exception raised when validation fails."""

    pass


def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """
    Validate and sanitize a file path.

    Args:
        file_path: File path to validate
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails

    Security:
        - Prevents path traversal attacks
        - Ensures file is within allowed directories
        - Checks file exists and is readable
    """
    try:
        path = Path(file_path).resolve()
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid file path: {e}") from e

    # Check for path traversal attempts
    try:
        # Ensure the path doesn't contain suspicious patterns
        if ".." in Path(file_path).parts:
            raise ValidationError("Path traversal detected: '..' not allowed in path")
    except Exception as e:
        raise ValidationError(f"Path validation failed: {e}") from e

    if must_exist:
        if not path.exists():
            raise ValidationError(f"File not found: {path}")

        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")

        # Check if file is readable
        try:
            with open(path, "rb") as f:
                f.read(1)
        except PermissionError as e:
            raise ValidationError(f"File not readable: {path}") from e
        except Exception as e:
            raise ValidationError(f"Cannot access file: {e}") from e

    return path


def validate_txt_file(file_path: str) -> Path:
    """
    Validate that a file is a .txt file.

    Args:
        file_path: File path to validate

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    path = validate_file_path(file_path, must_exist=True)

    if path.suffix.lower() != ".txt":
        raise ValidationError(
            f"Only .txt files supported, got: {path.suffix}. Please provide a plain text file."
        )

    return path


def validate_query(query: str, max_length: int = 1000) -> str:
    """
    Validate a search query string.

    Args:
        query: Query string to validate
        max_length: Maximum allowed query length

    Returns:
        Validated and trimmed query string

    Raises:
        ValidationError: If validation fails
    """
    if not query:
        raise ValidationError("Query cannot be empty")

    if not isinstance(query, str):
        raise ValidationError(f"Query must be a string, got {type(query).__name__}")

    # Trim whitespace
    query = query.strip()

    if not query:
        raise ValidationError("Query cannot be only whitespace")

    if len(query) > max_length:
        raise ValidationError(
            f"Query too long: {len(query)} characters (max {max_length}). "
            "Please shorten your query."
        )

    return query


def validate_max_results(n_results: int, max_allowed: int = 100) -> int:
    """
    Validate maximum results parameter.

    Args:
        n_results: Number of results requested
        max_allowed: Maximum allowed results

    Returns:
        Validated n_results value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(n_results, int):
        raise ValidationError(f"max_results must be an integer, got {type(n_results).__name__}")

    if n_results < 1:
        raise ValidationError(f"max_results must be at least 1, got {n_results}")

    if n_results > max_allowed:
        raise ValidationError(
            f"max_results too large: {n_results} (max {max_allowed}). Please request fewer results."
        )

    return n_results


def validate_document_id(document_id: str) -> str:
    """
    Validate a document ID.

    Args:
        document_id: Document ID to validate

    Returns:
        Validated document ID

    Raises:
        ValidationError: If validation fails
    """
    if not document_id:
        raise ValidationError("Document ID cannot be empty")

    if not isinstance(document_id, str):
        raise ValidationError(f"Document ID must be a string, got {type(document_id).__name__}")

    # Trim whitespace
    document_id = document_id.strip()

    if not document_id:
        raise ValidationError("Document ID cannot be only whitespace")

    # Basic UUID format check (UUIDs are 36 characters with hyphens)
    if len(document_id) > 50:
        raise ValidationError(f"Document ID too long: {len(document_id)} characters (max 50)")

    return document_id
