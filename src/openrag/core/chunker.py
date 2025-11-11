"""Text chunking with token-based splitting using tiktoken."""

from typing import Any

import tiktoken

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TextChunker:
    """
    Splits text into overlapping chunks using recursive character splitting.

    Uses tiktoken for accurate token counting and respects semantic boundaries
    where possible (paragraphs, sentences, etc.).
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        separators: list[str] | None = None,
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap size in tokens between chunks
            separators: Ordered list of separators to try (default: semantic hierarchy)

        Example:
            >>> chunker = TextChunker(chunk_size=400, chunk_overlap=60)
            >>> chunks = chunker.chunk_text("Long document text...")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators in order of semantic importance
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        # Initialize tiktoken encoder (cl100k_base is used by GPT-4, GPT-3.5-turbo)
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding: {e}")
            raise

        logger.info(
            f"Initialized TextChunker (chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, separators={len(self.separators)})"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks using recursive strategy.

        Algorithm:
        1. Try to split on semantic boundaries (paragraphs, sentences, etc.)
        2. Use tiktoken for accurate token counting
        3. Create overlapping chunks to maintain context
        4. Respect chunk size limits

        Args:
            text: Text to chunk

        Returns:
            List of text chunks

        Example:
            >>> chunker = TextChunker(chunk_size=100, chunk_overlap=20)
            >>> chunks = chunker.chunk_text("Long text...")
            >>> len(chunks)
            5
        """
        if not text or not text.strip():
            return []

        chunks: list[str] = []
        current_chunks = self._split_text_recursive(text, self.separators)

        # Merge small chunks and split large ones
        chunks = self._merge_and_split_chunks(current_chunks)

        logger.debug(
            f"Created {len(chunks)} chunks from text of {len(text)} characters "
            f"({self.count_tokens(text)} tokens)"
        )

        return chunks

    def _split_text_recursive(self, text: str, separators: list[str], depth: int = 0) -> list[str]:
        """
        Recursively split text using separator hierarchy.

        Args:
            text: Text to split
            separators: Remaining separators to try
            depth: Current recursion depth (for logging)

        Returns:
            List of text segments
        """
        if not separators:
            # Base case: no more separators, return text as is
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            # Split on current separator
            splits = text.split(separator)
        else:
            # Empty separator means split into characters
            splits = list(text)

        # Recursively process each split if needed
        final_splits: list[str] = []

        for i, split in enumerate(splits):
            if not split:
                continue

            # Re-add separator except for last split
            if i < len(splits) - 1 and separator:
                split = split + separator

            # Check if this split is still too large
            token_count = self.count_tokens(split)

            if token_count > self.chunk_size and remaining_separators:
                # Recursively split further
                subsplits = self._split_text_recursive(split, remaining_separators, depth + 1)
                final_splits.extend(subsplits)
            else:
                final_splits.append(split)

        return final_splits

    def _merge_and_split_chunks(self, splits: list[str]) -> list[str]:
        """
        Merge small splits and ensure chunks don't exceed chunk_size.

        Args:
            splits: Initial text splits

        Returns:
            List of properly sized chunks with overlap
        """
        if not splits:
            return []

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for split in splits:
            split_tokens = self.count_tokens(split)

            # If adding this split would exceed chunk_size, finalize current chunk
            if current_tokens + split_tokens > self.chunk_size and current_chunk:
                # Join current chunk and add to chunks
                chunk_text = "".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Start new chunk with overlap
                # Calculate how much of current chunk to keep for overlap
                overlap_chunks = self._get_overlap_content(current_chunk, current_tokens)
                current_chunk = overlap_chunks
                current_tokens = sum(self.count_tokens(c) for c in current_chunk)

            # Add split to current chunk
            current_chunk.append(split)
            current_tokens += split_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = "".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _get_overlap_content(self, chunks: list[str], total_tokens: int) -> list[str]:
        """
        Get content for overlap from end of chunks.

        Args:
            chunks: Current chunk segments
            total_tokens: Total tokens in chunks

        Returns:
            List of chunks to keep for overlap
        """
        if total_tokens <= self.chunk_overlap:
            return chunks

        # Take chunks from the end until we have enough overlap
        overlap_chunks: list[str] = []
        overlap_tokens = 0

        for chunk in reversed(chunks):
            chunk_tokens = self.count_tokens(chunk)
            if overlap_tokens + chunk_tokens > self.chunk_overlap:
                break
            overlap_chunks.insert(0, chunk)
            overlap_tokens += chunk_tokens

        return overlap_chunks

    def get_chunk_metadata(self, chunks: list[str]) -> list[dict[str, Any]]:
        """
        Get metadata for each chunk.

        Args:
            chunks: List of text chunks

        Returns:
            List of metadata dictionaries

        Example:
            >>> chunker = TextChunker()
            >>> chunks = chunker.chunk_text("Some text")
            >>> metadata = chunker.get_chunk_metadata(chunks)
            >>> metadata[0]['token_count']
            42
        """
        metadata = []

        for i, chunk in enumerate(chunks):
            metadata.append(
                {
                    "chunk_index": i,
                    "token_count": self.count_tokens(chunk),
                    "char_count": len(chunk),
                }
            )

        return metadata
