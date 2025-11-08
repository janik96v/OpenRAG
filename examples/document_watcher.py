"""Document auto-ingestion service for monitoring directories and automatically processing new documents."""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from ..models.contextual_schemas import RAGType
from .rag_engine import RAGEngine

logger = logging.getLogger(__name__)


class DocumentWatcherError(Exception):
    """Exception raised during document watcher operations."""

    pass


class DocumentWatcher:
    """
    Service for automatically watching directories and ingesting new documents.

    Monitors specified directories for new documents and automatically processes them
    through the RAG engine. Includes duplicate detection to avoid re-processing.
    """

    def __init__(
        self,
        rag_engine: RAGEngine,
        watched_directories: list[Path],
        check_interval: int = 5,
        supported_extensions: set[str] | None = None,
        tracking_file_path: Path | None = None,
        auto_ingest_rag_type: str = "traditional",
    ):
        """
        Initialize the document watcher.

        Args:
            rag_engine: RAG engine instance for document processing
            watched_directories: List of directories to monitor
            check_interval: How often to check for new files (seconds)
            supported_extensions: Set of supported file extensions (defaults to common document types)
            tracking_file_path: Path to the file tracking JSON file (defaults to chromadb_path/processed_files.json)
            auto_ingest_rag_type: RAG type for auto-ingestion ("traditional" or "contextual")
        """
        self.rag_engine = rag_engine
        self.watched_directories = watched_directories
        self.check_interval = check_interval
        self.supported_extensions = supported_extensions or {".pdf", ".txt", ".md", ".markdown"}

        # Convert string RAG type to enum
        if auto_ingest_rag_type.lower() == "contextual":
            self.auto_ingest_rag_type = RAGType.CONTEXTUAL
        else:
            self.auto_ingest_rag_type = RAGType.TRADITIONAL

        # Setup persistent file tracking
        if tracking_file_path is None:
            tracking_file_path = Path("data/processed_files.json")
        self.tracking_file_path = Path(tracking_file_path)
        self.tracking_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup error file tracking in ChromaDB persist directory
        from config import Settings
        settings = Settings()
        chromadb_path = settings.get_chromadb_path()
        chromadb_path.mkdir(parents=True, exist_ok=True)
        self.error_file_path = chromadb_path / "files_with_error.json"

        # Track processed files to avoid duplicates (loaded from persistent storage)
        self._processed_files: dict[str, dict[str, Any]] = self._load_processed_files()
        self._error_files: dict[str, dict[str, Any]] = self._load_error_files()
        self._is_running = False
        self._task: asyncio.Task[None] | None = None

        # Clean up orphaned entries on initialization
        self.cleanup_orphaned_entries()

        # Create watched directories if they don't exist
        for directory in self.watched_directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Watching directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create/access directory {directory}: {e}")

    def _load_processed_files(self) -> dict[str, dict[str, Any]]:
        """
        Load processed file tracking data from persistent storage.

        Returns:
            Dictionary of processed files with their metadata
        """
        try:
            if not self.tracking_file_path.exists():
                logger.info(f"Processed files tracking file not found at {self.tracking_file_path}, starting fresh")
                return {}

            with open(self.tracking_file_path, encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}

                data: dict[str, dict[str, Any]] = json.loads(content)
                logger.info(f"Loaded {len(data)} processed files from {self.tracking_file_path}")
                return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in processed files tracking file {self.tracking_file_path}: {e}")
            # Backup corrupted file
            backup_path = self.tracking_file_path.with_suffix('.json.backup')
            self.tracking_file_path.rename(backup_path)
            logger.info(f"Corrupted file backed up to {backup_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load processed files from {self.tracking_file_path}: {e}")
            return {}

    def _save_processed_files(self) -> None:
        """
        Save processed file tracking data to persistent storage.
        """
        try:
            with open(self.tracking_file_path, "w", encoding="utf-8") as f:
                json.dump(self._processed_files, f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f"Saved {len(self._processed_files)} processed files to {self.tracking_file_path}")
        except Exception as e:
            logger.error(f"Failed to save processed files to {self.tracking_file_path}: {e}")

    def _load_error_files(self) -> dict[str, dict[str, Any]]:
        """
        Load error file tracking data from persistent storage.

        Returns:
            Dictionary of error files with their metadata
        """
        try:
            if not self.error_file_path.exists():
                logger.info(f"Error files tracking file not found at {self.error_file_path}, starting fresh")
                return {}

            with open(self.error_file_path, encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}

                data: dict[str, dict[str, Any]] = json.loads(content)
                logger.info(f"Loaded {len(data)} error files from {self.error_file_path}")
                return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in error files tracking file {self.error_file_path}: {e}")
            # Backup corrupted file
            backup_path = self.error_file_path.with_suffix('.json.backup')
            self.error_file_path.rename(backup_path)
            logger.info(f"Corrupted error file backed up to {backup_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load error files from {self.error_file_path}: {e}")
            return {}

    def _save_error_files(self) -> None:
        """
        Save error file tracking data to persistent storage.
        """
        try:
            with open(self.error_file_path, "w", encoding="utf-8") as f:
                json.dump(self._error_files, f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f"Saved {len(self._error_files)} error files to {self.error_file_path}")
        except Exception as e:
            logger.error(f"Failed to save error files to {self.error_file_path}: {e}")

    def _add_error_file(self, file_path: Path, error_message: str) -> None:
        """
        Add a file to the error tracking list.

        Args:
            file_path: Path to the file that failed processing
            error_message: Error message describing why processing failed
        """
        try:
            file_key = str(file_path.absolute())

            self._error_files[file_key] = {
                "file_path": str(file_path),
                "error_message": error_message,
                "error_time": time.time(),
                "retry_count": self._error_files.get(file_key, {}).get("retry_count", 0) + 1,
            }

            # Save to persistent storage
            self._save_error_files()
            logger.info(f"Added file to error tracking: {file_path}")
        except Exception as e:
            logger.error(f"Failed to add error file {file_path}: {e}")

    def _is_file_in_error_list(self, file_path: Path) -> bool:
        """
        Check if a file is in the error tracking list.

        Args:
            file_path: Path to the file

        Returns:
            True if file is in error list, False otherwise
        """
        file_key = str(file_path.absolute())
        return file_key in self._error_files

    def _get_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of a file for duplicate detection.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash of the file content
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""

    def _is_file_processed(self, file_path: Path) -> bool:
        """
        Check if a file has already been processed.

        Args:
            file_path: Path to the file

        Returns:
            True if file has been processed, False otherwise
        """
        file_key = str(file_path.absolute())

        if file_key not in self._processed_files:
            return False

        # Check if file has been modified since last processing
        try:
            current_mtime = file_path.stat().st_mtime
            current_size = file_path.stat().st_size
            current_hash = self._get_file_hash(file_path)

            stored_info = self._processed_files[file_key]

            return (
                stored_info.get("mtime") == current_mtime
                and stored_info.get("size") == current_size
                and stored_info.get("hash") == current_hash
            )
        except Exception as e:
            logger.warning(f"Error checking file status for {file_path}: {e}")
            return False

    def _mark_file_processed(self, file_path: Path, document_id: str, rag_type: RAGType) -> None:
        """
        Mark a file as processed to avoid duplicate processing.

        Args:
            file_path: Path to the processed file
            document_id: ID of the processed document
            rag_type: Type of RAG processing used
        """
        try:
            file_stats = file_path.stat()
            file_hash = self._get_file_hash(file_path)

            self._processed_files[str(file_path.absolute())] = {
                "document_id": document_id,
                "mtime": file_stats.st_mtime,
                "size": file_stats.st_size,
                "hash": file_hash,
                "processed_at": time.time(),
                "rag_type": rag_type.value,
            }

            # Save to persistent storage
            self._save_processed_files()
        except Exception as e:
            logger.error(f"Failed to mark file as processed {file_path}: {e}")

    def _scan_directory(self, directory: Path) -> list[Path]:
        """
        Scan a directory for supported document files.

        Args:
            directory: Directory to scan

        Returns:
            List of document files found
        """
        documents: list[Path] = []

        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Directory does not exist or is not accessible: {directory}")
            return documents

        try:
            for file_path in directory.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in self.supported_extensions
                    and not self._is_file_processed(file_path)
                    and not self._is_file_in_error_list(file_path)
                ):
                    documents.append(file_path)

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")

        return documents

    async def _process_document(self, file_path: Path) -> bool:
        """
        Process a single document through the RAG engine.

        Args:
            file_path: Path to the document to process

        Returns:
            True if processing was successful, False otherwise
        """
        try:
            logger.info(f"Auto-ingesting document: {file_path.name} with {self.auto_ingest_rag_type.value} RAG")

            # Process document through RAG engine with configured RAG type
            document = self.rag_engine.ingest_document(file_path, self.auto_ingest_rag_type)

            # Mark as processed with RAG type
            self._mark_file_processed(file_path, document.document_id, self.auto_ingest_rag_type)

            logger.info(f"Successfully auto-ingested: {file_path.name} (ID: {document.document_id})")
            return True

        except Exception as e:
            error_str = str(e)

            # Check if this is a batch size exceeded error
            if ("batch size" in error_str.lower() and
                ("exceeded" in error_str.lower() or "greater than max batch size" in error_str.lower())):
                logger.error(f"Failed to auto-ingest document {file_path}: {error_str}")
                self._add_error_file(file_path, f"Max batch size exceeded: {error_str}")
                logger.info(f"Added {file_path.name} to error files list, continuing with other files")
                return False

            # Handle other errors normally
            logger.error(f"Failed to auto-ingest document {file_path}: {e}")
            return False

    async def _check_directories(self) -> None:
        """Check all watched directories for new documents."""
        all_documents = []

        # Scan all watched directories
        for directory in self.watched_directories:
            documents = self._scan_directory(directory)
            all_documents.extend(documents)

        if all_documents:
            logger.info(f"Found {len(all_documents)} new documents to process")

            # Process documents sequentially to avoid overwhelming the system
            for document_path in all_documents:
                await self._process_document(document_path)
                # Small delay between documents to prevent resource overload
                await asyncio.sleep(0.1)

    async def _watch_loop(self) -> None:
        """Main watching loop that periodically checks for new documents."""
        logger.info(f"Document watcher started, checking every {self.check_interval} seconds")

        while self._is_running:
            try:
                await self._check_directories()
            except Exception as e:
                logger.error(f"Error in document watcher loop: {e}")

            # Wait for next check interval
            await asyncio.sleep(self.check_interval)

        logger.info("Document watcher stopped")

    def start(self) -> None:
        """Start the document watcher."""
        if self._is_running:
            logger.warning("Document watcher is already running")
            return

        if not self.watched_directories:
            logger.warning("No directories to watch, document watcher not started")
            return

        self._is_running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Document watcher started")

    async def stop(self) -> None:
        """Stop the document watcher."""
        if not self._is_running:
            return

        self._is_running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Document watcher stopped")

    def add_directory(self, directory: Path) -> None:
        """
        Add a new directory to watch.

        Args:
            directory: Directory path to add
        """
        if directory not in self.watched_directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.watched_directories.append(directory)
                logger.info(f"Added watch directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to add watch directory {directory}: {e}")
                raise DocumentWatcherError(f"Failed to add watch directory: {e}") from e

    def remove_directory(self, directory: Path) -> None:
        """
        Remove a directory from watching.

        Args:
            directory: Directory path to remove
        """
        if directory in self.watched_directories:
            self.watched_directories.remove(directory)
            logger.info(f"Removed watch directory: {directory}")

    def cleanup_orphaned_entries(self) -> int:
        """
        Remove tracking entries for files that no longer exist.

        Returns:
            Number of orphaned entries removed
        """
        try:
            orphaned_files = []
            for file_path_str in list(self._processed_files.keys()):
                file_path = Path(file_path_str)
                if not file_path.exists():
                    orphaned_files.append(file_path_str)

            for orphaned_file in orphaned_files:
                del self._processed_files[orphaned_file]

            if orphaned_files:
                self._save_processed_files()
                logger.info(f"Cleaned up {len(orphaned_files)} orphaned file tracking entries")

            return len(orphaned_files)
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned entries: {e}")
            return 0

    def get_status(self) -> dict[str, Any]:
        """
        Get the current status of the document watcher.

        Returns:
            Dictionary containing watcher status information
        """
        return {
            "is_running": self._is_running,
            "watched_directories": [str(d) for d in self.watched_directories],
            "check_interval": self.check_interval,
            "supported_extensions": list(self.supported_extensions),
            "processed_files_count": len(self._processed_files),
            "error_files_count": len(self._error_files),
            "tracking_file_path": str(self.tracking_file_path),
            "error_file_path": str(self.error_file_path),
        }

    def force_scan(self) -> None:
        """Force an immediate scan of all watched directories."""
        if not self._is_running:
            logger.warning("Document watcher is not running, cannot force scan")
            return

        # Create a new task to run the scan immediately
        asyncio.create_task(self._check_directories())
        logger.info("Forced directory scan initiated")
