"""Background task manager with automatic cleanup to prevent task garbage collection."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """
    Manages background tasks with automatic cleanup.

    CRITICAL: Python's event loop only keeps weak references to tasks, which means
    tasks without external references can be garbage collected, causing the
    "disappearing task" bug. This manager maintains strong references using a set
    and automatically discards them when complete.

    Based on Python's official documentation pattern:
    https://docs.python.org/3/library/asyncio-task.html
    """

    def __init__(self):
        """Initialize the task manager with an empty task set."""
        self._tasks: set[asyncio.Task] = set()
        logger.info("Initialized BackgroundTaskManager")

    def create_task(self, coro, *, name: Optional[str] = None) -> asyncio.Task:
        """
        Create a task with automatic cleanup to prevent garbage collection.

        This uses the set + discard pattern from Python's official documentation:
        1. Create the task
        2. Add to set (strong reference prevents GC)
        3. Add done callback to discard from set when complete
        4. Add exception logging callback

        Args:
            coro: Coroutine to execute in the background
            name: Optional name for the task (useful for debugging)

        Returns:
            The created task
        """
        # Create task with optional name
        task = asyncio.create_task(coro, name=name)

        # Add to set to prevent garbage collection (CRITICAL!)
        self._tasks.add(task)

        # Remove from set when done (prevents memory leak)
        task.add_done_callback(self._tasks.discard)

        # Log any exceptions from background tasks
        task.add_done_callback(self._log_task_exception)

        logger.debug(f"Created background task: {name or task.get_name()}")

        return task

    def _log_task_exception(self, task: asyncio.Task) -> None:
        """
        Callback to log any exception from a completed task.

        Args:
            task: The completed task to check for exceptions
        """
        try:
            # Calling exception() retrieves the exception if task failed
            # This will raise if the task was successful
            task.result()
        except asyncio.CancelledError:
            # Task cancellation is normal, don't log as error
            logger.debug(f"Background task cancelled: {task.get_name()}")
        except Exception as e:
            # Log unexpected exceptions from background tasks
            logger.error(f"Background task failed: {task.get_name()}", exc_info=True)

    def cancel_all(self) -> None:
        """Cancel all pending background tasks."""
        logger.info(f"Cancelling {len(self._tasks)} background tasks")
        for task in self._tasks:
            task.cancel()

    @property
    def task_count(self) -> int:
        """
        Get the number of currently running background tasks.

        Returns:
            Number of active tasks
        """
        return len(self._tasks)
