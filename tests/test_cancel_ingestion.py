"""Tests for cancel_ingestion_tool and BackgroundTaskManager.cancel_by_prefix."""

import asyncio
from unittest.mock import MagicMock

import pytest

from openrag.tools.manage import cancel_ingestion_tool
from openrag.utils.async_tasks import BackgroundTaskManager


# ---------------------------------------------------------------------------
# BackgroundTaskManager.cancel_by_prefix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_by_prefix_cancels_matching_tasks():
    """Tasks whose names start with the given prefix are cancelled."""
    manager = BackgroundTaskManager()

    async def _noop():
        await asyncio.sleep(60)

    t1 = manager.create_task(_noop(), name="contextual_doc1")
    t2 = manager.create_task(_noop(), name="contextual_doc2")
    t3 = manager.create_task(_noop(), name="graph_doc1")

    cancelled = manager.cancel_by_prefix("contextual_")

    assert cancelled == 2
    assert t1.cancelled() or t1.cancelling() > 0
    assert t2.cancelled() or t2.cancelling() > 0
    # graph task must NOT be cancelled
    assert not t3.cancelled()
    assert t3.cancelling() == 0

    # Cleanup
    t3.cancel()
    await asyncio.gather(t1, t2, t3, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_by_prefix_no_matching_tasks():
    """Returns 0 when no tasks match the prefix."""
    manager = BackgroundTaskManager()

    async def _noop():
        await asyncio.sleep(60)

    t = manager.create_task(_noop(), name="graph_doc1")

    cancelled = manager.cancel_by_prefix("contextual_")

    assert cancelled == 0

    # Cleanup
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_by_prefix_skips_done_tasks():
    """Completed tasks are not counted as cancelled."""
    manager = BackgroundTaskManager()

    async def _fast():
        return

    t = manager.create_task(_fast(), name="contextual_doc1")
    await asyncio.sleep(0)  # Let the task complete

    cancelled = manager.cancel_by_prefix("contextual_")

    assert cancelled == 0


# ---------------------------------------------------------------------------
# cancel_ingestion_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_ingestion_tool_contextual():
    """cancel_ingestion_tool cancels contextual tasks and returns count."""
    mock_manager = MagicMock(spec=BackgroundTaskManager)
    mock_manager.cancel_by_prefix.return_value = 2

    result = await cancel_ingestion_tool(rag_type="contextual", task_manager=mock_manager)

    mock_manager.cancel_by_prefix.assert_called_once_with("contextual_")
    assert result["status"] == "success"
    assert result["rag_type"] == "contextual"
    assert result["tasks_cancelled"] == 2


@pytest.mark.asyncio
async def test_cancel_ingestion_tool_graph():
    """cancel_ingestion_tool cancels graph tasks and returns count."""
    mock_manager = MagicMock(spec=BackgroundTaskManager)
    mock_manager.cancel_by_prefix.return_value = 1

    result = await cancel_ingestion_tool(rag_type="graph", task_manager=mock_manager)

    mock_manager.cancel_by_prefix.assert_called_once_with("graph_")
    assert result["status"] == "success"
    assert result["rag_type"] == "graph"
    assert result["tasks_cancelled"] == 1


@pytest.mark.asyncio
async def test_cancel_ingestion_tool_invalid_rag_type():
    """Returns validation error for unsupported rag_type values."""
    mock_manager = MagicMock(spec=BackgroundTaskManager)

    result = await cancel_ingestion_tool(rag_type="traditional", task_manager=mock_manager)

    assert result["status"] == "error"
    assert result["error"] == "validation_error"
    mock_manager.cancel_by_prefix.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_ingestion_tool_no_task_manager():
    """Returns not_available error when task_manager is None."""
    result = await cancel_ingestion_tool(rag_type="contextual", task_manager=None)

    assert result["status"] == "error"
    assert result["error"] == "not_available"


@pytest.mark.asyncio
async def test_cancel_ingestion_tool_zero_tasks():
    """Returns success with tasks_cancelled=0 when nothing is running."""
    mock_manager = MagicMock(spec=BackgroundTaskManager)
    mock_manager.cancel_by_prefix.return_value = 0

    result = await cancel_ingestion_tool(rag_type="graph", task_manager=mock_manager)

    assert result["status"] == "success"
    assert result["tasks_cancelled"] == 0
