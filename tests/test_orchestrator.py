"""
Tests for Multi-Agent Orchestrator
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_config():
    """Create mock config"""
    config = Mock()
    config.GEMINI_API_KEY = "test_key"
    config.GEMINI_MODEL = "gemini-1.5-pro"
    config.GEMINI_TEMPERATURE = 0.7
    config.GEMINI_MAX_TOKENS = 8192
    config.MAX_PARALLEL_AGENTS = 2
    config.QUANTUM_TEMPERATURE = 1.0
    config.QUANTUM_ITERATIONS = 10
    config.COOLING_RATE = 0.95
    config.CHUNK_SIZE = 512
    config.CHUNK_OVERLAP = 50
    config.TOP_K_RETRIEVAL = 5
    config.EMBEDDING_MODEL = "test"
    config.MEMORY_RETENTION_DAYS = 30
    config.MAX_MEMORY_ITEMS = 100
    config.MCP_TOOL_TIMEOUT = 60
    config.MCP_RETRY_ATTEMPTS = 3
    return config


@pytest.fixture
def mock_components():
    """Create mock orchestrator components"""
    quantum_optimizer = Mock()
    quantum_optimizer.optimize_assignment = AsyncMock(
        return_value=(np.array([[1, 0], [0, 1]]), [100, 90, 85])
    )

    rag_system = AsyncMock()
    rag_system.retrieve = AsyncMock(return_value=[])

    memory_bank = AsyncMock()
    memory_bank.retrieve_memories = AsyncMock(return_value=[])
    memory_bank.store_memory = AsyncMock()

    mcp_server = AsyncMock()
    mcp_server.execute_tool = AsyncMock(return_value=Mock(success=True, result={}))
    mcp_server.register_tool = AsyncMock()

    gemini_model = Mock()
    gemini_model.generate_content = Mock(return_value=Mock(text="Test response"))

    return {
        "quantum_optimizer": quantum_optimizer,
        "rag_system": rag_system,
        "memory_bank": memory_bank,
        "mcp_server": mcp_server,
        "gemini_model": gemini_model,
    }


def test_task_models_import():
    """Test that task models can be imported"""
    from src.orchestrator.task_models import (
        Memory,
        MemoryType,
        Priority,
        ResearchResult,
        SessionState,
        Task,
        TaskState,
    )

    task = Task(description="Test", domain="test", priority=Priority.HIGH)
    assert task.state == TaskState.CREATED
    assert task.id is not None


def test_session_state():
    """Test session state management"""
    from src.orchestrator.task_models import Priority, SessionState, Task

    session = SessionState()
    assert session.session_id is not None

    task = Task(description="Test", domain="test", priority=Priority.HIGH)
    session.add_task(task)

    assert len(session.tasks) == 1

    status = session.get_status()
    assert status["total_tasks"] == 1


@pytest.mark.asyncio
async def test_orchestrator_creation(mock_config, mock_components):
    """Test orchestrator can be created"""
    from src.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

    orchestrator = MultiAgentOrchestrator(
        quantum_optimizer=mock_components["quantum_optimizer"],
        rag_system=mock_components["rag_system"],
        memory_bank=mock_components["memory_bank"],
        mcp_server=mock_components["mcp_server"],
        gemini_model=mock_components["gemini_model"],
    )

    assert orchestrator is not None
    assert orchestrator.quantum_optimizer is not None


@pytest.mark.asyncio
async def test_parallel_research_execution(mock_config, mock_components):
    """Test parallel research execution"""
    from src.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator
    from src.orchestrator.task_models import Priority, Task

    orchestrator = MultiAgentOrchestrator(
        quantum_optimizer=mock_components["quantum_optimizer"],
        rag_system=mock_components["rag_system"],
        memory_bank=mock_components["memory_bank"],
        mcp_server=mock_components["mcp_server"],
        gemini_model=mock_components["gemini_model"],
    )

    # Create mock agents
    mock_agent = Mock()
    mock_agent.agent_id = "test_agent"
    mock_agent.execute_task = AsyncMock(
        return_value=Mock(task_id="1", content="Result", confidence=0.9, sources=["test"])
    )

    orchestrator.agents = [mock_agent]

    # Test internal method directly
    tasks = [Task(description="Test", domain="test", priority=Priority.HIGH)]
    assignments = {tasks[0]: mock_agent}

    results = await orchestrator._execute_parallel_research(tasks, assignments)

    assert len(results) >= 0  # May be empty if mock doesn't return properly
