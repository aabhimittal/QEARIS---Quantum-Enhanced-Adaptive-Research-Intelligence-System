"""
Tests for Agent System
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.agents.research_agent import ResearchAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.orchestrator.task_models import Task, Priority


@pytest.fixture
def mock_gemini_model():
    """Mock Gemini model"""
    model = Mock()
    model.generate_content_async = AsyncMock(
        return_value=Mock(text="Test response", candidates=[])
    )
    return model


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server"""
    server = AsyncMock()
    server.execute_tool = AsyncMock(
        return_value=Mock(success=True, result={"data": "test"})
    )
    return server


@pytest.fixture
def mock_rag_system():
    """Mock RAG system"""
    system = AsyncMock()
    system.retrieve = AsyncMock(
        return_value=[
            {
                "content": "Test content",
                "similarity": 0.9,
                "metadata": {"source": "test.txt"}
            }
        ]
    )
    return system


@pytest.fixture
def mock_memory_bank():
    """Mock memory bank"""
    bank = AsyncMock()
    bank.retrieve_memories = AsyncMock(return_value=[])
    bank.store_memory = AsyncMock()
    return bank


@pytest.mark.asyncio
async def test_research_agent_execution(
    mock_gemini_model,
    mock_mcp_server,
    mock_rag_system,
    mock_memory_bank
):
    """Test research agent task execution"""
    agent = ResearchAgent(
        agent_id="test_researcher",
        gemini_model=mock_gemini_model,
        mcp_server=mock_mcp_server,
        rag_system=mock_rag_system,
        memory_bank=mock_memory_bank
    )
    
    task = Task(
        description="Test research task",
        domain="test",
        priority=Priority.HIGH
    )
    
    result = await agent.execute_task(task)
    
    assert result is not None
    assert result.task_id == task.id
    assert result.confidence > 0
    assert len(result.sources) > 0


@pytest.mark.asyncio
async def test_validation_agent_execution(mock_gemini_model, mock_mcp_server):
    """Test validation agent"""
    agent = ValidationAgent(
        agent_id="test_validator",
        gemini_model=mock_gemini_model,
        mcp_server=mock_mcp_server
    )
    
    # Mock research result
    research_result = Mock(
        task_id="task_123",
        content="Test content",
        sources=["source1", "source2"],
        confidence=0.9
    )
    
    validation = await agent.execute_task(research_result)
    
    assert validation is not None
    assert hasattr(validation, 'validated')


@pytest.mark.asyncio
async def test_synthesis_agent_execution(mock_gemini_model, mock_mcp_server):
    """Test synthesis agent"""
    # Mock the generate_content to return proper response
    mock_gemini_model.generate_content = Mock(
        return_value=Mock(text="This is a synthesized result with good structure.\n\n1. Point one\n2. Point two\n\nConclusion reached.")
    )
    
    agent = SynthesisAgent(
        agent_id="test_synthesizer",
        gemini_model=mock_gemini_model,
        mcp_server=mock_mcp_server
    )
    
    # Mock research results with proper structure
    from src.orchestrator.task_models import ResearchResult
    research_results = [
        ResearchResult(
            task_id="task_1",
            agent_id="agent_1",
            content="Result 1",
            sources=["src1"],
            confidence=0.9
        ),
        ResearchResult(
            task_id="task_2",
            agent_id="agent_2",
            content="Result 2",
            sources=["src2"],
            confidence=0.85
        )
    ]
    
    synthesis = await agent.execute_task({
        "research_results": research_results,
        "validation_results": [],
        "iteration": 0
    })
    
    assert synthesis is not None
    assert synthesis.confidence > 0


@pytest.mark.asyncio
async def test_agent_tool_usage(mock_gemini_model, mock_mcp_server, mock_rag_system, mock_memory_bank):
    """Test agent uses MCP tools correctly"""
    agent = ResearchAgent(
        agent_id="test_agent",
        gemini_model=mock_gemini_model,
        mcp_server=mock_mcp_server,
        rag_system=mock_rag_system,
        memory_bank=mock_memory_bank
    )
    
    # Use tool
    result = await agent.use_tool("search_knowledge_base", {"query": "test"})
    
    assert mock_mcp_server.execute_tool.called
    assert result is not None


@pytest.mark.asyncio
async def test_agent_memory_storage(
    mock_gemini_model,
    mock_mcp_server,
    mock_rag_system,
    mock_memory_bank
):
    """Test agent stores experiences in memory"""
    agent = ResearchAgent(
        agent_id="test_agent",
        gemini_model=mock_gemini_model,
        mcp_server=mock_mcp_server,
        rag_system=mock_rag_system,
        memory_bank=mock_memory_bank
    )
    
    await agent.store_experience("Test experience", importance=0.8)
    
    assert mock_memory_bank.store_memory.called


def test_agent_availability():
    """Test agent availability tracking"""
    from src.agents.base_agent import Agent, AgentType
    
    agent = Agent(
        name="Test",
        agent_type=AgentType.RESEARCHER,
        specialization="test",
        max_concurrent_tasks=2
    )
    
    # Initially available
    assert agent.is_available()
    
    # Assign tasks
    agent.assign_task("task1")
    assert agent.is_available()
    
    agent.assign_task("task2")
    assert not agent.is_available()
    
    # Complete task
    agent.complete_task("task1")
    assert agent.is_available()
