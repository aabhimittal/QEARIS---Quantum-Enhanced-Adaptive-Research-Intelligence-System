"""
Tests for Multi-Agent Orchestrator
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator
from src.config import settings


@pytest.fixture
async def orchestrator():
    """Create orchestrator instance"""
    with patch('src.orchestrator.multi_agent_orchestrator.genai') as mock_genai:
        mock_genai.configure = Mock()
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(
            return_value=Mock(text="Test", candidates=[])
        )
        mock_genai.GenerativeModel.return_value = mock_model
        
        orch = await MultiAgentOrchestrator.create(settings)
        return orch


@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initializes correctly"""
    assert orchestrator is not None
    assert len(orchestrator.agents) > 0
    assert orchestrator.quantum_optimizer is not None
    assert orchestrator.rag_system is not None
    assert orchestrator.memory_bank is not None
    assert orchestrator.mcp_server is not None


@pytest.mark.asyncio
async def test_workflow_execution(orchestrator):
    """Test complete workflow execution"""
    with patch.object(orchestrator, '_execute_parallel_research') as mock_parallel:
        with patch.object(orchestrator, '_execute_sequential_validation') as mock_validation:
            with patch.object(orchestrator, '_execute_synthesis_loop') as mock_synthesis:
                # Setup mocks
                mock_parallel.return_value = [
                    Mock(task_id="1", content="Result 1", confidence=0.9, sources=["src1"])
                ]
                mock_validation.return_value = [
                    Mock(validated=True, confidence=0.85)
                ]
                mock_synthesis.return_value = Mock(
                    content="Final report",
                    confidence=0.92,
                    sources=["src1", "src2"]
                )
                
                # Execute workflow
                results = await orchestrator.execute_research_workflow(
                    research_query="Test query",
                    domains=["test"],
                    max_agents=2
                )
                
                assert results is not None
                assert "session_id" in results
                assert "final_report" in results
                assert results["final_confidence"] > 0


@pytest.mark.asyncio
async def test_session_management(orchestrator):
    """Test session creation and retrieval"""
    # Execute workflow to create session
    with patch.object(orchestrator, '_execute_parallel_research', return_value=[]):
        with patch.object(orchestrator, '_execute_sequential_validation', return_value=[]):
            with patch.object(orchestrator, '_execute_synthesis_loop') as mock_synthesis:
                mock_synthesis.return_value = Mock(content="Test", confidence=0.8, sources=[])
                
                results = await orchestrator.execute_research_workflow(
                    research_query="Test",
                    domains=["test"]
                )
                
                session_id = results["session_id"]
                
                # Check session exists
                assert session_id in orchestrator.sessions


@pytest.mark.asyncio
async def test_quantum_optimization_integration(orchestrator):
    """Test quantum optimizer is used in workflow"""
    with patch.object(orchestrator.quantum_optimizer, 'optimize_assignment') as mock_optimize:
        mock_optimize.return_value = (
            [[1, 0], [0, 1]],  # Assignment matrix
            [100, 90, 85, 82, 80]  # Energy history
        )
        
        with patch.object(orchestrator, '_execute_parallel_research', return_value=[]):
            with patch.object(orchestrator, '_execute_sequential_validation', return_value=[]):
                with patch.object(orchestrator, '_execute_synthesis_loop') as mock_synthesis:
                    mock_synthesis.return_value = Mock(content="Test", confidence=0.8, sources=[])
                    
                    results = await orchestrator.execute_research_workflow(
                        research_query="Test",
                        domains=["test1", "test2"]
                    )
                    
                    # Verify optimizer was called
                    assert mock_optimize.called
                    assert "quantum_optimization" in results
